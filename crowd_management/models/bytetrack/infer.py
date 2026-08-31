"""ByteTrack-based motion estimation engine using YOLOv8 + ByteTrack tracker.

Replaces RAFT optical flow. Instead of pixel-level dense flow, this engine:
  1. Detects persons in each frame with YOLOv8
  2. Assigns persistent track IDs via ByteTrack (persist=True)
  3. Computes per-person velocity from track displacement × FPS
  4. Aggregates across all active tracks to produce crowd-level motion metrics

Per-camera YOLO model instances are maintained so that persist=True tracker
state is correctly isolated per video stream.

Output fields match the former RAFTResult interface consumed by Phase2Adapter:
    median_velocity, dominant_direction_deg, flow_consistency,
    active_motion_ratio, motion_confidence
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from crowd_management.ingestion.frame_packet import FramePacket
from crowd_management.models.contracts import ByteTrackResult

logger = logging.getLogger(__name__)

# Minimum displacement (pixels) for a track to be counted as "moving"
_DEFAULT_MOTION_THRESHOLD_PX = 2.0


@dataclass
class _TrackState:
    cx: float
    cy: float


class ByteTrackInference:
    """YOLOv8 + ByteTrack person tracking engine for crowd motion estimation.

    One YOLO model instance is created per camera_id so that each stream's
    ByteTrack state (persist=True) remains independent and correct.
    """

    def __init__(
        self,
        model_path: str = "yolov8l.pt",
        tracker_config: str = "bytetrack.yaml",
        confidence_threshold: float = 0.3,
        motion_threshold_px: float = _DEFAULT_MOTION_THRESHOLD_PX,
        device: str = "cpu",
    ) -> None:
        self._model_path = model_path
        self._tracker_config = tracker_config
        self._confidence_threshold = confidence_threshold
        self._motion_threshold_px = motion_threshold_px
        self._device = device

        # Per-camera YOLO model instances (lazy-initialized)
        self._models: dict[str, Any] = {}
        # Per-camera previous track centers: camera_id -> {track_id -> _TrackState}
        self._prev_centers: dict[str, dict[int, _TrackState]] = {}

        logger.info(
            "ByteTrackInference initialised: model=%s tracker=%s device=%s conf=%.2f",
            model_path, tracker_config, device, confidence_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer(self, packet: FramePacket) -> ByteTrackResult:
        """Run YOLOv8 + ByteTrack on a single frame and return crowd motion metrics.

        Args:
            packet: FramePacket containing the raw BGR frame and metadata.

        Returns:
            ByteTrackResult with aggregated velocity, direction, and confidence.
        """
        t_start = time.perf_counter()

        model = self._get_or_create_model(packet.camera_id)

        # Run YOLOv8 detection + ByteTrack tracking (person class = 0)
        try:
            results = model.track(
                packet.frame,
                persist=True,
                tracker=self._tracker_config,
                classes=[0],              # person only
                conf=self._confidence_threshold,
                verbose=False,
                device=self._device,
            )
        except Exception as e:
            logger.warning("ByteTrack inference failed on %s frame %s: %s", packet.camera_id, packet.frame_id, e)
            return self._empty_result(packet, elapsed_ms=0.0)

        inference_ms = (time.perf_counter() - t_start) * 1000.0

        # Extract tracks
        boxes_ids = self._extract_boxes_ids(results)
        fps = max(packet.stream_fps, 1.0)

        # Compute per-track velocities and directions
        velocities: list[float] = []
        displacement_vectors: list[tuple[float, float]] = []

        prev = self._prev_centers.get(packet.camera_id, {})
        new_centers: dict[int, _TrackState] = {}

        for (x1, y1, x2, y2), track_id in boxes_ids:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            new_centers[track_id] = _TrackState(cx=cx, cy=cy)

            if track_id in prev:
                px, py = prev[track_id].cx, prev[track_id].cy
                dx, dy = cx - px, cy - py
                dist = math.sqrt(dx * dx + dy * dy)
                vel_px_sec = dist * fps
                velocities.append(vel_px_sec)
                if dist > 0:
                    displacement_vectors.append((dx, dy))

        self._prev_centers[packet.camera_id] = new_centers

        # Aggregate metrics
        track_count = len(new_centers)
        result = self._aggregate(
            packet=packet,
            velocities=velocities,
            displacement_vectors=displacement_vectors,
            track_count=track_count,
            inference_ms=inference_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_model(self, camera_id: str) -> Any:
        """Return (or lazy-create) the YOLO model instance for this camera."""
        if camera_id not in self._models:
            try:
                from ultralytics import YOLO  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "ultralytics is required for ByteTrackInference. "
                    "Install it with: pip install ultralytics"
                ) from e

            logger.info("Creating YOLOv8 model instance for camera '%s' (model=%s)", camera_id, self._model_path)
            self._models[camera_id] = YOLO(self._model_path)

        return self._models[camera_id]

    @staticmethod
    def _extract_boxes_ids(results: list) -> list[tuple[tuple[float, float, float, float], int]]:
        """Extract (box_xyxy, track_id) pairs from Ultralytics results."""
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return []

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        return [((float(b[0]), float(b[1]), float(b[2]), float(b[3])), int(tid))
                for b, tid in zip(boxes, ids)]

    def _aggregate(
        self,
        packet: FramePacket,
        velocities: list[float],
        displacement_vectors: list[tuple[float, float]],
        track_count: int,
        inference_ms: float,
    ) -> ByteTrackResult:
        """Aggregate per-track velocities into crowd-level motion metrics."""
        if not velocities:
            return self._empty_result(packet, elapsed_ms=inference_ms)

        arr = np.array(velocities, dtype=np.float32)
        median_vel = float(np.median(arr))
        avg_vel = float(np.mean(arr))

        # Flow consistency: ||Σv|| / Σ||v||  (same formula as former RAFT)
        flow_consistency = 0.0
        dominant_dir_deg = 0.0
        if displacement_vectors:
            sum_dx = sum(d[0] for d in displacement_vectors)
            sum_dy = sum(d[1] for d in displacement_vectors)
            sum_mag = sum(math.sqrt(d[0] ** 2 + d[1] ** 2) for d in displacement_vectors)
            vec_mag = math.sqrt(sum_dx ** 2 + sum_dy ** 2)
            flow_consistency = float(min(1.0, vec_mag / sum_mag)) if sum_mag > 0 else 0.0
            dominant_dir_deg = float(math.atan2(sum_dy, sum_dx) * 180.0 / math.pi % 360.0)

        # Active motion ratio: fraction of tracks with velocity > threshold
        moving = sum(1 for v in velocities if v > self._motion_threshold_px)
        active_motion_ratio = float(moving / len(velocities)) if velocities else 0.0

        # Motion confidence: weighted combo of consistency, active ratio, track count
        track_norm = min(1.0, track_count / 20.0)
        motion_confidence = min(1.0, max(0.0,
            flow_consistency * 0.5
            + active_motion_ratio * 0.3
            + track_norm * 0.2
        ))

        return ByteTrackResult(
            camera_id=packet.camera_id,
            frame_id=packet.frame_id,
            source_frame_index=packet.source_frame_index,
            video_timestamp=packet.video_timestamp,
            wall_clock_timestamp=packet.wall_clock_timestamp,
            track_count=track_count,
            median_velocity=median_vel,
            average_velocity=avg_vel,
            dominant_direction_deg=dominant_dir_deg,
            flow_consistency=flow_consistency,
            active_motion_ratio=active_motion_ratio,
            motion_confidence=motion_confidence,
            inference_time_ms=inference_ms,
            model_name=self._model_path,
            device=self._device,
            result_timestamp=time.time(),
        )

    def _empty_result(self, packet: FramePacket, elapsed_ms: float) -> ByteTrackResult:
        """Return a zero-motion result when no tracks are detected."""
        return ByteTrackResult(
            camera_id=packet.camera_id,
            frame_id=packet.frame_id,
            source_frame_index=packet.source_frame_index,
            video_timestamp=packet.video_timestamp,
            wall_clock_timestamp=packet.wall_clock_timestamp,
            track_count=0,
            median_velocity=0.0,
            average_velocity=0.0,
            dominant_direction_deg=0.0,
            flow_consistency=0.0,
            active_motion_ratio=0.0,
            motion_confidence=0.0,
            inference_time_ms=elapsed_ms,
            model_name=self._model_path,
            device=self._device,
            result_timestamp=time.time(),
        )
