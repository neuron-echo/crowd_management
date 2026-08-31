"""Immutable result contracts for Phase 2 vision models with full provenance tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CSRNetResult:
    """Canonical result packet from CSRNet crowd density estimation."""

    camera_id: str
    frame_id: int
    source_frame_index: int
    video_timestamp: float
    wall_clock_timestamp: float
    crowd_count: float
    density_map: np.ndarray
    density_map_width: int
    density_map_height: int
    input_width: int
    input_height: int
    inference_time_ms: float
    model_name: str
    device: str
    checkpoint_identifier: str = "CSRNet-ShanghaiTech-A"
    result_timestamp: float = 0.0
    frame: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.density_map, np.ndarray):
            raise TypeError("density_map must be a numpy.ndarray")
        density = np.asarray(self.density_map, dtype=np.float32)
        if density.ndim != 2:
            raise ValueError(f"density_map must be 2D array, got shape {density.shape}")
        if density.shape[0] != self.density_map_height or density.shape[1] != self.density_map_width:
            raise ValueError(
                f"density_map shape {density.shape} does not match height {self.density_map_height} and width {self.density_map_width}"
            )
        if not np.isfinite(self.crowd_count):
            raise ValueError(f"crowd_count must be finite, got {self.crowd_count}")
        if self.crowd_count < 0:
            raise ValueError(f"crowd_count cannot be negative, got {self.crowd_count}")

        object.__setattr__(self, "density_map", density.copy())
        if self.frame is not None:
            object.__setattr__(self, "frame", np.asarray(self.frame, dtype=np.uint8).copy())
        object.__setattr__(self, "crowd_count", float(self.crowd_count))
        object.__setattr__(self, "video_timestamp", float(self.video_timestamp))
        object.__setattr__(self, "wall_clock_timestamp", float(self.wall_clock_timestamp))
        object.__setattr__(self, "inference_time_ms", float(self.inference_time_ms))
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "source_frame_index", int(self.source_frame_index))
        object.__setattr__(self, "input_width", int(self.input_width))
        object.__setattr__(self, "input_height", int(self.input_height))
        object.__setattr__(self, "checkpoint_identifier", str(self.checkpoint_identifier))
        if self.result_timestamp <= 0.0:
            object.__setattr__(self, "result_timestamp", time.time())
        else:
            object.__setattr__(self, "result_timestamp", float(self.result_timestamp))


@dataclass(frozen=True)
class ByteTrackResult:
    """Canonical result packet from YOLOv8 + ByteTrack crowd motion estimation.

    Replaces RAFTResult. Provides the same crowd-level motion fields consumed
    by Phase2Adapter and the Decision Engine, derived from per-person track
    displacement rather than dense optical flow.

    Fields:
        camera_id:              Source camera identifier.
        frame_id:               Monotonic frame counter from FrameProducer.
        source_frame_index:     Original frame index in the source video.
        video_timestamp:        Position in the source video (seconds).
        wall_clock_timestamp:   Wall-clock time when the frame was captured.
        track_count:            Number of active tracked persons in this frame.
        median_velocity:        50th-pct velocity (px/sec) across tracked persons.
        average_velocity:       Mean velocity (px/sec) across tracked persons.
        dominant_direction_deg: Crowd movement direction 0–360° (vector sum).
        flow_consistency:       ||Σv|| / Σ||v|| ∈ [0, 1] — 1 = perfectly uniform.
        active_motion_ratio:    Fraction of tracks with velocity > threshold ∈ [0, 1].
        motion_confidence:      Composite confidence score ∈ [0, 1].
        inference_time_ms:      YOLOv8 + ByteTrack inference wall-clock time.
        model_name:             YOLO model weights identifier (e.g. "yolov8l.pt").
        device:                 Inference device ("cpu" or "cuda").
        result_timestamp:       Wall-clock time of result creation.
    """

    camera_id: str
    frame_id: int
    source_frame_index: int
    video_timestamp: float
    wall_clock_timestamp: float
    track_count: int
    median_velocity: float
    average_velocity: float
    dominant_direction_deg: float
    flow_consistency: float
    active_motion_ratio: float
    motion_confidence: float
    inference_time_ms: float
    model_name: str
    device: str
    result_timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.dominant_direction_deg <= 360.0):
            raise ValueError(f"dominant_direction_deg must be in [0, 360], got {self.dominant_direction_deg}")
        if not (0.0 <= self.flow_consistency <= 1.0):
            raise ValueError(f"flow_consistency must be in [0, 1], got {self.flow_consistency}")
        if not (0.0 <= self.active_motion_ratio <= 1.0):
            raise ValueError(f"active_motion_ratio must be in [0, 1], got {self.active_motion_ratio}")
        if not (0.0 <= self.motion_confidence <= 1.0):
            raise ValueError(f"motion_confidence must be in [0, 1], got {self.motion_confidence}")

        object.__setattr__(self, "camera_id", str(self.camera_id))
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "source_frame_index", int(self.source_frame_index))
        object.__setattr__(self, "video_timestamp", float(self.video_timestamp))
        object.__setattr__(self, "wall_clock_timestamp", float(self.wall_clock_timestamp))
        object.__setattr__(self, "track_count", int(self.track_count))
        object.__setattr__(self, "median_velocity", float(self.median_velocity))
        object.__setattr__(self, "average_velocity", float(self.average_velocity))
        object.__setattr__(self, "dominant_direction_deg", float(self.dominant_direction_deg))
        object.__setattr__(self, "flow_consistency", float(self.flow_consistency))
        object.__setattr__(self, "active_motion_ratio", float(self.active_motion_ratio))
        object.__setattr__(self, "motion_confidence", float(self.motion_confidence))
        object.__setattr__(self, "inference_time_ms", float(self.inference_time_ms))
        if self.result_timestamp <= 0.0:
            object.__setattr__(self, "result_timestamp", time.time())
        else:
            object.__setattr__(self, "result_timestamp", float(self.result_timestamp))
