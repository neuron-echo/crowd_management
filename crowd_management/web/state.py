"""Thread-safe shared state for the web dashboard.

All pipeline callbacks write here; FastAPI reads from here.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class CameraState:
    camera_id: str
    gate_id: str
    gate_capacity: int = 500

    # Latest JPEG bytes for MJPEG streams
    frame_jpeg: bytes = b""
    density_jpeg: bytes = b""

    # Latest model results
    crowd_count: float = 0.0
    density_score: float = 0.0
    velocity: float = 0.0
    direction: float = 0.0
    track_count: int = 0
    flow_consistency: float = 0.0
    motion_confidence: float = 0.0
    motion_status: str = "UNAVAILABLE"

    # Decision
    action: str = "INITIALIZING"
    severity: str = "LOW"
    confidence: float = 0.0
    target_gate: str | None = None
    reason: str = ""

    # Congestion
    congestion_level: str = "LOW"
    congestion_score: float = 0.0
    trend: str = "STABLE"

    # Frame info
    frame_id: int = 0
    video_timestamp: float = 0.0
    csrnet_latency_ms: float = 0.0
    bytetrack_latency_ms: float = 0.0

    # History for sparkline chart (last 60 counts)
    count_history: list[float] = field(default_factory=list)
    score_history: list[float] = field(default_factory=list)

    is_eof: bool = False
    last_update: float = field(default_factory=time.time)


class DashboardState:
    """Singleton shared state. Thread-safe via per-camera locks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.cameras: dict[str, CameraState] = {}
        self.pipeline_running: bool = False
        self.start_time: float = time.time()
        self.total_evaluations: int = 0
        self.device: str = "cpu"

    def init_camera(self, camera_id: str, gate_id: str, gate_capacity: int) -> None:
        with self._lock:
            self.cameras[camera_id] = CameraState(
                camera_id=camera_id,
                gate_id=gate_id,
                gate_capacity=gate_capacity,
            )

    def update_frame(self, camera_id: str, bgr_frame: np.ndarray) -> None:
        with self._lock:
            cam = self.cameras.get(camera_id)
            if cam is None:
                return
            _, buf = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            cam.frame_jpeg = buf.tobytes()

    def update_density(self, camera_id: str, density_overlay: np.ndarray) -> None:
        with self._lock:
            cam = self.cameras.get(camera_id)
            if cam is None:
                return
            _, buf = cv2.imencode(".jpg", density_overlay, [cv2.IMWRITE_JPEG_QUALITY, 80])
            cam.density_jpeg = buf.tobytes()

    def update_csrnet(self, camera_id: str, result: Any) -> None:
        from crowd_management.utils.visualization import render_density_overlay
        with self._lock:
            cam = self.cameras.get(camera_id)
            if cam is None:
                return
            cam.crowd_count = result.crowd_count
            cam.frame_id = result.frame_id
            cam.video_timestamp = result.video_timestamp
            cam.csrnet_latency_ms = result.inference_time_ms
            cam.last_update = time.time()
            # Update count history (max 60 points)
            cam.count_history.append(result.crowd_count)
            if len(cam.count_history) > 60:
                cam.count_history.pop(0)
            # Store annotated frame JPEG
            if result.frame is not None:
                frame = result.frame.copy()
                # Burn crowd count onto frame
                cv2.putText(frame, f"Count: {result.crowd_count:.0f}",
                            (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame {result.frame_id:04d} | t={result.video_timestamp:.2f}s",
                            (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                cam.frame_jpeg = buf.tobytes()
                # Density overlay
                overlay = render_density_overlay(result.frame, result.density_map, alpha=0.5)
                cv2.putText(overlay, f"Density Map | Count: {result.crowd_count:.0f}",
                            (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                _, buf2 = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 80])
                cam.density_jpeg = buf2.tobytes()

    def update_bytetrack(self, camera_id: str, result: Any) -> None:
        with self._lock:
            cam = self.cameras.get(camera_id)
            if cam is None:
                return
            cam.track_count = result.track_count
            cam.velocity = result.median_velocity
            cam.direction = result.dominant_direction_deg
            cam.flow_consistency = result.flow_consistency
            cam.motion_confidence = result.motion_confidence
            cam.bytetrack_latency_ms = result.inference_time_ms

    def update_crowd_state(self, camera_id: str, gate_id: str, crowd_state: Any) -> None:
        with self._lock:
            cam = self.cameras.get(camera_id)
            if cam is None:
                return
            cam.crowd_count = crowd_state.crowd_count
            cam.density_score = crowd_state.density_score
            cam.velocity = crowd_state.velocity
            cam.congestion_level = crowd_state.congestion_level.value
            cam.congestion_score = crowd_state.congestion_score
            cam.trend = crowd_state.trend.value
            cam.motion_status = crowd_state.motion_status
            cam.score_history.append(crowd_state.congestion_score)
            if len(cam.score_history) > 60:
                cam.score_history.pop(0)

    def update_decision(self, camera_id: str, decision: Any) -> None:
        with self._lock:
            cam = self.cameras.get(camera_id)
            if cam is None:
                return
            cam.action = decision.action.value
            cam.severity = decision.severity.value
            cam.confidence = decision.confidence
            cam.target_gate = decision.target_gate_id
            cam.reason = decision.human_readable_reason[:80] if decision.human_readable_reason else ""
            self.total_evaluations += 1

    def snapshot(self) -> dict:
        """Thread-safe JSON-serialisable snapshot of the full state."""
        with self._lock:
            cameras = {}
            for cam_id, cam in self.cameras.items():
                cameras[cam_id] = {
                    "camera_id": cam.camera_id,
                    "gate_id": cam.gate_id,
                    "gate_capacity": cam.gate_capacity,
                    "crowd_count": round(cam.crowd_count, 1),
                    "density_score": round(cam.density_score, 3),
                    "velocity": round(cam.velocity, 1),
                    "direction": round(cam.direction, 1),
                    "track_count": cam.track_count,
                    "flow_consistency": round(cam.flow_consistency, 3),
                    "motion_confidence": round(cam.motion_confidence, 3),
                    "motion_status": cam.motion_status,
                    "congestion_level": cam.congestion_level,
                    "congestion_score": round(cam.congestion_score, 3),
                    "trend": cam.trend,
                    "action": cam.action,
                    "severity": cam.severity,
                    "confidence": round(cam.confidence, 2),
                    "target_gate": cam.target_gate,
                    "reason": cam.reason,
                    "frame_id": cam.frame_id,
                    "video_timestamp": round(cam.video_timestamp, 2),
                    "csrnet_latency_ms": round(cam.csrnet_latency_ms, 0),
                    "bytetrack_latency_ms": round(cam.bytetrack_latency_ms, 0),
                    "count_history": cam.count_history[-30:],
                    "score_history": cam.score_history[-30:],
                    "is_eof": cam.is_eof,
                    "last_update": cam.last_update,
                }
            return {
                "cameras": cameras,
                "pipeline_running": self.pipeline_running,
                "uptime_s": round(time.time() - self.start_time, 1),
                "total_evaluations": self.total_evaluations,
                "device": self.device,
                "timestamp": time.time(),
            }


# Global singleton
dashboard_state = DashboardState()
