"""Immutable result contracts for Phase 2 vision models."""

from __future__ import annotations

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


@dataclass(frozen=True)
class RAFTResult:
    """Canonical result packet from RAFT optical flow motion estimation."""

    camera_id: str
    previous_frame_id: int
    current_frame_id: int
    previous_source_frame_index: int
    current_source_frame_index: int
    previous_timestamp: float
    current_timestamp: float
    flow_width: int
    flow_height: int
    average_velocity: float
    median_velocity: float
    max_velocity: float
    velocity_variance: float
    dominant_direction_deg: float
    flow_consistency: float
    active_motion_ratio: float
    motion_confidence: float
    inference_time_ms: float
    model_name: str
    device: str
    flow_summary: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.previous_frame_id >= self.current_frame_id:
            raise ValueError(
                f"previous_frame_id ({self.previous_frame_id}) must be < current_frame_id ({self.current_frame_id})"
            )
        if not (0.0 <= self.dominant_direction_deg <= 360.0):
            raise ValueError(f"dominant_direction_deg must be in [0, 360], got {self.dominant_direction_deg}")
        if not (0.0 <= self.flow_consistency <= 1.0):
            raise ValueError(f"flow_consistency must be in [0, 1], got {self.flow_consistency}")
        if not (0.0 <= self.active_motion_ratio <= 1.0):
            raise ValueError(f"active_motion_ratio must be in [0, 1], got {self.active_motion_ratio}")

        object.__setattr__(self, "average_velocity", float(self.average_velocity))
        object.__setattr__(self, "median_velocity", float(self.median_velocity))
        object.__setattr__(self, "max_velocity", float(self.max_velocity))
        object.__setattr__(self, "velocity_variance", float(self.velocity_variance))
        object.__setattr__(self, "dominant_direction_deg", float(self.dominant_direction_deg))
        object.__setattr__(self, "flow_consistency", float(self.flow_consistency))
        object.__setattr__(self, "active_motion_ratio", float(self.active_motion_ratio))
        object.__setattr__(self, "motion_confidence", float(self.motion_confidence))
        object.__setattr__(self, "inference_time_ms", float(self.inference_time_ms))
        object.__setattr__(self, "previous_timestamp", float(self.previous_timestamp))
        object.__setattr__(self, "current_timestamp", float(self.current_timestamp))
        object.__setattr__(self, "previous_frame_id", int(self.previous_frame_id))
        object.__setattr__(self, "current_frame_id", int(self.current_frame_id))
        object.__setattr__(self, "previous_source_frame_index", int(self.previous_source_frame_index))
        object.__setattr__(self, "current_source_frame_index", int(self.current_source_frame_index))
        object.__setattr__(self, "flow_width", int(self.flow_width))
        object.__setattr__(self, "flow_height", int(self.flow_height))
        if self.flow_summary is not None:
            object.__setattr__(self, "flow_summary", np.asarray(self.flow_summary, dtype=np.float32).copy())
