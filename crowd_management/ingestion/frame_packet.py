"""Immutable frame packet representation for the ingestion stream."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FramePacket:
    """Canonical packet for a sampled frame from a video source.

    Frame format: NumPy ndarray
    Color format: BGR
    Data type: uint8
    """

    camera_id: str
    frame_id: int
    source_frame_index: int
    video_timestamp: float
    wall_clock_timestamp: float
    frame: np.ndarray
    source_fps: float
    stream_fps: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if not isinstance(self.frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray")
        frame = np.asarray(self.frame, dtype=np.uint8)
        if frame.ndim != 3:
            raise ValueError("frame must be a 3D BGR image array")
        object.__setattr__(self, "frame", frame.copy())
        object.__setattr__(self, "video_timestamp", float(self.video_timestamp))
        object.__setattr__(self, "wall_clock_timestamp", float(self.wall_clock_timestamp))
        object.__setattr__(self, "source_fps", float(self.source_fps))
        object.__setattr__(self, "stream_fps", float(self.stream_fps))
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        if self.frame.shape[0] != self.height or self.frame.shape[1] != self.width:
            raise ValueError("frame dimensions do not match width and height fields")
