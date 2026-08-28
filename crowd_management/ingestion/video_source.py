"""Concrete video source implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from crowd_management.ingestion.base import BaseFrameSource
from crowd_management.utils.exceptions import VideoSourceError


class VideoFileSource(BaseFrameSource):
    """A video file source that reads frames from disk in order."""

    source_type = "video"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._capture: cv2.VideoCapture | None = None
        self._opened = False
        self._eof = False
        self._frame_index = 0
        self._source_fps = 0.0
        self._width = 0
        self._height = 0
        self._frame_count = 0
        self._probe_metadata()

    def _probe_metadata(self) -> None:
        if self.path.exists():
            cap = cv2.VideoCapture(str(self.path))
            if cap.isOpened():
                self._source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                cap.release()

    @property
    def is_open(self) -> bool:
        return self._opened and self._capture is not None and self._capture.isOpened()

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "fps": self.source_fps,
            "width": self._width,
            "height": self._height,
            "frame_count": self.frame_count,
            "opened": self.is_open,
            "eof": self._eof,
        }

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def source_fps(self) -> float:
        return self._source_fps

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._width, self._height)

    def open(self) -> bool:
        if self.is_open:
            return True

        if not self.path.exists():
            raise VideoSourceError(f"Video file does not exist: {self.path}")

        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self._opened = False
            self._capture = None
            raise VideoSourceError(f"Could not open video source: {self.path}")

        self._source_fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._opened = True
        self._eof = False
        self._frame_index = 0
        return True

    def read_next(self) -> tuple[np.ndarray | None, int | None]:
        if not self.is_open:
            raise VideoSourceError(f"Source is not open: {self.path}")

        ret, frame = self._capture.read()
        if not ret or frame is None:
            self._eof = True
            return None, None

        source_index = self._frame_index
        self._frame_index += 1
        return np.asarray(frame, dtype=np.uint8), source_index

    def timestamp_for_index(self, source_frame_index: int) -> float:
        if self._source_fps <= 0:
            return float(source_frame_index)
        return float(source_frame_index) / float(self._source_fps)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._opened = False
        self._eof = True

    def __enter__(self) -> "VideoFileSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
