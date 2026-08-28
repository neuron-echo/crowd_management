"""Base interfaces for ingestion sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseFrameSource(ABC):
    """Abstract interface for all frame-producing video sources."""

    source_type: str = "abstract"

    @abstractmethod
    def open(self) -> bool:
        """Open the underlying source and verify that it is usable."""

    @abstractmethod
    def read_next(self) -> tuple[Any | None, int | None]:
        """Return (frame, source_frame_index) or (None, None) at end of stream."""

    @abstractmethod
    def close(self) -> None:
        """Release all resources tied to the source."""

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Whether the source is currently open and valid."""

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Runtime metadata describing the source."""

    @property
    @abstractmethod
    def frame_count(self) -> int:
        """Return the total number of frames if known."""

    @property
    @abstractmethod
    def source_fps(self) -> float:
        """Detected source frames-per-second."""

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """Return the source resolution as (width, height)."""
