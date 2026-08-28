"""Simple frame consumers for the ingestion queue."""

from __future__ import annotations

from queue import Empty

from crowd_management.ingestion.frame_buffer import FrameBuffer
from crowd_management.ingestion.frame_packet import FramePacket


class FrameConsumer:
    """Base interface for consuming FramePacket objects from a buffer."""

    def __init__(self, buffer: FrameBuffer) -> None:
        self.buffer = buffer

    def consume(self, timeout: float | None = None, block: bool = True) -> FramePacket:
        return self.buffer.get(block=block, timeout=timeout)


class QueueFrameConsumer(FrameConsumer):
    """Concrete consumer that uses a bounded buffer as the delivery queue."""

    def consume(self, timeout: float | None = None, block: bool = True) -> FramePacket:
        return super().consume(timeout=timeout, block=block)

    def try_consume(self) -> FramePacket | None:
        try:
            return self.buffer.get(block=False)
        except Empty:
            return None
