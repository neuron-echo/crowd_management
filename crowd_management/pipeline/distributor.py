"""Fan-out distributor for non-blocking multi-model frame broadcasting."""

from __future__ import annotations

import logging
import threading
from typing import Any

from crowd_management.ingestion.frame_buffer import FrameBuffer
from crowd_management.ingestion.frame_packet import FramePacket

logger = logging.getLogger(__name__)


class FrameDistributor:
    """Consumes FramePackets from one or more upstream buffers and broadcasts them

    to dedicated downstream queues (CSRNet and ByteTrack) without queue contention.
    """

    def __init__(
        self,
        csrnet_buffer_size: int = 15,
        bytetrack_buffer_size: int = 15,
        drop_policy: str = "oldest",
    ) -> None:
        self.csrnet_queue = FrameBuffer(capacity=csrnet_buffer_size, drop_policy=drop_policy)
        self.bytetrack_queue = FrameBuffer(capacity=bytetrack_buffer_size, drop_policy=drop_policy)
        self.drop_policy = drop_policy

        self._lock = threading.Lock()
        self._packets_routed = 0
        self._camera_packets: dict[str, int] = {}
        self._closed = False

    @property
    def packets_routed(self) -> int:
        with self._lock:
            return self._packets_routed

    def dispatch(self, packet: FramePacket) -> None:
        """Broadcast a single FramePacket to both CSRNet and ByteTrack queues.

        Args:
            packet: Input FramePacket from ingestion stream.
        """
        if self._closed:
            raise RuntimeError("Cannot dispatch on a closed FrameDistributor")

        if not isinstance(packet, FramePacket):
            raise TypeError("packet must be a FramePacket")

        # Push to CSRNet queue
        self.csrnet_queue.put(packet, block=False)

        # Push to ByteTrack queue
        self.bytetrack_queue.put(packet, block=False)

        with self._lock:
            self._packets_routed += 1
            cam = packet.camera_id
            self._camera_packets[cam] = self._camera_packets.get(cam, 0) + 1

    def close(self) -> None:
        """Close distributor and notify downstream queues."""
        with self._lock:
            self._closed = True
        self.csrnet_queue.close()
        self.bytetrack_queue.close()

    def stats(self) -> dict[str, Any]:
        """Return real-time diagnostic statistics for monitoring."""
        with self._lock:
            return {
                "packets_routed": self._packets_routed,
                "camera_distribution": dict(self._camera_packets),
                "csrnet_queue_size": len(self.csrnet_queue),
                "csrnet_frames_produced": self.csrnet_queue.frames_produced,
                "csrnet_frames_dropped": self.csrnet_queue.frames_dropped,
                "bytetrack_queue_size": len(self.bytetrack_queue),
                "bytetrack_frames_produced": self.bytetrack_queue.frames_produced,
                "bytetrack_frames_dropped": self.bytetrack_queue.frames_dropped,
            }
