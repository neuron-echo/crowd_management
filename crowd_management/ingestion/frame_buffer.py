"""Thread-safe bounded frame buffer for producer-consumer ingestion."""

from __future__ import annotations

import threading
import time
from collections import deque
from queue import Empty, Full
from typing import Deque

from crowd_management.ingestion.frame_packet import FramePacket
from crowd_management.utils.exceptions import FrameBufferError


class FrameBuffer:
    """Bounded queue with configurable overflow policy and monitoring."""

    def __init__(self, capacity: int = 30, drop_policy: str = "oldest") -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if drop_policy not in {"oldest", "newest", "drop_latest"}:
            raise ValueError("drop_policy must be one of oldest, newest, drop_latest")

        self.capacity = int(capacity)
        self.drop_policy = drop_policy
        self._queue: Deque[FramePacket] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._closed = False

        self.frames_produced = 0
        self.frames_consumed = 0
        self.frames_dropped = 0
        self.maximum_size = 0
        self._producer_count = 0
        self._consumer_count = 0
        self._producer_last_tick = None
        self._consumer_last_tick = None

    @property
    def current_size(self) -> int:
        return len(self._queue)

    @property
    def producer_fps(self) -> float:
        elapsed = max(time.monotonic() - (self._producer_last_tick or time.monotonic()), 1e-6)
        return self._producer_count / elapsed if self._producer_count else 0.0

    @property
    def consumer_fps(self) -> float:
        elapsed = max(time.monotonic() - (self._consumer_last_tick or time.monotonic()), 1e-6)
        return self._consumer_count / elapsed if self._consumer_count else 0.0

    def put(self, packet: FramePacket, block: bool = True, timeout: float | None = None) -> None:
        if not isinstance(packet, FramePacket):
            raise TypeError("packet must be a FramePacket")

        with self._not_full:
            if self._closed:
                raise FrameBufferError("Cannot enqueue into a closed buffer")

            self.frames_produced += 1
            self._producer_count += 1
            self._producer_last_tick = self._producer_last_tick or time.monotonic()

            if len(self._queue) >= self.capacity:
                self._handle_overflow(packet)
            else:
                self._queue.append(packet)

            self.maximum_size = max(self.maximum_size, len(self._queue))
            self._not_empty.notify()

    def get(self, block: bool = True, timeout: float | None = None) -> FramePacket:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._not_empty:
            while not self._queue:
                if self._closed:
                    raise Empty("FrameBuffer is closed and empty")
                if not block:
                    raise Empty("FrameBuffer is empty")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise Empty("FrameBuffer get timed out")
                self._not_empty.wait(timeout=remaining)

            packet = self._queue.popleft()
            self.frames_consumed += 1
            self._consumer_count += 1
            self._consumer_last_tick = self._consumer_last_tick or time.monotonic()
            self._not_full.notify()
            return packet

    def get_nowait(self) -> FramePacket:
        return self.get(block=False)

    def put_nowait(self, packet: FramePacket) -> None:
        self.put(packet, block=False)

    def close(self) -> None:
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def _handle_overflow(self, packet: FramePacket) -> None:
        if self.drop_policy == "oldest":
            self._queue.popleft()
            self.frames_dropped += 1
            self._queue.append(packet)
        elif self.drop_policy in {"newest", "drop_latest"}:
            self.frames_dropped += 1
        else:
            raise FrameBufferError(f"Unsupported drop policy: {self.drop_policy}")

    def __len__(self) -> int:
        return len(self._queue)

    def empty(self) -> bool:
        return len(self._queue) == 0

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
