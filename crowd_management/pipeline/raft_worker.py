"""Dedicated worker thread for RAFT optical flow execution with per-camera temporal state."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from queue import Empty
from typing import Callable, Deque

from crowd_management.ingestion.frame_buffer import FrameBuffer
from crowd_management.ingestion.frame_packet import FramePacket
from crowd_management.models.contracts import RAFTResult
from crowd_management.models.raft.infer import RAFTInference

logger = logging.getLogger(__name__)


class RAFTWorker:
    """Consumes FramePackets, tracks per-camera temporal pairs (t-1, t),

    and publishes RAFTResult objects to an output buffer.
    """

    def __init__(
        self,
        input_queue: FrameBuffer,
        engine: RAFTInference,
        output_capacity: int = 30,
        result_callback: Callable[[RAFTResult], None] | None = None,
    ) -> None:
        self.input_queue = input_queue
        self.engine = engine
        self.output_capacity = output_capacity
        self.result_callback = result_callback

        # Per-camera state: maps camera_id -> most recent FramePacket
        self._prev_frames: dict[str, FramePacket] = {}
        self._output_queue: Deque[RAFTResult] = deque(maxlen=output_capacity)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.pairs_processed = 0
        self.total_inference_time_ms = 0.0
        self.last_result: RAFTResult | None = None

    @property
    def avg_inference_time_ms(self) -> float:
        with self._lock:
            return (self.total_inference_time_ms / self.pairs_processed) if self.pairs_processed > 0 else 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="RAFTWorker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_result(self, block: bool = True, timeout: float | None = 0.5) -> RAFTResult | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._not_empty:
            while not self._output_queue:
                if self._stop_event.is_set():
                    return None
                if not block:
                    return None
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._not_empty.wait(timeout=remaining)

            return self._output_queue.popleft()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                packet = self.input_queue.get(block=True, timeout=0.1)
            except Empty:
                continue

            cam_id = packet.camera_id
            prev_packet = self._prev_frames.get(cam_id)

            # Store current frame as the latest frame for this camera
            self._prev_frames[cam_id] = packet

            # First frame received for this camera: cannot compute flow yet
            if prev_packet is None:
                continue

            # Ensure frames from the same camera are strictly ordered
            if prev_packet.frame_id >= packet.frame_id:
                logger.warning(
                    "Skipping non-sequential frame on camera %s: prev_id=%s >= curr_id=%s",
                    cam_id,
                    prev_packet.frame_id,
                    packet.frame_id,
                )
                continue

            try:
                result = self.engine.estimate_flow(prev_packet, packet)
            except Exception as e:
                logger.error(
                    "RAFTWorker inference error on camera %s frames %s->%s: %s",
                    cam_id,
                    prev_packet.frame_id,
                    packet.frame_id,
                    e,
                )
                continue

            with self._not_empty:
                self._output_queue.append(result)
                self.pairs_processed += 1
                self.total_inference_time_ms += result.inference_time_ms
                self.last_result = result
                self._not_empty.notify()

            if self.result_callback is not None:
                try:
                    self.result_callback(result)
                except Exception as e:
                    logger.warning("Error in RAFT result callback: %s", e)
