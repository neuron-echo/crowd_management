"""Dedicated worker thread for CSRNet inference execution."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from queue import Empty
from typing import Callable, Deque

from crowd_management.ingestion.frame_buffer import FrameBuffer
from crowd_management.models.contracts import CSRNetResult
from crowd_management.models.csrnet.infer import CSRNetInference

logger = logging.getLogger(__name__)


class CSRNetWorker:
    """Consumes FramePackets from an input buffer, runs CSRNet inference,

    and publishes CSRNetResult objects to an output buffer.
    """

    def __init__(
        self,
        input_queue: FrameBuffer,
        engine: CSRNetInference,
        output_capacity: int = 30,
        result_callback: Callable[[CSRNetResult], None] | None = None,
    ) -> None:
        self.input_queue = input_queue
        self.engine = engine
        self.output_capacity = output_capacity
        self.result_callback = result_callback

        self._output_queue: Deque[CSRNetResult] = deque(maxlen=output_capacity)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.frames_processed = 0
        self.total_inference_time_ms = 0.0
        self.last_result: CSRNetResult | None = None

    @property
    def avg_inference_time_ms(self) -> float:
        with self._lock:
            return (self.total_inference_time_ms / self.frames_processed) if self.frames_processed > 0 else 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="CSRNetWorker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_result(self, block: bool = True, timeout: float | None = 0.5) -> CSRNetResult | None:
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

            try:
                result = self.engine.estimate(packet)
            except Exception as e:
                logger.error("CSRNetWorker inference error on camera %s frame %s: %s", packet.camera_id, packet.frame_id, e)
                continue

            with self._not_empty:
                self._output_queue.append(result)
                self.frames_processed += 1
                self.total_inference_time_ms += result.inference_time_ms
                self.last_result = result
                self._not_empty.notify()

            if self.result_callback is not None:
                try:
                    self.result_callback(result)
                except Exception as e:
                    logger.warning("Error in CSRNet result callback: %s", e)
