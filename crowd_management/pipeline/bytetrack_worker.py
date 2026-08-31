"""Freshness-aware dedicated worker thread for ByteTrack (YOLOv8) inference execution.

Replaces raft_worker.py. Key differences vs RAFT:
  - Processes single frames (not pairs) — no consecutive-pair check needed
  - Drains backlogs per-camera and jumps to the latest frame to stay real-time
  - Calls ByteTrackInference.infer(packet) → ByteTrackResult per frame
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from queue import Empty
from typing import Callable, Deque

from crowd_management.ingestion.frame_buffer import FrameBuffer
from crowd_management.ingestion.frame_packet import FramePacket
from crowd_management.models.bytetrack.infer import ByteTrackInference
from crowd_management.models.contracts import ByteTrackResult

logger = logging.getLogger(__name__)


class ByteTrackWorker:
    """Real-time ByteTrack worker that processes the latest frame per camera.

    Drains incoming queue backlogs and always processes the most recent
    FramePacket per camera to maintain real-time motion estimation.
    """

    def __init__(
        self,
        input_queue: FrameBuffer,
        engine: ByteTrackInference,
        output_capacity: int = 30,
        result_callback: Callable[[ByteTrackResult], None] | None = None,
    ) -> None:
        self.input_queue = input_queue
        self.engine = engine
        self.output_capacity = output_capacity
        self.result_callback = result_callback

        # Per-camera latest packet cache (always process freshest)
        self._camera_latest: dict[str, FramePacket] = {}
        self._output_queue: Deque[ByteTrackResult] = deque(maxlen=output_capacity)
        self._latest_results: dict[str, ByteTrackResult] = {}

        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.frames_processed: int = 0
        self.stale_frames_discarded: int = 0
        self.total_inference_time_ms: float = 0.0
        self.last_result: ByteTrackResult | None = None

    @property
    def avg_inference_time_ms(self) -> float:
        with self._lock:
            return (self.total_inference_time_ms / self.frames_processed) if self.frames_processed > 0 else 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="ByteTrackWorker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_result(self, block: bool = True, timeout: float | None = 0.5) -> ByteTrackResult | None:
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

    def get_latest_result(self, camera_id: str) -> ByteTrackResult | None:
        """Return the most recently computed result for a specific camera."""
        with self._lock:
            return self._latest_results.get(camera_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _drain_queue(self) -> list[str]:
        """Drain the input queue, keeping only the latest packet per camera.

        Returns list of camera IDs that have a fresh packet ready.
        """
        drained: list[FramePacket] = []
        try:
            first = self.input_queue.get(block=True, timeout=0.05)
            drained.append(first)
            # Drain remaining (non-blocking) to jump to newest
            while True:
                pkt = self.input_queue.get(block=False)
                drained.append(pkt)
        except Empty:
            pass

        if not drained:
            return []

        # Keep only the latest packet per camera
        latest_by_cam: dict[str, FramePacket] = {}
        for pkt in drained:
            existing = latest_by_cam.get(pkt.camera_id)
            if existing is None or pkt.frame_id > existing.frame_id:
                latest_by_cam[pkt.camera_id] = pkt

        # Count how many packets were skipped (stale discards)
        skipped = len(drained) - len(latest_by_cam)
        if skipped > 0:
            with self._lock:
                self.stale_frames_discarded += skipped

        self._camera_latest.update(latest_by_cam)
        return list(latest_by_cam.keys())

    def _run(self) -> None:
        while not self._stop_event.is_set():
            active_cams = self._drain_queue()
            if not active_cams:
                time.sleep(0.01)
                continue

            for cam_id in active_cams:
                packet = self._camera_latest.pop(cam_id, None)
                if packet is None:
                    continue

                try:
                    result = self.engine.infer(packet)
                except Exception as e:
                    logger.error(
                        "ByteTrackWorker inference error on camera %s frame %s: %s",
                        cam_id, packet.frame_id, e,
                    )
                    continue

                with self._not_empty:
                    self._output_queue.append(result)
                    self._latest_results[cam_id] = result
                    self.frames_processed += 1
                    self.total_inference_time_ms += result.inference_time_ms
                    self.last_result = result
                    self._not_empty.notify()

                if self.result_callback is not None:
                    try:
                        self.result_callback(result)
                    except Exception as e:
                        logger.warning("Error in ByteTrack result callback: %s", e)
