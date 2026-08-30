"""Freshness-aware dedicated worker thread for RAFT optical flow execution."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from queue import Empty
from typing import Callable, Deque

from crowd_management.ingestion.frame_buffer import FrameBuffer
from crowd_management.ingestion.frame_packet import FramePacket
from crowd_management.models.contracts import RAFTResult
from crowd_management.models.raft.infer import RAFTInference

logger = logging.getLogger(__name__)


class RAFTWorker:
    """Freshness-aware RAFT optical flow worker.

    Prioritizes processing the latest valid frame pair from each camera,
    discarding stale backlogs to maintain real-time synchronization.
    """

    def __init__(
        self,
        input_queue: FrameBuffer,
        engine: RAFTInference,
        output_capacity: int = 30,
        result_callback: Callable[[RAFTResult], None] | None = None,
        max_allowed_frame_gap: int = 2,
    ) -> None:
        self.input_queue = input_queue
        self.engine = engine
        self.output_capacity = output_capacity
        self.result_callback = result_callback
        self.max_allowed_frame_gap = max_allowed_frame_gap

        # Per-camera rolling history: stores last 4 frames per camera
        self._camera_packets: dict[str, deque[FramePacket]] = defaultdict(lambda: deque(maxlen=4))
        self._output_queue: Deque[RAFTResult] = deque(maxlen=output_capacity)
        self._latest_results: dict[str, RAFTResult] = {}

        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.pairs_processed = 0
        self.stale_frames_discarded = 0
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
        self._thread = threading.Thread(target=self._run, name="FreshnessRAFTWorker", daemon=True)
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

    def get_latest_result(self, camera_id: str) -> RAFTResult | None:
        """Get the most recently computed RAFT result for a specific camera."""
        with self._lock:
            return self._latest_results.get(camera_id)

    def _drain_and_update_recent_packets(self) -> list[str]:
        """Drain incoming queue, store latest frames per camera, and return active camera IDs."""
        drained_packets: list[FramePacket] = []
        try:
            # First blocking get with short timeout
            first_pkt = self.input_queue.get(block=True, timeout=0.05)
            drained_packets.append(first_pkt)

            # Drain any additional queued packets non-blockingly to jump to the newest
            while True:
                pkt = self.input_queue.get(block=False)
                drained_packets.append(pkt)
        except Empty:
            pass

        if not drained_packets:
            return []

        # Group by camera_id and append to per-camera deque
        updated_cameras = set()
        for pkt in drained_packets:
            cam = pkt.camera_id
            self._camera_packets[cam].append(pkt)
            updated_cameras.add(cam)

        # Count skipped frames if queue had a backlog
        if len(drained_packets) > len(updated_cameras) * 2:
            with self._lock:
                self.stale_frames_discarded += (len(drained_packets) - len(updated_cameras) * 2)

        return list(updated_cameras)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            active_cams = self._drain_and_update_recent_packets()
            if not active_cams:
                time.sleep(0.01)
                continue

            for cam_id in active_cams:
                history = self._camera_packets[cam_id]
                if len(history) < 2:
                    continue  # Need at least two frames to form a pair

                prev_packet = history[-2]
                curr_packet = history[-1]

                # Verify temporal order and same camera
                if prev_packet.camera_id != curr_packet.camera_id:
                    continue
                if prev_packet.frame_id >= curr_packet.frame_id:
                    continue

                # If there's an excessively large gap between frames (e.g. from upstream drops > max_allowed_frame_gap),
                # do not compute flow across large time gap; wait for consecutive arrival.
                frame_gap = curr_packet.frame_id - prev_packet.frame_id
                if frame_gap > self.max_allowed_frame_gap:
                    logger.debug(
                        "Skipping wide frame gap on camera %s: %s -> %s (gap=%s)",
                        cam_id,
                        prev_packet.frame_id,
                        curr_packet.frame_id,
                        frame_gap,
                    )
                    continue

                try:
                    result = self.engine.estimate_flow(prev_packet, curr_packet)
                except Exception as e:
                    logger.error(
                        "RAFTWorker inference error on camera %s frames %s->%s: %s",
                        cam_id,
                        prev_packet.frame_id,
                        curr_packet.frame_id,
                        e,
                    )
                    continue

                with self._not_empty:
                    self._output_queue.append(result)
                    self._latest_results[cam_id] = result
                    self.pairs_processed += 1
                    self.total_inference_time_ms += result.inference_time_ms
                    self.last_result = result
                    self._not_empty.notify()

                if self.result_callback is not None:
                    try:
                        self.result_callback(result)
                    except Exception as e:
                        logger.warning("Error in RAFT result callback: %s", e)
