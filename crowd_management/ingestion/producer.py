"""Frame producer that reads video frames and pushes them into a buffer."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from crowd_management.ingestion.frame_buffer import FrameBuffer
from crowd_management.ingestion.frame_packet import FramePacket
from crowd_management.ingestion.video_source import VideoFileSource


class FrameProducer:
    """Worker that converts sampled video frames into FramePacket objects."""

    def __init__(
        self,
        camera_id: str,
        source: VideoFileSource,
        frame_buffer: FrameBuffer,
        target_fps: float | None = None,
        real_time_mode: bool = True,
        sampling_stride: int | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.source = source
        self.buffer = frame_buffer
        self.target_fps = float(target_fps) if target_fps is not None else None
        self.real_time_mode = real_time_mode
        self.sampling_stride = sampling_stride
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._started = False
        self._shutdown = False
        self._complete = False
        self._last_wall_time = time.monotonic()
        self._next_release_time = time.monotonic()
        self._frame_id = 0

    @property
    def started(self) -> bool:
        return self._started

    @property
    def complete(self) -> bool:
        return self._complete

    @property
    def source_fps(self) -> float:
        return self.source.source_fps

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._pause_event.clear()
        self._started = True
        self._shutdown = False
        self._thread = threading.Thread(target=self.run, name=f"Producer-{self.camera_id}", daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self._pause_event.set()

    def resume(self) -> None:
        self._pause_event.clear()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def shutdown(self) -> None:
        self.stop()
        self.buffer.close()
        self.source.close()
        self._shutdown = True

    def _get_sampling_stride(self) -> int:
        if self.sampling_stride is not None:
            return max(1, int(self.sampling_stride))
        if self.target_fps is not None and self.source.source_fps > 0:
            target = max(self.target_fps, 1e-9)
            stride = max(1, int(round(self.source.source_fps / target)))
            return stride
        return 1

    def _should_emit(self, source_frame_index: int, stride: int) -> bool:
        return source_frame_index % stride == 0

    def run(self) -> None:
        try:
            self.source.open()
            stride = self._get_sampling_stride()
            if self.real_time_mode and self.target_fps is not None and self.target_fps > 0:
                interval = 1.0 / self.target_fps
                self._next_release_time = time.monotonic()
            else:
                interval = 0.0

            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    time.sleep(0.01)
                    continue

                frame, source_index = self.source.read_next()
                if frame is None or source_index is None:
                    self._complete = True
                    break

                if not self._should_emit(source_index, stride):
                    continue

                if self.real_time_mode and interval > 0:
                    now = time.monotonic()
                    if now < self._next_release_time:
                        time.sleep(self._next_release_time - now)
                    self._next_release_time += interval

                self._frame_id += 1
                wall_time = time.time()
                packet = FramePacket(
                    camera_id=self.camera_id,
                    frame_id=self._frame_id,
                    source_frame_index=int(source_index),
                    video_timestamp=self.source.timestamp_for_index(int(source_index)),
                    wall_clock_timestamp=wall_time,
                    frame=frame,
                    source_fps=self.source.source_fps,
                    stream_fps=self.target_fps if self.target_fps is not None else self.source.source_fps,
                    width=int(self.source.resolution[0]),
                    height=int(self.source.resolution[1]),
                )
                self.buffer.put(packet, block=True, timeout=1.0)
        except Exception as e:
            logging.getLogger(__name__).warning("Producer encountered error on camera %s: %s", self.camera_id, e)
        finally:
            self._complete = True
            self.source.close()

    def __enter__(self) -> "FrameProducer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
