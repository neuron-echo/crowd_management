"""Multi-camera producer manager for independent video sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crowd_management.config.loader import load_ingestion_config
from crowd_management.config.schemas import AppConfig, CameraConfig, IngestionConfig
from crowd_management.ingestion.consumer import FrameConsumer
from crowd_management.ingestion.frame_buffer import FrameBuffer
from crowd_management.ingestion.frame_packet import FramePacket
from crowd_management.ingestion.producer import FrameProducer
from crowd_management.ingestion.video_source import VideoFileSource


class MultiCameraProducer:
    """Coordinate multiple independent camera producers and their buffers."""

    def __init__(self, config: AppConfig | str | Path):
        if isinstance(config, (str, Path)):
            config = load_ingestion_config(config)
        self.config = config
        self.producers: dict[str, FrameProducer] = {}
        self.buffers: dict[str, FrameBuffer] = {}
        self._camera_order: list[str] = []

        for camera_entry in self.config.cameras:
            if not camera_entry.enabled:
                continue
            buffer = FrameBuffer(capacity=self.config.ingestion.buffer_size, drop_policy=self.config.ingestion.drop_policy)
            source = VideoFileSource(camera_entry.source)
            producer = FrameProducer(
                camera_id=camera_entry.camera_id,
                source=source,
                frame_buffer=buffer,
                target_fps=self.config.ingestion.target_fps,
                real_time_mode=self.config.ingestion.real_time_mode,
            )
            self.buffers[camera_entry.camera_id] = buffer
            self.producers[camera_entry.camera_id] = producer
            self._camera_order.append(camera_entry.camera_id)

    @property
    def camera_ids(self) -> list[str]:
        return list(self._camera_order)

    def start_all(self) -> None:
        for camera_id in self._camera_order:
            self.producers[camera_id].start()

    def stop_all(self) -> None:
        for camera_id in self._camera_order:
            self.producers[camera_id].stop()

    def shutdown(self) -> None:
        for camera_id in self._camera_order:
            self.producers[camera_id].shutdown()

    def get_consumer(self, camera_id: str) -> FrameConsumer:
        if camera_id not in self.producers:
            raise KeyError(f"Unknown camera ID: {camera_id}")
        return FrameConsumer(self.buffers[camera_id])

    def iter_active(self):
        for camera_id in self._camera_order:
            yield camera_id, self.producers[camera_id]

    def camera_metadata(self) -> dict[str, dict[str, Any]]:
        metadata: dict[str, dict[str, Any]] = {}
        for camera_id in self._camera_order:
            producer = self.producers[camera_id]
            metadata[camera_id] = {
                "camera_id": camera_id,
                "source": producer.source.path,
                "source_fps": producer.source.source_fps,
                "resolution": producer.source.resolution,
                "buffer_size": self.buffers[camera_id].capacity,
                "drop_policy": self.buffers[camera_id].drop_policy,
                "real_time_mode": producer.real_time_mode,
                "target_fps": producer.target_fps,
            }
        return metadata
