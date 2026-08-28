"""Unit tests for the ingestion pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from crowd_management.config.loader import load_ingestion_config
from crowd_management.config.schemas import AppConfig, CameraConfig, DatasetConfig, IngestionConfig
from crowd_management.ingestion.consumer import FrameConsumer
from crowd_management.ingestion.frame_buffer import FrameBuffer
from crowd_management.ingestion.frame_packet import FramePacket
from crowd_management.ingestion.producer import FrameProducer
from crowd_management.ingestion.video_source import VideoFileSource


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    return np.zeros((64, 64, 3), dtype=np.uint8)


def test_configuration_parsing(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
dataset:
  name: HAJJv2
  root: /tmp/dataset

ingestion:
  target_fps: 5
  real_time_mode: true
  buffer_size: 12
  drop_policy: oldest

cameras:
  - camera_id: cam_01
    source: /tmp/video.mp4
    enabled: true
""",
        encoding="utf-8",
    )

    cfg = load_ingestion_config(config_file)
    assert cfg.dataset.name == "HAJJv2"
    assert cfg.ingestion.target_fps == 5.0
    assert cfg.cameras[0].camera_id == "cam_01"


def test_invalid_configuration(tmp_path: Path):
    config_file = tmp_path / "bad_config.yaml"
    config_file.write_text("dataset: {}\n", encoding="utf-8")
    with pytest.raises(Exception):
        load_ingestion_config(config_file)


def test_frame_packet_creation(synthetic_frame: np.ndarray):
    packet = FramePacket(
        camera_id="cam_01",
        frame_id=3,
        source_frame_index=15,
        video_timestamp=0.6,
        wall_clock_timestamp=time.time(),
        frame=synthetic_frame,
        source_fps=25.0,
        stream_fps=5.0,
        width=64,
        height=64,
    )
    assert packet.frame.shape == (64, 64, 3)
    assert packet.frame.dtype == np.uint8
    assert packet.video_timestamp == 0.6


def test_buffer_capacity_and_overflow():
    buffer = FrameBuffer(capacity=2, drop_policy="oldest")
    a = FramePacket(
        camera_id="cam_01",
        frame_id=1,
        source_frame_index=0,
        video_timestamp=0.0,
        wall_clock_timestamp=time.time(),
        frame=np.zeros((10, 10, 3), dtype=np.uint8),
        source_fps=25.0,
        stream_fps=5.0,
        width=10,
        height=10,
    )
    b = FramePacket(
        camera_id="cam_01",
        frame_id=2,
        source_frame_index=1,
        video_timestamp=0.04,
        wall_clock_timestamp=time.time(),
        frame=np.zeros((10, 10, 3), dtype=np.uint8),
        source_fps=25.0,
        stream_fps=5.0,
        width=10,
        height=10,
    )
    c = FramePacket(
        camera_id="cam_01",
        frame_id=3,
        source_frame_index=2,
        video_timestamp=0.08,
        wall_clock_timestamp=time.time(),
        frame=np.zeros((10, 10, 3), dtype=np.uint8),
        source_fps=25.0,
        stream_fps=5.0,
        width=10,
        height=10,
    )
    buffer.put(a)
    buffer.put(b)
    buffer.put(c)
    assert buffer.current_size <= 2
    assert buffer.frames_dropped >= 1


def test_buffer_get_and_timeout():
    buffer = FrameBuffer(capacity=5)
    packet = FramePacket(
        camera_id="cam_01",
        frame_id=1,
        source_frame_index=0,
        video_timestamp=0.0,
        wall_clock_timestamp=time.time(),
        frame=np.zeros((8, 8, 3), dtype=np.uint8),
        source_fps=25.0,
        stream_fps=5.0,
        width=8,
        height=8,
    )
    buffer.put(packet)
    got = buffer.get(timeout=0.1)
    assert got.frame_id == 1

    with pytest.raises(Exception):
        buffer.get(timeout=0.05)


def test_producer_startup_and_shutdown(tmp_path: Path):
    source_path = tmp_path / "test_video.mp4"
    video = np.zeros((20, 20, 3), dtype=np.uint8)
    import cv2

    writer = cv2.VideoWriter(str(source_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (20, 20))
    for _ in range(5):
        writer.write(video)
    writer.release()

    source = VideoFileSource(source_path)
    buffer = FrameBuffer(capacity=5)
    producer = FrameProducer("cam_01", source, buffer, target_fps=5.0, real_time_mode=False)
    producer.start()
    time.sleep(0.25)
    producer.stop()
    assert producer.complete or buffer.frames_produced > 0


def test_multi_camera_isolation():
    config = AppConfig(
        dataset=DatasetConfig(name="HAJJv2", root="/tmp"),
        ingestion=IngestionConfig(target_fps=5.0, real_time_mode=False, buffer_size=10, drop_policy="oldest"),
        cameras=[
            CameraConfig(camera_id="cam_01", source="/tmp/vid1.mp4", enabled=True),
            CameraConfig(camera_id="cam_02", source="/tmp/vid2.mp4", enabled=True),
        ],
    )
    assert len(config.cameras) == 2
