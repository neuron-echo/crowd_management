"""Integration tests for Phase 2 FanOutPipeline and concurrent worker lifecycles."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from crowd_management.config.schemas import AppConfig, CameraConfig, DatasetConfig, IngestionConfig
from crowd_management.models.contracts import ByteTrackResult, CSRNetResult
from crowd_management.models.manager import ModelManager
from crowd_management.pipeline.fanout_pipeline import FanOutPipeline


@pytest.fixture
def mock_video(tmp_path: Path) -> Path:
    vid_path = tmp_path / "mock_feed.mp4"
    writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    for i in range(12):
        frame = np.ones((64, 64, 3), dtype=np.uint8) * (i * 20)
        writer.write(frame)
    writer.release()
    return vid_path


def test_fanout_pipeline_integration(mock_video: Path):
    cfg = AppConfig(
        dataset=DatasetConfig(name="MockDataset", root=str(mock_video.parent)),
        ingestion=IngestionConfig(target_fps=5.0, real_time_mode=False, buffer_size=10, drop_policy="oldest"),
        cameras=[
            CameraConfig(camera_id="cam_01", source=str(mock_video), enabled=True),
        ],
    )

    csr_results: list[CSRNetResult] = []
    bt_results: list[ByteTrackResult] = []

    # Use fast lightweight resolutions for integration testing
    model_mgr = ModelManager(
        device="cpu",
        csrnet_resolution=(64, 64),
        bytetrack_model_path="yolov8n.pt",   # smallest model for test speed
    )

    pipeline = FanOutPipeline(
        config=cfg,
        model_manager=model_mgr,
        csrnet_callback=lambda r: csr_results.append(r),
        bytetrack_callback=lambda r: bt_results.append(r),
        queue_capacity=10,
    )

    pipeline.start()
    time.sleep(1.0)
    pipeline.stop()

    stats = pipeline.stats()
    assert stats["is_running"] is False
    assert stats["csrnet"]["frames_processed"] > 0
    assert len(csr_results) > 0
    assert all(r.camera_id == "cam_01" for r in csr_results)
