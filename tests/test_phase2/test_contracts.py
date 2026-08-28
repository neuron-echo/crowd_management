"""Unit tests for Phase 2 data contracts."""

from __future__ import annotations

import numpy as np
import pytest

from crowd_management.models.contracts import CSRNetResult, RAFTResult


def test_csrnet_result_valid():
    density = np.zeros((48, 64), dtype=np.float32)
    res = CSRNetResult(
        camera_id="cam_01",
        frame_id=10,
        source_frame_index=50,
        video_timestamp=2.0,
        wall_clock_timestamp=100.0,
        crowd_count=42.5,
        density_map=density,
        density_map_width=64,
        density_map_height=48,
        input_width=512,
        input_height=384,
        inference_time_ms=35.0,
        model_name="CSRNet-VGG16",
        device="cpu",
    )
    assert res.camera_id == "cam_01"
    assert res.crowd_count == 42.5
    assert res.density_map.shape == (48, 64)


def test_csrnet_result_invalid_count():
    density = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="cannot be negative"):
        CSRNetResult(
            camera_id="cam_01",
            frame_id=1,
            source_frame_index=0,
            video_timestamp=0.0,
            wall_clock_timestamp=0.0,
            crowd_count=-1.0,
            density_map=density,
            density_map_width=10,
            density_map_height=10,
            input_width=80,
            input_height=80,
            inference_time_ms=10.0,
            model_name="CSRNet",
            device="cpu",
        )


def test_raft_result_valid():
    res = RAFTResult(
        camera_id="cam_01",
        previous_frame_id=5,
        current_frame_id=6,
        previous_source_frame_index=25,
        current_source_frame_index=30,
        previous_timestamp=1.0,
        current_timestamp=1.2,
        flow_width=640,
        flow_height=480,
        average_velocity=2.5,
        median_velocity=2.1,
        max_velocity=10.0,
        velocity_variance=1.2,
        dominant_direction_deg=180.0,
        flow_consistency=0.85,
        active_motion_ratio=0.4,
        motion_confidence=0.6,
        inference_time_ms=45.0,
        model_name="RAFT",
        device="cpu",
    )
    assert res.previous_frame_id == 5
    assert res.current_frame_id == 6
    assert res.dominant_direction_deg == 180.0


def test_raft_result_invalid_ordering():
    with pytest.raises(ValueError, match="must be < current_frame_id"):
        RAFTResult(
            camera_id="cam_01",
            previous_frame_id=10,
            current_frame_id=10,
            previous_source_frame_index=50,
            current_source_frame_index=50,
            previous_timestamp=2.0,
            current_timestamp=2.0,
            flow_width=640,
            flow_height=480,
            average_velocity=2.0,
            median_velocity=2.0,
            max_velocity=2.0,
            velocity_variance=0.0,
            dominant_direction_deg=90.0,
            flow_consistency=0.5,
            active_motion_ratio=0.5,
            motion_confidence=0.5,
            inference_time_ms=10.0,
            model_name="RAFT",
            device="cpu",
        )
