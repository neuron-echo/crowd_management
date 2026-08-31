"""Unit tests for Phase 2 data contracts (CSRNetResult, ByteTrackResult)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from crowd_management.models.contracts import ByteTrackResult, CSRNetResult


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


def _make_bytetrack(
    camera_id: str = "cam_01",
    frame_id: int = 10,
    track_count: int = 5,
    median_velocity: float = 3.2,
    dominant_direction_deg: float = 90.0,
    flow_consistency: float = 0.8,
    active_motion_ratio: float = 0.6,
    motion_confidence: float = 0.75,
) -> ByteTrackResult:
    return ByteTrackResult(
        camera_id=camera_id,
        frame_id=frame_id,
        source_frame_index=frame_id * 5,
        video_timestamp=2.0,
        wall_clock_timestamp=time.time(),
        track_count=track_count,
        median_velocity=median_velocity,
        average_velocity=median_velocity,
        dominant_direction_deg=dominant_direction_deg,
        flow_consistency=flow_consistency,
        active_motion_ratio=active_motion_ratio,
        motion_confidence=motion_confidence,
        inference_time_ms=25.0,
        model_name="yolov8l.pt",
        device="cpu",
        result_timestamp=time.time(),
    )


def test_bytetrack_result_valid():
    res = _make_bytetrack()
    assert res.camera_id == "cam_01"
    assert res.track_count == 5
    assert res.median_velocity == 3.2
    assert res.dominant_direction_deg == 90.0
    assert res.flow_consistency == 0.8


def test_bytetrack_result_direction_range():
    """Direction must be in [0, 360]."""
    with pytest.raises(ValueError, match="dominant_direction_deg"):
        ByteTrackResult(
            camera_id="cam_01",
            frame_id=1,
            source_frame_index=5,
            video_timestamp=0.0,
            wall_clock_timestamp=0.0,
            track_count=3,
            median_velocity=2.0,
            average_velocity=2.0,
            dominant_direction_deg=400.0,  # invalid
            flow_consistency=0.5,
            active_motion_ratio=0.5,
            motion_confidence=0.5,
            inference_time_ms=10.0,
            model_name="yolov8l.pt",
            device="cpu",
        )


def test_bytetrack_result_flow_consistency_range():
    """flow_consistency must be in [0, 1]."""
    with pytest.raises(ValueError, match="flow_consistency"):
        ByteTrackResult(
            camera_id="cam_01",
            frame_id=1,
            source_frame_index=5,
            video_timestamp=0.0,
            wall_clock_timestamp=0.0,
            track_count=3,
            median_velocity=2.0,
            average_velocity=2.0,
            dominant_direction_deg=90.0,
            flow_consistency=1.5,  # invalid
            active_motion_ratio=0.5,
            motion_confidence=0.5,
            inference_time_ms=10.0,
            model_name="yolov8l.pt",
            device="cpu",
        )


def test_bytetrack_zero_tracks():
    """Zero tracks should produce zero velocities and confidence."""
    res = _make_bytetrack(track_count=0, median_velocity=0.0,
                          flow_consistency=0.0, active_motion_ratio=0.0, motion_confidence=0.0)
    assert res.track_count == 0
    assert res.median_velocity == 0.0
    assert res.motion_confidence == 0.0
