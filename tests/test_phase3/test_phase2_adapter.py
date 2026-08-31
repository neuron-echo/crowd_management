"""Tests for Phase 2 Adapter boundary (CSRNet + ByteTrack -> CrowdState)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from crowd_management.adapters.phase2_adapter import Phase2Adapter
from crowd_management.decision.config import DecisionConfig
from crowd_management.decision.contracts import CongestionLevel, Trend
from crowd_management.models.contracts import ByteTrackResult, CSRNetResult


@pytest.fixture
def adapter() -> Phase2Adapter:
    return Phase2Adapter(DecisionConfig())


def make_mock_csrnet_result(
    camera_id: str = "cam_01",
    crowd_count: float = 350.0,
    frame_id: int = 10,
    timestamp: float = 100.0,
) -> CSRNetResult:
    density = np.zeros((48, 64), dtype=np.float32)
    return CSRNetResult(
        camera_id=camera_id,
        frame_id=frame_id,
        source_frame_index=50,
        video_timestamp=2.0,
        wall_clock_timestamp=timestamp,
        crowd_count=crowd_count,
        density_map=density,
        density_map_width=64,
        density_map_height=48,
        input_width=512,
        input_height=384,
        inference_time_ms=35.0,
        model_name="CSRNet-VGG16",
        device="cpu",
        result_timestamp=timestamp,
    )


def make_mock_bytetrack_result(
    camera_id: str = "cam_01",
    frame_id: int = 10,
    median_velocity: float = 2.5,
    timestamp: float | None = None,
) -> ByteTrackResult:
    ts = timestamp if timestamp is not None else time.time()
    return ByteTrackResult(
        camera_id=camera_id,
        frame_id=frame_id,
        source_frame_index=frame_id * 5,
        video_timestamp=2.0,
        wall_clock_timestamp=ts,
        track_count=8,
        median_velocity=median_velocity,
        average_velocity=median_velocity,
        dominant_direction_deg=90.0,
        flow_consistency=0.85,
        active_motion_ratio=0.5,
        motion_confidence=0.8,
        inference_time_ms=25.0,
        model_name="yolov8l.pt",
        device="cpu",
        result_timestamp=ts,
    )


class TestPhase2Adapter:

    def test_full_fusion(self, adapter: Phase2Adapter):
        csr = make_mock_csrnet_result(camera_id="cam_01", crowd_count=400.0)
        bt = make_mock_bytetrack_result(camera_id="cam_01", frame_id=10, median_velocity=1.5)

        state = adapter.fuse(
            csrnet_result=csr,
            bytetrack_result=bt,
            gate_id="G1",
            gate_capacity=500,
            trend=Trend.INCREASING,
        )

        assert state.gate_id == "G1"
        assert state.crowd_count == 400.0
        assert state.velocity == 1.5
        assert state.has_density is True
        assert state.has_motion is True
        assert state.trend == Trend.INCREASING
        assert state.congestion_score > 0.5

    def test_missing_density_observation(self, adapter: Phase2Adapter):
        bt = make_mock_bytetrack_result(camera_id="cam_01", frame_id=10, median_velocity=3.0)

        state = adapter.fuse(
            csrnet_result=None,
            bytetrack_result=bt,
            gate_id="G1",
            gate_capacity=500,
        )

        assert state.has_density is False
        assert state.has_motion is True
        assert state.crowd_count == 0.0
        assert state.velocity == 3.0
        # Confidence should be degraded due to missing density
        assert state.confidence < 1.0

    def test_missing_motion_observation(self, adapter: Phase2Adapter):
        csr = make_mock_csrnet_result(camera_id="cam_01", crowd_count=250.0)

        state = adapter.fuse(
            csrnet_result=csr,
            bytetrack_result=None,
            gate_id="G1",
            gate_capacity=500,
        )

        assert state.has_density is True
        assert state.has_motion is False
        assert state.crowd_count == 250.0
        assert state.velocity == 0.0
        assert state.confidence < 1.0

    def test_cross_camera_rejection(self, adapter: Phase2Adapter):
        csr = make_mock_csrnet_result(camera_id="cam_01")
        bt = make_mock_bytetrack_result(camera_id="cam_02")

        with pytest.raises(ValueError, match="Cannot fuse results from different cameras"):
            adapter.fuse(
                csrnet_result=csr,
                bytetrack_result=bt,
                gate_id="G1",
                gate_capacity=500,
            )

    def test_stale_bytetrack_excluded_from_motion(self, adapter: Phase2Adapter):
        """When ByteTrack frame gap > stale threshold, motion is excluded."""
        csr = make_mock_csrnet_result(camera_id="cam_01", crowd_count=400.0, frame_id=50)
        # ByteTrack is at frame 5 -> gap = 45 > threshold (10)
        bt_stale = make_mock_bytetrack_result(
            camera_id="cam_01", frame_id=5, median_velocity=4.0
        )

        state = adapter.fuse(
            csrnet_result=csr,
            bytetrack_result=bt_stale,
            gate_id="G1",
            gate_capacity=500,
            stale_frame_gap=10,
        )

        assert state.gate_id == "G1"
        assert state.crowd_count == 400.0
        # Motion must be excluded from valid motion state
        assert state.has_motion is False
        assert state.velocity == 0.0
        assert state.motion_status == "STALE"
        assert state.is_motion_fresh is False

    def test_fresh_bytetrack_included_in_motion(self, adapter: Phase2Adapter):
        """When ByteTrack frame gap is within threshold, motion is valid."""
        csr = make_mock_csrnet_result(camera_id="cam_01", crowd_count=400.0, frame_id=10)
        bt_fresh = make_mock_bytetrack_result(
            camera_id="cam_01", frame_id=9, median_velocity=3.2   # gap=1 < threshold
        )

        state = adapter.fuse(
            csrnet_result=csr,
            bytetrack_result=bt_fresh,
            gate_id="G1",
            gate_capacity=500,
            stale_frame_gap=10,
        )

        assert state.has_motion is True
        assert state.velocity == 3.2
        assert state.motion_status in ("NEW", "OK")
        assert state.is_motion_fresh is True
