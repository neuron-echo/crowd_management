"""Tests for Phase 2 Adapter boundary."""

from __future__ import annotations

import time
import numpy as np
import pytest

from crowd_management.adapters.phase2_adapter import Phase2Adapter
from crowd_management.decision.config import DecisionConfig
from crowd_management.decision.contracts import CongestionLevel, Trend
from crowd_management.models.contracts import CSRNetResult, RAFTResult


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


def make_mock_raft_result(
    camera_id: str = "cam_01",
    median_velocity: float = 2.5,
    timestamp: float | None = None,
) -> RAFTResult:
    ts = timestamp if timestamp is not None else time.time()
    return RAFTResult(
        camera_id=camera_id,
        previous_frame_id=9,
        current_frame_id=10,
        previous_source_frame_index=45,
        current_source_frame_index=50,
        previous_timestamp=1.8,
        current_timestamp=2.0,
        flow_width=512,
        flow_height=384,
        average_velocity=2.5,
        median_velocity=median_velocity,
        max_velocity=8.0,
        velocity_variance=1.0,
        dominant_direction_deg=90.0,
        flow_consistency=0.85,
        active_motion_ratio=0.5,
        motion_confidence=0.8,
        inference_time_ms=50.0,
        model_name="RAFT",
        device="cpu",
        result_timestamp=ts,
    )


class TestPhase2Adapter:

    def test_full_fusion(self, adapter: Phase2Adapter):
        csr = make_mock_csrnet_result(camera_id="cam_01", crowd_count=400.0)
        raft = make_mock_raft_result(camera_id="cam_01", median_velocity=1.5)

        state = adapter.fuse(
            csrnet_result=csr,
            raft_result=raft,
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
        raft = make_mock_raft_result(camera_id="cam_01", median_velocity=3.0)

        state = adapter.fuse(
            csrnet_result=None,
            raft_result=raft,
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
            raft_result=None,
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
        raft = make_mock_raft_result(camera_id="cam_02")

        with pytest.raises(ValueError, match="Cannot fuse results from different cameras"):
            adapter.fuse(
                csrnet_result=csr,
                raft_result=raft,
                gate_id="G1",
                gate_capacity=500,
            )

    def test_stale_raft_excluded_from_motion(self, adapter: Phase2Adapter):
        """When RAFT result is older than stale threshold (video time), motion is excluded."""
        csr = make_mock_csrnet_result(camera_id="cam_01", crowd_count=400.0)
        # CSRNet is at video timestamp 2.0s
        # RAFT is at video timestamp 0.5s -> age = 1.5s > threshold (1.0s)
        density = np.zeros((48, 64), dtype=np.float32)
        raft_stale = RAFTResult(
            camera_id="cam_01",
            previous_frame_id=2,
            current_frame_id=3,
            previous_source_frame_index=10,
            current_source_frame_index=15,
            previous_timestamp=0.3,
            current_timestamp=0.5,
            flow_width=512, flow_height=384,
            average_velocity=4.0, median_velocity=4.0, max_velocity=8.0,
            velocity_variance=1.0, dominant_direction_deg=90.0,
            flow_consistency=0.8, active_motion_ratio=0.5, motion_confidence=0.9,
            inference_time_ms=40.0, model_name="RAFT", device="cpu",
            result_timestamp=100.0,
        )

        state = adapter.fuse(
            csrnet_result=csr,
            raft_result=raft_stale,
            gate_id="G1",
            gate_capacity=500,
            stale_threshold_s=1.0,
        )

        assert state.gate_id == "G1"
        assert state.crowd_count == 400.0
        # Motion must be excluded from valid motion state
        assert state.has_motion is False
        assert state.velocity == 0.0
        assert state.motion_status == "STALE"
        assert state.is_motion_fresh is False
        assert state.raft_result_age_s == 1.5

    def test_fresh_raft_included_in_motion(self, adapter: Phase2Adapter):
        """When RAFT result is within threshold (video time), motion is valid and active."""
        csr = make_mock_csrnet_result(camera_id="cam_01", crowd_count=400.0)
        # CSRNet is at video timestamp 2.0s
        # RAFT is at video timestamp 1.8s -> age = 0.2s <= threshold (1.0s)
        raft_fresh = RAFTResult(
            camera_id="cam_01",
            previous_frame_id=9,
            current_frame_id=10,
            previous_source_frame_index=45,
            current_source_frame_index=50,
            previous_timestamp=1.6,
            current_timestamp=1.8,
            flow_width=512, flow_height=384,
            average_velocity=3.2, median_velocity=3.2, max_velocity=8.0,
            velocity_variance=1.0, dominant_direction_deg=90.0,
            flow_consistency=0.85, active_motion_ratio=0.5, motion_confidence=0.9,
            inference_time_ms=40.0, model_name="RAFT", device="cpu",
            result_timestamp=100.0,
        )

        state = adapter.fuse(
            csrnet_result=csr,
            raft_result=raft_fresh,
            gate_id="G1",
            gate_capacity=500,
            stale_threshold_s=1.0,
        )

        assert state.has_motion is True
        assert state.velocity == 3.2
        assert state.motion_status in ("NEW", "OK")
        assert state.is_motion_fresh is True
        assert abs(state.raft_result_age_s - 0.2) < 1e-5

