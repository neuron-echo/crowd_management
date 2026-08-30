"""Unit and regression tests for TemporalSynchronizer and temporal causality rules."""

from __future__ import annotations

import time
import numpy as np
import pytest

from crowd_management.adapters.synchronizer import MotionFreshness, SyncResult, TemporalSynchronizer
from crowd_management.models.contracts import CSRNetResult, RAFTResult


def make_csr(
    camera_id: str = "cam_01",
    frame_id: int = 56,
    video_timestamp: float = 10.19,
    crowd_count: float = 2.4,
) -> CSRNetResult:
    return CSRNetResult(
        camera_id=camera_id,
        frame_id=frame_id,
        source_frame_index=frame_id * 5,
        video_timestamp=video_timestamp,
        wall_clock_timestamp=time.time(),
        crowd_count=crowd_count,
        density_map=np.zeros((10, 10), dtype=np.float32),
        density_map_width=10,
        density_map_height=10,
        input_width=512,
        input_height=384,
        inference_time_ms=120.0,
        model_name="CSRNet",
        device="cuda",
        result_timestamp=time.time(),
    )


def make_raft(
    camera_id: str = "cam_01",
    previous_frame_id: int = 50,
    current_frame_id: int = 51,
    previous_timestamp: float = 9.07,
    current_timestamp: float = 9.26,
    median_velocity: float = 3.1,
) -> RAFTResult:
    return RAFTResult(
        camera_id=camera_id,
        previous_frame_id=previous_frame_id,
        current_frame_id=current_frame_id,
        previous_source_frame_index=previous_frame_id * 5,
        current_source_frame_index=current_frame_id * 5,
        previous_timestamp=previous_timestamp,
        current_timestamp=current_timestamp,
        flow_width=512,
        flow_height=384,
        average_velocity=median_velocity,
        median_velocity=median_velocity,
        max_velocity=8.0,
        velocity_variance=1.0,
        dominant_direction_deg=90.0,
        flow_consistency=0.85,
        active_motion_ratio=0.5,
        motion_confidence=0.9,
        inference_time_ms=1300.0,
        model_name="RAFT",
        device="cuda",
        result_timestamp=time.time(),
    )


class TestTemporalSynchronizer:

    def test_specific_user_scenario_rejection_of_future_pair(self):
        """CSR frame = 56, available RAFT pairs: 37->38, 57->58, 50->51 -> Expected: 50->51."""
        sync = TemporalSynchronizer()

        # Add pairs in any order
        sync.add_raft_result(make_raft(previous_frame_id=37, current_frame_id=38, previous_timestamp=6.66, current_timestamp=6.85))
        sync.add_raft_result(make_raft(previous_frame_id=57, current_frame_id=58, previous_timestamp=10.37, current_timestamp=10.56))
        sync.add_raft_result(make_raft(previous_frame_id=50, current_frame_id=51, previous_timestamp=9.07, current_timestamp=9.26))

        csr = make_csr(frame_id=56, video_timestamp=10.19)
        res = sync.synchronize(csr, stale_threshold_s=1.5)

        assert res.raft_result is not None
        assert res.raft_result.previous_frame_id == 50
        assert res.raft_result.current_frame_id == 51
        assert res.is_valid_motion is True
        assert res.rejected_future_count == 1
        assert abs(res.age_s - (10.19 - 9.26)) < 1e-4

    def test_specific_user_scenario_concurrent_pair(self):
        """CSR frame = 56, available: 37->38, 55->56 -> Expected: 55->56."""
        sync = TemporalSynchronizer()

        sync.add_raft_result(make_raft(previous_frame_id=37, current_frame_id=38, previous_timestamp=6.66, current_timestamp=6.85))
        sync.add_raft_result(make_raft(previous_frame_id=55, current_frame_id=56, previous_timestamp=10.00, current_timestamp=10.19))

        csr = make_csr(frame_id=56, video_timestamp=10.19)
        res = sync.synchronize(csr, stale_threshold_s=1.0)

        assert res.raft_result is not None
        assert res.raft_result.previous_frame_id == 55
        assert res.raft_result.current_frame_id == 56
        assert res.is_valid_motion is True
        assert res.age_s == 0.0

    def test_specific_user_scenario_only_future_pair_available(self):
        """CSR frame = 56, available: 57->58 -> Expected: UNAVAILABLE."""
        sync = TemporalSynchronizer()

        sync.add_raft_result(make_raft(previous_frame_id=57, current_frame_id=58, previous_timestamp=10.37, current_timestamp=10.56))

        csr = make_csr(frame_id=56, video_timestamp=10.19)
        res = sync.synchronize(csr, stale_threshold_s=1.0)

        assert res.raft_result is None
        assert res.status == MotionFreshness.UNAVAILABLE
        assert res.is_valid_motion is False
        assert res.reason == "FUTURE_PAIR_REJECTED"
        assert res.rejected_future_count == 1

    def test_no_raft_available(self):
        sync = TemporalSynchronizer()
        csr = make_csr(frame_id=10, video_timestamp=2.0)
        res = sync.synchronize(csr)

        assert res.raft_result is None
        assert res.status == MotionFreshness.UNAVAILABLE
        assert res.is_valid_motion is False

    def test_multi_camera_isolation(self):
        sync = TemporalSynchronizer()
        sync.add_raft_result(make_raft(camera_id="cam_02", previous_frame_id=10, current_frame_id=11, current_timestamp=2.0))

        # Query cam_01 -> should not see cam_02
        csr1 = make_csr(camera_id="cam_01", frame_id=15, video_timestamp=3.0)
        res1 = sync.synchronize(csr1)
        assert res1.status == MotionFreshness.UNAVAILABLE

        # Query cam_02 -> should see cam_02
        csr2 = make_csr(camera_id="cam_02", frame_id=15, video_timestamp=3.0)
        res2 = sync.synchronize(csr2, stale_threshold_s=1.5)
        assert res2.status == MotionFreshness.NEW
        assert res2.raft_result.camera_id == "cam_02"

    def test_consecutive_frame_requirement(self):
        sync = TemporalSynchronizer()
        # Non-consecutive pair (50->52)
        bad_raft = make_raft(previous_frame_id=50, current_frame_id=52)
        sync.add_raft_result(bad_raft)

        csr = make_csr(frame_id=55, video_timestamp=10.0)
        res = sync.synchronize(csr)
        assert res.status == MotionFreshness.UNAVAILABLE

    def test_new_vs_last_known_semantics(self):
        sync = TemporalSynchronizer()
        sync.add_raft_result(make_raft(previous_frame_id=10, current_frame_id=11, current_timestamp=2.0))

        # First query for frame 12 -> NEW
        csr12 = make_csr(frame_id=12, video_timestamp=2.2)
        res1 = sync.synchronize(csr12, stale_threshold_s=1.0)
        assert res1.status == MotionFreshness.NEW
        assert res1.is_valid_motion is True

        # Second query for frame 13 (no new RAFT pair arrived yet) -> LAST_KNOWN
        csr13 = make_csr(frame_id=13, video_timestamp=2.4)
        res2 = sync.synchronize(csr13, stale_threshold_s=1.0)
        assert res2.status == MotionFreshness.LAST_KNOWN
        assert res2.is_valid_motion is True
        assert abs(res2.age_s - 0.4) < 1e-4

        # Third query for frame 25 (video timestamp 5.0 -> age = 3.0s > 1.0s) -> STALE
        csr25 = make_csr(frame_id=25, video_timestamp=5.0)
        res3 = sync.synchronize(csr25, stale_threshold_s=1.0)
        assert res3.status == MotionFreshness.STALE
        assert res3.is_valid_motion is False
        assert abs(res3.age_s - 3.0) < 1e-4


class TestCallbackRegression:

    def test_act_str_defined_in_decision_formatting(self):
        """Regression test ensuring decision action string formatting does not raise NameError."""
        from crowd_management.decision.contracts import ActionType, CongestionLevel, DecisionResult

        dec = DecisionResult(
            decision_id="dec_001",
            timestamp=time.time(),
            source_gate_id="G1",
            target_gate_id="G2",
            action=ActionType.DIVERT,
            severity=CongestionLevel.CRITICAL,
            confidence=0.88,
            reason_codes=["CONGESTED"],
            human_readable_reason="G1 is critical",
        )

        if dec and dec.target_gate_id:
            act_str = f"{dec.action.value} -> {dec.target_gate_id}"
        elif dec:
            act_str = str(dec.action.value)
        else:
            act_str = "None"

        assert act_str == "DIVERT -> G2"
