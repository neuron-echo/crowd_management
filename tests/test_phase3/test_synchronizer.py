"""Unit and regression tests for TemporalSynchronizer with ByteTrack results."""

from __future__ import annotations

import time

import numpy as np
import pytest

from crowd_management.adapters.synchronizer import MotionFreshness, SyncResult, TemporalSynchronizer
from crowd_management.models.contracts import ByteTrackResult, CSRNetResult


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


def make_bytetrack(
    camera_id: str = "cam_01",
    frame_id: int = 50,
    median_velocity: float = 3.1,
) -> ByteTrackResult:
    return ByteTrackResult(
        camera_id=camera_id,
        frame_id=frame_id,
        source_frame_index=frame_id * 5,
        video_timestamp=frame_id * 0.18,
        wall_clock_timestamp=time.time(),
        track_count=8,
        median_velocity=median_velocity,
        average_velocity=median_velocity,
        dominant_direction_deg=90.0,
        flow_consistency=0.85,
        active_motion_ratio=0.5,
        motion_confidence=0.9,
        inference_time_ms=25.0,
        model_name="yolov8l.pt",
        device="cuda",
        result_timestamp=time.time(),
    )


class TestTemporalSynchronizer:

    def test_future_frame_rejected(self):
        """ByteTrack frame > CSR frame must be rejected."""
        sync = TemporalSynchronizer()

        sync.add_bytetrack_result(make_bytetrack(frame_id=38))   # past
        sync.add_bytetrack_result(make_bytetrack(frame_id=58))   # future
        sync.add_bytetrack_result(make_bytetrack(frame_id=51))   # valid most-recent

        csr = make_csr(frame_id=56)
        res = sync.synchronize(csr)

        assert res.bytetrack_result is not None
        assert res.bytetrack_result.frame_id == 51
        assert res.is_valid_motion is True

    def test_concurrent_frame_selected(self):
        """ByteTrack at same frame_id as CSR is valid and preferred."""
        sync = TemporalSynchronizer()

        sync.add_bytetrack_result(make_bytetrack(frame_id=38))
        sync.add_bytetrack_result(make_bytetrack(frame_id=56))   # concurrent

        csr = make_csr(frame_id=56)
        res = sync.synchronize(csr)

        assert res.bytetrack_result is not None
        assert res.bytetrack_result.frame_id == 56
        assert res.is_valid_motion is True
        assert res.age_frames == 0

    def test_only_future_available_gives_unavailable(self):
        """Only future ByteTrack available -> UNAVAILABLE."""
        sync = TemporalSynchronizer()

        sync.add_bytetrack_result(make_bytetrack(frame_id=58))  # future only

        csr = make_csr(frame_id=56)
        res = sync.synchronize(csr)

        assert res.bytetrack_result is None
        assert res.status == MotionFreshness.UNAVAILABLE
        assert res.is_valid_motion is False

    def test_no_bytetrack_available(self):
        """No ByteTrack results at all -> UNAVAILABLE."""
        sync = TemporalSynchronizer()
        csr = make_csr(frame_id=10)
        res = sync.synchronize(csr)

        assert res.bytetrack_result is None
        assert res.status == MotionFreshness.UNAVAILABLE
        assert res.is_valid_motion is False

    def test_multi_camera_isolation(self):
        """cam_01 queries must not see cam_02 results."""
        sync = TemporalSynchronizer()
        sync.add_bytetrack_result(make_bytetrack(camera_id="cam_02", frame_id=11))

        csr1 = make_csr(camera_id="cam_01", frame_id=15)
        res1 = sync.synchronize(csr1)
        assert res1.status == MotionFreshness.UNAVAILABLE

        csr2 = make_csr(camera_id="cam_02", frame_id=15)
        res2 = sync.synchronize(csr2)
        assert res2.status == MotionFreshness.NEW
        assert res2.bytetrack_result.camera_id == "cam_02"

    def test_new_vs_last_known_semantics(self):
        """Same ByteTrack frame used twice -> second is LAST_KNOWN."""
        sync = TemporalSynchronizer()
        sync.add_bytetrack_result(make_bytetrack(frame_id=11))

        # First query -> NEW
        res1 = sync.synchronize(make_csr(frame_id=12))
        assert res1.status == MotionFreshness.NEW
        assert res1.is_valid_motion is True

        # Second query (no new ByteTrack arrived) -> LAST_KNOWN
        res2 = sync.synchronize(make_csr(frame_id=13))
        assert res2.status == MotionFreshness.LAST_KNOWN
        assert res2.is_valid_motion is True
        assert res2.age_frames == 2

    def test_stale_by_frame_gap(self):
        """Large frame gap beyond threshold -> STALE."""
        sync = TemporalSynchronizer(stale_frame_gap=10)
        sync.add_bytetrack_result(make_bytetrack(frame_id=10))

        csr = make_csr(frame_id=25)  # gap = 15 > threshold 10
        res = sync.synchronize(csr)

        assert res.status == MotionFreshness.STALE
        assert res.is_valid_motion is False
        assert res.age_frames == 15


class TestCallbackRegression:

    def test_act_str_defined_in_decision_formatting(self):
        """Regression: decision action string formatting must not raise NameError."""
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
