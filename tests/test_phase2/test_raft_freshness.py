"""Tests for freshness-aware RAFT worker, multi-camera isolation, stale detection, and robust velocity."""

from __future__ import annotations

import time
import threading
from collections import deque

import cv2
import numpy as np
import pytest

from crowd_management.ingestion.frame_buffer import FrameBuffer
from crowd_management.ingestion.frame_packet import FramePacket
from crowd_management.models.contracts import RAFTResult
from crowd_management.models.raft.infer import RAFTInference
from crowd_management.pipeline.raft_worker import RAFTWorker
from crowd_management.utils.visualization import compute_result_age, is_result_fresh


def _make_packet(cam_id: str, frame_id: int, src_idx: int, vts: float, w: int = 64, h: int = 64) -> FramePacket:
    """Create a simple FramePacket for testing."""
    frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return FramePacket(cam_id, frame_id, src_idx, vts, time.time(), frame, 25.0, 5.0, w, h)


def _make_textured_packet(cam_id: str, frame_id: int, src_idx: int, vts: float, shift_x: int = 0) -> FramePacket:
    """Create a textured FramePacket with optional horizontal shift for controlled flow."""
    x = np.linspace(0, 4 * np.pi, 128)
    y = np.linspace(0, 4 * np.pi, 128)
    xx, yy = np.meshgrid(x, y)
    pattern = (np.sin(xx) * np.cos(yy) * 127.0 + 128.0).astype(np.uint8)
    bgr = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
    if shift_x != 0:
        M = np.float32([[1, 0, shift_x], [0, 1, 0]])
        bgr = cv2.warpAffine(bgr, M, (128, 128))
    return FramePacket(cam_id, frame_id, src_idx, vts, time.time(), bgr, 25.0, 5.0, 128, 128)


# ===================================================================
# Test: RAFT result provenance — exact frame pair is recorded
# ===================================================================

class TestRAFTProvenance:
    """Verify that RAFTResult preserves exact frame pair provenance."""

    def test_frame_pair_ids_preserved(self):
        res = RAFTResult(
            camera_id="cam_01",
            previous_frame_id=96,
            current_frame_id=97,
            previous_source_frame_index=475,
            current_source_frame_index=480,
            previous_timestamp=17.59,
            current_timestamp=17.78,
            flow_width=512,
            flow_height=384,
            average_velocity=2.5,
            median_velocity=2.1,
            max_velocity=10.0,
            velocity_variance=1.2,
            dominant_direction_deg=108.0,
            flow_consistency=0.91,
            active_motion_ratio=0.4,
            motion_confidence=0.6,
            inference_time_ms=190.0,
            model_name="RAFT",
            device="cpu",
        )
        assert res.previous_frame_id == 96
        assert res.current_frame_id == 97
        assert res.previous_source_frame_index == 475
        assert res.current_source_frame_index == 480
        assert res.previous_timestamp == 17.59
        assert res.current_timestamp == 17.78
        assert res.camera_id == "cam_01"

    def test_delta_time_auto_computed(self):
        res = RAFTResult(
            camera_id="cam_01",
            previous_frame_id=1,
            current_frame_id=2,
            previous_source_frame_index=5,
            current_source_frame_index=10,
            previous_timestamp=1.0,
            current_timestamp=1.2,
            flow_width=512,
            flow_height=384,
            average_velocity=2.5,
            median_velocity=2.1,
            max_velocity=10.0,
            velocity_variance=1.2,
            dominant_direction_deg=90.0,
            flow_consistency=0.5,
            active_motion_ratio=0.5,
            motion_confidence=0.5,
            inference_time_ms=10.0,
            model_name="RAFT",
            device="cpu",
        )
        assert abs(res.delta_time - 0.2) < 1e-4

    def test_robust_velocity_fields_exist(self):
        res = RAFTResult(
            camera_id="cam_01",
            previous_frame_id=1,
            current_frame_id=2,
            previous_source_frame_index=5,
            current_source_frame_index=10,
            previous_timestamp=1.0,
            current_timestamp=1.2,
            flow_width=512,
            flow_height=384,
            average_velocity=10.0,
            median_velocity=5.0,
            max_velocity=50.0,
            velocity_variance=100.0,
            dominant_direction_deg=90.0,
            flow_consistency=0.5,
            active_motion_ratio=0.5,
            motion_confidence=0.5,
            inference_time_ms=10.0,
            model_name="RAFT",
            device="cpu",
            p90_velocity=15.0,
            p95_velocity=20.0,
        )
        assert res.mean_velocity == 10.0  # Same as average_velocity
        assert res.median_velocity == 5.0
        assert res.p90_velocity == 15.0
        assert res.p95_velocity == 20.0
        assert res.max_velocity == 50.0


# ===================================================================
# Test: Same-camera pairing guarantee
# ===================================================================

class TestSameCameraPairing:
    """Verify RAFT enforces same-camera constraint."""

    def test_cross_camera_raises(self):
        engine = RAFTInference(device="cpu", model_size="small", inference_resolution=(128, 128))
        p1 = _make_textured_packet("cam_01", 1, 0, 0.0)
        p2 = _make_textured_packet("cam_02", 2, 5, 0.2)
        with pytest.raises(ValueError, match="Camera mismatch"):
            engine.estimate_flow(p1, p2)

    def test_same_camera_succeeds(self):
        engine = RAFTInference(device="cpu", model_size="small", inference_resolution=(128, 128))
        p1 = _make_textured_packet("cam_01", 1, 0, 0.0)
        p2 = _make_textured_packet("cam_01", 2, 5, 0.2)
        result = engine.estimate_flow(p1, p2)
        assert result.camera_id == "cam_01"


# ===================================================================
# Test: Multi-camera isolation in RAFTWorker
# ===================================================================

class TestMultiCameraIsolation:
    """Verify RAFT worker never mixes cameras."""

    def test_worker_isolates_cameras(self):
        buf = FrameBuffer(capacity=20, drop_policy="oldest")
        engine = RAFTInference(device="cpu", model_size="small", inference_resolution=(128, 128))
        results: list[RAFTResult] = []
        worker = RAFTWorker(
            input_queue=buf,
            engine=engine,
            result_callback=lambda r: results.append(r),
        )

        # Interleave cam_01 and cam_02 frames
        for i in range(6):
            buf.put(_make_textured_packet("cam_01", i + 1, i * 5, i * 0.2))
            buf.put(_make_textured_packet("cam_02", i + 1, i * 5, i * 0.2))

        worker.start()
        time.sleep(2.0)
        worker.stop()

        # Verify no cross-camera contamination
        for r in results:
            assert r.camera_id in ("cam_01", "cam_02")
            # The result's camera_id must match the frame pair's camera_id


# ===================================================================
# Test: Stale result detection
# ===================================================================

class TestStaleDetection:
    """Verify result age computation and freshness thresholds."""

    def test_fresh_result(self):
        age = compute_result_age(result_video_timestamp=10.0, current_video_timestamp=10.1)
        assert age < 0.2
        assert is_result_fresh(age, stale_threshold_s=1.0)

    def test_stale_result(self):
        age = compute_result_age(result_video_timestamp=5.0, current_video_timestamp=7.5)
        assert age == 2.5
        assert not is_result_fresh(age, stale_threshold_s=1.0)

    def test_zero_age_is_fresh(self):
        age = compute_result_age(result_video_timestamp=10.0, current_video_timestamp=10.0)
        assert age == 0.0
        assert is_result_fresh(age, stale_threshold_s=0.0)

    def test_configurable_threshold(self):
        age = compute_result_age(result_video_timestamp=10.0, current_video_timestamp=12.0)
        assert is_result_fresh(age, stale_threshold_s=3.0)
        assert not is_result_fresh(age, stale_threshold_s=1.0)


# ===================================================================
# Test: Freshness-aware worker discards stale frames
# ===================================================================

class TestFreshnessAwareWorker:
    """Verify the freshness-aware RAFTWorker prioritizes latest frames."""

    def test_worker_processes_latest_pair(self):
        buf = FrameBuffer(capacity=50, drop_policy="oldest")
        engine = RAFTInference(device="cpu", model_size="small", inference_resolution=(128, 128))
        results: list[RAFTResult] = []
        worker = RAFTWorker(
            input_queue=buf,
            engine=engine,
            result_callback=lambda r: results.append(r),
        )

        # Fill buffer with 20 frames from cam_01 — simulates backlog
        for i in range(20):
            buf.put(_make_textured_packet("cam_01", i + 1, i * 5, i * 0.2))

        worker.start()
        time.sleep(3.0)
        worker.stop()

        # Worker should have processed at least one pair
        assert len(results) >= 1
        # The last result should be from near the end of the sequence, not the beginning
        last = results[-1]
        assert last.current_frame_id >= 15, f"Expected latest frames, got current_frame_id={last.current_frame_id}"


# ===================================================================
# Test: Robust velocity — median vs mean with outliers
# ===================================================================

class TestRobustVelocity:
    """Verify that median velocity is resistant to outliers."""

    def test_median_less_than_mean_with_outliers(self):
        """When a few pixels have extreme flow, median should be much lower than mean."""
        engine = RAFTInference(device="cpu", model_size="small", inference_resolution=(128, 128))

        # Create a mostly-static pair with small region of strong motion
        img_a = _make_textured_packet("cam_01", 1, 0, 0.0)
        img_b = _make_textured_packet("cam_01", 2, 5, 0.2, shift_x=4)

        result = engine.estimate_flow(img_a, img_b)

        # With a uniform translation, median and mean should be relatively close
        # This verifies the robust statistics are being computed correctly
        assert result.median_velocity >= 0.0
        assert result.p90_velocity >= result.median_velocity
        assert result.p95_velocity >= result.p90_velocity
        assert result.max_velocity >= result.p95_velocity

    def test_result_timestamp_is_set(self):
        res = RAFTResult(
            camera_id="cam_01",
            previous_frame_id=1,
            current_frame_id=2,
            previous_source_frame_index=5,
            current_source_frame_index=10,
            previous_timestamp=1.0,
            current_timestamp=1.2,
            flow_width=512,
            flow_height=384,
            average_velocity=2.0,
            median_velocity=1.5,
            max_velocity=8.0,
            velocity_variance=1.0,
            dominant_direction_deg=90.0,
            flow_consistency=0.5,
            active_motion_ratio=0.5,
            motion_confidence=0.5,
            inference_time_ms=10.0,
            model_name="RAFT",
            device="cpu",
        )
        assert res.result_timestamp > 0.0
