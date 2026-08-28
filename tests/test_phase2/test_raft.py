"""Unit and controlled mathematical sanity tests for RAFT optical flow engine."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from crowd_management.ingestion.frame_packet import FramePacket
from crowd_management.models.raft.infer import RAFTInference


@pytest.fixture(scope="module")
def raft_engine() -> RAFTInference:
    return RAFTInference(device="cpu", model_size="small", inference_resolution=(128, 128))


def create_texture_frame(width: int = 128, height: int = 128) -> np.ndarray:
    """Create a synthetic textured pattern with high spatial gradients for optical flow tracking."""
    x = np.linspace(0, 4 * np.pi, width)
    y = np.linspace(0, 4 * np.pi, height)
    xx, yy = np.meshgrid(x, y)
    pattern = (np.sin(xx) * np.cos(yy) * 127.0 + 128.0).astype(np.uint8)
    return cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)


def test_raft_temporal_validation(raft_engine: RAFTInference):
    frame = create_texture_frame()
    p1 = FramePacket("cam_01", 1, 0, 0.0, time.time(), frame, 25.0, 5.0, 128, 128)
    p2 = FramePacket("cam_02", 2, 5, 0.2, time.time(), frame, 25.0, 5.0, 128, 128)

    # Different cameras must raise ValueError
    with pytest.raises(ValueError, match="Camera mismatch"):
        raft_engine.estimate_flow(p1, p2)

    # Same frame ID must raise ValueError
    p3 = FramePacket("cam_01", 1, 0, 0.0, time.time(), frame, 25.0, 5.0, 128, 128)
    with pytest.raises(ValueError, match="Invalid temporal order"):
        raft_engine.estimate_flow(p1, p3)


def test_controlled_translation_right(raft_engine: RAFTInference):
    """Controlled test: translate textured image right by +8 pixels -> dominant direction ~0 degrees."""
    img_a = create_texture_frame()
    # Translate right (+8 in X)
    M = np.float32([[1, 0, 8], [0, 1, 0]])
    img_b = cv2.warpAffine(img_a, M, (128, 128))

    p_a = FramePacket("cam_01", 1, 0, 0.0, time.time(), img_a, 25.0, 5.0, 128, 128)
    p_b = FramePacket("cam_01", 2, 5, 0.2, time.time(), img_b, 25.0, 5.0, 128, 128)

    result = raft_engine.estimate_flow(p_a, p_b)

    assert result.average_velocity > 1.0
    # Expected angle near 0° (or 360°)
    angle = result.dominant_direction_deg
    assert angle < 45.0 or angle > 315.0, f"Expected dominant direction near 0°, got {angle}°"
    assert result.flow_consistency > 0.5


def test_controlled_translation_down(raft_engine: RAFTInference):
    """Controlled test: translate textured image down by +8 pixels -> dominant direction ~90 degrees."""
    img_a = create_texture_frame()
    # Translate down (+8 in Y)
    M = np.float32([[1, 0, 0], [0, 1, 8]])
    img_b = cv2.warpAffine(img_a, M, (128, 128))

    p_a = FramePacket("cam_01", 1, 0, 0.0, time.time(), img_a, 25.0, 5.0, 128, 128)
    p_b = FramePacket("cam_01", 2, 5, 0.2, time.time(), img_b, 25.0, 5.0, 128, 128)

    result = raft_engine.estimate_flow(p_a, p_b)

    assert result.average_velocity > 1.0
    # Expected angle near 90°
    angle = result.dominant_direction_deg
    assert 45.0 < angle < 135.0, f"Expected dominant direction near 90°, got {angle}°"
    assert result.flow_consistency > 0.5


def test_controlled_static_image(raft_engine: RAFTInference):
    """Controlled test: identical consecutive images -> velocity near zero."""
    img_a = create_texture_frame()

    p_a = FramePacket("cam_01", 1, 0, 0.0, time.time(), img_a, 25.0, 5.0, 128, 128)
    p_b = FramePacket("cam_01", 2, 5, 0.2, time.time(), img_a, 25.0, 5.0, 128, 128)

    result = raft_engine.estimate_flow(p_a, p_b)

    # Static frame pair should have minimal optical flow
    assert result.average_velocity < 1.0
