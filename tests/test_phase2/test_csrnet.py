"""Unit tests for CSRNet model and inference engine."""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from crowd_management.ingestion.frame_packet import FramePacket
from crowd_management.models.csrnet.architecture import CSRNet
from crowd_management.models.csrnet.infer import CSRNetInference


def test_csrnet_architecture_shape():
    model = CSRNet(load_vgg_weights=False)
    model.eval()

    # Input: (1, 3, 128, 128)
    dummy_input = torch.zeros((1, 3, 128, 128), dtype=torch.float32)
    with torch.no_grad():
        out = model(dummy_input)

    # Output spatial resolution should be 128 / 8 = 16
    assert out.shape == (1, 1, 16, 16)
    # Output must be non-negative
    assert torch.all(out >= 0.0)


def test_csrnet_infer_deterministic():
    engine = CSRNetInference(device="cpu", inference_resolution=(128, 128))

    frame = np.ones((128, 128, 3), dtype=np.uint8) * 128
    packet = FramePacket(
        camera_id="cam_01",
        frame_id=1,
        source_frame_index=0,
        video_timestamp=0.0,
        wall_clock_timestamp=time.time(),
        frame=frame,
        source_fps=25.0,
        stream_fps=5.0,
        width=128,
        height=128,
    )

    res1 = engine.estimate(packet)
    res2 = engine.estimate(packet)

    assert np.isclose(res1.crowd_count, res2.crowd_count, atol=1e-5)
    assert np.allclose(res1.density_map, res2.density_map, atol=1e-5)
    assert res1.density_map.shape == (16, 16)
    assert np.isfinite(res1.crowd_count)
    assert res1.crowd_count >= 0.0
