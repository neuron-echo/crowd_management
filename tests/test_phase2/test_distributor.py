"""Unit tests for FrameDistributor fan-out logic."""

from __future__ import annotations

import time

import numpy as np
import pytest

from crowd_management.ingestion.frame_packet import FramePacket
from crowd_management.pipeline.distributor import FrameDistributor


@pytest.fixture
def sample_packet() -> FramePacket:
    return FramePacket(
        camera_id="cam_01",
        frame_id=1,
        source_frame_index=0,
        video_timestamp=0.0,
        wall_clock_timestamp=time.time(),
        frame=np.zeros((64, 64, 3), dtype=np.uint8),
        source_fps=25.0,
        stream_fps=5.0,
        width=64,
        height=64,
    )


def test_distributor_broadcast(sample_packet: FramePacket):
    dist = FrameDistributor(csrnet_buffer_size=5, raft_buffer_size=5)
    dist.dispatch(sample_packet)

    assert len(dist.csrnet_queue) == 1
    assert len(dist.raft_queue) == 1
    assert dist.packets_routed == 1

    p1 = dist.csrnet_queue.get(block=False)
    p2 = dist.raft_queue.get(block=False)

    assert p1.frame_id == sample_packet.frame_id
    assert p2.frame_id == sample_packet.frame_id
    assert p1.camera_id == "cam_01"


def test_distributor_overflow_bounded():
    dist = FrameDistributor(csrnet_buffer_size=2, raft_buffer_size=2, drop_policy="oldest")

    for i in range(5):
        pkt = FramePacket(
            camera_id="cam_01",
            frame_id=i + 1,
            source_frame_index=i * 5,
            video_timestamp=float(i) * 0.2,
            wall_clock_timestamp=time.time(),
            frame=np.zeros((32, 32, 3), dtype=np.uint8),
            source_fps=25.0,
            stream_fps=5.0,
            width=32,
            height=32,
        )
        dist.dispatch(pkt)

    assert len(dist.csrnet_queue) <= 2
    assert len(dist.raft_queue) <= 2
    assert dist.csrnet_queue.frames_dropped >= 3


def test_distributor_multicamera_routing():
    dist = FrameDistributor(csrnet_buffer_size=10, raft_buffer_size=10)

    for cam in ["cam_01", "cam_02"]:
        pkt = FramePacket(
            camera_id=cam,
            frame_id=1,
            source_frame_index=0,
            video_timestamp=0.0,
            wall_clock_timestamp=time.time(),
            frame=np.zeros((32, 32, 3), dtype=np.uint8),
            source_fps=25.0,
            stream_fps=5.0,
            width=32,
            height=32,
        )
        dist.dispatch(pkt)

    stats = dist.stats()
    assert stats["camera_distribution"]["cam_01"] == 1
    assert stats["camera_distribution"]["cam_02"] == 1
    assert dist.packets_routed == 2
