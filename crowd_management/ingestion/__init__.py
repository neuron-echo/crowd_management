"""Ingestion subsystem for real-time video capture and buffering."""

from crowd_management.ingestion.consumer import FrameConsumer, QueueFrameConsumer
from crowd_management.ingestion.frame_buffer import FrameBuffer
from crowd_management.ingestion.frame_packet import FramePacket
from crowd_management.ingestion.multi_camera import MultiCameraProducer
from crowd_management.ingestion.producer import FrameProducer
from crowd_management.ingestion.video_source import BaseFrameSource, VideoFileSource

__all__ = [
    "BaseFrameSource",
    "FrameBuffer",
    "FrameConsumer",
    "FramePacket",
    "FrameProducer",
    "MultiCameraProducer",
    "QueueFrameConsumer",
    "VideoFileSource",
]
