"""Decision Engine module — model-agnostic crowd management logic.

This module has ZERO dependencies on:
- torch / torchvision
- cv2 / OpenCV
- CSRNet / RAFT
- FramePacket / FrameBuffer / FrameProducer

It consumes only CrowdState objects and VenueTopology.
"""
