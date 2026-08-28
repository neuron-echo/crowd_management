"""RAFT optical flow inference and crowd motion analytics engine."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large, raft_small

from crowd_management.ingestion.frame_packet import FramePacket
from crowd_management.models.contracts import RAFTResult


class RAFTInference:
    """Persistent RAFT optical flow engine for dense velocity and direction estimation."""

    def __init__(
        self,
        device: str | None = None,
        model_size: str = "large",
        inference_resolution: tuple[int, int] | None = (512, 384),
        model_name: str | None = None,
    ) -> None:
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_size = model_size
        self.inference_resolution = inference_resolution  # (width, height)
        self.model_name = model_name or f"torchvision-raft-{model_size}"

        # Load official torchvision RAFT
        if model_size == "small":
            self.model = raft_small(weights="DEFAULT")
        else:
            self.model = raft_large(weights=Raft_Large_Weights.DEFAULT)

        self.model.to(self.device)
        self.model.eval()

    def _preprocess_frame(self, bgr_frame: np.ndarray) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
        """Convert BGR frame into normalized PyTorch tensor in [-1.0, 1.0] range.

        Returns:
            Tuple of (tensor of shape (1, 3, H, W), (proc_w, proc_h), (orig_w, orig_h))
        """
        orig_h, orig_w = bgr_frame.shape[:2]

        if self.inference_resolution is not None:
            proc_w, proc_h = self.inference_resolution
        else:
            proc_w, proc_h = orig_w, orig_h

        # Dimensions must be divisible by 8 for RAFT encoders
        proc_w = max(8, (proc_w // 8) * 8)
        proc_h = max(8, (proc_h // 8) * 8)

        if (proc_w, proc_h) != (orig_w, orig_h):
            resized = cv2.resize(bgr_frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = bgr_frame

        # BGR -> RGB and normalize to [-1.0, 1.0]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        norm = (rgb / 127.5) - 1.0

        # HWC -> CHW -> NCHW
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor, (proc_w, proc_h), (orig_w, orig_h)

    def estimate_flow(
        self,
        prev_packet: FramePacket,
        curr_packet: FramePacket,
    ) -> RAFTResult:
        """Estimate optical flow and velocity analytics between two consecutive frames from the same camera.

        Args:
            prev_packet: FramePacket at time t - 1
            curr_packet: FramePacket at time t

        Returns:
            Immutable RAFTResult with computed velocity and direction statistics.
        """
        if prev_packet.camera_id != curr_packet.camera_id:
            raise ValueError(
                f"Camera mismatch in RAFT estimate_flow: {prev_packet.camera_id} vs {curr_packet.camera_id}"
            )
        if prev_packet.frame_id >= curr_packet.frame_id:
            raise ValueError(
                f"Invalid temporal order: prev_frame_id ({prev_packet.frame_id}) >= curr_frame_id ({curr_packet.frame_id})"
            )

        start_time = time.perf_counter()

        img1, (pw1, ph1), (orig_w, orig_h) = self._preprocess_frame(prev_packet.frame)
        img2, _, _ = self._preprocess_frame(curr_packet.frame)

        with torch.no_grad():
            list_of_flows = self.model(img1, img2)
            flow_tensor = list_of_flows[-1]  # Final predicted flow (1, 2, H, W)

        # Flow shape: (2, H, W) where flow[0] is u (horizontal) and flow[1] is v (vertical)
        flow_np = flow_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        u_flow = flow_np[0]
        v_flow = flow_np[1]

        # Rescale optical flow values to native resolution coordinates if resized
        scale_x = orig_w / float(pw1)
        scale_y = orig_h / float(ph1)
        u_flow_scaled = u_flow * scale_x
        v_flow_scaled = v_flow * scale_y

        # Velocity magnitude in pixels/frame
        magnitude = np.sqrt(u_flow_scaled**2 + v_flow_scaled**2)

        # Velocity statistics
        avg_velocity = float(np.mean(magnitude))
        median_velocity = float(np.median(magnitude))
        max_velocity = float(np.max(magnitude))
        velocity_var = float(np.var(magnitude))

        # Direction angle in degrees: 0° to 360° (0° = Right/East, 90° = Down/South, 180° = Left/West, 270° = Up/North)
        # Vector sum for dominant direction
        u_sum = float(np.sum(u_flow_scaled))
        v_sum = float(np.sum(v_flow_scaled))
        dom_rad = np.arctan2(v_sum, u_sum)
        dominant_deg = float(np.degrees(dom_rad) % 360.0)

        # Directional consistency: ||sum(v)|| / sum(||v||)
        sum_mag = float(np.sum(magnitude))
        vector_sum_mag = float(np.sqrt(u_sum**2 + v_sum**2))
        flow_consistency = float(vector_sum_mag / (sum_mag + 1e-6))
        flow_consistency = float(np.clip(flow_consistency, 0.0, 1.0))

        # Active motion ratio (motion > 0.5 pixels)
        active_motion_pixels = np.count_nonzero(magnitude > 0.5)
        active_motion_ratio = float(active_motion_pixels / float(magnitude.size))

        # Motion confidence score
        motion_confidence = float(np.clip(active_motion_ratio * (0.5 + 0.5 * flow_consistency), 0.0, 1.0))

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        # Downsample flow field (e.g. 32x24 grid) for lightweight visual inspection
        summary_grid_w, summary_grid_h = 32, 24
        u_summary = cv2.resize(u_flow_scaled, (summary_grid_w, summary_grid_h), interpolation=cv2.INTER_AREA)
        v_summary = cv2.resize(v_flow_scaled, (summary_grid_w, summary_grid_h), interpolation=cv2.INTER_AREA)
        flow_summary = np.stack([u_summary, v_summary], axis=-1)  # (24, 32, 2)

        return RAFTResult(
            camera_id=curr_packet.camera_id,
            previous_frame_id=prev_packet.frame_id,
            current_frame_id=curr_packet.frame_id,
            previous_source_frame_index=prev_packet.source_frame_index,
            current_source_frame_index=curr_packet.source_frame_index,
            previous_timestamp=prev_packet.video_timestamp,
            current_timestamp=curr_packet.video_timestamp,
            flow_width=orig_w,
            flow_height=orig_h,
            average_velocity=avg_velocity,
            median_velocity=median_velocity,
            max_velocity=max_velocity,
            velocity_variance=velocity_var,
            dominant_direction_deg=dominant_deg,
            flow_consistency=flow_consistency,
            active_motion_ratio=active_motion_ratio,
            motion_confidence=motion_confidence,
            inference_time_ms=inference_time_ms,
            model_name=self.model_name,
            device=self.device,
            flow_summary=flow_summary,
        )
