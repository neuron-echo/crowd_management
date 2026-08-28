"""CSRNet crowd density inference engine."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from crowd_management.ingestion.frame_packet import FramePacket
from crowd_management.models.contracts import CSRNetResult
from crowd_management.models.csrnet.architecture import CSRNet
from crowd_management.models.csrnet.weights import build_csrnet_model


class CSRNetInference:
    """Persistent inference worker for CSRNet crowd density estimation."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        inference_resolution: tuple[int, int] | None = None,
        model_name: str = "CSRNet-VGG16",
    ) -> None:
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.inference_resolution = inference_resolution  # (width, height) or None for full
        self.model_name = model_name
        self.model: CSRNet = build_csrnet_model(
            checkpoint_path=checkpoint_path,
            device=self.device,
            load_vgg_weights=True,
        )

        # ImageNet mean and std for standard VGG-16 normalization
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def preprocess(self, bgr_frame: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        """Preprocess BGR numpy frame into normalized PyTorch tensor.

        Returns:
            Tuple of (tensor of shape (1, 3, H, W), (orig_w, orig_h))
        """
        orig_h, orig_w = bgr_frame.shape[:2]

        if self.inference_resolution is not None:
            proc_w, proc_h = self.inference_resolution
            # Ensure dimensions are divisible by 8 for 3 max-pooling layers
            proc_w = (proc_w // 8) * 8
            proc_h = (proc_h // 8) * 8
            frame_resized = cv2.resize(bgr_frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        else:
            # Pad or crop to multiple of 8 if necessary
            proc_w = (orig_w // 8) * 8
            proc_h = (orig_h // 8) * 8
            if proc_w != orig_w or proc_h != orig_h:
                frame_resized = cv2.resize(bgr_frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
            else:
                frame_resized = bgr_frame

        # BGR -> RGB
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self._mean) / self._std

        # HWC -> CHW -> NCHW
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor, (orig_w, orig_h)

    def estimate(self, packet: FramePacket) -> CSRNetResult:
        """Run CSRNet density estimation on a single FramePacket.

        Args:
            packet: Input FramePacket from ingestion stream.

        Returns:
            Immutable CSRNetResult object.
        """
        start_time = time.perf_counter()

        tensor, (orig_w, orig_h) = self.preprocess(packet.frame)

        with torch.no_grad():
            density_tensor = self.model(tensor)

        # density_tensor is (1, 1, H // 8, W // 8)
        density_map = density_tensor.squeeze().cpu().numpy().astype(np.float32)
        if density_map.ndim == 0:
            density_map = np.array([[float(density_map)]], dtype=np.float32)
        elif density_map.ndim == 1:
            density_map = density_map.reshape(1, -1)

        # Crowd count is strictly the integral (sum) of the density map
        crowd_count = float(np.sum(density_map))

        # Clamp tiny negative numerical noise to zero
        crowd_count = max(0.0, crowd_count)

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        map_h, map_w = density_map.shape[:2]

        return CSRNetResult(
            camera_id=packet.camera_id,
            frame_id=packet.frame_id,
            source_frame_index=packet.source_frame_index,
            video_timestamp=packet.video_timestamp,
            wall_clock_timestamp=packet.wall_clock_timestamp,
            crowd_count=crowd_count,
            density_map=density_map,
            density_map_width=map_w,
            density_map_height=map_h,
            input_width=orig_w,
            input_height=orig_h,
            inference_time_ms=inference_time_ms,
            model_name=self.model_name,
            device=self.device,
            frame=packet.frame,
        )
