"""CSRNet crowd density inference engine — paper-accurate multi-scale inference.

Inference method matches csrnetb.py (original paper pipeline):
  - Scales: [0.7, 1.0, 1.3]
  - Scale correction: raw_count / (s * s)
  - Final count: mean across scales
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from crowd_management.ingestion.frame_packet import FramePacket
from crowd_management.models.contracts import CSRNetResult
from crowd_management.models.csrnet.architecture import CSRNet
from crowd_management.models.csrnet.weights import build_csrnet_model, resolve_checkpoint_path

# Multi-scale factors — identical to original csrnetb.py
_SCALES = [0.7, 1.0, 1.3]


class CSRNetInference:
    """Persistent inference worker for CSRNet crowd density estimation.

    Uses the same multi-scale inference as the original paper (csrnetb.py):
      scales=[0.7, 1.0, 1.3], scale correction count/(s*s), final=mean.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        inference_resolution: tuple[int, int] | None = None,
        model_name: str = "CSRNet-VGG16",
        strict_checkpoint: bool = True,
    ) -> None:
        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.inference_resolution = inference_resolution  # (width, height) optional override
        self.model_name = model_name
        self.resolved_checkpoint = resolve_checkpoint_path(checkpoint_path)
        self.checkpoint_identifier = (
            self.resolved_checkpoint.name if self.resolved_checkpoint else "CSRNet-ShanghaiTech-A"
        )

        self.model: CSRNet = build_csrnet_model(
            checkpoint_path=checkpoint_path,
            device=self.device,
            strict_checkpoint=strict_checkpoint,
        )

        # Transform — identical to csrnetb.py
        self._transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _multiscale_count(
        self, pil_img: Image.Image
    ) -> tuple[float, np.ndarray]:
        """Run multi-scale inference identical to csrnetb.py.

        Returns:
            (final_crowd_count, density_map_at_scale_1)
        """
        orig_w, orig_h = pil_img.size
        counts = []
        density_at_full_scale: np.ndarray | None = None

        for s in _SCALES:
            new_w = int(orig_w * s)
            new_h = int(orig_h * s)
            # Enforce multiple of 8 (3 max-pool layers → 8× downsampling)
            new_w = new_w - new_w % 8
            new_h = new_h - new_h % 8

            scaled_img = pil_img.resize((new_w, new_h))
            tensor = self._transform(scaled_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                density_tensor = self.model(tensor)

            density_np = density_tensor.squeeze().cpu().numpy().astype(np.float32)
            if density_np.ndim == 0:
                density_np = np.array([[float(density_np)]], dtype=np.float32)

            raw_count = float(np.sum(density_np))
            # Scale correction — same as csrnetb.py: count / (s * s)
            corrected_count = raw_count / (s * s)
            counts.append(corrected_count)

            # Keep the density map at scale 1.0 for visualization
            if abs(s - 1.0) < 1e-6:
                density_at_full_scale = density_np

        final_count = float(np.mean(counts))
        final_count = max(0.0, final_count)

        if density_at_full_scale is None:
            density_at_full_scale = np.zeros((orig_h // 8, orig_w // 8), dtype=np.float32)

        return final_count, density_at_full_scale

    def estimate(self, packet: FramePacket) -> CSRNetResult:
        """Run multi-scale CSRNet density estimation on a single FramePacket.

        Args:
            packet: Input FramePacket from ingestion stream.

        Returns:
            Immutable CSRNetResult with crowd_count from multi-scale inference.
        """
        start_time = time.perf_counter()

        bgr_frame = packet.frame
        orig_h, orig_w = bgr_frame.shape[:2]

        # Convert BGR → RGB → PIL (same pipeline as csrnetb.py)
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Optional resolution override (resize before multi-scale)
        if self.inference_resolution is not None:
            proc_w, proc_h = self.inference_resolution
            proc_w = max(8, (proc_w // 8) * 8)
            proc_h = max(8, (proc_h // 8) * 8)
            pil_img = pil_img.resize((proc_w, proc_h))

        crowd_count, density_map = self._multiscale_count(pil_img)
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
            checkpoint_identifier=self.checkpoint_identifier,
            frame=packet.frame,
        )
