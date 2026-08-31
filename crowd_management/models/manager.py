"""Central model lifecycle manager for Phase 2 neural network models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from crowd_management.config.schemas import ModelsConfig
from crowd_management.models.bytetrack.infer import ByteTrackInference
from crowd_management.models.csrnet.infer import CSRNetInference

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages lifecycle, memory residency, and execution device for vision models."""

    def __init__(
        self,
        device: str | None = None,
        csrnet_checkpoint: str | Path | None = None,
        csrnet_resolution: tuple[int, int] | None = (720, 540),
        bytetrack_model_path: str = "yolov8l.pt",
        bytetrack_tracker_config: str = "bytetrack.yaml",
        bytetrack_confidence: float = 0.3,
        bytetrack_motion_threshold: float = 2.0,
        enable_csrnet: bool = True,
        enable_bytetrack: bool = True,
        strict_checkpoint: bool = True,
        models_config: ModelsConfig | None = None,
    ) -> None:
        # Override from models_config if provided
        if models_config is not None:
            device = device or models_config.device
            enable_csrnet = models_config.csrnet.enabled
            csrnet_checkpoint = csrnet_checkpoint or models_config.csrnet.checkpoint_path
            csrnet_resolution = csrnet_resolution or models_config.csrnet.inference_resolution
            enable_bytetrack = models_config.bytetrack.enabled
            bytetrack_model_path = models_config.bytetrack.model_path
            bytetrack_tracker_config = models_config.bytetrack.tracker_config
            bytetrack_confidence = models_config.bytetrack.confidence_threshold
            bytetrack_motion_threshold = models_config.bytetrack.motion_threshold_px

        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._print_startup_device_report()

        self._csrnet: CSRNetInference | None = None
        self._bytetrack: ByteTrackInference | None = None

        if enable_csrnet:
            logger.info("Instantiating CSRNet engine (resolution=%s)...", csrnet_resolution)
            self._csrnet = CSRNetInference(
                checkpoint_path=csrnet_checkpoint,
                device=self.device,
                inference_resolution=csrnet_resolution,
                strict_checkpoint=strict_checkpoint,
            )

        if enable_bytetrack:
            logger.info(
                "Instantiating ByteTrack engine (model=%s tracker=%s)...",
                bytetrack_model_path, bytetrack_tracker_config,
            )
            self._bytetrack = ByteTrackInference(
                model_path=bytetrack_model_path,
                tracker_config=bytetrack_tracker_config,
                confidence_threshold=bytetrack_confidence,
                motion_threshold_px=bytetrack_motion_threshold,
                device=self.device,
            )

    def _print_startup_device_report(self) -> None:
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
        report = (
            f"\n{'=' * 60}\n"
            f"DEVICE HARDWARE REPORT:\n"
            f"    Configured device   : {self.device}\n"
            f"    CUDA available      : {cuda_avail}\n"
            f"    GPU Name            : {gpu_name}\n"
            f"    CSRNet device       : {self.device}\n"
            f"    ByteTrack device    : {self.device}\n"
            f"{'=' * 60}"
        )
        print(report)
        logger.info(report)

    @property
    def csrnet(self) -> CSRNetInference:
        if self._csrnet is None:
            raise RuntimeError("CSRNet is not enabled on this ModelManager")
        return self._csrnet

    @property
    def bytetrack(self) -> ByteTrackInference:
        if self._bytetrack is None:
            raise RuntimeError("ByteTrack is not enabled on this ModelManager")
        return self._bytetrack

    @property
    def device_info(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "csrnet_enabled": self._csrnet is not None,
            "bytetrack_enabled": self._bytetrack is not None,
        }
