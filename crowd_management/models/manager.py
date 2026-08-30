"""Central model lifecycle manager for Phase 2 neural network models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from crowd_management.config.schemas import ModelsConfig
from crowd_management.models.csrnet.infer import CSRNetInference
from crowd_management.models.raft.infer import RAFTInference

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages lifecycle, memory residency, and execution device for vision models."""

    def __init__(
        self,
        device: str | None = None,
        csrnet_checkpoint: str | Path | None = None,
        csrnet_resolution: tuple[int, int] | None = (720, 540),
        raft_model_size: str = "large",
        raft_resolution: tuple[int, int] | None = (512, 384),
        enable_csrnet: bool = True,
        enable_raft: bool = True,
        strict_checkpoint: bool = True,
        models_config: ModelsConfig | None = None,
    ) -> None:
        # Override from models_config if provided
        if models_config is not None:
            device = device or models_config.device
            enable_csrnet = models_config.csrnet.enabled
            csrnet_checkpoint = csrnet_checkpoint or models_config.csrnet.checkpoint_path
            csrnet_resolution = csrnet_resolution or models_config.csrnet.inference_resolution
            enable_raft = models_config.raft.enabled
            raft_model_size = raft_model_size or models_config.raft.model_size
            raft_resolution = raft_resolution or models_config.raft.inference_resolution

        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._print_startup_device_report()

        self._csrnet: CSRNetInference | None = None
        self._raft: RAFTInference | None = None

        if enable_csrnet:
            logger.info("Instantiating CSRNet engine (resolution=%s)...", csrnet_resolution)
            self._csrnet = CSRNetInference(
                checkpoint_path=csrnet_checkpoint,
                device=self.device,
                inference_resolution=csrnet_resolution,
                strict_checkpoint=strict_checkpoint,
            )

        if enable_raft:
            logger.info("Instantiating RAFT optical flow engine (size=%s, resolution=%s)...", raft_model_size, raft_resolution)
            self._raft = RAFTInference(
                device=self.device,
                model_size=raft_model_size,
                inference_resolution=raft_resolution,
            )

    def _print_startup_device_report(self) -> None:
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
        report = (
            f"\n{'=' * 60}\n"
            f"DEVICE HARDWARE REPORT:\n"
            f"    Configured device : {self.device}\n"
            f"    CUDA available    : {cuda_avail}\n"
            f"    GPU Name          : {gpu_name}\n"
            f"    CSRNet device     : {self.device}\n"
            f"    RAFT device       : {self.device}\n"
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
    def raft(self) -> RAFTInference:
        if self._raft is None:
            raise RuntimeError("RAFT is not enabled on this ModelManager")
        return self._raft

    @property
    def device_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "csrnet_enabled": self._csrnet is not None,
            "raft_enabled": self._raft is not None,
        }
        return info
