"""Central model lifecycle manager for Phase 2 neural network models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from crowd_management.models.csrnet.infer import CSRNetInference
from crowd_management.models.raft.infer import RAFTInference

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages lifecycle, memory residency, and execution device for vision models."""

    def __init__(
        self,
        device: str | None = None,
        csrnet_checkpoint: str | Path | None = None,
        csrnet_resolution: tuple[int, int] | None = None,
        raft_model_size: str = "large",
        raft_resolution: tuple[int, int] | None = (512, 384),
        enable_csrnet: bool = True,
        enable_raft: bool = True,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Initializing ModelManager on device: %s", self.device)

        self._csrnet: CSRNetInference | None = None
        self._raft: RAFTInference | None = None

        if enable_csrnet:
            logger.info("Instantiating CSRNet engine (resolution=%s)...", csrnet_resolution)
            self._csrnet = CSRNetInference(
                checkpoint_path=csrnet_checkpoint,
                device=self.device,
                inference_resolution=csrnet_resolution,
            )

        if enable_raft:
            logger.info("Instantiating RAFT optical flow engine (size=%s, resolution=%s)...", raft_model_size, raft_resolution)
            self._raft = RAFTInference(
                device=self.device,
                model_size=raft_model_size,
                inference_resolution=raft_resolution,
            )

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
            "csrnet_enabled": self._csrnet is not None,
            "raft_enabled": self._raft is not None,
        }
        if torch.cuda.is_available() and self.device == "cuda":
            info["device_name"] = torch.cuda.get_device_name(0)
        return info
