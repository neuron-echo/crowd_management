"""Weight loading, verification, and fail-fast management utilities for CSRNet."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from crowd_management.models.csrnet.architecture import CSRNet

logger = logging.getLogger(__name__)


class CSRNetCheckpointError(RuntimeError):
    """Raised when a CSRNet crowd-counting checkpoint is missing or invalid."""


def resolve_checkpoint_path(path: str | Path | None) -> Path | None:
    """Resolve checkpoint path relative to workspace root if not absolute."""
    if path is None or str(path) in ["default", "auto"]:
        default_chk = Path.cwd() / "checkpoints" / "csrnet_shanghaitech_part_a.pth"
        if default_chk.exists():
            return default_chk
        return None

    p = Path(path)
    if p.is_absolute():
        return p

    # Check relative to cwd
    cwd_path = Path.cwd() / p
    if cwd_path.exists():
        return cwd_path

    return p


def build_csrnet_model(
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    load_vgg_weights: bool = True,
    strict_checkpoint: bool = True,
) -> CSRNet:
    """Build and initialize CSRNet model with verified crowd-counting weights.

    Args:
        checkpoint_path: Path to trained CSRNet checkpoint file (`.pth`).
        device: PyTorch device ('cpu' or 'cuda').
        load_vgg_weights: Load VGG-16 frontend if checkpoint is not provided (test only).
        strict_checkpoint: If True, fail fast if checkpoint is missing or invalid.

    Returns:
        Verified, eval-mode CSRNet instance.

    Raises:
        CSRNetCheckpointError: If checkpoint is missing, corrupted, or has uninitialized head.
    """
    resolved_path = resolve_checkpoint_path(checkpoint_path)

    if strict_checkpoint and (resolved_path is None or not resolved_path.exists()):
        err_msg = (
            f"\n{'=' * 60}\n"
            f"ERROR: CSRNet crowd-counting checkpoint is missing or invalid!\n"
            f"Target path: {checkpoint_path} (resolved: {resolved_path})\n"
            f"Cannot produce valid crowd-count predictions without trained regression weights.\n"
            f"Please ensure 'checkpoints/csrnet_shanghaitech_part_a.pth' exists.\n"
            f"{'=' * 60}"
        )
        logger.error(err_msg)
        raise CSRNetCheckpointError(err_msg)

    # Instantiate base architecture
    model = CSRNet(load_vgg_weights=load_vgg_weights if resolved_path is None else False)

    weights_loaded = False
    if resolved_path is not None and resolved_path.exists():
        logger.info("Loading CSRNet checkpoint from: %s", resolved_path)
        try:
            try:
                state_dict = torch.load(str(resolved_path), map_location="cpu", weights_only=False)
            except Exception:
                state_dict = torch.load(str(resolved_path), map_location="cpu")

            if isinstance(state_dict, dict):
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            # Check that backend regression layers were loaded
            backend_missing = [k for k in missing if k.startswith("backend") or k.startswith("output_layer")]
            if backend_missing:
                raise CSRNetCheckpointError(
                    f"Checkpoint is missing crowd-density regression head keys: {backend_missing[:3]}"
                )

            weights_loaded = True
        except Exception as e:
            if strict_checkpoint:
                raise CSRNetCheckpointError(f"Failed to load valid CSRNet checkpoint from {resolved_path}: {e}") from e
            logger.warning("Failed to load checkpoint: %s; fallback allowed only for unit testing", e)

    param_count = sum(p.numel() for p in model.parameters())

    banner = (
        f"\n{'=' * 60}\n"
        f"CSRNet initialization:\n"
        f"    architecture           : CSRNet (VGG-16 + 6 Dilated Conv Backend)\n"
        f"    checkpoint             : {resolved_path if weights_loaded else 'NONE (Random/VGG fallback)'}\n"
        f"    crowd-counting weights : {'LOADED' if weights_loaded else 'UNINITIALIZED'}\n"
        f"    device                 : {device}\n"
        f"    parameter count        : {param_count:,}\n"
        f"{'=' * 60}"
    )
    logger.info(banner)
    print(banner)

    model.to(device)
    model.eval()
    return model
