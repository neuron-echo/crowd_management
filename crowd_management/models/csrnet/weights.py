"""Weight loading and management utilities for CSRNet."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from crowd_management.models.csrnet.architecture import CSRNet

logger = logging.getLogger(__name__)


def build_csrnet_model(
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    load_vgg_weights: bool = True,
) -> CSRNet:
    """Build and initialize CSRNet model with specified checkpoint or VGG-16 backbone.

    Args:
        checkpoint_path: Optional path to a trained CSRNet `.pth` state dictionary.
        device: PyTorch device ('cpu' or 'cuda').
        load_vgg_weights: If True and checkpoint_path is None, load VGG-16 pretrained frontend.

    Returns:
        Initialized and eval-mode CSRNet instance.
    """
    model = CSRNet(load_vgg_weights=load_vgg_weights)

    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if p.exists():
            logger.info("Loading CSRNet checkpoint from: %s", p)
            state_dict = torch.load(str(p), map_location="cpu", weights_only=True)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning("Checkpoint path %s does not exist; using default VGG backbone", p)

    model.to(device)
    model.eval()
    return model
