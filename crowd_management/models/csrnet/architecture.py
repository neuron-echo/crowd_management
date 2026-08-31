"""CSRNet PyTorch architecture — matches the original training implementation (model.py).

Architecture from: Li et al., "CSRNet: Dilated Convolutional Neural Networks for
Understanding the Highly Congested Scenes", CVPR 2018.

Frontend : VGG-16 first 10 conv layers (3 max-pooling stages → 8× spatial downsampling)
Backend  : 6 dilated conv layers (dilation rate = 2)
Output   : 1×1 conv producing a single-channel density map

NOTE: The forward pass does NOT apply relu on the output — this matches the original
training code and the PartAmodel_best.pth checkpoint exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def make_layers(
    cfg: list,
    in_channels: int = 3,
    batch_norm: bool = False,
    dilation: bool = False,
) -> nn.Sequential:
    """Build a sequential block of Conv-ReLU (or Conv-BN-ReLU) layers.

    Args:
        cfg:         List of channel sizes or 'M' for MaxPool2d.
        in_channels: Number of input channels.
        batch_norm:  Insert BatchNorm2d between Conv and ReLU.
        dilation:    Use dilation rate 2 (for backend dilated convs).
    """
    d_rate = 2 if dilation else 1
    layers: list[nn.Module] = []
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(
                in_channels,
                int(v),
                kernel_size=3,
                padding=d_rate,
                dilation=d_rate,
            )
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(int(v)), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = int(v)
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    """CSRNet crowd density estimation model.

    Exactly matches the original training architecture so checkpoint keys align
    with state_dict keys in PartAmodel_best.pth.
    """

    def __init__(self, load_weights: bool = False) -> None:
        """
        Args:
            load_weights: If False, initialise the frontend from pretrained
                          VGG-16 weights (same behaviour as original model.py).
                          Set True when loading a full CSRNet checkpoint so
                          the VGG download is skipped.
        """
        super().__init__()

        self.frontend_feat = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
        self.backend_feat  = [512, 512, 512, 256, 128, 64]

        self.frontend     = make_layers(self.frontend_feat)
        self.backend      = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self._initialize_weights()

        if not load_weights:
            # Pre-fill frontend with ImageNet-pretrained VGG-16 weights
            self._load_vgg_frontend()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass → density map of shape (B, 1, H/8, W/8)."""
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x   # NO relu — matches original training code and checkpoint

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _load_vgg_frontend(self) -> None:
        """Copy matching VGG-16 weights into the frontend (original model.py logic)."""
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            frontend_dict = self.frontend.state_dict()
            vgg_dict = vgg.features.state_dict()
            for k in frontend_dict:
                if k in vgg_dict:
                    frontend_dict[k] = vgg_dict[k]
            self.frontend.load_state_dict(frontend_dict)
        except Exception:
            pass  # fallback to random init if download fails
