"""Official CSRNet PyTorch architecture for crowd density estimation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16


class CSRNet(nn.Module):
    """Congested Scene Recognition Network (CSRNet) as proposed by Li et al. (CVPR 2018).

    Frontend: VGG-16 first 10 convolutional layers with 3 max-pooling stages (downsampling by 8x).
    Backend: 6 dilated convolutional layers with dilation rate = 2.
    Output: 1x1 convolution producing a single-channel spatial density map.
    """

    def __init__(self, load_vgg_weights: bool = True) -> None:
        super().__init__()
        self.seen = 0

        # Frontend: VGG-16 first 10 conv layers (features[0:23])
        self.frontend_feat = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = self._make_layers(self.frontend_feat, in_channels=3)
        self.backend = self._make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self._initialize_weights()

        if load_vgg_weights:
            self._load_vgg_frontend()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor of shape (B, 3, H, W)

        Returns:
            Density map tensor of shape (B, 1, H // 8, W // 8)
        """
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        # Density values must be non-negative in physical crowd density formulation
        return torch.relu(x)

    def _make_layers(
        self,
        cfg: list[int | str],
        in_channels: int = 3,
        batch_norm: bool = False,
        dilation: bool = False,
    ) -> nn.Sequential:
        d_rate = 2 if dilation else 1
        layers: list[nn.Module] = []
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
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

    def _initialize_weights(self) -> None:
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)

    def _load_vgg_frontend(self) -> None:
        try:
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
            vgg_state = vgg.features.state_dict()
            frontend_state = self.frontend.state_dict()
            matched_state = {}
            for (k_front, v_front), (k_vgg, v_vgg) in zip(frontend_state.items(), vgg_state.items()):
                if v_front.shape == v_vgg.shape:
                    matched_state[k_front] = v_vgg
            self.frontend.load_state_dict(matched_state, strict=False)
        except Exception:
            # Fallback to random initialization if download fails
            pass
