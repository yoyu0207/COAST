"""Learn an online ecological prior from T1 under GWDA posterior soft supervision."""

import torch
import torch.nn as nn


class EcologicalPriorEncoder(nn.Module):

    def __init__(self, in_channels: int = 8):
        super().__init__()
        self.eco_score = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_diffusion = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, kernel_size=3, padding=4,
                      dilation=4, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 1, kernel_size=3, padding=8,
                      dilation=8, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eco   = self.eco_score(x)
        prior = self.spatial_diffusion(eco)
        return prior                                # [B, 1, H, W]

    def get_channel_weights(self) -> torch.Tensor:
        """Return the learned input-channel weights."""
        return self.eco_score[0].weight.squeeze().detach()
