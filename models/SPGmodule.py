"""Residual spatial prior gating: F_out = F + gamma * (F * attention)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPriorGate(nn.Module):
    def __init__(self, in_channels):
        super(SpatialPriorGate, self).__init__()
        self.prior_proj = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid() 
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feature, spatial_prior):
        if spatial_prior.shape[2:] != feature.shape[2:]:
            prior_aligned = F.interpolate(
                spatial_prior, 
                size=feature.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        else:
            prior_aligned = spatial_prior
        attention = self.prior_proj(prior_aligned)
        out = feature + self.gamma * (feature * attention)
        return out
