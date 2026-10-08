"""Boundary-enhanced COAST with multi-scale bitemporal decoding."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision.models import resnet18

from models.SPGmodule import SpatialPriorGate
from models.ecological_prior import EcologicalPriorEncoder
from models.transformer_block import TransformerBlock


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ChannelAttention(nn.Module):
    """Channel attention for the concatenated multi-scale decoder features."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        average = self.mlp(functional.adaptive_avg_pool2d(feature, 1))
        maximum = self.mlp(torch.amax(feature, dim=(2, 3), keepdim=True))
        return torch.sigmoid(average + maximum)


class COAST(nn.Module):
    """Online ecological-prior-guided bitemporal Transformer.

    The ecological prior still gates only the 1/16 semantic features. The
    decoder progressively fuses bitemporal absolute differences from 1/8,
    1/4, 1/2, and full input resolution. A separate boundary head is used only
    for auxiliary supervision during training.
    """

    def __init__(self, in_channels: int = 8, num_classes: int = 1):
        super().__init__()
        base = resnet18(weights=None)
        self.stem_conv = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem_bn = base.bn1
        self.stem_relu = base.relu
        self.pool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3

        self.embed_dim = 128
        self.project = nn.Conv2d(256, self.embed_dim, kernel_size=1)
        self.transformer = TransformerBlock(
            dim=self.embed_dim, heads=8, mlp_dim=256)
        self.prior_encoder = EcologicalPriorEncoder(in_channels=in_channels)
        self.spg1 = SpatialPriorGate(self.embed_dim)
        self.spg2 = SpatialPriorGate(self.embed_dim)

        self.deep_fusion = ConvBlock(self.embed_dim * 2, 128)
        self.diff8 = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        self.decode8 = ConvBlock(128 + 64, 96)
        self.diff4 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        self.decode4 = ConvBlock(96 + 32, 64)
        self.diff2 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        self.decode2 = ConvBlock(64 + 32, 48)
        self.raw_diff = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.decode1 = ConvBlock(48 + 16, 32)
        self.scale8_projection = nn.Conv2d(96, 16, kernel_size=1, bias=False)
        self.scale4_projection = nn.Conv2d(64, 16, kernel_size=1, bias=False)
        self.scale2_projection = nn.Conv2d(48, 16, kernel_size=1, bias=False)
        self.scale1_projection = nn.Conv2d(32, 16, kernel_size=1, bias=False)
        self.ecam = ChannelAttention(64, reduction=8)
        self.segmentation_head = nn.Sequential(
            ConvBlock(64, 32),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )
        self.boundary_head = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    @staticmethod
    def _resize(feature: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return functional.interpolate(
            feature, size=reference.shape[-2:], mode="bilinear",
            align_corners=False)

    def _encode(self, image: torch.Tensor):
        stem = self.stem_relu(self.stem_bn(self.stem_conv(image)))
        level4 = self.layer1(self.pool(stem))
        level8 = self.layer2(level4)
        level16 = self.layer3(level8)
        batch, _, height, width = level16.shape
        semantic = self.project(level16).flatten(2).transpose(1, 2)
        semantic = self.transformer(semantic)
        semantic = semantic.transpose(1, 2).reshape(
            batch, self.embed_dim, height, width)
        return stem, level4, level8, semantic

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor,
                return_prior: bool = False):
        prior = self.prior_encoder(image_a)
        stem_a, level4_a, level8_a, semantic_a = self._encode(image_a)
        stem_b, level4_b, level8_b, semantic_b = self._encode(image_b)
        semantic_a = self.spg1(semantic_a, prior)
        semantic_b = self.spg2(semantic_b, prior)

        decoded16 = self.deep_fusion(torch.cat([semantic_a, semantic_b], dim=1))
        difference8 = self.diff8(torch.abs(level8_a - level8_b))
        decoded8 = self.decode8(torch.cat([
            self._resize(decoded16, difference8), difference8], dim=1))
        difference4 = self.diff4(torch.abs(level4_a - level4_b))
        decoded4 = self.decode4(torch.cat([
            self._resize(decoded8, difference4), difference4], dim=1))
        difference2 = self.diff2(torch.abs(stem_a - stem_b))
        decoded2 = self.decode2(torch.cat([
            self._resize(decoded4, difference2), difference2], dim=1))

        raw_difference = self.raw_diff(torch.abs(image_a - image_b))
        decoded1 = self.decode1(torch.cat([
            self._resize(decoded2, raw_difference), raw_difference], dim=1))
        output_size = decoded1.shape[-2:]
        multi_scale = torch.cat([
            functional.interpolate(
                self.scale8_projection(decoded8), size=output_size,
                mode="bilinear", align_corners=False),
            functional.interpolate(
                self.scale4_projection(decoded4), size=output_size,
                mode="bilinear", align_corners=False),
            functional.interpolate(
                self.scale2_projection(decoded2), size=output_size,
                mode="bilinear", align_corners=False),
            self.scale1_projection(decoded1),
        ], dim=1)
        fused = multi_scale * self.ecam(multi_scale) + multi_scale
        logits = self.segmentation_head(fused)
        boundary_logits = self._resize(self.boundary_head(decoded2), logits)
        if return_prior:
            return logits, prior, boundary_logits
        return logits
