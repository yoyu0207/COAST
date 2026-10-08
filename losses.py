"""Detection and auxiliary supervision losses used by COAST."""

import torch
from torch import nn
from torch.nn import functional as F


class BCEHybridLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight, self.dice_weight = bce_weight, dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce = self.bce(inputs, targets)
        probabilities = torch.sigmoid(inputs).reshape(-1)
        targets = targets.reshape(-1)
        intersection = (probabilities * targets).sum()
        dice = 1 - (2 * intersection + 1e-6) / (probabilities.sum() + targets.sum() + 1e-6)
        return self.bce_weight * bce + self.dice_weight * dice


def coast_loss(outputs, labels, posterior, criterion, alpha=0.1, boundary_weight=0.2):
    logits, online_prior, boundary_logits = outputs
    detection = criterion(logits, labels)
    prior = F.mse_loss(online_prior, posterior)
    dilated = F.max_pool2d(labels, 3, 1, 1)
    eroded = -F.max_pool2d(-labels, 3, 1, 1)
    boundary_target = ((dilated - eroded) > 0).to(labels.dtype)
    boundary = criterion(boundary_logits, boundary_target)
    return detection + alpha * prior + boundary_weight * boundary
