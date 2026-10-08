"""Reproducible initialization, checkpoint loading and pooled pixel metrics."""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random

import numpy as np
import torch


def set_global_seed(seed, warn_only=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_model(checkpoint, device):
    from models import COAST
    model = COAST().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model.eval()


class MetricTracker:
    def __init__(self, threshold=0.5):
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = self.tn = self.fp = self.fn = 0

    @torch.no_grad()
    def update(self, inputs, targets):
        predictions = (torch.sigmoid(inputs) > self.threshold).long()
        targets = targets.long()
        self.tp += (predictions * targets).sum().item()
        self.tn += ((1 - predictions) * (1 - targets)).sum().item()
        self.fp += (predictions * (1 - targets)).sum().item()
        self.fn += ((1 - predictions) * targets).sum().item()

    def get_metrics(self):
        epsilon = 1e-7
        precision = self.tp / (self.tp + self.fp + epsilon)
        recall = self.tp / (self.tp + self.fn + epsilon)
        return {
            "Precision": precision, "Recall": recall,
            "F1": 2 * precision * recall / (precision + recall + epsilon),
            "IoU": self.tp / (self.tp + self.fp + self.fn + epsilon),
            "OA": (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + epsilon),
        }
