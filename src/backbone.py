"""DINO ResNet-50 backbone + PCA-whitening wrapped as a single feature extractor.

The backbone phi: [B, 3, H, W] -> [B, d] is built from:
  1. A pretrained DINO ResNet-50 (fc stripped, global-average-pooled).
  2. A learned PCA-whitening layer (mean subtraction + linear whitening),
     producing centered features with unit covariance (Section 3.1).

Images fed to phi are assumed to be in [0, 1] (float). ImageNet mean/std
normalization is applied inside the module so that gradients flow all the way
from the loss to the raw pixel tensor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PCAWhitening(nn.Module):
    """Applies y = W (x - mu). Parameters are frozen (not trainable)."""

    def __init__(self, mean: torch.Tensor, weight: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("weight", weight.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) @ self.weight.T

    @classmethod
    def load(cls, path: str | Path) -> "PCAWhitening":
        state = torch.load(str(path), map_location="cpu")
        return cls(state["mean"], state["weight"])

    def save(self, path: str | Path) -> None:
        torch.save({"mean": self.mean.detach().cpu(),
                    "weight": self.weight.detach().cpu()}, str(path))


def _load_dino_resnet50() -> nn.Module:
    """Load DINO ResNet-50 from torch.hub and strip its classification head."""
    model = torch.hub.load("facebookresearch/dino:main", "dino_resnet50",
                           trust_repo=True)
    if hasattr(model, "fc"):
        model.fc = nn.Identity()
    return model


class Backbone(nn.Module):
    """phi(I) = whitening(dino_resnet50(normalize(I)))."""

    def __init__(
        self,
        whitening: Optional[PCAWhitening] = None,
        feat_dim: int = 2048,
    ):
        super().__init__()
        self.resnet = _load_dino_resnet50()
        for p in self.resnet.parameters():
            p.requires_grad_(False)
        self.resnet.eval()
        self.whitening = whitening
        self.feat_dim = feat_dim

        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean)
        self.register_buffer("imagenet_std", std)

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.imagenet_mean) / self.imagenet_std

    def features(self, images: torch.Tensor) -> torch.Tensor:
        """Backbone features before whitening (used to fit PCA)."""
        return self.resnet(self._normalize(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.features(images)
        if self.whitening is not None:
            feats = self.whitening(feats)
        return feats

    def attach_whitening(self, whitening: PCAWhitening) -> None:
        self.whitening = whitening

    def train(self, mode: bool = True):  # keep resnet in eval
        super().train(mode)
        self.resnet.eval()
        return self
