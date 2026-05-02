"""Fit PCA-whitening over a pile of feature vectors.

Given features F in R^{N x d}, we compute
    mu = mean(F, axis=0)
    C  = cov(F)
    C  = U diag(eig) U^T
    W  = diag(1 / sqrt(eig + eps)) U^T
so that (F - mu) @ W.T has zero mean and (approximately) unit covariance,
matching the "PCA-sphering" layer described in Section 3.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from .backbone import Backbone, PCAWhitening


def fit_pca_whitening(features: torch.Tensor, eps: float = 1e-6) -> PCAWhitening:
    """Fit PCA-whitening from a dense feature tensor [N, d]."""
    assert features.ndim == 2, "features must be [N, d]"
    features = features.double()
    mean = features.mean(dim=0)
    centered = features - mean
    cov = (centered.T @ centered) / max(1, centered.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp_min(eps)
    weight = (eigvecs / eigvals.sqrt()).T
    return PCAWhitening(mean.float(), weight.float())


@torch.no_grad()
def collect_features(
    backbone: Backbone,
    image_loader: Iterable[torch.Tensor],
    device: str = "cuda",
) -> torch.Tensor:
    """Run ``backbone.features`` on a loader that yields image batches in [0,1]."""
    backbone.eval().to(device)
    feats = []
    for batch in image_loader:
        batch = batch.to(device, non_blocking=True)
        feats.append(backbone.features(batch).cpu())
    return torch.cat(feats, dim=0)


def save(whitening: PCAWhitening, path: str | Path) -> None:
    whitening.save(path)


def load(path: str | Path) -> PCAWhitening:
    return PCAWhitening.load(path)
