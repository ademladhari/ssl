"""Watermarking losses and FPR <-> hypercone-angle conversion.

Zero-bit detection region (Eq. 3):
    D = { x in R^d : |x^T a| > ||x|| cos(theta) }
with key ``a`` on the unit sphere. The paper's "robustness estimate" (Eq. 5)
    r(x) = (x^T a)^2 - ||x||^2 cos^2(theta)
is positive iff x is inside the hypercone, so we *minimize* ``-r(x)``.

Multi-bit embedding (Eq. 7) uses a hinge loss with margin mu so that each
projection (x^T a_i) reaches sign m_i by at least mu.

The FPR of the hypercone detector (Eq. 4) is
    FPR = 1 - I_{cos^2(theta)}(1/2, (d-1)/2)
        = I_{sin^2(theta)}((d-1)/2, 1/2),
which we invert with ``scipy.special.betaincinv``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import betaincinv, betainc


def theta_from_fpr(fpr: float, d: int = 2048) -> float:
    """Return the cone angle theta (radians) achieving the requested FPR."""
    assert 0.0 < fpr < 1.0
    sin2 = betaincinv((d - 1) / 2.0, 0.5, fpr)
    sin2 = float(np.clip(sin2, 0.0, 1.0))
    return math.asin(math.sqrt(sin2))


def fpr_from_theta(theta: float, d: int = 2048) -> float:
    """Forward direction of Eq. 4 (useful for sanity checks)."""
    sin2 = math.sin(theta) ** 2
    return float(betainc((d - 1) / 2.0, 0.5, sin2))


def zero_bit_loss(x: torch.Tensor, a: torch.Tensor, cos_theta: float) -> torch.Tensor:
    """Minimize to push each row of x into the dual hypercone of key ``a``.

    Args:
        x: feature tensor [B, d].
        a: unit-norm carrier [d].
        cos_theta: cos(theta) for the target FPR.
    """
    proj = x @ a
    norm_sq = (x * x).sum(dim=-1)
    robustness = proj.pow(2) - norm_sq * (cos_theta ** 2)
    return -robustness.mean()


def multi_bit_loss(
    x: torch.Tensor,
    carriers: torch.Tensor,
    messages: torch.Tensor,
    margin: float = 5.0,
) -> torch.Tensor:
    """Hinge loss (Eq. 7). ``carriers`` is [k, d], ``messages`` is [B, k] in {-1, +1}."""
    proj = x @ carriers.T  # [B, k]
    hinge = F.relu(margin - proj * messages)
    return hinge.mean()


def image_mse_loss(image: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """Per-pixel MSE between two image tensors of shape [B, C, H, W] in [0, 1]."""
    return F.mse_loss(image, original)
