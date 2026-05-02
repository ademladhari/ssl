"""Detection (zero-bit) and decoding (multi-bit)."""

from __future__ import annotations

import math
from typing import Tuple

import torch

from .backbone import Backbone
from .losses import theta_from_fpr


@torch.no_grad()
def detect_zero_bit(
    backbone: Backbone,
    images: torch.Tensor,
    key: torch.Tensor,
    fpr: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (detected_bool [B], score [B]).

    Score > 0 <=> x in the dual hypercone (x^T a)^2 > ||x||^2 cos^2(theta).
    """
    d = backbone.feat_dim
    theta = theta_from_fpr(fpr, d=d)
    cos_theta = math.cos(theta)

    key = key.to(images.device).float()
    key = key / key.norm().clamp_min(1e-12)

    x = backbone(images)
    proj = x @ key
    norm_sq = (x * x).sum(dim=-1)
    score = proj.pow(2) - norm_sq * (cos_theta ** 2)
    return score > 0, score


@torch.no_grad()
def decode_multi_bit(
    backbone: Backbone,
    images: torch.Tensor,
    carriers: torch.Tensor,
) -> torch.Tensor:
    """Return [B, k] tensor of decoded bits in {-1, +1}."""
    carriers = carriers.to(images.device).float()
    x = backbone(images)
    proj = x @ carriers.T
    return torch.sign(proj).clamp(min=-1.0)


def bit_error_rate(decoded: torch.Tensor, truth: torch.Tensor) -> float:
    """BER over all bits (both tensors in {-1, +1}, shapes [B, k])."""
    errs = (decoded.sign() != truth.sign()).float()
    return errs.mean().item()


def word_error_rate(decoded: torch.Tensor, truth: torch.Tensor) -> float:
    """Fraction of messages with at least one bit wrong."""
    per_word = (decoded.sign() != truth.sign()).any(dim=-1).float()
    return per_word.mean().item()
