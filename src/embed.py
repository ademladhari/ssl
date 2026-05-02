"""Algorithm 1: watermark an image by gradient descent on its pixels."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import torch

from .augmentations import sample_and_apply
from .backbone import Backbone
from .constraints import apply_constraints
from .losses import (
    image_mse_loss,
    multi_bit_loss,
    theta_from_fpr,
    zero_bit_loss,
)


@dataclass
class EmbedConfig:
    target_psnr: float = 40.0
    n_iter: int = 100
    lr: float = 0.01
    lambda_w: float = 1.0
    ssim_window: int = 17
    init_noise_std: float = 1e-3


def _clone_for_optim(image: torch.Tensor, init_noise_std: float = 0.0) -> torch.Tensor:
    img = image.detach().clone()
    if init_noise_std > 0.0:
        img = (img + init_noise_std * torch.randn_like(img)).clamp(0.0, 1.0)
    img = img.requires_grad_(True)
    return img


def _round_to_uint8(image: torch.Tensor) -> torch.Tensor:
    return (image.clamp(0.0, 1.0) * 255.0).round() / 255.0


def embed_zero_bit(
    backbone: Backbone,
    images: torch.Tensor,
    key: torch.Tensor,
    fpr: float = 1e-6,
    config: Optional[EmbedConfig] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Watermark ``images`` (batch in [0, 1]) so each feature lies in cone(a, theta).

    Returns the marked images in [0, 1], snapped to the uint8 grid.
    """
    config = config or EmbedConfig()
    device = images.device
    key = key.to(device).float()
    key = key / key.norm().clamp_min(1e-12)

    d = backbone.feat_dim
    theta = theta_from_fpr(fpr, d=d)
    cos_theta = math.cos(theta)

    rng = random.Random(seed)

    iw = _clone_for_optim(images, init_noise_std=config.init_noise_std)
    original = images.detach()
    opt = torch.optim.Adam([iw], lr=config.lr)

    for _ in range(config.n_iter):
        opt.zero_grad(set_to_none=True)
        projected = apply_constraints(iw, original, config.target_psnr,
                                      ssim_window=config.ssim_window)
        transformed = sample_and_apply(projected, rng=rng)
        x = backbone(transformed)
        loss_w = zero_bit_loss(x, key, cos_theta)
        loss_i = image_mse_loss(projected, original)
        loss = config.lambda_w * loss_w + loss_i
        loss.backward()
        opt.step()

    with torch.no_grad():
        final = apply_constraints(iw, original, config.target_psnr,
                                  ssim_window=config.ssim_window)
        final = _round_to_uint8(final)
    return final


def embed_multi_bit(
    backbone: Backbone,
    images: torch.Tensor,
    carriers: torch.Tensor,
    messages: torch.Tensor,
    margin: float = 5.0,
    config: Optional[EmbedConfig] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Embed a k-bit message per image. ``messages`` is [B, k] in {-1, +1}."""
    config = config or EmbedConfig(lambda_w=5e4)
    device = images.device
    carriers = carriers.to(device).float()
    messages = messages.to(device).float()
    if messages.ndim == 1:
        messages = messages.unsqueeze(0).expand(images.shape[0], -1)

    rng = random.Random(seed)

    iw = _clone_for_optim(images, init_noise_std=config.init_noise_std)
    original = images.detach()
    opt = torch.optim.Adam([iw], lr=config.lr)

    for _ in range(config.n_iter):
        opt.zero_grad(set_to_none=True)
        projected = apply_constraints(iw, original, config.target_psnr,
                                      ssim_window=config.ssim_window)
        transformed = sample_and_apply(projected, rng=rng)
        x = backbone(transformed)
        loss_w = multi_bit_loss(x, carriers, messages, margin=margin)
        loss_i = image_mse_loss(projected, original)
        loss = config.lambda_w * loss_w + loss_i
        loss.backward()
        opt.step()

    with torch.no_grad():
        final = apply_constraints(iw, original, config.target_psnr,
                                  ssim_window=config.ssim_window)
        final = _round_to_uint8(final)
    return final
