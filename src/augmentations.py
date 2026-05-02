"""Differentiable augmentations applied during marking (Section 4.1).

At each gradient step we sample one transformation t in T where
    T = { identity, rotation, gaussian_blur, center_crop, resize }
and, independently, horizontally flip the image with probability 0.5.
All operations are differentiable w.r.t. the pixel tensor so that gradients
propagate back to I_w.
"""

from __future__ import annotations

import math
import random
from typing import Callable, List

import numpy as np
import torch
import torch.nn.functional as F


def _rotate(image: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate by ``angle_deg`` around the image center with bilinear sampling."""
    b, c, h, w = image.shape
    theta = math.radians(angle_deg)
    cos, sin = math.cos(theta), math.sin(theta)
    mat = torch.tensor(
        [[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=image.dtype, device=image.device
    ).unsqueeze(0).expand(b, -1, -1)
    grid = F.affine_grid(mat, image.shape, align_corners=False)
    return F.grid_sample(image, grid, align_corners=False, padding_mode="zeros")


def _gaussian_blur(image: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    if kernel_size <= 1:
        return image
    coords = torch.arange(kernel_size, device=image.device, dtype=image.dtype)
    coords = coords - (kernel_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kx = g.view(1, 1, 1, kernel_size)
    ky = g.view(1, 1, kernel_size, 1)
    c = image.shape[1]
    kx = kx.expand(c, 1, 1, kernel_size)
    ky = ky.expand(c, 1, kernel_size, 1)
    pad = kernel_size // 2
    out = F.conv2d(image, kx, padding=(0, pad), groups=c)
    out = F.conv2d(out, ky, padding=(pad, 0), groups=c)
    return out


def _center_crop(image: torch.Tensor, scale: float, aspect: float) -> torch.Tensor:
    """Differentiable center crop followed by resize back to original shape."""
    _, _, h, w = image.shape
    area = scale * h * w
    ch = int(round(math.sqrt(area / aspect)))
    cw = int(round(math.sqrt(area * aspect)))
    ch = max(1, min(ch, h))
    cw = max(1, min(cw, w))
    top = (h - ch) // 2
    left = (w - cw) // 2
    return image[:, :, top:top + ch, left:left + cw]


def _resize(image: torch.Tensor, scale: float) -> torch.Tensor:
    _, _, h, w = image.shape
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    return F.interpolate(image, size=(nh, nw), mode="bilinear", align_corners=False)


def _hflip(image: torch.Tensor) -> torch.Tensor:
    return torch.flip(image, dims=[-1])


def _sample_rotation_angle_deg(rng: random.Random) -> float:
    """alpha ~ VonMises(mu=0, kappa=1), divided by 2. Sampled with numpy for speed."""
    alpha = np.random.vonmises(mu=0.0, kappa=1.0) / 2.0
    return math.degrees(alpha)


def sample_and_apply(
    image: torch.Tensor,
    rng: random.Random | None = None,
    flip_prob: float = 0.5,
) -> torch.Tensor:
    """Sample one transformation from T and apply it. Differentiable."""
    if rng is None:
        rng = random

    choice = rng.choice(["identity", "rotation", "blur", "crop", "resize"])

    if choice == "identity":
        out = image
    elif choice == "rotation":
        out = _rotate(image, _sample_rotation_angle_deg(rng))
    elif choice == "blur":
        k = rng.choice([1, 3, 5, 7, 9, 11, 13, 15])
        sigma = 0.15 * k + 0.35
        out = _gaussian_blur(image, k, sigma)
    elif choice == "crop":
        scale = rng.uniform(0.2, 1.0)
        aspect = rng.uniform(3 / 4, 4 / 3)
        out = _center_crop(image, scale, aspect)
    else:  # resize
        scale = rng.uniform(0.2, 1.0)
        out = _resize(image, scale)

    if rng.random() < flip_prob:
        out = _hflip(out)
    return out


__all__: List[str] = [
    "sample_and_apply",
    "_rotate",
    "_gaussian_blur",
    "_center_crop",
    "_resize",
    "_hflip",
]
