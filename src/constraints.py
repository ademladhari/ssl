"""Admissible-image set C(I_o) from Section 3.2.

Two constraints are chained at every marking iteration:

1. SSIM heatmap attenuation (paper-strict): compute a per-pixel SSIM map
   between the current image I and original I_o with C1=0.01^2, C2=0.03^2 and
   17x17 Gaussian windows, then scale perturbation delta = I - I_o by
   (1 - ssim_map). This concentrates changes in perceptually less visible areas.

2. PSNR clipping: if the resulting PSNR is below the target, delta is scaled
   down so that PSNR == target (using ||delta||_2^2 <= h*w*c * 10^(-PSNR/10)
   for inputs in [0, 1]).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _gaussian_kernel(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.unsqueeze(0) * g.unsqueeze(1)  # [w, w]


def ssim_map(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 17,
    sigma: float = 1.5,
    c1: float = 0.01 ** 2,
    c2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Standard per-pixel SSIM map aggregated over channels, clamped to [0, 1].

    Inputs: [B, C, H, W] in [0, 1]. Output: [B, 1, H, W]. Kept for diagnostics
    and tests; the marking constraint itself uses :func:`texture_mask`.
    """
    assert img1.shape == img2.shape
    b, c, _, _ = img1.shape
    kernel = _gaussian_kernel(window_size, sigma, img1.device, img1.dtype)
    kernel = kernel.expand(c, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    def _conv(x):
        return F.conv2d(x, kernel, padding=pad, groups=c)

    mu1 = _conv(img1)
    mu2 = _conv(img2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = _conv(img1 * img1) - mu1_sq
    sigma2_sq = _conv(img2 * img2) - mu2_sq
    sigma12 = _conv(img1 * img2) - mu1_mu2

    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim = num / den
    ssim = ssim.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
    return ssim


def texture_mask(
    image: torch.Tensor,
    window_size: int = 17,
    sigma: float = 1.5,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Per-pixel texture mask in [0, 1] from ``image`` alone (shape [B, C, H, W]).

    Computes the local standard deviation over 17x17 Gaussian tiles per channel,
    sums across channels, and normalizes per-image to [0, 1]. Flat regions get
    ~0 (changes would be visible), textured regions get ~1. This serves as the
    SSIM-style attenuation for :func:`apply_constraints`.
    """
    _, c, _, _ = image.shape
    kernel = _gaussian_kernel(window_size, sigma, image.device, image.dtype)
    kernel = kernel.expand(c, 1, window_size, window_size).contiguous()
    pad = window_size // 2
    mu = F.conv2d(image, kernel, padding=pad, groups=c)
    mu_sq = mu * mu
    mu2 = F.conv2d(image * image, kernel, padding=pad, groups=c)
    var = (mu2 - mu_sq).clamp_min(0.0)
    std = var.sqrt().sum(dim=1, keepdim=True)
    max_per = std.flatten(1).max(dim=1).values.view(-1, 1, 1, 1).clamp_min(eps)
    mask = (std / max_per).clamp(0.0, 1.0)
    return mask


def psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR in dB. Inputs [B, C, H, W] in [0, 1]."""
    mse = (img1 - img2).pow(2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


def _psnr_l2_budget(shape: torch.Size, target_psnr: float) -> float:
    """Max ||delta||_2 allowed so PSNR(I_o, I_o + delta) >= target_psnr.

    For inputs in [0, 1], MAX^2 = 1, so MSE = ||delta||^2 / N, and
    PSNR = -10 log10(MSE). PSNR >= target <=> ||delta||^2 <= N * 10^{-target/10}.
    """
    _, c, h, w = shape
    n = c * h * w
    return math.sqrt(n * (10 ** (-target_psnr / 10)))


def apply_constraints(
    image: torch.Tensor,
    original: torch.Tensor,
    target_psnr: float,
    ssim_window: int = 17,
) -> torch.Tensor:
    """Project ``image`` into C(original) via SSIM attenuation + PSNR clip."""
    delta = image - original

    with torch.no_grad():
        smap = ssim_map(image.detach(), original.detach(), window_size=ssim_window)
    delta = delta * (1.0 - smap)

    max_norm = _psnr_l2_budget(image.shape, target_psnr)
    flat = delta.flatten(1)
    norm = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(max_norm / norm, max=1.0)
    delta = (flat * scale).view_as(delta)

    return (original + delta).clamp(0.0, 1.0)
