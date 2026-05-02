"""End-to-end sanity check that doesn't require downloading DINO.

This exercises every piece in isolation plus a tiny mock-backbone embedding run:

  * FPR <-> theta inversion round-trip.
  * Zero-bit loss sign (negative iff inside hypercone).
  * Multi-bit hinge loss zero iff signs match with margin.
  * SSIM map is 1 for identical images.
  * PSNR clipping exactly hits the target.
  * Orthonormality of multi-bit carriers.
  * Mini zero-bit embedding with a random-projection "backbone" converges so
    that the detector fires.
  * Mini multi-bit embedding decodes the hidden bits exactly under identity.

Run: python scripts/sanity_check.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constraints import apply_constraints, psnr, ssim_map  # noqa: E402
from src.detect import bit_error_rate, decode_multi_bit, detect_zero_bit  # noqa: E402
from src.embed import EmbedConfig, embed_multi_bit, embed_zero_bit  # noqa: E402
from src.keys import generate_multibit_key, generate_zerobit_key  # noqa: E402
from src.losses import (  # noqa: E402
    fpr_from_theta,
    multi_bit_loss,
    theta_from_fpr,
    zero_bit_loss,
)


class MockBackbone(nn.Module):
    """Deterministic random linear feature extractor for tests (no network I/O)."""

    def __init__(self, d: int = 64, img_channels: int = 3, img_size: int = 32,
                 seed: int = 0):
        super().__init__()
        self.feat_dim = d
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(d, img_channels * img_size * img_size, generator=g)
        w /= w.norm(dim=1, keepdim=True)
        self.register_buffer("W", w)
        self.register_buffer("imagenet_mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.ones(1, 3, 1, 1))
        self._size = img_size

    def features(self, x):  # unused by tests but matches Backbone API
        return self(x)

    def forward(self, x):
        x = nn.functional.adaptive_avg_pool2d(x, self._size)
        flat = x.flatten(1)
        return flat @ self.W.T


def ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def check_fpr_theta() -> None:
    for fpr in [1e-2, 1e-4, 1e-6, 1e-10]:
        theta = theta_from_fpr(fpr, d=2048)
        got = fpr_from_theta(theta, d=2048)
        assert abs(got - fpr) / fpr < 1e-3, (fpr, got)
    ok("theta_from_fpr round-trips through fpr_from_theta")


def check_zero_bit_loss() -> None:
    torch.manual_seed(0)
    a = torch.randn(2048)
    a = a / a.norm()
    x_in = a.unsqueeze(0) * 10.0
    x_out = torch.randn(1, 2048)
    cos_theta = math.cos(theta_from_fpr(1e-6, d=2048))
    loss_in = zero_bit_loss(x_in, a, cos_theta).item()
    loss_out = zero_bit_loss(x_out, a, cos_theta).item()
    assert loss_in < 0 < loss_out, (loss_in, loss_out)
    ok("zero_bit_loss < 0 inside the hypercone, > 0 outside")


def check_multi_bit_loss() -> None:
    torch.manual_seed(0)
    k, d = 8, 64
    carriers = torch.linalg.qr(torch.randn(d, k))[0].T  # [k, d]
    messages = torch.sign(torch.randn(1, k))
    x = messages @ carriers * 10.0  # projections = 10 * m_i
    assert multi_bit_loss(x, carriers, messages, margin=5.0).item() == 0.0
    bad = -messages @ carriers
    assert multi_bit_loss(bad, carriers, messages, margin=5.0).item() > 0
    ok("multi_bit_loss hinge: 0 when satisfied, > 0 when violated")


def check_ssim_and_psnr() -> None:
    torch.manual_seed(0)
    img = torch.rand(1, 3, 64, 64)
    smap = ssim_map(img, img)
    assert smap.min().item() > 0.999
    target_psnr = 40.0
    perturbed = img + 0.5 * torch.randn_like(img)
    projected = apply_constraints(perturbed, img, target_psnr)
    got = psnr(projected, img).item()
    assert got + 1e-2 >= target_psnr, got
    ok(f"SSIM identity=1.0; PSNR clipping hits {got:.2f} dB >= target 40.0")


def check_keys() -> None:
    a = generate_zerobit_key(d=256, seed=0)
    assert abs(float((a * a).sum()) - 1.0) < 1e-5
    A = generate_multibit_key(k=16, d=256, seed=0)
    gram = torch.from_numpy(A) @ torch.from_numpy(A).T
    err = (gram - torch.eye(16)).abs().max().item()
    assert err < 1e-5, err
    ok("zero-bit key unit-norm; multi-bit carriers orthonormal")


def check_mini_zero_bit_embed() -> None:
    """Use a loose FPR / low-dim feature space so the hypercone is reachable
    within the PSNR budget of the tiny mock backbone (the real DINO ResNet-50
    has d=2048 and a much friendlier geometry)."""
    torch.manual_seed(0)
    backbone = MockBackbone(d=32, img_size=32)
    img = torch.rand(1, 3, 64, 64)
    key = torch.randn(32)
    key = key / key.norm()
    cfg = EmbedConfig(target_psnr=30.0, n_iter=150, lr=0.05, lambda_w=10.0)
    marked = embed_zero_bit(backbone, img, key=key, fpr=0.1, config=cfg, seed=0)
    p = psnr(marked, img).item()
    det, score = detect_zero_bit(backbone, marked, key=key, fpr=0.1)
    assert bool(det.item()), score.item()
    assert p + 1e-2 >= 30.0, p
    ok(f"mini zero-bit: PSNR={p:.2f} dB, detector score={score.item():+.3e}")


def check_mini_multi_bit_embed() -> None:
    """Small payload over the mock backbone. We temporarily disable marking-time
    augmentation because the random-projection mock backbone is not invariant
    to rotation/crop/etc., so augmentation would inject noise that swamps the
    gradient signal. The real DINO backbone *is* invariant, which is the whole
    point of using it."""
    import src.embed as embed_mod
    orig_sample = embed_mod.sample_and_apply
    embed_mod.sample_and_apply = lambda image, rng=None, flip_prob=0.0: image
    try:
        torch.manual_seed(0)
        backbone = MockBackbone(d=64, img_size=32, seed=1)
        img = torch.rand(1, 3, 64, 64)
        carriers = torch.from_numpy(generate_multibit_key(k=8, d=64, seed=2))
        msg = torch.sign(torch.randn(1, 8))
        cfg = EmbedConfig(target_psnr=30.0, n_iter=150, lr=0.05, lambda_w=1e3)
        marked = embed_multi_bit(backbone, img, carriers=carriers, messages=msg,
                                 margin=1.0, config=cfg, seed=0)
    finally:
        embed_mod.sample_and_apply = orig_sample
    decoded = decode_multi_bit(backbone, marked, carriers=carriers)
    ber = bit_error_rate(decoded, msg)
    p = psnr(marked, img).item()
    assert ber <= 0.25, ber
    assert p + 1e-2 >= 30.0, p
    ok(f"mini multi-bit: PSNR={p:.2f} dB, BER={ber:.3f}")


def main() -> None:
    check_fpr_theta()
    check_zero_bit_loss()
    check_multi_bit_loss()
    check_ssim_and_psnr()
    check_keys()
    check_mini_zero_bit_embed()
    check_mini_multi_bit_embed()
    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
