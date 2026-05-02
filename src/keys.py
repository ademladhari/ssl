"""Secret-key generation for zero-bit and multi-bit watermarking.

Zero-bit uses a single unit-norm carrier ``a``. Multi-bit uses ``k`` orthonormal
carriers stacked as a matrix of shape [k, d]. Both are stored as ``.npy``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def generate_zerobit_key(d: int = 2048, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(d)
    return (g / np.linalg.norm(g)).astype(np.float32)


def generate_multibit_key(k: int, d: int = 2048, seed: int | None = None) -> np.ndarray:
    """k orthonormal carriers of dimension d (``k <= d``)."""
    assert k <= d, f"cannot pick {k} > {d} orthonormal directions"
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((d, k))
    q, _ = np.linalg.qr(g)
    return q.T.astype(np.float32)  # [k, d]


def save_zerobit_key(path: str | Path, d: int = 2048, seed: int | None = None) -> np.ndarray:
    a = generate_zerobit_key(d=d, seed=seed)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, a)
    return a


def save_multibit_key(path: str | Path, k: int, d: int = 2048, seed: int | None = None) -> np.ndarray:
    A = generate_multibit_key(k=k, d=d, seed=seed)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, A)
    return A


def load_key(path: str | Path) -> torch.Tensor:
    arr = np.load(str(path))
    return torch.from_numpy(arr).float()


def parse_message(bits: str) -> torch.Tensor:
    """Parse a string of '0'/'1' into a [-1, +1] tensor of shape [k]."""
    cleaned = [c for c in bits if c in "01"]
    if not cleaned:
        raise ValueError("empty bit string")
    out = torch.tensor([1.0 if c == "1" else -1.0 for c in cleaned])
    return out


def format_message(signs: Iterable[int]) -> str:
    return "".join("1" if int(s) > 0 else "0" for s in signs)
