"""Small image I/O helpers shared by the CLI scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image


IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)


def load_image(path: str | Path) -> torch.Tensor:
    img = Image.open(str(path)).convert("RGB")
    return TF.to_tensor(img)  # [3, H, W] in [0, 1]


def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    if tensor.ndim == 4:
        tensor = tensor[0]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(tensor.clamp(0, 1).cpu()).save(str(path))


def iter_batches(
    paths: Iterable[Path],
    batch_size: int,
    resize_to: Tuple[int, int] | None = None,
) -> Iterator[Tuple[List[Path], torch.Tensor]]:
    """Yield (paths, tensor[B, 3, H, W]) batches. When ``resize_to`` is None we
    only batch images that share the same resolution (watermarking is per-image)."""
    buf_paths: List[Path] = []
    buf_tensors: List[torch.Tensor] = []
    current_shape: Tuple[int, int] | None = None

    def _flush():
        nonlocal buf_paths, buf_tensors, current_shape
        if buf_tensors:
            yield buf_paths, torch.stack(buf_tensors, dim=0)
        buf_paths, buf_tensors, current_shape = [], [], None

    for p in paths:
        img = load_image(p)
        if resize_to is not None:
            img = TF.resize(img, list(resize_to), antialias=True)
        shape = tuple(img.shape[-2:])
        if current_shape is None:
            current_shape = shape
        if shape != current_shape or len(buf_tensors) >= batch_size:
            yield from _flush()
            current_shape = shape
        buf_paths.append(p)
        buf_tensors.append(img)

    yield from _flush()
