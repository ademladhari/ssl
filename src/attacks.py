"""Evaluation-time attacks applied to watermarked images.

All operate on PIL-friendly tensors [B, 3, H, W] in [0, 1]. They do not need
gradients. ``augly`` is optional: Meme / Screenshot degrade to a no-op if it
is not installed (the other attacks still run).
"""

from __future__ import annotations

import io
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

try:  # augly is optional
    import augly.image as imaugs  # type: ignore
    _HAS_AUGLY = True
except Exception:  # pragma: no cover - optional dep
    imaugs = None  # type: ignore
    _HAS_AUGLY = False


def _to_pil_list(images: torch.Tensor) -> List[Image.Image]:
    out = []
    for i in range(images.shape[0]):
        arr = (images[i].clamp(0, 1) * 255.0).round().to(torch.uint8)
        out.append(TF.to_pil_image(arr.cpu()))
    return out


def _from_pil_list(pils: List[Image.Image], device, dtype) -> torch.Tensor:
    tensors = [TF.to_tensor(p).to(device=device, dtype=dtype) for p in pils]
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    padded = []
    for t in tensors:
        pad = (0, max_w - t.shape[2], 0, max_h - t.shape[1])
        padded.append(F.pad(t, pad))
    return torch.stack(padded, dim=0)


def _pil_attack(images: torch.Tensor, fn: Callable[[Image.Image], Image.Image]) -> torch.Tensor:
    pils = _to_pil_list(images)
    out = [fn(p) for p in pils]
    return _from_pil_list(out, images.device, images.dtype)


def rotation(images: torch.Tensor, angle: float) -> torch.Tensor:
    return _pil_attack(images, lambda p: p.rotate(angle, resample=Image.BILINEAR))


def center_crop(images: torch.Tensor, ratio: float) -> torch.Tensor:
    def _fn(p: Image.Image) -> Image.Image:
        w, h = p.size
        nh, nw = int(round(h * (ratio ** 0.5))), int(round(w * (ratio ** 0.5)))
        left = (w - nw) // 2
        top = (h - nh) // 2
        return p.crop((left, top, left + nw, top + nh))

    return _pil_attack(images, _fn)


def resize(images: torch.Tensor, scale: float) -> torch.Tensor:
    def _fn(p: Image.Image) -> Image.Image:
        w, h = p.size
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        return p.resize((nw, nh), Image.BILINEAR)

    return _pil_attack(images, _fn)


def gaussian_blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    from PIL import ImageFilter
    return _pil_attack(images, lambda p: p.filter(ImageFilter.GaussianBlur(radius=sigma)))


def jpeg(images: torch.Tensor, quality: int) -> torch.Tensor:
    def _fn(p: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        p.convert("RGB").save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB").copy()

    return _pil_attack(images, _fn)


def brightness(images: torch.Tensor, factor: float) -> torch.Tensor:
    from PIL import ImageEnhance
    return _pil_attack(images, lambda p: ImageEnhance.Brightness(p).enhance(factor))


def contrast(images: torch.Tensor, factor: float) -> torch.Tensor:
    from PIL import ImageEnhance
    return _pil_attack(images, lambda p: ImageEnhance.Contrast(p).enhance(factor))


def hue(images: torch.Tensor, shift: float) -> torch.Tensor:
    """Hue shift in [-0.5, 0.5] (torchvision convention)."""
    return torch.stack([TF.adjust_hue(img, shift) for img in images], dim=0)


def meme(images: torch.Tensor, text: str = "HELLO") -> torch.Tensor:
    if not _HAS_AUGLY:
        return images
    def _fn(p: Image.Image) -> Image.Image:
        return imaugs.meme_format(p, text=text)
    return _pil_attack(images, _fn)


def screenshot(images: torch.Tensor) -> torch.Tensor:
    if not _HAS_AUGLY:
        return images
    def _fn(p: Image.Image) -> Image.Image:
        return imaugs.overlay_onto_screenshot(p)
    return _pil_attack(images, _fn)


def identity(images: torch.Tensor) -> torch.Tensor:
    return images.clone()


# Ordered list of (name, fn) used by scripts/evaluate.py. Matches the paper's
# Table 1/2 defaults (FPR = 1e-6, PSNR = 40 dB setup).
DEFAULT_ATTACKS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "identity":       lambda x: identity(x),
    "rotation_25":    lambda x: rotation(x, 25.0),
    "crop_0.5":       lambda x: center_crop(x, 0.5),
    "crop_0.1":       lambda x: center_crop(x, 0.1),
    "resize_0.7":     lambda x: resize(x, 0.7),
    "blur_2.0":       lambda x: gaussian_blur(x, 2.0),
    "jpeg_50":        lambda x: jpeg(x, 50),
    "brightness_2":   lambda x: brightness(x, 2.0),
    "contrast_2":     lambda x: contrast(x, 2.0),
    "hue_0.25":       lambda x: hue(x, 0.25),
    "meme":           lambda x: meme(x),
    "screenshot":     lambda x: screenshot(x),
}
