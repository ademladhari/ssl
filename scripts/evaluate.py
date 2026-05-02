"""Mark a folder of images, apply each attack and compute TPR / BER / WER."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchvision.transforms.functional as TF
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attacks import DEFAULT_ATTACKS  # noqa: E402
from src.backbone import Backbone, PCAWhitening  # noqa: E402
from src.constraints import psnr  # noqa: E402
from src.detect import (  # noqa: E402
    bit_error_rate,
    decode_multi_bit,
    detect_zero_bit,
    word_error_rate,
)
from src.embed import EmbedConfig, embed_multi_bit, embed_zero_bit  # noqa: E402
from src.io_utils import list_images, load_image  # noqa: E402
from src.keys import generate_multibit_key, load_key  # noqa: E402


def _load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _prepare(image_path: Path, image_size: int | None, device: str) -> torch.Tensor:
    img = load_image(image_path)
    if image_size is not None:
        img = TF.resize(img, [image_size, image_size], antialias=True)
    return img.unsqueeze(0).to(device)


def _random_messages(n: int, k: int, device: str) -> torch.Tensor:
    return torch.randint(0, 2, (n, k), device=device).float() * 2.0 - 1.0


def evaluate_zero_bit(args, cfg, backbone, paths: List[Path]) -> Dict[str, float]:
    key = load_key(args.key).to(args.device)
    fpr = cfg["fpr"]
    cfg_embed = EmbedConfig(target_psnr=cfg["target_psnr"], n_iter=cfg["n_iter"],
                            lr=cfg["lr"], lambda_w=cfg["lambda_w"])

    results: Dict[str, List[int]] = {name: [] for name in DEFAULT_ATTACKS}
    psnrs: List[float] = []

    for path in tqdm(paths, desc="zero-bit"):
        img = _prepare(path, args.image_size, args.device)
        marked = embed_zero_bit(backbone, img, key=key, fpr=fpr, config=cfg_embed)
        psnrs.append(psnr(marked, img).mean().item())
        for name, fn in DEFAULT_ATTACKS.items():
            attacked = fn(marked)
            det, _ = detect_zero_bit(backbone, attacked, key=key, fpr=fpr)
            results[name].append(int(det.item()))

    summary = {name: sum(v) / max(1, len(v)) for name, v in results.items()}
    summary["_psnr_mean"] = sum(psnrs) / max(1, len(psnrs))
    return summary


def evaluate_multi_bit(args, cfg, backbone, paths: List[Path]) -> Dict[str, Tuple[float, float]]:
    key = load_key(args.key).to(args.device)
    k = cfg["n_bits"]
    cfg_embed = EmbedConfig(target_psnr=cfg["target_psnr"], n_iter=cfg["n_iter"],
                            lr=cfg["lr"], lambda_w=cfg["lambda_w"])

    ber_acc: Dict[str, List[float]] = {n: [] for n in DEFAULT_ATTACKS}
    wer_acc: Dict[str, List[float]] = {n: [] for n in DEFAULT_ATTACKS}
    psnrs: List[float] = []

    for path in tqdm(paths, desc="multi-bit"):
        img = _prepare(path, args.image_size, args.device)
        msg = _random_messages(1, k, args.device)
        marked = embed_multi_bit(backbone, img, carriers=key, messages=msg,
                                 margin=cfg["margin"], config=cfg_embed)
        psnrs.append(psnr(marked, img).mean().item())
        for name, fn in DEFAULT_ATTACKS.items():
            attacked = fn(marked)
            decoded = decode_multi_bit(backbone, attacked, carriers=key)
            ber_acc[name].append(bit_error_rate(decoded, msg))
            wer_acc[name].append(word_error_rate(decoded, msg))

    summary = {name: (sum(ber) / max(1, len(ber)), sum(wer) / max(1, len(wer)))
               for name, (ber, wer) in ((n, (ber_acc[n], wer_acc[n])) for n in DEFAULT_ATTACKS)}
    summary["_psnr_mean"] = (sum(psnrs) / max(1, len(psnrs)), 0.0)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["zero_bit", "multi_bit"], required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--whitening", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--max-images", type=int, default=50)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    default_cfg_path = ROOT / "configs" / f"{args.mode}.yaml"
    cfg = _load_cfg(args.config or default_cfg_path)

    whitening = PCAWhitening.load(args.whitening)
    backbone = Backbone(whitening=whitening, feat_dim=cfg["feat_dim"]).to(args.device).eval()

    paths = list_images(args.images_dir)[: args.max_images]
    if not paths:
        raise SystemExit(f"no images in {args.images_dir}")

    if args.mode == "zero_bit":
        summary = evaluate_zero_bit(args, cfg, backbone, paths)
        print("\n=== zero-bit TPR ===")
        for name, tpr in summary.items():
            print(f"  {name:<14s} {tpr:.3f}")
    else:
        summary = evaluate_multi_bit(args, cfg, backbone, paths)
        print("\n=== multi-bit BER / WER ===")
        for name, (ber, wer) in summary.items():
            print(f"  {name:<14s} BER={ber:.3f}  WER={wer:.3f}")


if __name__ == "__main__":
    main()
