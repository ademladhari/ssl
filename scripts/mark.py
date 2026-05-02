"""Watermark one image (zero-bit or multi-bit)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backbone import Backbone, PCAWhitening  # noqa: E402
from src.constraints import psnr  # noqa: E402
from src.embed import EmbedConfig, embed_multi_bit, embed_zero_bit  # noqa: E402
from src.io_utils import load_image, save_image  # noqa: E402
from src.keys import load_key, parse_message  # noqa: E402


def _load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["zero_bit", "multi_bit"], required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--whitening", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--message", default=None, help="bit string for multi-bit mode")
    p.add_argument("--config", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    default_cfg_path = ROOT / "configs" / f"{args.mode}.yaml"
    cfg = _load_cfg(args.config or default_cfg_path)

    whitening = PCAWhitening.load(args.whitening)
    backbone = Backbone(whitening=whitening, feat_dim=cfg["feat_dim"]).to(args.device).eval()

    image = load_image(args.image).unsqueeze(0).to(args.device)
    original = image.clone()

    key = load_key(args.key).to(args.device)

    embed_cfg = EmbedConfig(
        target_psnr=cfg["target_psnr"],
        n_iter=cfg["n_iter"],
        lr=cfg["lr"],
        lambda_w=cfg["lambda_w"],
    )

    if args.mode == "zero_bit":
        marked = embed_zero_bit(backbone, image, key=key, fpr=cfg["fpr"],
                                config=embed_cfg)
    else:
        if args.message is None:
            raise SystemExit("--message is required in multi_bit mode")
        msg = parse_message(args.message)
        k = cfg["n_bits"]
        if msg.numel() != k:
            raise SystemExit(f"message length {msg.numel()} != n_bits {k}")
        marked = embed_multi_bit(backbone, image, carriers=key,
                                 messages=msg.unsqueeze(0),
                                 margin=cfg["margin"], config=embed_cfg)

    got_psnr = psnr(marked, original).item()
    print(f"[mark] PSNR = {got_psnr:.2f} dB (target {cfg['target_psnr']})")
    save_image(marked, args.out)
    print(f"[mark] saved -> {args.out}")


if __name__ == "__main__":
    main()
