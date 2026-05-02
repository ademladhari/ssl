"""Detect or decode a watermark in an image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backbone import Backbone, PCAWhitening  # noqa: E402
from src.detect import decode_multi_bit, detect_zero_bit  # noqa: E402
from src.io_utils import load_image  # noqa: E402
from src.keys import format_message, load_key  # noqa: E402


def _load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["zero_bit", "multi_bit"], required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--whitening", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--fpr", type=float, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    default_cfg_path = ROOT / "configs" / f"{args.mode}.yaml"
    cfg = _load_cfg(args.config or default_cfg_path)
    fpr = args.fpr if args.fpr is not None else cfg.get("fpr", 1e-6)

    whitening = PCAWhitening.load(args.whitening)
    backbone = Backbone(whitening=whitening, feat_dim=cfg["feat_dim"]).to(args.device).eval()

    image = load_image(args.image).unsqueeze(0).to(args.device)
    key = load_key(args.key).to(args.device)

    if args.mode == "zero_bit":
        detected, score = detect_zero_bit(backbone, image, key=key, fpr=fpr)
        print(f"[detect] detected={bool(detected.item())}, score={score.item():+.4e}, fpr={fpr}")
    else:
        bits = decode_multi_bit(backbone, image, carriers=key)[0].tolist()
        print(f"[decode] {format_message(bits)}")


if __name__ == "__main__":
    main()
