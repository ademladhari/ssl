"""Fit PCA-whitening on a folder of natural images and save it as a .pt file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backbone import Backbone  # noqa: E402
from src.io_utils import list_images, load_image  # noqa: E402
from src.pca import fit_pca_whitening  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Fit PCA-whitening over image features")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--out", default="checkpoints/whitening.pt")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-images", type=int, default=10_000)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    paths = list_images(args.images_dir)[: args.max_images]
    if not paths:
        raise SystemExit(f"no images found in {args.images_dir}")
    print(f"[fit_pca] {len(paths)} images; device={args.device}")

    backbone = Backbone().to(args.device).eval()
    feats = []
    with torch.no_grad():
        batch: list = []
        for i, path in enumerate(paths, 1):
            img = load_image(path)
            img = TF.resize(img, [args.image_size, args.image_size], antialias=True)
            batch.append(img)
            if len(batch) == args.batch_size or i == len(paths):
                x = torch.stack(batch, 0).to(args.device)
                feats.append(backbone.features(x).cpu())
                batch = []
    feats = torch.cat(feats, dim=0)
    print(f"[fit_pca] features shape = {tuple(feats.shape)}")

    whitening = fit_pca_whitening(feats)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    whitening.save(args.out)
    print(f"[fit_pca] saved -> {args.out}")


if __name__ == "__main__":
    main()
