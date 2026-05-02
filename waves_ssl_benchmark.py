from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SSL_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SSL_ROOT) not in sys.path:
    sys.path.insert(0, str(SSL_ROOT))

from waves_benchmark_common import run_waves_benchmark  # noqa: E402
from src.backbone import Backbone, PCAWhitening  # noqa: E402
from src.detect import detect_zero_bit  # noqa: E402
from src.embed import EmbedConfig, embed_zero_bit  # noqa: E402
from src.io_utils import list_images, load_image  # noqa: E402
from src.keys import load_key  # noqa: E402


def _require_path(name: str, value: str) -> Path:
    if not value.strip():
        raise ValueError(f"{name} must be set to a valid path.")
    path = Path(value)
    if not path.is_absolute():
        path = (SSL_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def _load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config (expected mapping): {path}")
    return cfg


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def aggregate_runs(per_run_dirs: list[Path], out_dir: Path) -> None:
    leaderboards: list[pd.DataFrame] = []
    per_strength: list[pd.DataFrame] = []
    for run_dir in per_run_dirs:
        leaderboards.append(pd.read_csv(run_dir / "waves_ssl_leaderboard.csv"))
        per_strength.append(pd.read_csv(run_dir / "waves_ssl_per_strength.csv"))

    lb_all = pd.concat(leaderboards, ignore_index=True)
    ps_all = pd.concat(per_strength, ignore_index=True)

    lb_grouped = (
        lb_all.groupby(["Attack", "attack_key"], as_index=False)[["Q@0.7P", "Q@0.4P", "Avg P", "Avg Q"]]
        .mean()
        .sort_values(by="Avg P", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    lb_grouped.insert(1, "Rank", np.arange(1, len(lb_grouped) + 1))
    lb_grouped = lb_grouped[["Attack", "Rank", "Q@0.7P", "Q@0.4P", "Avg P", "Avg Q", "attack_key"]]

    ps_grouped = (
        ps_all.groupby(["attack_key", "attack_label", "strength"], as_index=False)[["P", "Q", "raw_similarity"]]
        .mean()
        .sort_values(by=["attack_key", "strength"], kind="stable")
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    lb_grouped.to_csv(out_dir / "waves_ssl_leaderboard.csv", index=False)
    ps_grouped.to_csv(out_dir / "waves_ssl_per_strength.csv", index=False)


def main() -> None:
    image_dir_raw = os.getenv("WAVES_SSL_IMAGE_DIR", "").strip()
    image_path_raw = os.getenv("WAVES_SSL_IMAGE", "").strip()
    key_path = _require_path("WAVES_SSL_KEY", os.getenv("WAVES_SSL_KEY", "checkpoints/key_zero.npy"))
    whitening_path = _require_path(
        "WAVES_SSL_WHITENING", os.getenv("WAVES_SSL_WHITENING", "checkpoints/whitening.pt")
    )
    config_path = _require_path("WAVES_SSL_CONFIG", os.getenv("WAVES_SSL_CONFIG", "configs/zero_bit.yaml"))

    out_dir = Path(os.getenv("WAVES_SSL_OUT_DIR", "outputs_waves_ssl"))
    device = os.getenv("WAVES_SSL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    seed = int(os.getenv("WAVES_SSL_SEED", "42"))
    max_images = int(os.getenv("WAVES_SSL_MAX_IMAGES", "100"))
    strengths = [float(s.strip()) for s in os.getenv("WAVES_SSL_STRENGTHS", "0.2,0.4,0.6,0.8,1.0").split(",")]
    waves_root = Path(os.getenv("WAVES_ROOT", str(PROJECT_ROOT / "dct" / "WAVES")))

    cfg = _load_cfg(config_path)
    embed_cfg = EmbedConfig(
        target_psnr=float(cfg.get("target_psnr", 40.0)),
        n_iter=int(cfg.get("n_iter", 100)),
        lr=float(cfg.get("lr", 0.01)),
        lambda_w=float(cfg.get("lambda_w", 1.0)),
    )
    fpr = float(cfg.get("fpr", 1e-6))
    feat_dim = int(cfg.get("feat_dim", 2048))

    excluded = {
        "adv_cls_wm1_wm2_0.04_200_warm",
        "kl_vae",
        "4x_regen_kl_vae",
    }

    whitening = PCAWhitening.load(whitening_path)
    backbone = Backbone(whitening=whitening, feat_dim=feat_dim).to(device).eval()
    image = load_image(image_path).unsqueeze(0).to(device)
    key = load_key(key_path).to(device)

    if image_dir_raw:
        image_dir = Path(image_dir_raw)
        if not image_dir.is_absolute():
            image_dir = (SSL_ROOT / image_dir).resolve()
        if not image_dir.exists():
            raise FileNotFoundError(f"WAVES_SSL_IMAGE_DIR not found: {image_dir}")
        all_images = list_images(image_dir)
        if not all_images:
            raise RuntimeError(f"No images found under WAVES_SSL_IMAGE_DIR: {image_dir}")
        image_paths = all_images[:max_images]
    elif image_path_raw:
        image_paths = [_require_path("WAVES_SSL_IMAGE", image_path_raw)]
    else:
        raise ValueError("Set WAVES_SSL_IMAGE_DIR (recommended) or WAVES_SSL_IMAGE.")

    out_dir.mkdir(parents=True, exist_ok=True)
    per_run_root = out_dir / "_per_run"
    per_run_root.mkdir(parents=True, exist_ok=True)
    per_run_dirs: list[Path] = []

    print(f"Starting SSL WAVES benchmark for {len(image_paths)} image(s)...")
    for run_idx, image_path in enumerate(image_paths, start=1):
        image = load_image(image_path).unsqueeze(0).to(device)
        with torch.no_grad():
            marked = embed_zero_bit(backbone, image, key=key, fpr=fpr, config=embed_cfg, seed=seed + run_idx - 1)

        watermarked_rgb = (marked[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.float64)
        original_gray = np.asarray(Image.fromarray(_to_uint8(watermarked_rgb)).convert("L"), dtype=np.float64)

        def score_fn(candidate_gray: np.ndarray) -> float:
            candidate_rgb = np.stack([_to_uint8(candidate_gray)] * 3, axis=-1)
            candidate_tensor = (
                torch.from_numpy(candidate_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            )
            _, score = detect_zero_bit(backbone, candidate_tensor, key=key, fpr=fpr)
            return float(score.item())

        run_dir = per_run_root / f"run_{run_idx:03d}"
        per_run_dirs.append(run_dir)
        print(f"[{run_idx}/{len(image_paths)}] {image_path.name} -> {run_dir}")
        run_waves_benchmark(
            original=original_gray,
            watermarked=original_gray,
            score_fn=score_fn,
            strengths=strengths,
            out_dir=run_dir,
            output_prefix="waves_ssl",
            waves_root=waves_root,
            mode_name="WAVES-style SSL benchmark",
            excluded_attacks=excluded,
            parameters={
                "image_path": str(image_path),
                "run_index": run_idx,
                "num_images": len(image_paths),
                "key_path": str(key_path),
                "whitening_path": str(whitening_path),
                "config_path": str(config_path),
                "device": device,
                "seed": seed + run_idx - 1,
                "strengths": strengths,
                "target_psnr": embed_cfg.target_psnr,
                "n_iter": embed_cfg.n_iter,
                "lr": embed_cfg.lr,
                "lambda_w": embed_cfg.lambda_w,
                "fpr": fpr,
                "feat_dim": feat_dim,
            },
            notes=[
                "SSL benchmark uses zero-bit embedding and detection score from Fernandez et al. reproduction.",
                "Each run embeds watermark on one source image, applies WAVES attacks, then evaluates detector score.",
                "Distortion attacks come from WAVES distortions module via shared benchmark runner.",
            ],
        )

    aggregate_runs(per_run_dirs, out_dir)
    print(f"Done. Leaderboard: {out_dir / 'waves_ssl_leaderboard.csv'}")


if __name__ == "__main__":
    main()
