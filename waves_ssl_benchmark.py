from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
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
from src.io_utils import load_image  # noqa: E402
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


def main() -> None:
    image_path = _require_path("WAVES_SSL_IMAGE", os.getenv("WAVES_SSL_IMAGE", ""))
    key_path = _require_path("WAVES_SSL_KEY", os.getenv("WAVES_SSL_KEY", "checkpoints/key_zero.npy"))
    whitening_path = _require_path(
        "WAVES_SSL_WHITENING", os.getenv("WAVES_SSL_WHITENING", "checkpoints/whitening.pt")
    )
    config_path = _require_path("WAVES_SSL_CONFIG", os.getenv("WAVES_SSL_CONFIG", "configs/zero_bit.yaml"))

    out_dir = Path(os.getenv("WAVES_SSL_OUT_DIR", "outputs_waves_ssl"))
    device = os.getenv("WAVES_SSL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    seed = int(os.getenv("WAVES_SSL_SEED", "42"))
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

    with torch.no_grad():
        marked = embed_zero_bit(backbone, image, key=key, fpr=fpr, config=embed_cfg, seed=seed)

    watermarked_rgb = (marked[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.float64)
    original_gray = np.asarray(Image.fromarray(_to_uint8(watermarked_rgb)).convert("L"), dtype=np.float64)

    def score_fn(candidate_gray: np.ndarray) -> float:
        candidate_rgb = np.stack([_to_uint8(candidate_gray)] * 3, axis=-1)
        candidate_tensor = (
            torch.from_numpy(candidate_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        )
        _, score = detect_zero_bit(backbone, candidate_tensor, key=key, fpr=fpr)
        return float(score.item())

    outputs = run_waves_benchmark(
        original=original_gray,
        watermarked=original_gray,
        score_fn=score_fn,
        strengths=strengths,
        out_dir=out_dir,
        output_prefix="waves_ssl",
        waves_root=waves_root,
        mode_name="WAVES-style SSL benchmark",
        excluded_attacks=excluded,
        parameters={
            "image_path": str(image_path),
            "key_path": str(key_path),
            "whitening_path": str(whitening_path),
            "config_path": str(config_path),
            "device": device,
            "seed": seed,
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
            "Embedding is run once; attacks are applied on the watermarked image for WAVES robustness evaluation.",
            "Distortion attacks come from WAVES distortions module via shared benchmark runner.",
        ],
    )

    print(f"Done. Leaderboard: {outputs['leaderboard_csv']}")


if __name__ == "__main__":
    main()
