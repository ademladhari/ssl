from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter

SSL_ROOT = Path(__file__).resolve().parent
if str(SSL_ROOT) not in sys.path:
    sys.path.insert(0, str(SSL_ROOT))

from src.backbone import Backbone, PCAWhitening  # noqa: E402
from src.detect import detect_zero_bit  # noqa: E402
from src.embed import EmbedConfig, embed_zero_bit  # noqa: E402
from src.io_utils import list_images, load_image  # noqa: E402
from src.keys import load_key  # noqa: E402


ATTACK_NAMES = {
    "distortion_single_rotation": "Dist-Rotation",
    "distortion_single_resizedcrop": "Dist-RCrop",
    "distortion_single_erasing": "Dist-Erase",
    "distortion_single_brightness": "Dist-Bright",
    "distortion_single_contrast": "Dist-Contrast",
    "distortion_single_blurring": "Dist-Blur",
    "distortion_single_noise": "Dist-Noise",
    "distortion_single_jpeg": "Dist-JPEG",
    "distortion_combo_geometric": "Dist-Com-Geo",
    "distortion_combo_photometric": "Dist-Com-Photo",
    "distortion_combo_degradation": "Dist-Com-Deg",
    "distortion_combo_all": "Dist-Com-All",
    "regen_diffusion": "Regen-Diffusion",
    "regen_diffusion_prompt": "Regen-Diffusion&P",
    "regen_vae": "Regen-VAE",
    "kl_vae": "Regen-KLVAE",
    "2x_regen": "Regen-2xDiffusion",
    "4x_regen": "Regen-4xDiffusion",
    "4x_regen_bmshj": "Regen-4xVAE",
    "4x_regen_kl_vae": "Regen-4xKLVAE",
    "adv_emb_resnet18_untg": "AdvEmb-RN18",
    "adv_emb_clip_untg_alphaRatio_0.05_step_200": "AdvEmb-CLIP",
    "adv_emb_same_vae_untg": "AdvEmb-KLVAE8",
    "adv_emb_klf16_vae_untg": "AdvEmb-KLVAE16",
    "adv_emb_sdxl_vae_untg": "AdvEmb-SdxlVAE",
    "adv_cls_unwm_wm_0.01_50_warm_train3k": "AdvCls-UnWM-WM",
    "adv_cls_real_wm_0.01_50_warm": "AdvCls-Real-WM",
    "adv_cls_wm1_wm2_0.01_50_warm": "AdvCls-WM1-WM2",
    "adv_cls_wm1_wm2_0.04_200_warm": "abandon",
}


@dataclass
class AttackSummary:
    attack_key: str
    attack_label: str
    q_at_07p: float
    q_at_04p: float
    avg_p: float
    avg_q: float
    rank: int = 0


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


def _to_pil_gray(img: np.ndarray) -> Image.Image:
    return Image.fromarray(_to_uint8(img)).convert("L")


def _to_np_gray(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"), dtype=np.float64)


def _attack_jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    pil = _to_pil_gray(img)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    out = Image.open(buf).convert("L")
    return np.array(out, dtype=np.float64)


def ssim_gray(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    mu_a = gaussian_filter(a, sigma=1.5)
    mu_b = gaussian_filter(b, sigma=1.5)
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_a_sq = gaussian_filter(a * a, sigma=1.5) - mu_a_sq
    sigma_b_sq = gaussian_filter(b * b, sigma=1.5) - mu_b_sq
    sigma_ab = gaussian_filter(a * b, sigma=1.5) - mu_ab
    num = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    den = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    return float(np.mean(num / (den + 1e-12)))


def _resize_to_shape(img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    h, w = target_shape
    return np.array(_to_pil_gray(img).resize((w, h), Image.Resampling.BICUBIC), dtype=np.float64)


def _single_distortion(img: np.ndarray, distortion_type: str, strength: float) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    pil = _to_pil_gray(img)
    w, h = pil.size
    if distortion_type == "rotation":
        angle = strength * 25.0
        return _to_np_gray(pil.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False))
    if distortion_type == "resizedcrop":
        crop_ratio = 1.0 - 0.35 * strength
        cw, ch = max(1, int(w * crop_ratio)), max(1, int(h * crop_ratio))
        left = (w - cw) // 2
        top = (h - ch) // 2
        cropped = pil.crop((left, top, left + cw, top + ch))
        return _to_np_gray(cropped.resize((w, h), Image.Resampling.BICUBIC))
    if distortion_type == "erasing":
        arr = np.array(pil, dtype=np.float64)
        bw, bh = max(1, int(w * 0.25 * strength)), max(1, int(h * 0.25 * strength))
        left = (w - bw) // 2
        top = (h - bh) // 2
        arr[top : top + bh, left : left + bw] = 0
        return arr
    if distortion_type == "brightness":
        return _to_np_gray(ImageEnhance.Brightness(pil).enhance(1.0 + 0.8 * strength))
    if distortion_type == "contrast":
        return _to_np_gray(ImageEnhance.Contrast(pil).enhance(1.0 + 0.9 * strength))
    if distortion_type == "blurring":
        return _to_np_gray(pil.filter(ImageFilter.GaussianBlur(radius=0.4 + 2.0 * strength)))
    if distortion_type == "noise":
        arr = np.array(pil, dtype=np.float64)
        sigma = 8.0 + 20.0 * strength
        return np.clip(arr + np.random.default_rng(123).normal(0.0, sigma, arr.shape), 0, 255)
    if distortion_type == "compression":
        q = int(np.clip(95 - 80 * strength, 10, 95))
        return _attack_jpeg(img, quality=q)
    raise ValueError(f"Unknown distortion type: {distortion_type}")


def _combo_distortion(img: np.ndarray, combo: list[tuple[str, float]], global_strength: float) -> np.ndarray:
    out = img.copy()
    for dtype, weight in combo:
        out = _single_distortion(out, dtype, min(1.0, max(0.0, global_strength * weight)))
    return out


def _approx_regen(img: np.ndarray, strength: float, rounds: int, prompt: bool = False) -> np.ndarray:
    out = img.copy()
    quality = int(np.clip(90 - strength * 70, 10, 95))
    blur_radius = float(strength * (1.5 if prompt else 1.0))
    noise_sigma = float(strength * (10.0 if prompt else 7.0))
    for _ in range(rounds):
        out = _attack_jpeg(out, quality=quality)
        out = np.array(_to_pil_gray(out).filter(ImageFilter.GaussianBlur(radius=blur_radius)), dtype=np.float64)
        noise = np.random.default_rng(123).normal(0.0, noise_sigma, out.shape)
        out = np.clip(out + noise, 0, 255)
    return out


def _approx_adv(img: np.ndarray, strength: float, mode: str) -> np.ndarray:
    out = img.copy()
    rng = np.random.default_rng(777)
    if mode.startswith("adv_emb"):
        sigma = 3.0 + strength * 10.0
        high = out - np.array(_to_pil_gray(out).filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float64)
        out = np.clip(out + 0.55 * high + rng.normal(0, sigma, out.shape), 0, 255)
    else:
        sigma = 4.0 + strength * 12.0
        local = np.array(
            _to_pil_gray(out).filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=3)),
            dtype=np.float64,
        )
        out = np.clip(0.7 * out + 0.3 * local + rng.normal(0, sigma, out.shape), 0, 255)
    return out


def attack_fn(attack_name: str):
    if attack_name == "distortion_single_rotation":
        return lambda img, s: _single_distortion(img, "rotation", s)
    if attack_name == "distortion_single_resizedcrop":
        return lambda img, s: _single_distortion(img, "resizedcrop", s)
    if attack_name == "distortion_single_erasing":
        return lambda img, s: _single_distortion(img, "erasing", s)
    if attack_name == "distortion_single_brightness":
        return lambda img, s: _single_distortion(img, "brightness", s)
    if attack_name == "distortion_single_contrast":
        return lambda img, s: _single_distortion(img, "contrast", s)
    if attack_name == "distortion_single_blurring":
        return lambda img, s: _single_distortion(img, "blurring", s)
    if attack_name == "distortion_single_noise":
        return lambda img, s: _single_distortion(img, "noise", s)
    if attack_name == "distortion_single_jpeg":
        return lambda img, s: _single_distortion(img, "compression", s)
    if attack_name == "distortion_combo_geometric":
        return lambda img, s: _combo_distortion(img, [("rotation", 1.0), ("resizedcrop", 1.0), ("erasing", 1.0)], s)
    if attack_name == "distortion_combo_photometric":
        return lambda img, s: _combo_distortion(img, [("brightness", 1.0), ("contrast", 1.0)], s)
    if attack_name == "distortion_combo_degradation":
        return lambda img, s: _combo_distortion(img, [("blurring", 1.0), ("noise", 1.0), ("compression", 1.0)], s)
    if attack_name == "distortion_combo_all":
        return lambda img, s: _combo_distortion(
            img,
            [
                ("rotation", 1.0),
                ("resizedcrop", 1.0),
                ("erasing", 0.8),
                ("brightness", 0.7),
                ("contrast", 0.7),
                ("blurring", 0.8),
                ("noise", 0.8),
                ("compression", 0.8),
            ],
            s,
        )
    if attack_name in {"regen_diffusion", "regen_diffusion_prompt"}:
        return lambda img, s: _approx_regen(img, s, rounds=1, prompt=(attack_name == "regen_diffusion_prompt"))
    if attack_name in {"regen_vae", "kl_vae"}:
        return lambda img, s: _approx_regen(img, s, rounds=1, prompt=False)
    if attack_name == "2x_regen":
        return lambda img, s: _approx_regen(img, s, rounds=2, prompt=False)
    if attack_name in {"4x_regen", "4x_regen_bmshj", "4x_regen_kl_vae"}:
        return lambda img, s: _approx_regen(img, s, rounds=4, prompt=False)
    if attack_name.startswith("adv_emb"):
        return lambda img, s: _approx_adv(img, s, mode="adv_emb")
    if attack_name.startswith("adv_cls"):
        return lambda img, s: _approx_adv(img, s, mode="adv_cls")
    raise ValueError(f"Unsupported attack name: {attack_name}")


def q_at_threshold(p_values: list[float], q_values: list[float], threshold: float) -> float:
    candidates = [q for p, q in zip(p_values, q_values) if p >= threshold]
    return float(max(candidates)) if candidates else float("-inf")


def run_benchmark_local(
    *,
    original: np.ndarray,
    watermarked: np.ndarray,
    score_fn,
    strengths: list[float],
    out_dir: Path,
    output_prefix: str,
    excluded_attacks: set[str] | None = None,
    parameters: dict[str, object] | None = None,
    notes: list[str] | None = None,
) -> None:
    excluded = excluded_attacks or set()
    selected_attacks = [k for k in ATTACK_NAMES.keys() if k not in excluded]
    if len(selected_attacks) != 26:
        raise RuntimeError(f"Expected 26 attacks, got {len(selected_attacks)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    clean_score = float(score_fn(watermarked))
    if clean_score <= 0:
        raise RuntimeError(f"Invalid clean score ({clean_score}); score must be positive on clean image.")

    per_strength_rows: list[dict[str, object]] = []
    summaries: list[AttackSummary] = []

    for attack_key in selected_attacks:
        fn = attack_fn(attack_key)
        p_values: list[float] = []
        q_values: list[float] = []
        for strength in strengths:
            attacked = _resize_to_shape(fn(watermarked, strength), original.shape)
            score = float(score_fn(attacked))
            p = float(np.clip(score / clean_score, 0.0, 1.0))
            q = float(np.clip(ssim_gray(original, attacked), 0.0, 1.0))
            p_values.append(p)
            q_values.append(q)
            per_strength_rows.append(
                {
                    "attack_key": attack_key,
                    "attack_label": ATTACK_NAMES[attack_key],
                    "strength": strength,
                    "P": p,
                    "Q": q,
                    "raw_similarity": score,
                }
            )

        summaries.append(
            AttackSummary(
                attack_key=attack_key,
                attack_label=ATTACK_NAMES[attack_key],
                q_at_07p=q_at_threshold(p_values, q_values, threshold=0.7),
                q_at_04p=q_at_threshold(p_values, q_values, threshold=0.4),
                avg_p=float(np.mean(p_values)),
                avg_q=float(np.mean(q_values)),
            )
        )

    summaries.sort(key=lambda x: x.avg_p, reverse=True)
    for i, summary in enumerate(summaries, start=1):
        summary.rank = i

    leaderboard_df = pd.DataFrame(
        [
            {
                "Attack": s.attack_label,
                "Rank": s.rank,
                "Q@0.7P": s.q_at_07p,
                "Q@0.4P": s.q_at_04p,
                "Avg P": s.avg_p,
                "Avg Q": s.avg_q,
                "attack_key": s.attack_key,
            }
            for s in summaries
        ]
    )
    strength_df = pd.DataFrame(per_strength_rows)
    leaderboard_csv = out_dir / f"{output_prefix}_leaderboard.csv"
    strengths_csv = out_dir / f"{output_prefix}_per_strength.csv"
    report_json = out_dir / f"{output_prefix}_report.json"
    leaderboard_df.to_csv(leaderboard_csv, index=False)
    strength_df.to_csv(strengths_csv, index=False)
    report = {
        "mode": "WAVES-style SSL benchmark (standalone)",
        "clean_similarity": clean_score,
        "num_attacks": len(selected_attacks),
        "excluded_attacks": sorted(list(excluded)),
        "parameters": parameters or {},
        "outputs": {"leaderboard_csv": str(leaderboard_csv), "per_strength_csv": str(strengths_csv)},
        "notes": notes or [],
    }
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def _score_to_similarity(score: float) -> float:
    clipped = float(np.clip(score, -30.0, 30.0))
    return float(1.0 / (1.0 + np.exp(-clipped)))


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
            return _score_to_similarity(float(score.item()))

        run_dir = per_run_root / f"run_{run_idx:03d}"
        per_run_dirs.append(run_dir)
        print(f"[{run_idx}/{len(image_paths)}] {image_path.name} -> {run_dir}")
        run_benchmark_local(
            original=original_gray,
            watermarked=original_gray,
            score_fn=score_fn,
            strengths=strengths,
            out_dir=run_dir,
            output_prefix="waves_ssl",
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
                "Standalone benchmark implementation: no dependency on repository-level waves_benchmark_common.py.",
                "Local WAVES-like attack suite is implemented inside this file for Colab portability.",
            ],
        )

    aggregate_runs(per_run_dirs, out_dir)
    print(f"Done. Leaderboard: {out_dir / 'waves_ssl_leaderboard.csv'}")


if __name__ == "__main__":
    main()
