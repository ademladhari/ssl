# SSL Watermarking

Reproduction of Fernandez et al., "Watermarking Images in Self-Supervised Latent Spaces"
(arXiv:2112.09581). Pixels are optimized with Adam so that the feature of a frozen
DINO ResNet-50 (+ PCA-whitening) lands in a hypercone (zero-bit) or matches a set
of carrier signs (multi-bit), under SSIM-masked PSNR constraints and augmentation.

## Install

```
pip install -r requirements.txt
```

## Fit PCA-whitening

Collect a folder of natural images (e.g. ~1k-10k of YFCC/COCO) and fit PCA-whitening:

```
python scripts/fit_pca.py --images-dir path/to/images --out checkpoints/whitening.pt
```

## Generate a key

Zero-bit (single unit-norm carrier):

```
python -c "from src.keys import save_zerobit_key; save_zerobit_key('checkpoints/key_zero.npy', d=2048)"
```

Multi-bit (k orthonormal carriers):

```
python -c "from src.keys import save_multibit_key; save_multibit_key('checkpoints/key_multi.npy', k=30, d=2048)"
```

## Mark an image

Zero-bit:

```
python scripts/mark.py --mode zero_bit --image path/to/img.png --key checkpoints/key_zero.npy \
    --whitening checkpoints/whitening.pt --out out.png
```

Multi-bit:

```
python scripts/mark.py --mode multi_bit --image path/to/img.png --key checkpoints/key_multi.npy \
    --whitening checkpoints/whitening.pt --message 010110... --out out.png
```

## Detect / decode

```
python scripts/detect.py --mode zero_bit --image out.png --key checkpoints/key_zero.npy \
    --whitening checkpoints/whitening.pt --fpr 1e-6
```

## Evaluate

```
python scripts/evaluate.py --mode zero_bit --images-dir path/to/eval \
    --key checkpoints/key_zero.npy --whitening checkpoints/whitening.pt
```

## Layout

- `src/backbone.py` - DINO ResNet-50 + PCA-whitening wrapper.
- `src/pca.py` - fit PCA-whitening from image features.
- `src/augmentations.py` - differentiable marking-time augmentations.
- `src/constraints.py` - SSIM attenuation + PSNR clipping.
- `src/losses.py` - zero-bit/multi-bit Lw, image loss, FPR -> theta.
- `src/keys.py` - secret key / carrier generation and I/O.
- `src/embed.py` - Algorithm 1 marking loop.
- `src/detect.py` - detection and decoding.
- `src/attacks.py` - eval-time attacks.
- `configs/` - default hyperparameters.
- `scripts/` - CLI entry points.
