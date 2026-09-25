"""Rebuild the Qatar and CIDRZ image pickles with CLAHE contrast normalization.

CLAHE (Contrast-Limited Adaptive Histogram Equalization) locally re-normalizes an image's
intensity histogram, which should reduce differences in exposure/contrast/windowing
between the two datasets' X-ray acquisition pipelines - a very plausible contributor to
the observed domain shift, and widely used as a CXR-specific preprocessing step in the
TB-screening literature.

Rather than re-downloading and re-extracting from the raw DICOMs/PNGs (Qatar: 4200 PNGs,
CIDRZ: 364 DICOMs, both already fully cached locally - see download.py/features.py for
why re-scanning CIDRZ specifically was expensive the first time, due to Kaggle's per-file
download endpoint flakiness), this recovers the pre-ImageNet-normalization [0,1] pixel
values directly from the EXISTING pickles: `extract_image_features_pretrained` computes
`(replicated_[0,1]_image - IMAGENET_MEAN) / IMAGENET_STD`, an invertible per-channel affine
map, and since the 3 channels are identical (channel-replicated from grayscale), inverting
it on channel 0 alone exactly recovers the resized [0,1] grayscale image CLAHE should
operate on. This guarantees byte-for-byte the same participant set/split/order as the
original pickles (no re-sampling risk) and requires no network access at all.

Usage:
    python -m datasets.tb_cxr_qatar.clahe_preprocess \
        --qatar-in data/tb_cxr_qatar/tb_cxr_qatar.pkl \
        --qatar-out data/tb_cxr_qatar/tb_cxr_qatar_clahe.pkl \
        --cidrz-in data/google_health/tb_dataset.pkl \
        --cidrz-out data/google_health/tb_dataset_clahe.pkl
"""
import argparse
import pickle

import numpy as np

from datasets.google_health.features import IMAGENET_MEAN, IMAGENET_STD


def apply_clahe_to_normalized_batch(images: np.ndarray, clip_limit: float = 0.01) -> np.ndarray:
    """images: (N, 3, H, W) float32, already ImageNet-normalized & channel-replicated.

    Returns an array of the same shape/normalization convention, with CLAHE applied to
    the recovered grayscale image before re-normalizing.
    """
    from skimage.exposure import equalize_adapthist

    mean = IMAGENET_MEAN[:, None, None].astype(np.float32)
    std = IMAGENET_STD[:, None, None].astype(np.float32)
    out = np.empty_like(images)
    for i in range(images.shape[0]):
        raw01 = images[i] * std + mean  # (3, H, W); channels are identical (replicated)
        gray = np.clip(raw01[0], 0.0, 1.0).astype(np.float64)
        eq = equalize_adapthist(gray, clip_limit=clip_limit).astype(np.float32)
        eq3 = np.repeat(eq[None, :, :], 3, axis=0)
        out[i] = (eq3 - mean) / std
    return out


def _rebuild(in_path: str, out_path: str, clip_limit: float) -> None:
    with open(in_path, "rb") as f:
        data = pickle.load(f)
    rebuilt = {}
    for split in ("train", "valid", "test"):
        images = apply_clahe_to_normalized_batch(data[split]["image"].astype(np.float32), clip_limit)
        rebuilt[split] = {**data[split], "image": images}
        print(f"  {in_path} [{split}]: {images.shape[0]} images CLAHE-processed")
    with open(out_path, "wb") as f:
        pickle.dump(rebuilt, f)
    print(f"  saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qatar-in", default="data/tb_cxr_qatar/tb_cxr_qatar.pkl")
    parser.add_argument("--qatar-out", default="data/tb_cxr_qatar/tb_cxr_qatar_clahe.pkl")
    parser.add_argument("--cidrz-in", default="data/google_health/tb_dataset.pkl")
    parser.add_argument("--cidrz-out", default="data/google_health/tb_dataset_clahe.pkl")
    parser.add_argument("--clip-limit", type=float, default=0.01)
    args = parser.parse_args()

    print("Rebuilding Qatar pickle with CLAHE...")
    _rebuild(args.qatar_in, args.qatar_out, args.clip_limit)
    print("Rebuilding CIDRZ pickle with CLAHE...")
    _rebuild(args.cidrz_in, args.cidrz_out, args.clip_limit)


if __name__ == "__main__":
    main()
