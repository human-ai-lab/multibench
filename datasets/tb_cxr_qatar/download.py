"""Fetch and preprocess the Qatar University TB Chest X-ray Database from Kaggle.

https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset (695MB,
4,200 PNG chest X-rays: 3,500 normal / 700 tuberculosis-positive, 512x512 RGB, no per-file
rate-limiting quirks like the google_health dataset - a single `kagglehub.dataset_download`
call fetches everything).

Built as a single-modality (image-only) counterpart to `datasets/google_health`: that
dataset's real bottleneck for TB classification was sample size (364 participants, 42
positive) - this one has 16x more positive examples (700) at a similar ~17% positive rate,
letting the same encoder/training choices be tested with data no longer the constraint.

Requires the `google_health` extra (`pip install -e '.[google_health]'` - just needs
`kagglehub`/`kaggle`, already covers this) and a Kaggle API token at
`~/.kaggle/kaggle.json`.

Usage:
    python -m datasets.tb_cxr_qatar.download --output data/tb_cxr_qatar/tb_cxr_qatar.pkl
"""
import argparse
import glob
import os
import pickle

import numpy as np

from datasets.google_health.features import extract_image_features_pretrained, extract_image_features_xrv
from datasets.google_health.download import _stratified_split

DATASET_REF = "tawsifurrahman/tuberculosis-tb-chest-xray-dataset"


def _load_pixel_array(png_path: str) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(png_path).convert("L"))


def build_dataset(
    output: str,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    image_size: int = 224,
    image_encoder: str = "vgg11_slim",
):
    """Download the full dataset, extract image features, stratified-split, and pickle it.

    `image_encoder`: "vgg11_slim" (3-channel, ImageNet-normalized) or "xrv" (1-channel,
    torchxrayvision-normalized) - must match `configs/tb_cxr_qatar.yaml`'s image encoder.
    """
    import kagglehub

    print("Downloading dataset (695MB, one-time)...")
    local_dir = kagglehub.dataset_download(DATASET_REF)
    base = os.path.join(local_dir, "TB_Chest_Radiography_Database")

    normal_paths = sorted(glob.glob(os.path.join(base, "Normal", "*.png")))
    tb_paths = sorted(glob.glob(os.path.join(base, "Tuberculosis", "*.png")))
    print(f"Found {len(normal_paths)} normal, {len(tb_paths)} tuberculosis images.")

    paths = normal_paths + tb_paths
    labels = np.array([0] * len(normal_paths) + [1] * len(tb_paths), dtype=np.int64)

    feature_fn = extract_image_features_xrv if image_encoder == "xrv" else extract_image_features_pretrained

    image_feats = []
    for i, path in enumerate(paths, start=1):
        image_feats.append(feature_fn(_load_pixel_array(path), size=image_size))
        if i % 500 == 0:
            print(f"  extracted {i}/{len(paths)}")
    image_feats = np.stack(image_feats)

    train_idx, val_idx, test_idx = _stratified_split(labels, val_frac, test_frac, seed)

    def _subset(idx):
        return {"image": image_feats[idx], "label": labels[idx]}

    data = {"train": _subset(train_idx), "valid": _subset(val_idx), "test": _subset(test_idx)}
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(data, f)
    print(f"Wrote {output}: train={len(train_idx)} valid={len(val_idx)} test={len(test_idx)}")
    for name, idx in (("train", train_idx), ("valid", val_idx), ("test", test_idx)):
        print(f"  {name}: {int(labels[idx].sum())} pos / {int((labels[idx] == 0).sum())} neg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/tb_cxr_qatar/tb_cxr_qatar.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-encoder", choices=["vgg11_slim", "xrv"], default="vgg11_slim")
    args = parser.parse_args()
    build_dataset(args.output, seed=args.seed, image_encoder=args.image_encoder)


if __name__ == "__main__":
    main()
