"""Fetch and preprocess the Montgomery + Shenzhen TB Chest X-ray sets from Kaggle.

https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-montgomery (138 images:
80 normal / 58 TB, from Montgomery County, Maryland, USA)
https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen (662 images:
326 normal / 336 TB, from Shenzhen No.3 Hospital, China)

Built as a better-documented alternative to `datasets/tb_cxr_qatar` for cross-dataset
evaluation against CIDRZ (`datasets/google_health`). That investigation found Qatar's
"Tuberculosis" class is itself a pool of NLM's Montgomery+Shenzhen sets (clinical/
radiological reading) mixed with the Belarus TB Portal set (bacteriologically-confirmed
drug-resistance study) - two different reference standards folded into one undocumented
class, with the "Normal" class's provenance not stated at all on the dataset page.

Montgomery and Shenzhen, by contrast, each ship a metadata CSV (`montgomery_metadata.csv`,
`shenzhen_metadata.csv`) with a `findings` column: "normal" or free-text radiological/
clinical findings (Montgomery's are notably rich - many explicitly mention AFB smear/
culture positivity, e.g. "extensive cavitary TB smear and culture positive..."; Shenzhen's
are terser radiological readings, e.g. "bilateral PTB", "Right PTB"). Every non-normal
label here is traceable to readable text, not an opaque folder name - a real (if still not
perfectly homogeneous - Montgomery mixes active and explicitly inactive/treated TB under
the same positive label) improvement in verifiability over Qatar.

Much smaller than Qatar (800 vs 4,200 images total), the classic paired benchmark in the
TB-CAD literature (Jaeger et al. 2014).

Requires the `google_health` extra (kagglehub) and a Kaggle API token at
`~/.kaggle/kaggle.json`.

Usage:
    python -m datasets.tb_cxr_mont_shen.download --output data/tb_cxr_mont_shen/tb_cxr_mont_shen.pkl
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

from datasets.google_health.download import _stratified_split
from datasets.google_health.features import extract_image_features_pretrained, extract_image_features_xrv

MONTGOMERY_REF = "raddar/tuberculosis-chest-xrays-montgomery"
SHENZHEN_REF = "raddar/tuberculosis-chest-xrays-shenzhen"


def _load_pixel_array(png_path: str) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(png_path).convert("L"))


def _load_source(local_dir: str, metadata_name: str, source_name: str):
    metadata = pd.read_csv(os.path.join(local_dir, metadata_name))
    base = os.path.join(local_dir, "images", "images")
    paths = [os.path.join(base, study_id) for study_id in metadata["study_id"]]
    labels = (metadata["findings"].str.strip().str.lower() != "normal").astype(np.int64).values
    sources = [source_name] * len(paths)
    return paths, labels, sources


def build_dataset(
    output: str,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    image_size: int = 224,
    image_encoder: str = "vgg11_slim",
):
    """Download both sets, extract image features, stratified-split, and pickle it.

    `image_encoder`: "vgg11_slim" (3-channel, ImageNet-normalized) or "xrv" (1-channel,
    torchxrayvision-normalized) - must match the training config's image encoder.
    """
    import kagglehub

    print("Downloading Montgomery set...")
    montgomery_dir = kagglehub.dataset_download(MONTGOMERY_REF)
    print("Downloading Shenzhen set...")
    shenzhen_dir = kagglehub.dataset_download(SHENZHEN_REF)

    m_paths, m_labels, m_sources = _load_source(montgomery_dir, "montgomery_metadata.csv", "montgomery")
    s_paths, s_labels, s_sources = _load_source(shenzhen_dir, "shenzhen_metadata.csv", "shenzhen")
    print(f"Montgomery: {len(m_paths)} images ({int(m_labels.sum())} TB / {int((m_labels == 0).sum())} normal)")
    print(f"Shenzhen: {len(s_paths)} images ({int(s_labels.sum())} TB / {int((s_labels == 0).sum())} normal)")

    paths = m_paths + s_paths
    labels = np.concatenate([m_labels, s_labels])
    sources = np.array(m_sources + s_sources)

    feature_fn = extract_image_features_xrv if image_encoder == "xrv" else extract_image_features_pretrained

    image_feats = []
    for i, path in enumerate(paths, start=1):
        image_feats.append(feature_fn(_load_pixel_array(path), size=image_size))
        if i % 200 == 0:
            print(f"  extracted {i}/{len(paths)}")
    image_feats = np.stack(image_feats)

    train_idx, val_idx, test_idx = _stratified_split(labels, val_frac, test_frac, seed)

    def _subset(idx):
        return {"image": image_feats[idx], "label": labels[idx], "source": sources[idx]}

    data = {"train": _subset(train_idx), "valid": _subset(val_idx), "test": _subset(test_idx)}
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(data, f)
    print(f"Wrote {output}: train={len(train_idx)} valid={len(val_idx)} test={len(test_idx)}")
    for name, idx in (("train", train_idx), ("valid", val_idx), ("test", test_idx)):
        print(f"  {name}: {int(labels[idx].sum())} pos / {int((labels[idx] == 0).sum())} neg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/tb_cxr_mont_shen/tb_cxr_mont_shen.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-encoder", choices=["vgg11_slim", "xrv"], default="vgg11_slim")
    args = parser.parse_args()
    build_dataset(args.output, seed=args.seed, image_encoder=args.image_encoder)


if __name__ == "__main__":
    main()
