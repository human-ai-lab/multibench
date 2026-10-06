"""Standardize every TB chest X-ray corpus to one on-disk format.

Per corpus, writes `data/tb_multi/<corpus>.npy` (uint8, N x 224 x 224 grayscale, per-image min-max
then bilinear resize - the same preprocessing as `features.extract_image_features_pretrained`
before ImageNet normalization) and `data/tb_multi/<corpus>_meta.csv` with one row per image:
id, label (1 = active TB; see `label_note`), corpus, orig_h, orig_w, file ext, plus corpus-specific
columns (e.g. TBX11K category). Downstream code uses `load_corpus(name)`.

Label conventions: label=1 is TB-positive as the corpus defines it. TBX11K also has "sick but
non-TB" (label 0, category `sick`) and "latent/uncertain TB" images that are kept in the
metadata but flagged `use_for_binary=False` for latent/uncertain (no clean binary label).

Usage:
    python -m datasets.tb_multi.standardize --corpora qatar montgomery shenzhen tbx11k pakistan cidrz
"""
import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
from PIL import Image

OUT_DIR = "data/tb_multi"
SIZE = 224
IMAGENET_MEAN, IMAGENET_STD = 0.485, 0.229  # channel 0 stats; CIDRZ pickle stores ImageNet-normalized copies


def to_u8(gray: np.ndarray) -> np.ndarray:
    arr = gray.astype(np.float32)
    arr -= arr.min()
    peak = arr.max()
    if peak > 0:
        arr /= peak
    img = Image.fromarray((arr * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _read_gray(path):
    im = Image.open(path)
    w, h = im.size
    return np.array(im.convert("L")), h, w


def _from_paths(corpus, paths, labels, ids=None, extra=None):
    imgs, rows = [], []
    for k, p in enumerate(paths):
        g, h, w = _read_gray(p)
        imgs.append(to_u8(g))
        r = dict(id=ids[k] if ids is not None else os.path.basename(p), label=int(labels[k]), corpus=corpus,
                 orig_h=h, orig_w=w, ext=os.path.splitext(p)[1].lower())
        if extra is not None:
            r.update({key: v[k] for key, v in extra.items()})
        rows.append(r)
        if (k + 1) % 1000 == 0:
            print(f"  {corpus}: {k + 1}/{len(paths)}", flush=True)
    return np.stack(imgs), pd.DataFrame(rows)


def build_qatar():
    import kagglehub
    base = os.path.join(kagglehub.dataset_download("tawsifurrahman/tuberculosis-tb-chest-xray-dataset"),
                        "TB_Chest_Radiography_Database")
    n = sorted(glob.glob(os.path.join(base, "Normal", "*.png")))
    t = sorted(glob.glob(os.path.join(base, "Tuberculosis", "*.png")))
    return _from_paths("qatar", n + t, [0] * len(n) + [1] * len(t))


def _mont_shen(corpus, ref, meta_name):
    import kagglehub
    d = kagglehub.dataset_download(ref)
    m = pd.read_csv(os.path.join(d, meta_name))
    paths = [os.path.join(d, "images", "images", s) for s in m["study_id"]]
    labels = (m["findings"].str.strip().str.lower() != "normal").astype(int).values
    return _from_paths(corpus, paths, labels, ids=list(m["study_id"]), extra={"findings": m["findings"].values})


def build_montgomery():
    return _mont_shen("montgomery", "raddar/tuberculosis-chest-xrays-montgomery", "montgomery_metadata.csv")


def build_shenzhen():
    return _mont_shen("shenzhen", "raddar/tuberculosis-chest-xrays-shenzhen", "shenzhen_metadata.csv")


def build_pakistan():
    d = "data/raw/pakistan/images"
    paths = sorted(glob.glob(os.path.join(d, "*.jpg")))
    labels = [1 if os.path.basename(p).startswith("TB") else 0 for p in paths]
    return _from_paths("pakistan", paths, labels)


def build_cidrz():
    """CIDRZ images come from the DICOM-derived pickle (polarity already fixed); recover uint8 by
    undoing ImageNet normalization on channel 0."""
    with open("data/google_health/tb_dataset.pkl", "rb") as f:
        data = pickle.load(f)
    imgs, labels = [], []
    for s in ("train", "valid", "test"):
        x = data[s]["image"][:, 0] * IMAGENET_STD + IMAGENET_MEAN
        imgs.append(np.clip(np.round(x * 255), 0, 255).astype(np.uint8))
        labels.append(data[s]["label"])
    imgs, labels = np.concatenate(imgs), np.concatenate(labels)
    meta = pd.DataFrame(dict(id=[f"cidrz_{i}" for i in range(len(imgs))], label=labels.astype(int), corpus="cidrz",
                             orig_h=SIZE, orig_w=SIZE, ext=".dcm"))
    return imgs, meta


def build_tbx11k():
    from datasets.tb_multi.tbx11k import build
    return build(_from_paths)


BUILDERS = dict(qatar=build_qatar, montgomery=build_montgomery, shenzhen=build_shenzhen, tbx11k=build_tbx11k,
                pakistan=build_pakistan, cidrz=build_cidrz)


def load_corpus(name):
    return np.load(os.path.join(OUT_DIR, f"{name}.npy")), pd.read_csv(os.path.join(OUT_DIR, f"{name}_meta.csv"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="+", default=list(BUILDERS))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for c in args.corpora:
        print(f"building {c}...", flush=True)
        img, meta = BUILDERS[c]()
        np.save(os.path.join(OUT_DIR, f"{c}.npy"), img)
        meta.to_csv(os.path.join(OUT_DIR, f"{c}_meta.csv"), index=False)
        print(f"  {c}: n={len(meta)} pos={int(meta['label'].sum())} neg={int((meta['label'] == 0).sum())}")


if __name__ == "__main__":
    main()
