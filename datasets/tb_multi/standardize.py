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

SIZE = int(os.environ.get("TB_RES", 224))  # TB_RES=448 rebuilds the whole pipeline at native-ish resolution
OUT_DIR = "data/tb_multi" if SIZE == 224 else f"data/tb_multi_{SIZE}"
IMAGENET_MEAN, IMAGENET_STD = 0.485, 0.229  # channel 0 stats; CIDRZ pickle stores ImageNet-normalized copies


def to_u8(gray: np.ndarray, size: int = None) -> np.ndarray:
    size = size or SIZE
    arr = gray.astype(np.float32)
    arr -= arr.min()
    peak = arr.max()
    if peak > 0:
        arr /= peak
    img = Image.fromarray((arr * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
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


def _build_cidrz_hires():
    """Native-resolution CIDRZ: re-render every cached DICOM (polarity fixed) at SIZE and put the images in the order of the
    224 pickle by matching each DICOM's 224 render to the pickle's uint8 images (exact up to rounding)."""
    import pydicom
    from datasets.google_health.download import dicom_pixels
    base = glob.glob(os.path.expanduser("~/.cache/kagglehub/datasets/googlehealthai/google-health-ai/versions/*"))[0]
    files = sorted(glob.glob(os.path.join(base, "**", "*.dcm"), recursive=True))
    lo, meta = np.load("data/tb_multi/cidrz.npy"), pd.read_csv("data/tb_multi/cidrz_meta.csv")
    hi = np.zeros((len(meta), SIZE, SIZE), np.uint8)
    seen = np.zeros(len(meta), bool)
    flat = lo.reshape(len(lo), -1).astype(np.float32)
    for f in files:
        arr = dicom_pixels(pydicom.dcmread(f))
        a = to_u8(arr, 224).reshape(1, -1).astype(np.float32)
        d = np.abs(flat - a).mean(1)
        j = int(d.argmin())
        assert d[j] < 2.0, f"no 224 match for {f} (mean abs diff {d[j]:.2f})"
        assert not seen[j], f"two DICOMs matched pickle row {j}"
        seen[j] = True
        hi[j] = to_u8(arr, SIZE)
    assert seen.all(), f"{int((~seen).sum())} pickle rows have no DICOM"
    return hi, meta.assign(orig_h=SIZE, orig_w=SIZE)


def build_cidrz():
    """CIDRZ images come from the DICOM-derived pickle (polarity already fixed); recover uint8 by
    undoing ImageNet normalization on channel 0."""
    if SIZE != 224:
        return _build_cidrz_hires()
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


def build_india():
    """Kaggle `mohdabdulrahman1/indian-tb-and-normal-x-rays` (512x512 JPEG, train+val folders merged).
    Provenance is not documented by the uploader; used only as an extra external target, after
    `results/tb_multi_india_report.md` showed 0 near-duplicates against every other corpus."""
    import kagglehub
    root = kagglehub.dataset_download("mohdabdulrahman1/indian-tb-and-normal-x-rays")
    paths, labels, split = [], [], []
    for sp in ("train", "val"):
        for cls, lab in (("normal", 0), ("tb", 1)):
            ps = sorted(glob.glob(os.path.join(root, sp, cls, "*")))
            paths += ps; labels += [lab] * len(ps); split += [sp] * len(ps)
    ids = [os.path.relpath(p, root) for p in paths]
    return _from_paths("india", paths, labels, ids=ids, extra=dict(split=split))


VINDR_ROOT = "/data/physionet.org/files/vindr-cxr/1.0.0"
VINDR_FINDINGS = ["Pulmonary fibrosis", "Calcification", "Lung cavity", "Nodule/Mass", "Infiltration", "Consolidation",
                  "Pleural effusion", "Pleural thickening", "No finding"]


def _vindr_decode(job):
    import pydicom
    from datasets.google_health.download import dicom_pixels
    path, size = job
    ds = pydicom.dcmread(path)
    arr = dicom_pixels(ds)  # MONOCHROME1 inverted to the MONOCHROME2 convention, as for CIDRZ
    return to_u8(arr, size), arr.shape[0], arr.shape[1], str(ds.PhotometricInterpretation), int(ds.BitsStored)


def build_vindr():
    """VinDr-CXR 1.0.0 (PhysioNet, credentialed), the subset in `results/vindr_selection.csv` (`download_vindr`).
    Labels are radiologist read-outs: train = >=2 of 3 'Tuberculosis' reads (exactly 1 read excluded as ambiguous),
    test = the consensus label. The negatives were sampled 50/50 from 'No finding' and other-findings images, so each
    row carries `w`, the inverse sampling probability (release count of its split x label x stratum / sampled count),
    which makes weighted metrics representative of the release's TB-free + TB-positive images (prevalence ~3.6%)."""
    from multiprocessing import Pool
    sel = pd.read_csv("results/vindr_selection.csv")
    ann = f"{VINDR_ROOT}/annotations"
    tr, te = pd.read_csv(f"{ann}/image_labels_train.csv"), pd.read_csv(f"{ann}/image_labels_test.csv").set_index("image_id")
    g = tr.groupby("image_id")
    votes, maj = g["Tuberculosis"].sum(), (g[VINDR_FINDINGS].sum() >= 2).astype(int)
    nofind_tr = maj["No finding"].astype(bool)
    pop = {("train", "nofind"): int(((votes == 0) & nofind_tr).sum()), ("train", "other"): int(((votes == 0) & ~nofind_tr).sum()),
           ("test", "nofind"): int(((te.Tuberculosis == 0) & (te["No finding"] == 1)).sum()),
           ("test", "other"): int(((te.Tuberculosis == 0) & (te["No finding"] == 0)).sum())}
    stratum = []
    for i, s, lab in zip(sel.image_id, sel.split, sel.label):
        nf = bool(nofind_tr[i]) if s == "train" else bool(te.loc[i, "No finding"])
        stratum.append("tb" if lab else ("nofind" if nf else "other"))
    sel = sel.assign(stratum=stratum)
    n_sel = sel[sel.label == 0].groupby(["split", "stratum"]).size().to_dict()
    sel["w"] = [1.0 if lab else pop[(s, st)] / n_sel[(s, st)] for s, st, lab in zip(sel.split, sel.stratum, sel.label)]
    with Pool(8) as pool:
        out = pool.map(_vindr_decode, [(f"{VINDR_ROOT}/{s}/{i}.dicom", SIZE) for i, s in zip(sel.image_id, sel.split)], chunksize=8)
    imgs = np.stack([o[0] for o in out])
    meta = pd.DataFrame(dict(id=sel.image_id.values, label=sel.label.values, corpus="vindr", orig_h=[o[1] for o in out],
                             orig_w=[o[2] for o in out], ext=".dicom", split=sel.split.values, tb_votes=sel.tb_votes.values,
                             stratum=sel.stratum.values, w=sel.w.values, photometric=[o[3] for o in out],
                             bits=[o[4] for o in out]))
    for f in VINDR_FINDINGS:  # per-image finding flags (train: >=2 of 3 reads) for the label-validity check
        meta["f_" + f.replace(" ", "_").replace("/", "_")] = [int(maj.loc[i, f]) if s == "train" else int(te.loc[i, f])
                                                              for i, s in zip(sel.image_id, sel.split)]
    return imgs, meta


BUILDERS = dict(qatar=build_qatar, montgomery=build_montgomery, shenzhen=build_shenzhen, tbx11k=build_tbx11k,
                pakistan=build_pakistan, cidrz=build_cidrz, india=build_india, vindr=build_vindr)


def load_corpus(name):
    return np.load(os.path.join(OUT_DIR, f"{name}.npy")), pd.read_csv(os.path.join(OUT_DIR, f"{name}_meta.csv"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="+", default=[c for c in BUILDERS if c not in ("india", "vindr")])
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
