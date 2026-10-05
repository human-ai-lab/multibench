"""Rebuild the CIDRZ pickle's images with MONOCHROME1 DICOMs inverted to normal polarity.

`download.build_dataset` used to pass `ds.pixel_array` straight to the image preprocessor, so
every MONOCHROME1 DICOM (320 of 365, all FUJIFILM) was stored as a negative. The pickle keeps
no participant IDs, so each row is matched back to its DICOM by recomputing that DICOM with the
old (raw-pixel) preprocessing and taking the nearest stored image; the match distance is
printed so a wrong mapping is visible. Corrected images replace only the "image" field, so
text, audio, labels and row order are unchanged.

Also writes a per-row CSV (split, row, DICOM file, PhotometricInterpretation, manufacturer,
label, match distance) for confound checks.

Usage:
    python -m datasets.google_health.fix_dicom_polarity \
        --in data/google_health/tb_dataset_monochrome1_bug.pkl \
        --out data/google_health/tb_dataset.pkl
"""
import argparse
import csv
import glob
import os
import pickle

import numpy as np

from datasets.google_health.download import dicom_pixels
from datasets.google_health.features import extract_image_features_pretrained

DICOM_GLOB = "~/.cache/kagglehub/datasets/googlehealthai/**/*.dcm"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", default="data/google_health/tb_dataset_monochrome1_bug.pkl")
    parser.add_argument("--out", default="data/google_health/tb_dataset.pkl")
    parser.add_argument("--meta-csv", default="results/cidrz_dicom_meta.csv")
    parser.add_argument("--max-dist", type=float, default=1e-3,
                        help="Largest acceptable mean abs difference for a row-DICOM match.")
    args = parser.parse_args()

    import pydicom

    files = sorted(glob.glob(os.path.expanduser(DICOM_GLOB), recursive=True))
    old, new, meta = [], [], []
    for f in files:
        ds = pydicom.dcmread(f)
        old.append(extract_image_features_pretrained(ds.pixel_array))
        new.append(extract_image_features_pretrained(dicom_pixels(ds)))
        meta.append((f, str(ds.PhotometricInterpretation), str(getattr(ds, "Manufacturer", ""))))
    old = np.stack(old).reshape(len(files), -1)
    print(f"Recomputed {len(files)} DICOMs")

    with open(args.in_path, "rb") as fh:
        data = pickle.load(fh)

    rows, used = [], set()
    fixed = {}
    for split in ("train", "valid", "test"):
        images = data[split]["image"].astype(np.float32)
        out = np.empty_like(images)
        for i, img in enumerate(images):
            dists = np.abs(old - img.reshape(1, -1)).mean(axis=1)
            j = int(np.argmin(dists))
            if dists[j] > args.max_dist or j in used:
                raise RuntimeError(f"{split}[{i}]: no unique DICOM match (best dist {dists[j]:.4g}, file {j})")
            used.add(j)
            out[i] = new[j]
            rows.append((split, i, *meta[j], int(data[split]["label"][i]), float(dists[j])))
        fixed[split] = {**data[split], "image": out}
        n_inv = sum(r[3] == "MONOCHROME1" for r in rows if r[0] == split)
        print(f"  {split}: {len(images)} rows matched, {n_inv} inverted (MONOCHROME1), "
              f"max match dist {max(r[6] for r in rows if r[0] == split):.2e}")

    with open(args.out, "wb") as fh:
        pickle.dump(fixed, fh)
    print(f"saved -> {args.out}  ({len(files) - len(used)} DICOM(s) not in the pickle)")

    os.makedirs(os.path.dirname(args.meta_csv) or ".", exist_ok=True)
    with open(args.meta_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "row", "file", "photometric", "manufacturer", "label", "match_dist"])
        w.writerows(rows)
    print(f"per-row metadata -> {args.meta_csv}")


if __name__ == "__main__":
    main()
