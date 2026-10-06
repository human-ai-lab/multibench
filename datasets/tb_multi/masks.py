"""Lung masks (torchxrayvision ChestX-Det PSPNet, same recipe as tb_cxr_qatar/lung_mask_preprocess.py) for
every standardized corpus. Writes <corpus>_mask.npy (bool, N x 224 x 224) and <corpus>_lung.npy (uint8,
outside-lung = 0). Segmentation failures (< --min-area of the frame) are kept unmasked."""
import argparse
import os

import numpy as np

from datasets.tb_cxr_qatar.lung_mask_preprocess import refine_mask, segment_lungs
from datasets.tb_multi.standardize import OUT_DIR, load_corpus
from utils.device import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+", default=["qatar", "montgomery", "shenzhen", "tbx11k", "pakistan", "cidrz"])
    ap.add_argument("--dilate", type=int, default=6)
    ap.add_argument("--min-area", type=float, default=0.05)
    args = ap.parse_args()
    device = get_device()
    for c in args.corpora:
        img, _ = load_corpus(c)
        raw = segment_lungs(img.astype(np.float32) / 255.0, device)
        masks = np.stack([refine_mask(m, args.dilate) for m in raw])
        failed = masks.mean(axis=(1, 2)) < args.min_area
        masks[failed] = True
        print(f"{c}: n={len(img)} mask area mean {masks[~failed].mean():.3f}, failures {int(failed.sum())}", flush=True)
        np.save(os.path.join(OUT_DIR, f"{c}_mask.npy"), masks)
        np.save(os.path.join(OUT_DIR, f"{c}_lung.npy"), np.where(masks, img, 0).astype(np.uint8))


if __name__ == "__main__":
    main()
