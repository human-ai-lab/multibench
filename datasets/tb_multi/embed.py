"""Frozen chest-X-ray embeddings for every standardized corpus (torchxrayvision DenseNet121 'all').

Writes data/tb_multi/<corpus>_xrv.npy (N x 1024 float32, global-pooled features). Used for the
frozen-probe LOCO baselines, the corpus-identity probe, and embedding-based near-duplicate checks.
"""
import argparse
import os

import numpy as np
import torch
import torchxrayvision as xrv

from datasets.tb_multi.standardize import OUT_DIR, load_corpus
from utils.device import get_device


def embed(img_u8: np.ndarray, device, batch_size=64, weights="densenet121-res224-all"):
    model = xrv.models.DenseNet(weights=weights).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(img_u8), batch_size):
            x = torch.from_numpy(img_u8[i:i + batch_size]).float().to(device)
            x = ((2 * (x / 255.0) - 1.0) * 1024)[:, None]
            f = model.features(x)
            f = torch.relu(f)
            out.append(torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+", default=["qatar", "montgomery", "shenzhen", "tbx11k", "pakistan", "cidrz"])
    args = ap.parse_args()
    device = get_device()
    for c in args.corpora:
        img, _ = load_corpus(c)
        np.save(os.path.join(OUT_DIR, f"{c}_xrv.npy"), embed(img, device))
        print("embedded", c, flush=True)


if __name__ == "__main__":
    main()
