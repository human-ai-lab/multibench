"""AdaBN (Li, Wang, Shi, Liu, Hou, Tian, 2016, "Revisiting Batch Normalization for
Practical Domain Adaptation", arXiv:1603.04779) applied to the Qatar-trained TB CXR model.

The Qatar-trained model's frozen VGG11-BN backbone has 8 BatchNorm2d layers, each holding
running mean/variance statistics computed from Qatar's own pixel intensity distribution.
Evaluated zero-shot on CIDRZ (a different population/scanner/imaging pipeline), those
statistics are mismatched with CIDRZ's actual distribution - which is a plausible
contributor to the observed collapse (UA 0.973 in-distribution -> 0.527 cross-dataset).

AdaBN recalibrates ONLY those running statistics using unlabeled target-domain (CIDRZ)
images - no gradient updates, no labels, no change to any learned weight. This is the
cheapest, most directly-mechanism-matched domain-adaptation technique available here: it
requires nothing but a forward pass over the target images already on disk.

Usage:
    python -m datasets.tb_cxr_qatar.adabn_eval \
        --model results/models/tb_cxr_qatar_to_cidrz.pt \
        --cidrz-path data/google_health/tb_dataset.pkl \
        --qatar-config configs/tb_cxr_qatar.yaml
"""
import argparse
import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eval_scripts.performance import compute_metrics
from utils.config import load_config
from utils.device import get_device


def _pool_images_labels(path: str) -> TensorDataset:
    with open(path, "rb") as f:
        data = pickle.load(f)
    images = np.concatenate([data[split]["image"] for split in ("train", "valid", "test")], axis=0)
    labels = np.concatenate([data[split]["label"] for split in ("train", "valid", "test")], axis=0)
    return TensorDataset(
        torch.from_numpy(images.astype(np.float32)), torch.from_numpy(labels.astype(np.int64))
    )


def adapt_batchnorm(model: nn.Module, unlabeled_loader: DataLoader, device: torch.device) -> None:
    """Recompute every BatchNorm2d's running mean/var from `unlabeled_loader`'s images only
    (reset first, then a cumulative - not exponential-moving-average - mean over all
    batches, so the result reflects the target domain alone, not a blend with the source
    statistics the model was trained with). No labels are used; no weights change.
    """
    n_bn = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # None => cumulative moving average instead of EMA
            n_bn += 1
    print(f"Recalibrating {n_bn} BatchNorm2d layers on the target domain...")

    model.train()  # BN layers must be in train mode to update running stats on forward
    with torch.no_grad():
        for batch in unlabeled_loader:
            inputs = [batch[0].float().to(device)]
            model(inputs)
    model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="results/models/tb_cxr_qatar_to_cidrz.pt")
    parser.add_argument("--cidrz-path", default="data/google_health/tb_dataset.pkl")
    parser.add_argument("--qatar-config", default="configs/tb_cxr_qatar.yaml")
    args = parser.parse_args()

    config = load_config(args.qatar_config)
    device = get_device()
    model = torch.load(args.model, weights_only=False).to(device)

    cidrz_dataset = _pool_images_labels(args.cidrz_path)
    cidrz_loader = DataLoader(cidrz_dataset, batch_size=16, shuffle=False)

    adapt_batchnorm(model, cidrz_loader, device)

    truths, preds = [], []
    with torch.no_grad():
        for batch in cidrz_loader:
            inputs = [batch[0].float().to(device)]
            labels = batch[1]
            outs = model(inputs)
            preds.append(torch.argmax(outs, dim=1).cpu())
            truths.append(labels)
    truths = torch.cat(truths)
    preds = torch.cat(preds)
    results = compute_metrics(truths, preds, config.get("evaluation", ["UA", "WA", "F1"]))
    print("AdaBN-adapted evaluation results (Qatar-trained, BN recalibrated on CIDRZ, tested on CIDRZ):")
    for name, value in results.items():
        print(f"  {name}: {value}")
    return results


if __name__ == "__main__":
    main()
