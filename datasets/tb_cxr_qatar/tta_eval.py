"""Test-time augmentation (TTA) via prediction averaging, applied to an already-trained
Qatar model evaluated on CIDRZ - no retraining, no weight/BN-statistic changes.

Views: identity, horizontal flip, and a 90%-scale center crop (resized back to 224) with
and without the flip - 4 views total, averaged in probability space. All 4 transforms
(flip, crop+resize) commute with the existing per-channel ImageNet normalization already
baked into the stored tensors (both are spatial/elementwise operations independent of the
per-channel affine normalization), so they can be applied directly to the pre-normalized
tensors already in the pickle without needing the raw pixel values back.

This is the "safe" TTA variant Zhou/Zhang et al. and the TTA-in-medical-imaging literature
recommend over entropy-based or learned/adaptive TTA-weighting for small eval sets: fixed,
data-independent transforms, simple averaging, no risk of a learned aggregator overfitting
to this project's 55-image CIDRZ test split.

Usage:
    python -m datasets.tb_cxr_qatar.tta_eval --model results/models/tb_cxr_qatar_to_cidrz.pt [--adapt-bn]
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from datasets.tb_cxr_qatar.cross_dataset_analysis import (
    _load_split, _qatar_source_prior, adapt_batchnorm, bootstrap_ci, calibrate_threshold_to_prior,
    mcnemar_exact, print_result,
)
from eval_scripts.performance import compute_metrics
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from utils.device import get_device


def _views(x: torch.Tensor):
    """x: (B, 3, 224, 224), already normalized. Yields 4 augmented versions."""
    yield x
    yield torch.flip(x, dims=[3])
    crop_size = int(224 * 0.9)
    offset = (224 - crop_size) // 2
    cropped = x[:, :, offset:offset + crop_size, offset:offset + crop_size]
    resized = F.interpolate(cropped, size=(224, 224), mode="bilinear", align_corners=False)
    yield resized
    yield torch.flip(resized, dims=[3])


def predict_probs_tta(model, loader, device):
    model.eval()
    probs, truth = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].float().to(device)
            view_probs = [torch.softmax(model([v]), dim=1)[:, 1] for v in _views(x)]
            avg_probs = torch.stack(view_probs, dim=0).mean(dim=0)
            probs.append(avg_probs.cpu().numpy())
            truth.append(batch[1].numpy())
    return np.concatenate(probs), np.concatenate(truth)


def evaluate_tta(model_path, cidrz_path="data/google_health/tb_dataset.pkl",
                  qatar_pkl="data/tb_cxr_qatar/tb_cxr_qatar.pkl", adapt_bn=False,
                  adapt_splits=("train", "valid"), eval_splits=("test",), device=None):
    device = device or get_device()
    model = torch.load(model_path, weights_only=False).to(device)

    if adapt_bn:
        adapt_loader = DataLoader(_load_split(cidrz_path, adapt_splits), batch_size=16, shuffle=False)
        adapt_batchnorm(model, adapt_loader, device)

    eval_loader = DataLoader(_load_split(cidrz_path, eval_splits), batch_size=16, shuffle=False)
    probs, truth = predict_probs_tta(model, eval_loader, device)

    n = len(truth)
    default_preds = (probs > 0.5).astype(np.int64)
    default_metrics = compute_metrics(torch.tensor(truth), torch.tensor(default_preds), ["UA", "WA", "F1"])
    prior = _qatar_source_prior(qatar_pkl)
    t_star = calibrate_threshold_to_prior(probs, prior)
    calibrated_preds = (probs > t_star).astype(np.int64)
    calibrated_metrics = compute_metrics(torch.tensor(truth), torch.tensor(calibrated_preds), ["UA", "WA", "F1"])

    return {
        "n": n, "positive_rate": float(truth.mean()), "majority_acc": float(max(truth.mean(), 1 - truth.mean())),
        "auroc": float(roc_auc_score(truth, probs)),
        "auroc_95ci": bootstrap_ci(truth, probs, lambda t, p: roc_auc_score(t, p)),
        "default": default_metrics, "calibrated": calibrated_metrics,
        "calibration_threshold": t_star,
        "mcnemar_p": mcnemar_exact(default_preds, calibrated_preds, truth),
        "adapt_bn": adapt_bn,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="results/models/tb_cxr_qatar_to_cidrz.pt")
    parser.add_argument("--cidrz-path", default="data/google_health/tb_dataset.pkl")
    parser.add_argument("--qatar-pkl", default="data/tb_cxr_qatar/tb_cxr_qatar.pkl")
    parser.add_argument("--adapt-bn", action="store_true")
    args = parser.parse_args()

    r = evaluate_tta(args.model, args.cidrz_path, args.qatar_pkl, args.adapt_bn)
    label = f"{args.model} + TTA" + (" + AdaBN" if args.adapt_bn else "")
    print_result(label, r)


if __name__ == "__main__":
    main()
