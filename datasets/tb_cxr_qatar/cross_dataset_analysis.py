"""Rigorous cross-dataset evaluation for Qatar-trained TB CXR models on CIDRZ.

Earlier scripts in this project (cross_dataset_eval.py, adabn_eval.py) reported only
threshold-0.5 UA/WA/F1 on all 364 pooled CIDRZ images. Two problems with that, caught
before committing to any conclusion drawn from it:

1. CIDRZ is ~84% negative (60/364 positive), so ACC/WA alone is a weak metric: an
   always-predict-negative classifier already scores ACC=0.835, and a threshold that makes
   the model predict positive at roughly CIDRZ's own base rate can clear ACC>0.7 with a
   classifier that has NO real discriminative signal (a random-ranking classifier
   thresholded to ~16.5% positive rate gets ACC~0.72 by construction). ACC/WA alone can
   never be used to claim "significant improvement" - AUROC (threshold-free, rank-based)
   is required alongside it to show real signal exists.
2. adabn_eval.py recalibrated BatchNorm statistics using ALL 364 CIDRZ images, including
   the ones then scored - transductive leakage. This module instead adapts BN stats using
   only CIDRZ's "train"+"valid" images (309, unlabeled) and always scores on CIDRZ's "test"
   split alone (55 images, disjoint from adaptation - never touched until scoring), which
   also keeps this evaluation split comparable to few_shot_finetune.py's held-out set.

Every evaluation here reports, side by side: n, majority-class ("always negative") ACC as
a floor, AUROC (with a percentile bootstrap CI - the only metric here immune to threshold
gaming), and both a default (p>0.5) and a source-prior-calibrated decision threshold's
ACC/UA/F1, plus a McNemar exact test comparing the two thresholds' predictions (paired,
same items, only the decision rule differs) - so a threshold change can be labeled
"significant" or not rather than asserted.

Usage:
    python -m datasets.tb_cxr_qatar.cross_dataset_analysis \
        --model results/models/tb_cxr_qatar_to_cidrz.pt [--adapt-bn]
"""
import argparse
import pickle
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eval_scripts.performance import compute_metrics
from utils.config import load_config
from utils.device import get_device


def _load_split(path: str, splits: Tuple[str, ...]) -> TensorDataset:
    with open(path, "rb") as f:
        data = pickle.load(f)
    images = np.concatenate([data[s]["image"] for s in splits], axis=0)
    labels = np.concatenate([data[s]["label"] for s in splits], axis=0)
    return TensorDataset(
        torch.from_numpy(images.astype(np.float32)), torch.from_numpy(labels.astype(np.int64))
    )


def _qatar_source_prior(pkl_path: str = "data/tb_cxr_qatar/tb_cxr_qatar.pkl") -> float:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    labels = np.concatenate([data[s]["label"] for s in ("train", "valid", "test")], axis=0)
    return float(labels.mean())


def calibrate_threshold_to_prior(probs_pos: np.ndarray, target_prior: float) -> float:
    """Choose t such that P(probs_pos > t) ~= target_prior (no target labels used)."""
    return float(np.quantile(probs_pos, 1.0 - target_prior))


def adapt_batchnorm(model: nn.Module, unlabeled_loader: DataLoader, device: torch.device) -> None:
    """AdaBN (Li et al. 2016): recompute BatchNorm2d running stats from unlabeled target
    images only. No labels, no gradient updates, no weight changes."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    with torch.no_grad():
        for batch in unlabeled_loader:
            model([batch[0].float().to(device)])
    model.eval()


def predict_probs(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, truth = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model([batch[0].float().to(device)])
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            truth.append(batch[1].numpy())
    return np.concatenate(probs), np.concatenate(truth)


def bootstrap_ci(truth: np.ndarray, probs: np.ndarray, metric_fn, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(truth)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        t, p = truth[idx], probs[idx]
        if len(np.unique(t)) < 2:
            continue
        vals.append(metric_fn(t, p))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap_auroc_diff(truth: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray,
                                 n_boot: int = 2000, seed: int = 0):
    """95% CI for AUROC(a) - AUROC(b), resampling the SAME indices for both (paired) -
    the correct test for "does method A beat method B on this eval set", vs. two
    independently-computed CIs which don't answer that question."""
    rng = np.random.default_rng(seed)
    n = len(truth)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        t = truth[idx]
        if len(np.unique(t)) < 2:
            continue
        diffs.append(roc_auc_score(t, probs_a[idx]) - roc_auc_score(t, probs_b[idx]))
    point = roc_auc_score(truth, probs_a) - roc_auc_score(truth, probs_b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def mcnemar_exact(pred_a: np.ndarray, pred_b: np.ndarray, truth: np.ndarray) -> float:
    """Exact two-sided McNemar test p-value on paired correctness (pred_a vs pred_b)."""
    correct_a = pred_a == truth
    correct_b = pred_b == truth
    b = int(np.sum(correct_a & ~correct_b))  # a right, b wrong
    c = int(np.sum(~correct_a & correct_b))  # a wrong, b right
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue


def evaluate(
    model_path: str,
    cidrz_path: str = "data/google_health/tb_dataset.pkl",
    qatar_pkl: str = "data/tb_cxr_qatar/tb_cxr_qatar.pkl",
    qatar_config: str = "configs/tb_cxr_qatar.yaml",
    adapt_bn: bool = False,
    adapt_splits: Tuple[str, ...] = ("train", "valid"),
    eval_splits: Tuple[str, ...] = ("test",),
    device=None,
) -> Dict:
    """`adapt_splits`/`eval_splits` must be disjoint. Defaults match the primary protocol
    (adapt on CIDRZ train+valid, score on CIDRZ test - the split used for model/technique
    selection). Pass `adapt_splits=("valid","test")`, `eval_splits=("train",)` for the
    CONFIRMATION run on the 254-image split never used to pick a technique."""
    config = load_config(qatar_config)
    device = device or get_device()
    model = torch.load(model_path, weights_only=False).to(device)

    if adapt_bn:
        adapt_loader = DataLoader(_load_split(cidrz_path, adapt_splits), batch_size=16, shuffle=False)
        adapt_batchnorm(model, adapt_loader, device)

    eval_ds = _load_split(cidrz_path, eval_splits)
    eval_loader = DataLoader(eval_ds, batch_size=16, shuffle=False)
    probs, truth = predict_probs(model, eval_loader, device)

    n = len(truth)
    majority_acc = max(truth.mean(), 1 - truth.mean())
    auroc = roc_auc_score(truth, probs)
    auroc_ci = bootstrap_ci(truth, probs, lambda t, p: roc_auc_score(t, p))

    default_preds = (probs > 0.5).astype(np.int64)
    default_metrics = compute_metrics(torch.tensor(truth), torch.tensor(default_preds), ["UA", "WA", "F1"])

    prior = _qatar_source_prior(qatar_pkl)
    t_star = calibrate_threshold_to_prior(probs, prior)
    calibrated_preds = (probs > t_star).astype(np.int64)
    calibrated_metrics = compute_metrics(torch.tensor(truth), torch.tensor(calibrated_preds), ["UA", "WA", "F1"])

    p_value = mcnemar_exact(default_preds, calibrated_preds, truth)

    result = {
        "n": n, "positive_rate": float(truth.mean()), "majority_acc": float(majority_acc),
        "auroc": float(auroc), "auroc_95ci": auroc_ci,
        "default": default_metrics, "calibrated": calibrated_metrics,
        "calibration_threshold": t_star, "mcnemar_p": p_value,
        "adapt_bn": adapt_bn,
    }
    return result


def print_result(label: str, r: Dict) -> None:
    print(f"\n=== {label} (CIDRZ test split, n={r['n']}, positive rate={r['positive_rate']:.3f}) ===")
    print(f"  Majority-class ('always negative') ACC floor: {r['majority_acc']:.3f}")
    print(f"  AUROC: {r['auroc']:.3f}  (95% bootstrap CI: [{r['auroc_95ci'][0]:.3f}, {r['auroc_95ci'][1]:.3f}])")
    print(f"  Default threshold (0.5):        {r['default']}")
    print(f"  Prior-calibrated threshold ({r['calibration_threshold']:.3f}): {r['calibrated']}")
    print(f"  McNemar p (default vs calibrated): {r['mcnemar_p']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="results/models/tb_cxr_qatar_to_cidrz.pt")
    parser.add_argument("--cidrz-path", default="data/google_health/tb_dataset.pkl")
    parser.add_argument("--qatar-pkl", default="data/tb_cxr_qatar/tb_cxr_qatar.pkl")
    parser.add_argument("--qatar-config", default="configs/tb_cxr_qatar.yaml")
    parser.add_argument("--adapt-bn", action="store_true")
    args = parser.parse_args()

    r = evaluate(args.model, args.cidrz_path, args.qatar_pkl, args.qatar_config, args.adapt_bn)
    label = f"{args.model}" + (" + AdaBN" if args.adapt_bn else "")
    print_result(label, r)
    return r


if __name__ == "__main__":
    main()
