"""Confirmation-split significance test: does the best technique stack found on CIDRZ's
"test" split (55 images - used to pick "augmented + AdaBN + TTA" out of ~12 configs, so
any conclusion drawn there is subject to selection bias) still beat the zero-shot baseline
on CIDRZ's "train" split (254 images, 42 positive) - never scored during technique
selection?

Both the baseline and the chosen stack are zero-shot (no CIDRZ labels used anywhere - only
unlabeled images for AdaBN), so adapting BN on "valid"+"test" (unlabeled) and scoring on
"train" is a legitimate, disjoint confirmation protocol; this is the one comparison this
project treats as its headline result, everything else is exploratory.

Usage:
    python -m datasets.tb_cxr_qatar.confirmation_eval \
        --baseline results/models/tb_cxr_qatar_to_cidrz.pt \
        --candidate results/models/tb_cxr_qatar_augmented_to_cidrz.pt
"""
import argparse

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from datasets.tb_cxr_qatar.cross_dataset_analysis import (
    _load_split, adapt_batchnorm, paired_bootstrap_auroc_diff,
)
from datasets.tb_cxr_qatar.tta_eval import predict_probs_tta
from utils.device import get_device


def get_tta_probs(model_path, cidrz_path, adapt_splits, eval_splits, adapt_bn, device):
    model = torch.load(model_path, weights_only=False).to(device)
    if adapt_bn:
        adapt_loader = DataLoader(_load_split(cidrz_path, adapt_splits), batch_size=16, shuffle=False)
        adapt_batchnorm(model, adapt_loader, device)
    eval_loader = DataLoader(_load_split(cidrz_path, eval_splits), batch_size=16, shuffle=False)
    return predict_probs_tta(model, eval_loader, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="results/models/tb_cxr_qatar_to_cidrz.pt")
    parser.add_argument("--candidate", default="results/models/tb_cxr_qatar_augmented_to_cidrz.pt")
    parser.add_argument("--cidrz-path", default="data/google_health/tb_dataset.pkl")
    parser.add_argument("--adapt-splits", nargs="+", default=["valid", "test"])
    parser.add_argument("--eval-splits", nargs="+", default=["train"])
    args = parser.parse_args()

    device = get_device()
    print(f"CONFIRMATION protocol: adapt on {args.adapt_splits} (unlabeled), score on {args.eval_splits} "
          f"(never used to select a technique)")

    base_probs, base_truth = get_tta_probs(args.baseline, args.cidrz_path, tuple(args.adapt_splits),
                                            tuple(args.eval_splits), adapt_bn=False, device=device)
    cand_probs, cand_truth = get_tta_probs(args.candidate, args.cidrz_path, tuple(args.adapt_splits),
                                            tuple(args.eval_splits), adapt_bn=True, device=device)
    assert (base_truth == cand_truth).all(), "eval sets must match for a paired comparison"

    print(f"n={len(base_truth)}, positive rate={base_truth.mean():.3f}")
    print("TTA applied to both arms, so the comparison isolates augmentation + AdaBN.")
    print(f"Baseline (+ TTA only) AUROC: {roc_auc_score(base_truth, base_probs):.3f}")
    print(f"Candidate (augmented + AdaBN + TTA) AUROC: {roc_auc_score(cand_truth, cand_probs):.3f}")

    point, lo, hi = paired_bootstrap_auroc_diff(base_truth, cand_probs, base_probs)
    print(f"\nPaired bootstrap AUROC difference (candidate - baseline): {point:.3f}  95% CI: [{lo:.3f}, {hi:.3f}]")
    if lo > 0:
        print("=> CI excludes 0: candidate significantly beats baseline on this held-out confirmation split.")
    else:
        print("=> CI includes 0: NOT significant on this held-out confirmation split.")


if __name__ == "__main__":
    main()
