"""Late fusion on CIDRZ: a Qatar-trained image expert (cross-dataset, label-free) combined
with in-domain clinical-text and cough-audio experts (trained on CIDRZ labels).

Qatar has no audio or clinical fields, so only the image branch can transfer; audio and text
must be learned from CIDRZ itself. Decision-level fusion makes that possible without any
paired multimodal source data - but the result is "cross-dataset image + in-domain text/
audio", not zero-shot, and the question it answers is whether the transferred image branch
adds anything on top of what CIDRZ's own non-image modalities already give.

Protocol (all participants pooled, n=364, 60 positive):
- Image expert scores are computed once and label-free: folds are split on indices only, BN
  statistics are adapted on the other folds' unlabeled images (transductive AdaBN), and each
  held-out fold is scored with 4-view TTA. The score is the pre-softmax logit margin - this
  expert's softmax saturates near 1 on CIDRZ, so probabilities are numerically useless.
- Text and audio experts are separate class-balanced logistic regressions (C fixed at 0.1).
- Nested stacking inside repeated stratified 5-fold CV: inner out-of-fold expert scores
  train a class-balanced logistic-regression stacker per outer fold; the decision threshold
  is fixed at 0.5 (never tuned). An equal-weight z-score average is reported as a
  no-learned-weights check.
- Every arm uses identical folds/seeds, so differences are paired; significance comes from
  a paired participant bootstrap and a Nadeau-Bengio corrected repeated-CV t-test.

Usage:
    python -m datasets.tb_cxr_qatar.multimodal_late_fusion \
        --image-model results/models/tb_cxr_qatar_partial_unfreeze.pt \
        --image-cache results/cidrz_image_scores_partial_unfreeze.npy
"""
import argparse
import os
import pickle

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from datasets.tb_cxr_qatar.tta_eval import _views
from utils.device import get_device

ARMS = {
    "image": ("image",),
    "audio": ("audio",),
    "text": ("text",),
    "text+audio": ("text", "audio"),
    "text+image": ("text", "image"),
    "text+audio+image": ("text", "audio", "image"),
}
AVG_ARMS = {
    "text+image (avg)": ("text", "image"),
    "text+audio+image (avg)": ("text", "audio", "image"),
}
REFERENCE_ARM = "text"


def load_cidrz(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    splits = ("train", "valid", "test")
    return {k: np.concatenate([data[s][k] for s in splits]) for k in ("text", "audio", "image", "label")}


def adapt_bn_only(model: nn.Module, images: np.ndarray, device, batch_size: int = 16) -> None:
    """AdaBN with only the BatchNorm layers in train mode. VGG11Slim inserts Dropout after
    every ReLU, so putting the whole model in train() mode would recompute BN statistics on
    dropout-perturbed activations that never occur at inference."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
            m.train()
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            model([torch.from_numpy(images[i:i + batch_size]).float().to(device)])
    model.eval()


def tta_margin(model: nn.Module, images: np.ndarray, device, batch_size: int = 16) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            x = torch.from_numpy(images[i:i + batch_size]).float().to(device)
            margins = []
            for v in _views(x):
                logits = model([v])
                margins.append(logits[:, 1] - logits[:, 0])
            out.append(torch.stack(margins).mean(0).cpu().numpy())
    return np.concatenate(out)


def image_expert_scores(model_path: str, images: np.ndarray, device, n_folds: int = 5, seed: int = 0):
    scores = np.zeros(len(images), dtype=np.float32)
    folds = KFold(n_folds, shuffle=True, random_state=seed).split(images)
    for k, (adapt_idx, score_idx) in enumerate(folds):
        model = torch.load(model_path, weights_only=False).to(device)
        adapt_bn_only(model, images[adapt_idx], device)
        scores[score_idx] = tta_margin(model, images[score_idx], device)
        print(f"  image expert fold {k + 1}/{n_folds} scored", flush=True)
    return scores


def _expert():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))


def _stacker():
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)


def _zscore(train_vals: np.ndarray, test_vals: np.ndarray):
    mu, sd = train_vals.mean(), train_vals.std() + 1e-8
    return (train_vals - mu) / sd, (test_vals - mu) / sd


def run_cv(feats: dict, y: np.ndarray, n_repeats: int = 10, n_folds: int = 5, inner_folds: int = 5):
    n = len(y)
    all_arms = list(ARMS) + list(AVG_ARMS)
    probs = {arm: np.zeros((n_repeats, n)) for arm in all_arms}
    fold_uar = {arm: [] for arm in all_arms}
    fold_sizes = []
    for r in range(n_repeats):
        outer = StratifiedKFold(n_folds, shuffle=True, random_state=r)
        for tr, te in outer.split(feats["text"], y):
            fold_sizes.append((len(tr), len(te)))
            inner = StratifiedKFold(inner_folds, shuffle=True, random_state=1000 + r)
            tr_s, te_s = {}, {}
            for m in ("text", "audio"):
                raw_tr = cross_val_predict(_expert(), feats[m][tr], y[tr], cv=inner, method="decision_function")
                raw_te = _expert().fit(feats[m][tr], y[tr]).decision_function(feats[m][te])
                tr_s[m], te_s[m] = _zscore(raw_tr, raw_te)
            tr_s["image"], te_s["image"] = _zscore(feats["image"][tr], feats["image"][te])

            inputs = {arm: ([tr_s[m] for m in mods], [te_s[m] for m in mods]) for arm, mods in ARMS.items()}
            for arm, mods in AVG_ARMS.items():
                inputs[arm] = ([np.mean([tr_s[m] for m in mods], axis=0)], [np.mean([te_s[m] for m in mods], axis=0)])

            for arm, (xtr, xte) in inputs.items():
                p = _stacker().fit(np.column_stack(xtr), y[tr]).predict_proba(np.column_stack(xte))[:, 1]
                probs[arm][r, te] = p
                fold_uar[arm].append(balanced_accuracy_score(y[te], p > 0.5))
    return probs, fold_uar, fold_sizes


def _bootstrap(y, p_a, p_b=None, n_boot=2000, seed=0):
    """95% CIs over participants for UAR/AUROC of p_a, or of (p_a - p_b) when paired."""
    rng = np.random.default_rng(seed)
    uar, auc = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        t = y[idx]
        if len(np.unique(t)) < 2:
            continue
        u = balanced_accuracy_score(t, p_a[idx] > 0.5)
        a = roc_auc_score(t, p_a[idx])
        if p_b is not None:
            u -= balanced_accuracy_score(t, p_b[idx] > 0.5)
            a -= roc_auc_score(t, p_b[idx])
        uar.append(u)
        auc.append(a)
    return np.percentile(uar, [2.5, 97.5]), np.percentile(auc, [2.5, 97.5])


def _nadeau_bengio(diffs, fold_sizes):
    """Corrected repeated-CV t-test (Nadeau & Bengio 2003) on per-fold paired differences."""
    diffs = np.asarray(diffs)
    n_tr, n_te = np.mean(fold_sizes, axis=0)
    var = diffs.var(ddof=1)
    if var == 0:
        return 1.0
    t = diffs.mean() / np.sqrt((1 / len(diffs) + n_te / n_tr) * var)
    return float(2 * stats.t.sf(abs(t), df=len(diffs) - 1))


def report(probs, fold_uar, fold_sizes, y):
    ref_avg = probs[REFERENCE_ARM].mean(0)
    print(f"\n{'arm':24s} {'UAR mean+/-sd':>16s} {'AUROC mean+/-sd':>16s}  UAR 95% CI   "
          f"| vs text-only: dUAR [95% CI]  dAUROC [95% CI]  NB p(UAR)")
    for arm, p in probs.items():
        uars = [balanced_accuracy_score(y, pr > 0.5) for pr in p]
        aucs = [roc_auc_score(y, pr) for pr in p]
        p_avg = p.mean(0)
        (u_lo, u_hi), _ = _bootstrap(y, p_avg)
        line = (f"{arm:24s} {np.mean(uars):.3f}+/-{np.std(uars):.3f}  {np.mean(aucs):.3f}+/-{np.std(aucs):.3f}  "
                f"[{u_lo:.3f},{u_hi:.3f}]")
        if arm != REFERENCE_ARM:
            (du_lo, du_hi), (da_lo, da_hi) = _bootstrap(y, p_avg, ref_avg)
            du = balanced_accuracy_score(y, p_avg > 0.5) - balanced_accuracy_score(y, ref_avg > 0.5)
            da = roc_auc_score(y, p_avg) - roc_auc_score(y, ref_avg)
            nb = _nadeau_bengio(np.array(fold_uar[arm]) - np.array(fold_uar[REFERENCE_ARM]), fold_sizes)
            line += f" | {du:+.3f} [{du_lo:+.3f},{du_hi:+.3f}]  {da:+.3f} [{da_lo:+.3f},{da_hi:+.3f}]  p={nb:.3f}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cidrz-path", default="data/google_health/tb_dataset.pkl")
    parser.add_argument("--image-model", default="results/models/tb_cxr_qatar_partial_unfreeze.pt")
    parser.add_argument("--image-cache", default=None,
                        help="Where to cache the out-of-fold image-expert scores (.npy); reused if present.")
    parser.add_argument("--n-repeats", type=int, default=10)
    args = parser.parse_args()

    data = load_cidrz(args.cidrz_path)
    y = data["label"]
    print(f"CIDRZ pooled: n={len(y)}, positives={int(y.sum())}")

    if args.image_cache and os.path.exists(args.image_cache):
        image_scores = np.load(args.image_cache)
        print(f"Loaded cached image-expert scores from {args.image_cache}")
    else:
        print(f"Scoring image expert {args.image_model} (label-free, out-of-fold AdaBN + TTA)...")
        image_scores = image_expert_scores(args.image_model, data["image"], get_device())
        if args.image_cache:
            os.makedirs(os.path.dirname(args.image_cache) or ".", exist_ok=True)
            np.save(args.image_cache, image_scores)
    print(f"Image expert alone (label-free margin) AUROC on all 364: {roc_auc_score(y, image_scores):.3f}")

    feats = {"text": data["text"], "audio": data["audio"], "image": image_scores}
    probs, fold_uar, fold_sizes = run_cv(feats, y, n_repeats=args.n_repeats)
    print(f"\nImage expert: {args.image_model}")
    report(probs, fold_uar, fold_sizes, y)


if __name__ == "__main__":
    main()
