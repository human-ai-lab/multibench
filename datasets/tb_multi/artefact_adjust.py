"""Artefact-adjusted AUROC: how much of a held-out score is pathology rather than acquisition artefact?

The artefact predictor sees only lung-mask area, the 28x28 lung silhouette and the outer-border pixels (no lung
texture). Two variants bound the artefact signal in a target corpus:
  src : logistic regression trained on the *other* corpora and applied to the target (what a source-trained model
        could exploit); the usual shortcut_audit control.
  cv  : 5-fold cross-validated logistic regression fitted *inside* the target (target-specific artefacts; the
        conservative bound).
Reported per model score s and target:
  raw        AUROC(s)
  art_src / art_cv   AUROC of the artefact predictor alone
  strat_*    artefact-stratified AUROC: target images are cut into K quantile bins of the artefact score and the AUROC
             is taken over positive/negative pairs inside the same bin (pair-weighted). If s only ranked cases by
             silhouette it falls to 0.5; the share of pairs inside a bin is reported as `pairs_kept`.
  add_cv     AUROC(logistic[s, artefact]) - AUROC(artefact), 5-fold CV inside the target: what the model adds beyond
             the artefact features.
Caveat: the `cv` artefact predictor can also learn real pathology (the border ring contains peripheral lung and the
silhouette is a segmentation of diseased lungs), so it is an upper bound on artefact signal and the `cv`-stratified
AUROC a lower bound on pathology signal; where the artefact AUROC is ~1 (pakistan) the stratified score is not identifiable.
Usage: python -m datasets.tb_multi.artefact_adjust --methods bb_raddino ms_vit [--seeds 0 1] [--out results/artefact_adjusted.md]
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from datasets.tb_multi.common import ALL, EXTERNAL, SOURCES, load_all
from datasets.tb_multi.standardize import OUT_DIR

K_BINS = 10


def _small(a):
    return np.asarray(Image.fromarray(a.astype(np.uint8)).resize((28, 28), Image.BILINEAR), dtype=np.float32) / 255


def artefact_features(meta):
    mask = np.concatenate([np.load(f"{OUT_DIR}/{c}_mask.npy", mmap_mode="r")[meta[meta.corpus == c].row.values] for c in ALL])
    gray = np.concatenate([np.load(f"{OUT_DIR}/{c}.npy", mmap_mode="r")[meta[meta.corpus == c].row.values] for c in ALL])
    ring = np.ones((224, 224), bool)
    ring[40:184, 40:184] = False
    area = mask.mean((1, 2))[:, None]
    sil = np.stack([_small(m * 255) for m in mask]).reshape(len(mask), -1)
    border = np.stack([_small(np.where(ring, g, 0)) for g in gray]).reshape(len(gray), -1)
    return np.hstack([area, sil, border])


def _fit(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return lambda X: m.decision_function(sc.transform(X))


def cv_scores(X, y, seed=0, folds=5):
    out = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        out[te] = _fit(X[tr], y[tr])(X[te])
    return out


def strat_auc(y, s, art, k=K_BINS):
    """Pair-weighted AUROC of s inside quantile bins of `art`; returns (auc, share of pos-neg pairs kept)."""
    edges = np.quantile(art, np.linspace(0, 1, k + 1)[1:-1])
    b = np.searchsorted(edges, art)
    num = den = 0.0
    for j in range(k):
        i = b == j
        npos, nneg = int(y[i].sum()), int((1 - y[i]).sum())
        if npos and nneg:
            num += roc_auc_score(y[i], s[i]) * npos * nneg
            den += npos * nneg
    total = float(y.sum()) * float((1 - y).sum())
    return (num / den if den else float("nan")), den / total


def boot_strat(y, s, art, n=200, seed=0):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(strat_auc(y[i], s[i], art[i])[0])
    return tuple(np.nanpercentile(v, [2.5, 97.5]))


def load_scores(method, target, seeds, dirs=("results/loco", "results/loco_final", "results/loco_ext")):
    out = []
    for sd in seeds:
        for d in dirs:
            for stem in (f"{method}_{target}_s{sd}", f"{method}_ext-{target}_s{sd}"):
                f = os.path.join(d, stem + "_scores.npy")
                if os.path.exists(f):
                    out.append(np.load(f)); break
            else:
                continue
            break
    return np.mean(out, 0) if out else None, len(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--targets", nargs="+", default=SOURCES + EXTERNAL)
    ap.add_argument("--out", default="results/artefact_adjusted.md")
    args = ap.parse_args()

    _, meta = load_all(None, corpora=ALL)
    F = artefact_features(meta)
    dom, y_all = meta.domain.values, meta.label.values
    lines = ["| method | target | n seeds | raw AUROC | artefact-only (src) | artefact-only (cv) | stratified (src) [pairs kept] | stratified (cv) "
             "[95% CI] [pairs kept] | model adds over artefact (cv) |", "|---|---|---|---|---|---|---|---|---|---|"]
    cache = {}
    for t in args.targets:
        te = dom == t
        y, Xt = y_all[te], F[te]
        tr = np.isin(dom, [s for s in SOURCES if s != t])
        art_src = _fit(F[tr], y_all[tr])(Xt)
        art_cv = np.mean([cv_scores(Xt, y, seed=r) for r in range(3)], 0)
        cache[t] = (y, art_src, art_cv)
        print(f"{t}: artefact-only src {roc_auc_score(y, art_src):.3f} cv {roc_auc_score(y, art_cv):.3f}", flush=True)
    for m in args.methods:
        for t in args.targets:
            s, ns = load_scores(m, t, args.seeds)
            if s is None:
                continue
            y, art_src, art_cv = cache[t]
            raw = roc_auc_score(y, s)
            ss, kept_src = strat_auc(y, s, art_src)
            sc, kept = strat_auc(y, s, art_cv)
            lo, hi = boot_strat(y, s, art_cv)
            both = np.column_stack([s, art_cv])
            comb = cv_scores(both, y, seed=1)
            add = roc_auc_score(y, comb) - roc_auc_score(y, art_cv)
            lines.append(f"| {m} | {t} | {ns} | {raw:.3f} | {roc_auc_score(y, art_src):.3f} | {roc_auc_score(y, art_cv):.3f} | "
                         f"{ss:.3f} [{kept_src:.2f}] | {sc:.3f} [{lo:.3f},{hi:.3f}] [{kept:.2f}] | {add:+.3f} |")
            print(lines[-1], flush=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
