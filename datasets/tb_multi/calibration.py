"""Calibration / threshold-transfer analysis of the LOCO image experts on the held-out corpus.

For each (method, target), per seed then averaged, from the saved raw logit margins m (p = sigmoid(m)):
  ECE / Brier        raw probabilities (training used source-balanced cells, i.e. an implicit 50% prior)
  ECE / Brier (pi)   after shifting the logit by log(pi/(1-pi)) with the TARGET's true prevalence (oracle diagnostic:
                     isolates prior shift from everything else; not deployable)
  slope / intercept  logistic regression of y on m on the target (oracle diagnostic; slope<1 = over-confident,
                     intercept = remaining offset)
  transfer sens/spec target sensitivity/specificity at the threshold giving 90% sensitivity on held-in validation
                     data (label use only on source corpora) -- the deployable operating point
ECE uses 10 equal-mass bins.
Targets with per-row sampling weights `w` (VinDr-CXR, an enriched sample; `--targets ... vindr --dir results/loco_vindr`) get
weighted metrics (weight-mass bins, weighted logistic fit, weighted transfer sens/spec at the stored held-in threshold) and the
weighted prevalence; the others are unchanged.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

from datasets.tb_multi.common import ALL, SOURCES, load_all

T = SOURCES + ["cidrz"]


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def ece(y, p, bins=10, w=None):
    order = np.argsort(p)
    if w is None:
        parts = np.array_split(order, bins)
        return float(sum(len(i) / len(y) * abs(y[i].mean() - p[i].mean()) for i in parts))
    cuts = np.searchsorted(np.cumsum(w[order]) / w.sum(), np.linspace(0, 1, bins + 1)[1:-1])
    return float(sum(w[i].sum() / w.sum() * abs(np.average(y[i], weights=w[i]) - np.average(p[i], weights=w[i]))
                     for i in np.split(order, cuts) if len(i) and w[i].sum() > 0))


def metrics(y, m, w=None):
    w = np.ones(len(y)) if w is None else w
    wt = None if np.all(w == 1) else w
    pi = np.average(y, weights=w)
    p = sigmoid(m)
    pc = sigmoid(m + np.log(pi / (1 - pi)))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(m[:, None], y, sample_weight=wt)
    return dict(ece=ece(y, p, w=wt), brier=float(np.average((p - y) ** 2, weights=w)), ece_pi=ece(y, pc, w=wt),
                brier_pi=float(np.average((pc - y) ** 2, weights=w)), slope=float(lr.coef_[0, 0]), intercept=float(lr.intercept_[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="+", default=["erm", "sbal", "sbal_aug_lung", "sbal_aug_lung_dstd", "bb_raddino",
                                                      "ms_raddino", "lesion_raddino"])
    ap.add_argument("--dir", default="results/loco")
    ap.add_argument("--targets", nargs="+", default=T)
    ap.add_argument("--out", default="results/calibration.md")
    args = ap.parse_args()
    _, meta = load_all(None, corpora=ALL)
    y = {t: meta[meta.domain == t].label.values for t in args.targets}
    wts = {t: (meta[meta.domain == t].w.fillna(1.0).values if "w" in meta else np.ones(len(y[t]))) for t in args.targets}
    rows = ["| method | target | prevalence | ECE | Brier | ECE (π-shifted) | Brier (π-shifted) | slope | intercept | transfer sens | transfer spec |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in args.methods:
        for t in args.targets:
            fs = sorted(glob.glob(f"{args.dir}/{name}_{t}_s*.json"))
            if not fs:
                continue
            w, weighted = wts[t], bool(np.any(wts[t] != 1))
            sc = [np.load(f.replace(".json", "_scores.npy")).astype(np.float64) for f in fs]
            ms = [metrics(y[t], m, w) for m in sc]
            js = [json.load(open(f)) for f in fs]
            g = lambda k: np.mean([m[k] for m in ms])
            if weighted:  # stored transfer values are unweighted: redo them with the weights at the stored held-in threshold
                ts = [np.average(m[y[t] == 1] >= j["thr"], weights=w[y[t] == 1]) for m, j in zip(sc, js)]
                tp = [np.average(m[y[t] == 0] < j["thr"], weights=w[y[t] == 0]) for m, j in zip(sc, js)]
            else:
                ts, tp = [j["transfer_sens"] for j in js], [j["transfer_spec"] for j in js]
            rows.append(f"| {name} | {t} | {np.average(y[t], weights=w):.2f} | {g('ece'):.3f} | {g('brier'):.3f} | {g('ece_pi'):.3f} | "
                        f"{g('brier_pi'):.3f} | {g('slope'):.2f} | {g('intercept'):+.2f} | {np.mean(ts):.2f} | {np.mean(tp):.2f} |")
    open(args.out, "w").write("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
