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
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

from datasets.tb_multi.common import SOURCES, load_all

T = SOURCES + ["cidrz"]


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def ece(y, p, bins=10):
    order = np.argsort(p)
    parts = np.array_split(order, bins)
    return float(sum(len(i) / len(y) * abs(y[i].mean() - p[i].mean()) for i in parts))


def metrics(y, m):
    pi = y.mean()
    p = sigmoid(m)
    pc = sigmoid(m + np.log(pi / (1 - pi)))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(m[:, None], y)
    return dict(ece=ece(y, p), brier=float(np.mean((p - y) ** 2)), ece_pi=ece(y, pc),
                brier_pi=float(np.mean((pc - y) ** 2)), slope=float(lr.coef_[0, 0]), intercept=float(lr.intercept_[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="+", default=["erm", "sbal", "sbal_aug_lung", "sbal_aug_lung_dstd", "bb_raddino",
                                                      "ms_raddino", "lesion_raddino"])
    ap.add_argument("--out", default="results/calibration.md")
    args = ap.parse_args()
    _, meta = load_all(None)
    y = {t: meta[meta.domain == t].label.values for t in T}
    rows = ["| method | target | prevalence | ECE | Brier | ECE (π-shifted) | Brier (π-shifted) | slope | intercept | transfer sens | transfer spec |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in args.methods:
        for t in T:
            fs = sorted(glob.glob(f"results/loco/{name}_{t}_s*.json"))
            if not fs:
                continue
            ms = [metrics(y[t], np.load(f.replace(".json", "_scores.npy")).astype(np.float64)) for f in fs]
            js = [json.load(open(f)) for f in fs]
            g = lambda k: np.mean([m[k] for m in ms])
            rows.append(f"| {name} | {t} | {y[t].mean():.2f} | {g('ece'):.3f} | {g('brier'):.3f} | {g('ece_pi'):.3f} | "
                        f"{g('brier_pi'):.3f} | {g('slope'):.2f} | {g('intercept'):+.2f} | "
                        f"{np.mean([j['transfer_sens'] for j in js]):.2f} | {np.mean([j['transfer_spec'] for j in js]):.2f} |")
    open(args.out, "w").write("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
