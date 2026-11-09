"""RAD-DINO at 448 px vs 224 px on the five LOCO targets, five seeds each, same tuned configuration (`results/run_448.sh` vs
`run_final.sh`, method f_raddino). Per-seed AUROC (mean +/- sd), AUROC of the seed-averaged scores, and a paired bootstrap of the
difference (448 minus 224). VinDr-CXR is in vindr_448.py (weighted). Usage: python -m datasets.tb_multi.res448_compare"""
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from datasets.tb_multi.artefact_adjust import load_scores
from datasets.tb_multi.common import ALL, SOURCES, load_all

SEEDS = [0, 1, 2, 3, 4]


def zs(s):
    return (s - s.mean()) / s.std()


def main():
    _, meta = load_all(None, corpora=ALL)
    L = ["| target | 224 px per-seed | 448 px per-seed | 224 px seed-avg | 448 px seed-avg | 448 minus 224 [95% CI] |", "|---|---|---|---|---|---|"]
    rng = np.random.default_rng(0)
    for t in SOURCES + ["cidrz"]:
        y = meta[meta.domain == t].label.values
        s448 = [np.load(f"results/loco_448/r448_{t}_s{s}_scores.npy") for s in SEEDS]
        s224 = []
        for s in SEEDS:
            m, n = load_scores("f_raddino", t, [s])
            assert n == 1, (t, s)
            s224.append(m)
        p224, p448 = [[roc_auc_score(y, x) for x in S] for S in (s224, s448)]
        a224, a448 = [np.mean([zs(x) for x in S], 0) for S in (s224, s448)]
        d = []
        for _ in range(1000):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() != y[i].max():
                d.append(roc_auc_score(y[i], a448[i]) - roc_auc_score(y[i], a224[i]))
        L.append(f"| {t} | {np.mean(p224):.3f}±{np.std(p224):.3f} | {np.mean(p448):.3f}±{np.std(p448):.3f} | {roc_auc_score(y, a224):.3f} | "
                 f"{roc_auc_score(y, a448):.3f} | {roc_auc_score(y, a448) - roc_auc_score(y, a224):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] |")
    open("results/res448_compare.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
