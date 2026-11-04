"""Bootstrap uncertainty of the oracle WHO-TPP operating points on a labelled target (default CIDRZ, 60 positives).

The ceiling is sens at 70% specificity and spec at 90% sensitivity on the seed-averaged scores (same definition as
local_recalibration.py). Cases are resampled with replacement; we report the 95% interval and the share of resamples whose
ceiling reaches the profile (sens >= 0.90, spec >= 0.70), i.e. how much of 'the model misses the target' is sampling noise.
Usage: python -m datasets.tb_multi.ceiling_bootstrap --methods f_raddino f_vit
"""
import argparse

import numpy as np

from datasets.tb_multi.artefact_adjust import load_scores
from datasets.tb_multi.common import ALL, load_all


def ceilings(m, y):
    pos, neg = m[y == 1], m[y == 0]
    return (pos > np.quantile(neg, 0.70)).mean(), (neg < np.quantile(pos, 0.10)).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="+", default=["f_raddino", "f_ms_raddino", "f_vit"])
    ap.add_argument("--target", default="cidrz")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--out", default="results/ceiling_bootstrap.md")
    args = ap.parse_args()
    _, meta = load_all(None, corpora=ALL)
    y = meta[meta.domain == args.target].label.values
    rng = np.random.default_rng(0)
    L = ["| method | target | n (pos) | sens@spec70 [95% CI] | spec@sens90 [95% CI] | P(sens>=.90) | P(spec>=.70) | P(both) |", "|---|---|---|---|---|---|---|---|"]
    for mth in args.methods:
        m, _ = load_scores(mth, args.target, args.seeds)
        if m is None:
            continue
        p = ceilings(m, y)
        B = []
        while len(B) < args.reps:
            i = rng.integers(0, len(y), len(y))
            if y[i].sum() >= 3 and (1 - y[i]).sum() >= 3:
                B.append(ceilings(m[i], y[i]))
        B = np.array(B)
        lo, hi = np.percentile(B, 2.5, 0), np.percentile(B, 97.5, 0)
        L.append(f"| {mth} | {args.target} | {len(y)} ({int(y.sum())}) | {p[0]:.2f} [{lo[0]:.2f}, {hi[0]:.2f}] | {p[1]:.2f} [{lo[1]:.2f}, {hi[1]:.2f}] | "
                 f"{(B[:, 0] >= .9).mean():.3f} | {(B[:, 1] >= .7).mean():.3f} | {((B[:, 0] >= .9) & (B[:, 1] >= .7)).mean():.3f} |")
        print(L[-1], flush=True)
    open(args.out, "w").write("## Bootstrap of the oracle WHO-TPP ceiling\n\n" + "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
