"""Do the fused (text / audio + cross-corpus image expert) scores reach the WHO triage profile on CIDRZ?

Reuses the out-of-fold scores of `fusion_study.run` (repeated stratified 5-fold CV, same folds and experts) and reads the oracle
operating points off them: sens at 70% specificity and spec at 90% sensitivity (same definition as ceiling_bootstrap.py), averaged
over the CV repeats. Participants are resampled with replacement for the 95% interval and for the share of resamples whose
ceiling reaches the profile (sens >= 0.90, spec >= 0.70). The text/audio experts are fit inside CV on CIDRZ (label use on train
folds only); the image expert is never fit on CIDRZ.
Usage: python -m datasets.tb_multi.fusion_who --image-scores raddino5=results/cidrz_img_final_raddino5.npy
"""
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

from datasets.tb_multi.ceiling_bootstrap import ceilings
from datasets.tb_multi.fusion_study import load_cidrz, run

ARMS = ["image", "text", "stack|text+image", "relw|text+image", "stack|text+audio+image", "relw|text+audio+image"]


def rep_ceilings(P, y):
    return np.mean([ceilings(p, y) for p in P], 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-scores", nargs="+", required=True, help="name=path.npy (seed-averaged z-scored margins)")
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default="results/fusion_who.md")
    args = ap.parse_args()
    data = load_cidrz("data/google_health/tb_dataset.pkl")
    y = data["label"]
    L = ["| image expert | arm | AUROC | sens@spec70 [95% CI] | spec@sens90 [95% CI] | P(sens>=.90) | P(spec>=.70) | P(both) |",
         "|---|---|---|---|---|---|---|---|"]
    rng = np.random.default_rng(0)
    for spec in args.image_scores:
        name, path = spec.split("=")
        arms = run(data, y, np.load(path).astype(np.float64), args.n_repeats)
        for arm in ARMS:
            P = arms[arm]
            pt, auc = rep_ceilings(P, y), np.mean([roc_auc_score(y, p) for p in P])
            B = []
            while len(B) < args.reps:
                i = rng.integers(0, len(y), len(y))
                if y[i].sum() >= 3 and (1 - y[i]).sum() >= 3:
                    B.append(rep_ceilings(P[:, i], y[i]))
            B = np.array(B)
            lo, hi = np.percentile(B, 2.5, 0), np.percentile(B, 97.5, 0)
            L.append(f"| {name} | {arm} | {auc:.3f} | {pt[0]:.2f} [{lo[0]:.2f}, {hi[0]:.2f}] | {pt[1]:.2f} [{lo[1]:.2f}, {hi[1]:.2f}] | "
                     f"{(B[:, 0] >= .9).mean():.3f} | {(B[:, 1] >= .7).mean():.3f} | {((B[:, 0] >= .9) & (B[:, 1] >= .7)).mean():.3f} |")
            print(L[-1], flush=True)
    open(args.out, "w").write("## Fused models against the WHO triage profile (CIDRZ, oracle operating points)\n\n" + "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
