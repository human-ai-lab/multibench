"""Aggregate results/loco/*.json: per-method x target AUROC (mean +/- sd over seeds), macro-average over the
four LOCO corpora (+ separately the external CIDRZ target), and paired bootstrap of the seed-averaged scores
against a reference method. Targets with per-row sampling weights `w` (VinDr-CXR) are reported with weighted AUROC
(`--targets ... vindr --dir results/loco_vindr`); the others are unchanged."""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from datasets.tb_multi.common import ALL, SOURCES, load_all

T = SOURCES + ["cidrz"]


def zs(s):
    return (s - s.mean()) / (s.std() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="sbal")
    ap.add_argument("--dir", default="results/loco")
    ap.add_argument("--targets", nargs="+", default=T)
    ap.add_argument("--out", default="results/loco_summary.md")
    args = ap.parse_args()
    T_ = args.targets
    _, meta = load_all(None, corpora=ALL)
    ytrue = {t: meta[meta.domain == t].label.values for t in T_}
    wt = {t: (meta[meta.domain == t].w.fillna(1.0).values if "w" in meta else np.ones(len(ytrue[t]))) for t in T_}
    R, S = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        m = re.match(r"(.+)_(qatar|nlm|tbx11k|pakistan|cidrz|india|vindr)_s(\d+)\.json", os.path.basename(f))
        if not m or m.group(2) not in T_:
            continue
        name, t, _ = m.groups()
        sc = np.load(f.replace(".json", "_scores.npy"))
        r = json.load(open(f))
        if np.any(wt[t] != 1):  # stored AUROC is unweighted: recompute it with the sampling weights
            r["auc"] = roc_auc_score(ytrue[t], sc, sample_weight=wt[t])
        R[name][t].append(r)
        S[name][t].append(zs(sc))
    lines = ["| method | " + " | ".join(T_) + " | LOCO mean (4) | seeds |", "|---|" + "---|" * (len(T_) + 2)]
    avg = {}
    for name in sorted(R, key=lambda n: -np.mean([np.mean([r["auc"] for r in R[n][t]]) for t in (SOURCES if set(SOURCES) & set(T_) else T_) if t in R[n]] or [0])):
        cells, per = [], []
        for t in T_:
            if t in R[name]:
                a = [r["auc"] for r in R[name][t]]
                cells.append(f"{np.mean(a):.3f}±{np.std(a):.3f}")
                if t in SOURCES:
                    per.append(np.mean(a))
            else:
                cells.append("–")
        avg[name] = np.mean(per) if len(per) == 4 else float("nan")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {'–' if np.isnan(avg[name]) else f'{avg[name]:.3f}'} | {min(len(R[name][t]) for t in R[name])} |")
    lines += ["", f"Paired bootstrap of seed-averaged AUROC vs `{args.ref}` (diff, 95% CI):", "",
              "| method | " + " | ".join(T_) + " |", "|---|" + "---|" * len(T_)]
    rng = np.random.default_rng(0)
    for name in R:
        if name == args.ref or args.ref not in R:
            continue
        cells = []
        for t in T_:
            if t in S[name] and t in S[args.ref]:
                a, b, y, w = np.mean(S[name][t], 0), np.mean(S[args.ref][t], 0), ytrue[t], wt[t]
                d = []
                for _ in range(500):
                    i = rng.integers(0, len(y), len(y))
                    if y[i].min() != y[i].max():
                        d.append(roc_auc_score(y[i], a[i], sample_weight=w[i]) - roc_auc_score(y[i], b[i], sample_weight=w[i]))
                cells.append(f"{roc_auc_score(y, a, sample_weight=w) - roc_auc_score(y, b, sample_weight=w):+.3f} [{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}]")
            else:
                cells.append("–")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
