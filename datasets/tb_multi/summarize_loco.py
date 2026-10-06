"""Aggregate results/loco/*.json: per-method x target AUROC (mean +/- sd over seeds), macro-average over the
four LOCO corpora (+ separately the external CIDRZ target), and paired bootstrap of the seed-averaged scores
against a reference method."""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from datasets.tb_multi.common import SOURCES, load_all

T = SOURCES + ["cidrz"]


def zs(s):
    return (s - s.mean()) / (s.std() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="sbal")
    ap.add_argument("--dir", default="results/loco")
    ap.add_argument("--out", default="results/loco_summary.md")
    args = ap.parse_args()
    _, meta = load_all(None)
    ytrue = {t: meta[meta.domain == t].label.values for t in T}
    R, S = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        m = re.match(r"(.+)_(qatar|nlm|tbx11k|pakistan|cidrz)_s(\d+)\.json", os.path.basename(f))
        if not m:
            continue
        name, t, _ = m.groups()
        R[name][t].append(json.load(open(f)))
        S[name][t].append(zs(np.load(f.replace(".json", "_scores.npy"))))
    lines = ["| method | " + " | ".join(T) + " | LOCO mean (4) | seeds |", "|---|" + "---|" * (len(T) + 2)]
    avg = {}
    for name in sorted(R, key=lambda n: -np.mean([np.mean([r["auc"] for r in R[n][t]]) for t in SOURCES if t in R[n]] or [0])):
        cells, per = [], []
        for t in T:
            if t in R[name]:
                a = [r["auc"] for r in R[name][t]]
                cells.append(f"{np.mean(a):.3f}±{np.std(a):.3f}")
                if t in SOURCES:
                    per.append(np.mean(a))
            else:
                cells.append("–")
        avg[name] = np.mean(per) if len(per) == 4 else float("nan")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {avg[name]:.3f} | {min(len(R[name][t]) for t in R[name])} |")
    lines += ["", f"Paired bootstrap of seed-averaged AUROC vs `{args.ref}` (diff, 95% CI):", "",
              "| method | " + " | ".join(T) + " |", "|---|" + "---|" * len(T)]
    rng = np.random.default_rng(0)
    for name in R:
        if name == args.ref or args.ref not in R:
            continue
        cells = []
        for t in T:
            if t in S[name] and t in S[args.ref]:
                a, b, y = np.mean(S[name][t], 0), np.mean(S[args.ref][t], 0), ytrue[t]
                d = []
                for _ in range(500):
                    i = rng.integers(0, len(y), len(y))
                    if y[i].min() != y[i].max():
                        d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
                cells.append(f"{roc_auc_score(y, a) - roc_auc_score(y, b):+.3f} [{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}]")
            else:
                cells.append("–")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
