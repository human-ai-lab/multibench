"""Paired stratified-bootstrap differences in weighted VinDr AUROC between seed-averaged method scores and a reference.
Usage: python -m datasets.tb_multi.vindr_paired --ref f_vit --methods f_raddino ... --out results/vindr_paired.md"""
import argparse

import numpy as np

from datasets.tb_multi.common import ALL, load_all
from datasets.tb_multi.vindr_report import load, wauc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default="f_vit")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default="results/vindr_paired.md")
    args = ap.parse_args()
    _, meta = load_all(None, corpora=ALL)
    v = meta[meta.domain == "vindr"].reset_index(drop=True)
    y, w = v.label.values, v.w.values.astype(float)
    cells = np.array([f"{a}|{b}|{c}" for a, b, c in zip(v.split, y, v.stratum)])
    groups = [np.flatnonzero(cells == c) for c in np.unique(cells)]
    ref = np.mean(load(args.ref, args.seeds)[0], 0)
    rng = np.random.default_rng(0)
    idx = [np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(args.n)]
    lines = [f"| method | AUROC diff vs {args.ref} (weighted) [95% CI] |", "|---|---|"]
    for m in args.methods:
        s = np.mean(load(m, args.seeds)[0], 0)
        d = [wauc(y[i], s[i], w[i]) - wauc(y[i], ref[i], w[i]) for i in idx]
        lines.append(f"| {m} | {wauc(y, s, w) - wauc(y, ref, w):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] |")
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
