"""RAD-DINO at 448 px vs 224 px on VinDr-CXR (weighted), same tuned config and seeds (`results/run_vindr448.sh` vs `run_vindr.sh`).
Per-seed weighted AUROC, the seed-averaged weighted AUROC, and a paired bootstrap (448 minus 224) with weighted AUROC; also the
weighted oracle WHO-TPP points (sens at 70% spec, spec at 90% sens) on the seed average and per single seed, with a bootstrap that
resamples within the (split, label, stratum) sampling cells and the share of resamples meeting both targets.
Usage: python -m datasets.tb_multi.vindr_448"""
import numpy as np
from sklearn.metrics import roc_auc_score

from datasets.tb_multi.artefact_adjust import load_scores
from datasets.tb_multi.common import ALL, load_all
from datasets.tb_multi.vindr_report import wquantile

SEEDS = [0, 1, 2, 3, 4]


def zs(s):
    return (s - s.mean()) / s.std()


def main():
    _, meta = load_all(None, corpora=ALL)
    m = meta[meta.domain == "vindr"]
    y, w = m.label.values, m.w.values
    s224 = [np.load(f"results/loco_vindr/f_raddino_vindr_s{s}_scores.npy") for s in SEEDS]
    s448 = [np.load(f"results/loco_vindr448/r448_vindr_s{s}_scores.npy") for s in SEEDS]
    cells = np.array([f"{a}|{b}|{c}" for a, b, c in zip(m.split, y, m.stratum)])
    groups = [np.flatnonzero(cells == c) for c in np.unique(cells)]

    def ceil(a, y, w):
        pos, neg = a[y == 1], a[y == 0]
        return (np.average(pos > wquantile(neg, w[y == 0], 0.70)), np.average(neg < np.quantile(pos, 0.10), weights=w[y == 0]))

    L = ["| arm | seed AUROCs (weighted) | seed-avg AUROC | sens@spec70 [95% CI] | spec@sens90 [95% CI] | P(both) |", "|---|---|---|---|---|---|"]
    avg, rng = {}, np.random.default_rng(0)
    for name, S in (("224 px", s224), ("448 px", s448)):
        a = np.mean([zs(x) for x in S], 0)
        avg[name] = a
        B = np.array([ceil(a[i], y[i], w[i]) for i in (np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(1000))])
        lo, hi = np.percentile(B, 2.5, 0), np.percentile(B, 97.5, 0)
        sens, spec = ceil(a, y, w)
        L.append(f"| {name} ({len(S)}-seed mean) | {', '.join(f'{roc_auc_score(y, x, sample_weight=w):.3f}' for x in S)} | "
                 f"{roc_auc_score(y, a, sample_weight=w):.3f} | {sens:.2f} [{lo[0]:.2f}, {hi[0]:.2f}] | {spec:.2f} [{lo[1]:.2f}, {hi[1]:.2f}] | "
                 f"{((B[:, 0] >= .9) & (B[:, 1] >= .7)).mean():.3f} |")
        single = np.array([ceil(x, y, w) for x in S])  # single models: no ensembling
        L.append(f"| {name} single seeds, min--max | | | {single[:, 0].min():.2f}--{single[:, 0].max():.2f} | "
                 f"{single[:, 1].min():.2f}--{single[:, 1].max():.2f} | {int(((single[:, 0] >= .9) & (single[:, 1] >= .7)).sum())}/{len(S)} seeds meet both |")
    d = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], avg["448 px"][i], sample_weight=w[i]) - roc_auc_score(y[i], avg["224 px"][i], sample_weight=w[i]))
    diff = roc_auc_score(y, avg["448 px"], sample_weight=w) - roc_auc_score(y, avg["224 px"], sample_weight=w)
    L += ["", f"448 minus 224 px (paired bootstrap, weighted): {diff:+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]"]
    open("results/vindr_448.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
