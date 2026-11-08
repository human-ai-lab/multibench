"""Label audit of the VinDr-CXR sample: does the score follow the radiologists' agreement, and which images should a
human reader re-examine?

1. AUROC of a model against the TB-free images of the same split, per positive tier (train 2 of 3 reads, train 3 of 3,
   test consensus). If the 2-read tier scores clearly below the unanimous tier, the majority rule is admitting noisier labels.
2. A review list: the lowest-scoring positives (possible false labels or subtle disease) and the highest-scoring negatives
   (possible missed TB), with the findings the radiologists did record. CSV plus a contact sheet rendered from the DICOMs.
The model only prioritises images; it cannot say which label is right.
Usage: python -m datasets.tb_multi.vindr_label_review --method f_raddino
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from datasets.tb_multi.artefact_adjust import load_scores
from datasets.tb_multi.common import ALL, load_all
from datasets.tb_multi.standardize import VINDR_ROOT, _vindr_decode

FLAGS = ["Lung_cavity", "Infiltration", "Consolidation", "Nodule_Mass", "Pulmonary_fibrosis", "Calcification", "Pleural_effusion", "Pleural_thickening", "No_finding"]


def sheet(df, title, path, size=256, cols=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = int(np.ceil(len(df) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.75), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    for a, (_, r) in zip(ax.ravel(), df.iterrows()):
        img, *_ = _vindr_decode((f"{VINDR_ROOT}/{r.split}/{r.id}.dicom", size))
        a.imshow(img, cmap="gray")
        fl = [f.replace("_", " ")[:12] for f in FLAGS[:-1] if r[f"f_{f}"] > 0][:2]
        a.set_title(f"{r.id[:6]}  {'test consensus' if r.split == 'test' and r.label else 'votes ' + str(r.tb_votes)}\nscore pct {r.pct:.0f}  {', '.join(fl) or ('no finding' if r.f_No_finding else '-')}", fontsize=6.5)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", default="f_raddino")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n", type=int, default=24, help="images per review group")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    _, meta = load_all(None, corpora=ALL)
    m = meta[meta.domain == "vindr"].reset_index(drop=True)
    vm = pd.read_csv("data/tb_multi/vindr_meta.csv")
    assert (m.id.values == vm.id.values).all()
    s, _ = load_scores(args.method, "vindr", args.seeds, dirs=("results/loco_vindr",))
    assert s is not None, "no VinDr score files for " + args.method
    df = vm.assign(score=s)
    df["pct"] = df.score.rank(pct=True) * 100
    L = [f"## VinDr-CXR label audit ({args.method}, seed-averaged score)", "",
         "| positive tier | n | AUROC vs TB-free images of the same split | median score pct |", "|---|---|---|---|"]
    for name, split, votes in [("train, 2 of 3 reads", "train", 2), ("train, 3 of 3 reads", "train", 3), ("test, consensus", "test", 1)]:
        d = df[(df.split == split) & ((df.tb_votes == votes) | (df.label == 0))]
        pos = d[d.label == 1]
        L.append(f"| {name} | {len(pos)} | {roc_auc_score(d.label, d.score):.3f} | {pos.pct.median():.0f} |")
    tr = df[df.split == "train"]
    neg, p2, p3 = tr[tr.label == 0].score.values, tr[tr.tb_votes == 2].score.values, tr[tr.tb_votes == 3].score.values
    au = lambda p, n: roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n])
    rs, d = np.random.default_rng(0), []
    for _ in range(2000):
        n_ = rs.choice(neg, len(neg)); d.append(au(rs.choice(p3, len(p3)), n_) - au(rs.choice(p2, len(p2)), n_))
    L += ["", f"Unanimous minus 2-of-3 AUROC on the training split: {au(p3, neg) - au(p2, neg):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] (bootstrap, shared negatives)."]
    for g, q in (("positives, low-scoring", df[df.label == 1].nsmallest(args.n, "score")),
                 ("negatives, high-scoring", df[df.label == 0].nlargest(args.n, "score"))):
        L += ["", f"{g}: mean other-finding rate = " + ", ".join(f"{f.replace('_', ' ')} {q[f'f_{f}'].mean():.2f}" for f in FLAGS)]
    rng = np.random.default_rng(0)
    rand = df[(df.label == 1) & (df.tb_votes >= 2)].sample(args.n, random_state=0)
    groups = {"low_pos": df[df.label == 1].nsmallest(args.n, "score"), "high_neg": df[df.label == 0].nlargest(args.n, "score"), "random_pos": rand}
    rows = []
    for k, g in groups.items():
        rows.append(g.assign(group=k))
        sheet(g, {"low_pos": "Lowest-scoring TB-positive images (check the label)", "high_neg": "Highest-scoring TB-free images (check for missed TB)",
                  "random_pos": "Random positives (2 or 3 reads, or test consensus), for reference"}[k], os.path.join(args.outdir, f"vindr_review_{k}.png"))
    pd.concat(rows)[["group", "id", "split", "label", "tb_votes", "score", "pct"] + [f"f_{f}" for f in FLAGS]].to_csv(os.path.join(args.outdir, "vindr_label_review.csv"), index=False)
    open(os.path.join(args.outdir, "vindr_label_review.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
