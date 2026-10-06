"""Multimodal fusion study on CIDRZ (n=364, 60 TB): does a cross-corpus image expert add to in-domain
clinical text + cough audio, and which fusion strategy uses it best -- including when a modality is missing?

The image expert's scores come from `loco_train.py --target cidrz` (trained on Qatar/NLM/TBX11K/Pakistan only,
seed-averaged z-scored logit margins, never fit to CIDRZ); text/audio experts are fit inside CV on CIDRZ.
Repeated stratified 5-fold CV (10 repeats), identical folds across arms. Fusion strategies:
  stack   nested stacking: LR on inner-OOF z-scored expert scores (the earlier late-fusion baseline)
  concat  early fusion: LR on standardized [text, audio, image-score]
  relw    reliability-weighted average: weights = max(inner-OOF AUROC - 0.5, 0) of each expert
  mdmlp   small MLP with per-modality encoders + modality dropout (masked mean fusion); can run with a
          modality absent at test time
Reported: AUROC / UAR(0.5 threshold) per arm; image contribution = (text+audio+image) - (text+audio) with a
paired participant bootstrap; robustness = AUROC with audio (or text) masked at test time.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from datasets.tb_cxr_qatar.multimodal_late_fusion import _expert, _zscore, load_cidrz

torch.set_num_threads(4)


class MDMLP(nn.Module):
    def __init__(self, dt, da, d=16):
        super().__init__()
        self.enc = nn.ModuleList([nn.Sequential(nn.Linear(dt, d), nn.ReLU()), nn.Sequential(nn.Linear(da, d), nn.ReLU()),
                                  nn.Sequential(nn.Linear(1, d), nn.ReLU())])
        self.out = nn.Linear(d, 2)

    def forward(self, xs, mask):  # xs: list of 3 tensors; mask: (B,3) in {0,1}
        z = torch.stack([e(x) for e, x in zip(self.enc, xs)], 1) * mask.unsqueeze(-1)
        return self.out(z.sum(1) / mask.sum(1, keepdim=True).clamp(min=1))


def fit_mdmlp(tr, y, p_drop=0.3, epochs=150, seed=0, use=(1, 1, 1)):
    torch.manual_seed(seed)
    xs = [torch.tensor(a, dtype=torch.float32) for a in tr]
    yt = torch.tensor(y)
    m = MDMLP(xs[0].shape[1], xs[1].shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=5e-2)
    w = torch.tensor([1.0, float((y == 0).sum() / (y == 1).sum())])
    base = torch.tensor(use, dtype=torch.float32).expand(len(y), 3)
    for _ in range(epochs):
        keep = (torch.rand(len(y), 3) > p_drop).float() * base
        empty = keep.sum(1) == 0
        keep[empty] = base[empty]
        loss = nn.functional.cross_entropy(m(xs, keep), yt, weight=w)
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def predict_mdmlp(m, te, use):
    xs = [torch.tensor(a, dtype=torch.float32) for a in te]
    mask = torch.tensor(use, dtype=torch.float32).expand(len(xs[0]), 3)
    with torch.no_grad():
        lg = m(xs, mask)
    return (lg[:, 1] - lg[:, 0]).numpy()


def run(data, y, img, n_repeats, n_folds=5):
    arms = {}
    def put(name, r, te, s):
        arms.setdefault(name, np.zeros((n_repeats, len(y))))[r, te] = s
    for r in range(n_repeats):
        for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=r).split(data["text"], y):
            inner = StratifiedKFold(5, shuffle=True, random_state=1000 + r)
            ztr, zte, auc_oof = {}, {}, {}
            for m in ("text", "audio"):
                raw_tr = cross_val_predict(_expert(), data[m][tr], y[tr], cv=inner, method="decision_function")
                raw_te = _expert().fit(data[m][tr], y[tr]).decision_function(data[m][te])
                ztr[m], zte[m] = _zscore(raw_tr, raw_te)
                auc_oof[m] = roc_auc_score(y[tr], raw_tr)
            ztr["image"], zte["image"] = _zscore(img[tr], img[te])
            auc_oof["image"] = roc_auc_score(y[tr], img[tr])  # label use on train folds only, as for other experts
            for m in ("text", "audio", "image"):
                put(m, r, te, zte[m])
            for arm, mods in (("text+audio", ("text", "audio")), ("text+image", ("text", "image")),
                              ("text+audio+image", ("text", "audio", "image"))):
                st = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(
                    np.column_stack([ztr[m] for m in mods]), y[tr])
                put(f"stack|{arm}", r, te, st.decision_function(np.column_stack([zte[m] for m in mods])))
                wts = np.array([max(auc_oof[m] - 0.5, 0) for m in mods])
                put(f"relw|{arm}", r, te, np.column_stack([zte[m] for m in mods]) @ (wts / wts.sum()))
                F = lambda idx, d: np.column_stack([d[m][idx] if m == "image" else d[m][idx] for m in mods])
                Xtr = np.column_stack([data[m][tr] if m != "image" else img[tr] for m in mods])
                Xte = np.column_stack([data[m][te] if m != "image" else img[te] for m in mods])
                cl = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000)).fit(Xtr, y[tr])
                put(f"concat|{arm}", r, te, cl.decision_function(Xte))
            # MD-MLP: standardized inputs; modality dropout during training
            sc_t, sc_a = StandardScaler().fit(data["text"][tr]), StandardScaler().fit(data["audio"][tr])
            itr, ite = _zscore(img[tr], img[te])
            tr_in = [sc_t.transform(data["text"][tr]), sc_a.transform(data["audio"][tr]), itr[:, None]]
            te_in = [sc_t.transform(data["text"][te]), sc_a.transform(data["audio"][te]), ite[:, None]]
            for arm, use in (("text+audio", (1, 1, 0)), ("text+audio+image", (1, 1, 1))):
                mm = fit_mdmlp(tr_in, y[tr], seed=r, use=use)
                put(f"mdmlp|{arm}", r, te, predict_mdmlp(mm, te_in, use))
                if arm == "text+audio+image":  # same trained model, modality missing at test time
                    put("mdmlp|full, audio missing at test", r, te, predict_mdmlp(mm, te_in, (1, 0, 1)))
                    put("mdmlp|full, text missing at test", r, te, predict_mdmlp(mm, te_in, (0, 1, 1)))
                    put("mdmlp|full, image missing at test", r, te, predict_mdmlp(mm, te_in, (1, 1, 0)))
    return arms


def boot(y, a, b, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return np.percentile(d, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-scores", nargs="+", required=True, help="name=path.npy (seed-averaged z-scored margins)")
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--out", default="results/fusion_study.md")
    args = ap.parse_args()
    data = load_cidrz("data/google_health/tb_dataset.pkl")
    y = data["label"]
    out = []
    for spec in args.image_scores:
        name, path = spec.split("=")
        img = np.load(path).astype(np.float64)
        assert len(img) == len(y)
        arms = run(data, y, img, args.n_repeats)
        out += [f"\n### image expert: {name}  (alone AUROC {roc_auc_score(y, img):.3f})\n",
                "| arm | AUROC | UAR@0.5 or sign | ΔAUROC vs same-method w/o image |", "|---|---|---|---|"]
        for arm, p in arms.items():
            aucs = [roc_auc_score(y, q) for q in p]
            thr = 0.5 if False else 0.0
            uars = [balanced_accuracy_score(y, q > np.median(q) * 0 + (0.0 if "|" in arm else 0.0)) for q in p]
            delta = ""
            if arm.endswith("text+audio+image") or arm.endswith("text+image"):
                ref = arm.split("|")[0] + ("|text+audio" if arm.endswith("audio+image") else "|text")
                if ref in arms:
                    a, b = p.mean(0), arms[ref].mean(0)
                    lo, hi = boot(y, a, b)
                    delta = f"{roc_auc_score(y, a) - roc_auc_score(y, b):+.3f} [{lo:+.3f},{hi:+.3f}]"
            out.append(f"| {arm} | {np.mean(aucs):.3f}±{np.std(aucs):.3f} | {np.mean(uars):.3f} | {delta} |")
        print("\n".join(out[-len(arms) - 4:]), flush=True)
    open(args.out, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
