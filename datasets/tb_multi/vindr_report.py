"""Evaluation of the LOCO image experts on VinDr-CXR, the second external cohort (`results/loco_vindr/`).

The VinDr sample is enriched (all 646 TB positives, 1,500 sampled negatives, half 'No finding'); every row carries `w`, the
inverse sampling probability, so *weighted* metrics describe the release's TB-free + TB-positive images (prevalence ~3.6%).
Per method, on the seed-averaged raw margins unless noted:
  AUROC            per-seed mean +/- sd (weighted), weighted and unweighted on the seed average, 95% CI (stratified bootstrap within
                   split x label x stratum cells), per label source (train majority vote / test consensus), and against
                   'No finding' negatives only vs 'other findings' negatives only
  positives        AUROC for positives with an active-pattern finding (cavity/infiltration/consolidation/nodule) vs none
                   (the latter are the proxy for inactive TB: fibrosis/calcification only)
  WHO TPP          ceiling sens@70% spec and spec@90% sens (weighted), the deployable transfer threshold (held-in validation),
                   and PPV at the weighted prevalence
  calibration      weighted ECE/Brier/slope/intercept of sigmoid(margin); after Platt / intercept-only recalibration on k labelled
                   local cases (drawn from the enriched sample, fitted with the weights), evaluated on the rest
  artefact         weighted artefact-stratified AUROC (10 quantile bins of an artefact score fitted inside VinDr with 5-fold CV, or
                   trained on the four sources) and the cross-validated gain of [score, artefact] over the artefact alone
Usage: python -m datasets.tb_multi.vindr_report --methods f_raddino f_vit ... --seeds 0 1 2 3 4
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from datasets.tb_multi.artefact_adjust import _fit, artefact_features, cv_scores
from datasets.tb_multi.calibration import sigmoid
from datasets.tb_multi.common import ALL, SOURCES, load_all

DIR = "results/loco_vindr"
KS = (50, 100, 200, 400)
ACTIVE = ["f_Lung_cavity", "f_Infiltration", "f_Consolidation", "f_Nodule_Mass"]


def wauc(y, s, w):
    return float(roc_auc_score(y, s, sample_weight=w))


def wquantile(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return float(x[o][np.searchsorted(c, q).clip(0, len(x) - 1)])


def wece(y, p, w, bins=10):
    o = np.argsort(p)
    cuts = np.searchsorted(np.cumsum(w[o]) / w.sum(), np.linspace(0, 1, bins + 1)[1:-1])
    return float(sum(w[i].sum() / w.sum() * abs(np.average(y[i], weights=w[i]) - np.average(p[i], weights=w[i]))
                     for i in np.split(o, cuts) if len(i) and w[i].sum() > 0))


def cal_metrics(y, m, w):
    p = sigmoid(m)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(m[:, None], y, sample_weight=w)
    return dict(ece=wece(y, p, w), brier=float(np.average((p - y) ** 2, weights=w)), slope=float(lr.coef_[0, 0]),
                intercept=float(lr.intercept_[0]))


def intercept_only(m, y, w):
    f = lambda b: -np.sum(w * (y * np.log(sigmoid(m + b) + 1e-12) + (1 - y) * np.log(1 - sigmoid(m + b) + 1e-12)))
    return minimize_scalar(f, bounds=(-15, 15), method="bounded").x


def strat_wauc(y, s, art, w, k=10):
    edges = np.quantile(art, np.linspace(0, 1, k + 1)[1:-1])
    b = np.searchsorted(edges, art)
    num = den = 0.0
    for j in range(k):
        i = b == j
        if y[i].sum() and (1 - y[i]).sum():
            pw = w[i][y[i] == 1].sum() * w[i][y[i] == 0].sum()
            num += wauc(y[i], s[i], w[i]) * pw
            den += pw
    tot = w[y == 1].sum() * w[y == 0].sum()
    return (num / den if den else float("nan")), den / tot


def boot_ci(fn, cells, y, n=300, seed=0):
    """Stratified bootstrap: resample inside each (split, label, stratum) cell so the sampling design is preserved."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(cells == c) for c in np.unique(cells)]
    v = []
    for _ in range(n):
        i = np.concatenate([rng.choice(g, len(g)) for g in groups])
        v.append(fn(i))
    return tuple(np.percentile(v, [2.5, 97.5]))


def load(method, seeds, d=DIR):
    S, J = [], []
    for sd in seeds:
        f = f"{d}/{method}_vindr_s{sd}"
        if os.path.exists(f + ".json"):
            S.append(np.load(f + "_scores.npy").astype(np.float64)); J.append(json.load(open(f + ".json")))
    return S, J


def local_sweep(y, m, w, thr, reps, rng):
    out = {}
    for k in KS:
        acc = {a: [] for a in ("sens", "spec", "tpp", "ece_raw", "ece_int", "ece_platt", "br_raw", "br_int", "br_platt")}
        for _ in range(reps):
            idx = rng.permutation(len(y)); loc, rest = idx[:k], idx[k:]
            yl, ml, wl, yr, mr, wr = y[loc], m[loc], w[loc], y[rest], m[rest], w[rest]
            if yl.sum() < 5 or (1 - yl).sum() < 5:
                continue
            t = np.quantile(ml[yl == 1], 0.10)
            se, sp = float((mr[yr == 1] >= t).mean()), float(np.average(mr[yr == 0] < t, weights=wr[yr == 0]))
            acc["sens"].append(se); acc["spec"].append(sp); acc["tpp"].append(float(se >= 0.9 and sp >= 0.7))
            for tag, p in (("raw", sigmoid(mr)),
                           ("int", sigmoid(mr + intercept_only(ml, yl, wl))),
                           ("platt", LogisticRegression(C=1.0, max_iter=500).fit(ml[:, None], yl, sample_weight=wl).predict_proba(mr[:, None])[:, 1])):
                acc["ece_" + tag].append(wece(yr, p, wr)); acc["br_" + tag].append(float(np.average((p - yr) ** 2, weights=wr)))
        out[k] = {a: (float(np.mean(v)) if v else float("nan")) for a, v in acc.items()} | {"n_ok": len(acc["sens"])}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--reps", type=int, default=150)
    ap.add_argument("--out", default="results/vindr_eval.md")
    args = ap.parse_args()
    _, meta = load_all(None, corpora=ALL)
    v = meta[meta.domain == "vindr"].reset_index(drop=True)
    y, w = v.label.values, v.w.values.astype(np.float64)
    split, strat = v.split.values, v.stratum.values
    cells = np.array([f"{a}|{b}|{c}" for a, b, c in zip(split, y, strat)])
    prev = float(np.average(y, weights=w))
    active = (v[ACTIVE].sum(1) > 0).values
    F = artefact_features(meta)
    Fv, dom = F[(meta.domain == "vindr").values], meta.domain.values
    art_src = _fit(F[np.isin(dom, SOURCES)], meta.label.values[np.isin(dom, SOURCES)])(Fv)
    art_cv = np.mean([cv_scores(Fv, y, seed=r) for r in range(3)], 0)
    A = ["## Artefact-only baselines on VinDr (weighted AUROC)", "",
         f"artefact predictor trained on the 4 sources: {wauc(y, art_src, w):.3f}; cross-validated inside VinDr: {wauc(y, art_cv, w):.3f} "
         f"(unweighted {roc_auc_score(y, art_cv):.3f})", ""]
    T1 = ["| method | seeds | AUROC per-seed (weighted) | AUROC seed-avg weighted [95% CI] | unweighted | vs No-finding negs | vs other-findings negs | "
          "train-label rows | test-label rows | active-pattern pos | no-active-pattern pos |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    T2 = ["| method | ceiling sens@spec70 | ceiling spec@sens90 | transfer thr: sens | spec | PPV @ prev "
          f"{prev:.3f} | ECE | Brier | slope | intercept |", "|---|---|---|---|---|---|---|---|---|---|"]
    T3 = ["| method | k | local-threshold sens | spec | P(meets TPP) | ECE raw | ECE intercept-only | ECE Platt | Brier raw | Brier Platt |", "|---|---|---|---|---|---|---|---|---|---|"]
    T4 = ["| method | raw (weighted) | artefact-only (cv) | stratified (cv) [pairs kept] | stratified (src) [pairs kept] | model adds over artefact (cv) |", "|---|---|---|---|---|---|"]
    rng = np.random.default_rng(0)
    summary = {}
    for mth in args.methods:
        S, J = load(mth, args.seeds)
        if not S:
            continue
        sa = np.mean(S, 0)
        per = [wauc(y, s, w) for s in S]
        lo, hi = boot_ci(lambda i: wauc(y[i], sa[i], w[i]), cells, y)
        sub = lambda mask: wauc(y[mask], sa[mask], w[mask])
        nf, ot = (strat == "nofind") | (y == 1), (strat == "other") | (y == 1)
        T1.append(f"| {mth} | {len(S)} | {np.mean(per):.3f}±{np.std(per):.3f} | {wauc(y, sa, w):.3f} [{lo:.3f},{hi:.3f}] | {roc_auc_score(y, sa):.3f} | "
                  f"{roc_auc_score(y[nf], sa[nf]):.3f} | {roc_auc_score(y[ot], sa[ot]):.3f} | {sub(split == 'train'):.3f} | {sub(split == 'test'):.3f} | "
                  f"{roc_auc_score(y[(y == 0) | active], sa[(y == 0) | active]):.3f} (n={int(((y == 1) & active).sum())}) | "
                  f"{roc_auc_score(y[(y == 0) | ~active], sa[(y == 0) | ~active]):.3f} (n={int(((y == 1) & ~active).sum())}) |")
        pos, neg, wn = sa[y == 1], sa[y == 0], w[y == 0]
        c70, c90 = float((pos > wquantile(neg, wn, 0.70)).mean()), float(np.average(neg < np.quantile(pos, 0.10), weights=wn))
        thr = float(np.mean([j["thr"] for j in J]))
        ts, tp = float((pos >= thr).mean()), float(np.average(neg < thr, weights=wn))
        ppv = ts * prev / (ts * prev + (1 - tp) * (1 - prev))
        cm = cal_metrics(y, sa, w)
        T2.append(f"| {mth} | {c70:.2f} | {c90:.2f} | {ts:.2f} | {tp:.2f} | {ppv:.2f} | {cm['ece']:.3f} | {cm['brier']:.3f} | {cm['slope']:.2f} | {cm['intercept']:+.2f} |")
        for k, r in local_sweep(y, sa, w, thr, args.reps, rng).items():
            T3.append(f"| {mth} | {k} | {r['sens']:.2f} | {r['spec']:.2f} | {r['tpp']:.2f} | {r['ece_raw']:.3f} | {r['ece_int']:.3f} | {r['ece_platt']:.3f} | "
                      f"{r['br_raw']:.3f} | {r['br_platt']:.3f} |")
        sc, kc = strat_wauc(y, sa, art_cv, w); ss, ks = strat_wauc(y, sa, art_src, w)
        comb = cv_scores(np.column_stack([sa, art_cv]), y, seed=1)
        T4.append(f"| {mth} | {wauc(y, sa, w):.3f} | {wauc(y, art_cv, w):.3f} | {sc:.3f} [{kc:.2f}] | {ss:.3f} [{ks:.2f}] | {wauc(y, comb, w) - wauc(y, art_cv, w):+.3f} |")
        summary[mth] = dict(auc_w=wauc(y, sa, w), ci=[lo, hi], per_seed=per, strat_cv=sc, pairs_kept=kc, ceiling_sens70=c70, ceiling_spec90=c90,
                            transfer_sens=ts, transfer_spec=tp, ppv=ppv, **cm)
        print(mth, f"AUROC {wauc(y, sa, w):.3f} [{lo:.3f},{hi:.3f}] strat {sc:.3f}", flush=True)
    txt = ["# VinDr-CXR external evaluation", "", f"n={len(y)} (pos {int(y.sum())}), weighted prevalence {prev:.3f}, one within-corpus duplicate pair kept", ""]
    txt += A + ["## Discrimination", ""] + T1 + ["", "## WHO triage profile (sens>=90%, spec>=70%) and calibration, seed-averaged margins", ""] + T2
    txt += ["", "## Local recalibration with k labelled cases (weights = inverse sampling probability)", ""] + T3
    txt += ["", "## Artefact-adjusted AUROC (weighted)", ""] + T4
    open(args.out, "w").write("\n".join(txt) + "\n")
    json.dump(summary, open(args.out.replace(".md", ".json"), "w"), indent=1)
    print("\n".join(txt))


if __name__ == "__main__":
    main()
