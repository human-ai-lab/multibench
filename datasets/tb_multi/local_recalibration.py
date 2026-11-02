"""Deployment-oriented evaluation of the held-out image experts: WHO target-product-profile (TPP) operating points and
what a small labelled local sample buys.

WHO TPP for a TB triage test: sensitivity >= 90% and specificity >= 70%. For every (method, target), on the seed-averaged
raw logit margins m, we report:
  ceiling      sens at 70% spec and spec at 90% sens on the whole target (an oracle that knows every local label)
  transfer     the deployable zero-label operating point: threshold giving 90% sensitivity on held-in validation corpora
  local-k      k labelled local cases are drawn at random (R repeats, natural prevalence); the threshold is set at the
               10% quantile of the local positives' scores (90% sensitivity on the local sample; needs >=1 positive)
               and evaluated on the remaining cases: mean sens / spec and the fraction of repeats meeting the TPP.
  calibration  Platt (slope+intercept) and intercept-only recalibration fitted on the k local cases, evaluated on the
               rest: ECE and Brier versus the raw probabilities sigmoid(m) (decision-neutral for ranking).
Usage: python -m datasets.tb_multi.local_recalibration --methods bb_raddino --seeds 0 1 2 3 4
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

from datasets.tb_multi.artefact_adjust import load_scores
from datasets.tb_multi.calibration import ece, sigmoid
from datasets.tb_multi.common import ALL, EXTERNAL, SOURCES, load_all

KS = (0, 25, 50, 100, 200, 400)


def _json(method, target, seeds, dirs=("results/loco", "results/loco_final", "results/loco_ext")):
    out = []
    for sd in seeds:
        for d in dirs:
            for stem in (f"{method}_{target}_s{sd}", f"{method}_ext-{target}_s{sd}"):
                f = os.path.join(d, stem + ".json")
                if os.path.exists(f):
                    out.append(json.load(open(f))); break
            else:
                continue
            break
    return out


def intercept_only(m, y):
    f = lambda b: -np.sum(y * np.log(sigmoid(m + b) + 1e-12) + (1 - y) * np.log(1 - sigmoid(m + b) + 1e-12))
    return minimize_scalar(f, bounds=(-15, 15), method="bounded").x


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def evaluate(y, m, thr_transfer, reps, rng):
    pos, neg = m[y == 1], m[y == 0]
    res = dict(ceiling_sens70=float((pos > np.quantile(neg, 0.70)).mean()), ceiling_spec90=float((neg < np.quantile(pos, 0.10)).mean()),
               transfer_sens=float((pos >= thr_transfer).mean()), transfer_spec=float((neg < thr_transfer).mean()), k={})
    res["transfer_tpp"] = float(res["transfer_sens"] >= 0.9 and res["transfer_spec"] >= 0.7)
    for k in KS:
        if k == 0 or k > 0.6 * len(y):
            continue
        acc = dict(sens=[], spec=[], tpp=[], ece_raw=[], ece_platt=[], ece_int=[], br_raw=[], br_platt=[], br_int=[])
        for _ in range(reps):
            idx = rng.permutation(len(y))
            loc, rest = idx[:k], idx[k:]
            yl, ml, yr, mr = y[loc], m[loc], y[rest], m[rest]
            if yl.sum() < 3 or yr.sum() < 1 or (1 - yr).sum() < 1:
                continue
            t = np.quantile(ml[yl == 1], 0.10)
            se, sp = float((mr[yr == 1] >= t).mean()), float((mr[yr == 0] < t).mean())
            acc["sens"].append(se); acc["spec"].append(sp); acc["tpp"].append(float(se >= 0.9 and sp >= 0.7))
            praw = sigmoid(mr)
            acc["ece_raw"].append(ece(yr, praw)); acc["br_raw"].append(brier(yr, praw))
            if (1 - yl).sum() >= 3:
                lr = LogisticRegression(C=1.0, max_iter=500).fit(ml[:, None], yl)
                pp = lr.predict_proba(mr[:, None])[:, 1]
                acc["ece_platt"].append(ece(yr, pp)); acc["br_platt"].append(brier(yr, pp))
                pi = sigmoid(mr + intercept_only(ml, yl))
                acc["ece_int"].append(ece(yr, pi)); acc["br_int"].append(brier(yr, pi))
        res["k"][k] = {a: (float(np.mean(v)) if v else float("nan")) for a, v in acc.items()} | {"n_ok": len(acc["sens"])}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--targets", nargs="+", default=SOURCES + EXTERNAL)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--out", default="results/local_recalibration.md")
    args = ap.parse_args()
    _, meta = load_all(None, corpora=ALL)
    rng = np.random.default_rng(0)
    L1 = ["| method | target | prev | ceiling: sens@spec70 / spec@sens90 | transfer sens/spec (meets TPP?) | k | local-threshold sens | spec | P(meets TPP) |",
          "|---|---|---|---|---|---|---|---|---|"]
    L2 = ["| method | target | k | ECE raw | ECE intercept-only | ECE Platt | Brier raw | Brier intercept-only | Brier Platt |", "|---|---|---|---|---|---|---|---|---|"]
    for mth in args.methods:
        for t in args.targets:
            m, ns = load_scores(mth, t, args.seeds)
            if m is None:
                continue
            y = meta[meta.domain == t].label.values
            js = _json(mth, t, args.seeds)
            if all("thr" in j for j in js):
                thr = float(np.mean([j["thr"] for j in js]))
            else:  # older runs did not store the threshold: invert the stored transfer sensitivity on the positives
                thr = float(np.quantile(m[y == 1], 1 - np.mean([j["transfer_sens"] for j in js])))
            r = evaluate(y, m, thr, args.reps, rng)
            head = f"| {mth} | {t} | {y.mean():.2f} | {r['ceiling_sens70']:.2f} / {r['ceiling_spec90']:.2f} | {r['transfer_sens']:.2f}/{r['transfer_spec']:.2f} ({'yes' if r['transfer_tpp'] else 'no'}) |"
            for k, v in r["k"].items():
                L1.append(head + f" {k} | {v['sens']:.2f} | {v['spec']:.2f} | {v['tpp']:.2f} |")
                L2.append(f"| {mth} | {t} | {k} | {v['ece_raw']:.3f} | {v['ece_int']:.3f} | {v['ece_platt']:.3f} | {v['br_raw']:.3f} | {v['br_int']:.3f} | {v['br_platt']:.3f} |")
            print(t, mth, "done", flush=True)
    with open(args.out, "w") as f:
        f.write("## Operating points under local labelled samples\n\n" + "\n".join(L1) + "\n\n## Probability calibration after local recalibration\n\n" + "\n".join(L2) + "\n")


if __name__ == "__main__":
    main()
