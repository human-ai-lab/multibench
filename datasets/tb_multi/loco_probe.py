"""Leave-one-corpus-out linear probes on frozen embeddings + corpus-fingerprint audits.

Domains: qatar, nlm (Montgomery+Shenzhen), tbx11k (active TB vs healthy/sick-non-TB), pakistan. Each is
held out in turn (train on the other three); `cidrz` (external real-world target) is scored by a model
trained on all four, with NO tuning on it (C chosen by inner leave-one-source-out on held-in data only).

Variants: raw | dstd (label-free per-domain feature standardization: each source standardized by its own
stats, target by its own) | sbal (sample weights equalizing class prior inside every source, so source
identity cannot predict the label) | dstd+sbal.

Audit probes (LOCO AUROC / identity accuracy): metadata-only (orig size/aspect/ext), corpus identity from
embeddings.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from datasets.tb_multi.common import SOURCES, boot_auc_ci, load_all, op_points


def standardize_by_domain(X, dom):
    X = X.copy()
    for d in np.unique(dom):
        i = dom == d
        X[i] = (X[i] - X[i].mean(0)) / (X[i].std(0) + 1e-6)
    return X


def source_balanced_weights(y, dom):
    """Each (source, class) cell gets equal total weight, so the source carries no label prior."""
    w = np.zeros(len(y))
    doms = np.unique(dom)
    for d in doms:
        for c in (0, 1):
            i = (dom == d) & (y == c)
            if i.any():
                w[i] = len(y) / (2 * len(doms)) / i.sum()
    return w * len(y) / w.sum()


def fit_score(Xtr, ytr, dtr, Xte, dte, dstd, sbal, C=None):
    if dstd:
        Xtr, Xte = standardize_by_domain(Xtr, dtr), standardize_by_domain(Xte, dte)
    else:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    w = source_balanced_weights(ytr, dtr) if sbal else None
    if C is None:  # inner leave-one-source-out on held-in data only
        best, C = -1, 0.01
        for c in (0.001, 0.01, 0.1):
            aucs = []
            for d in np.unique(dtr):
                tr, va = dtr != d, dtr == d
                if len(np.unique(ytr[va])) < 2:
                    continue
                m = LogisticRegression(C=c, max_iter=2000, class_weight=None if sbal else "balanced")
                m.fit(Xtr[tr], ytr[tr], sample_weight=None if w is None else w[tr])
                aucs.append(roc_auc_score(ytr[va], m.decision_function(Xtr[va])))
            if np.mean(aucs) > best:
                best, C = np.mean(aucs), c
    m = LogisticRegression(C=C, max_iter=2000, class_weight=None if sbal else "balanced")
    m.fit(Xtr, ytr, sample_weight=w)
    s_tr = m.decision_function(Xtr)
    thr = np.quantile(s_tr[ytr == 1], 0.10)  # 90% sensitivity on held-in data
    return m.decision_function(Xte), thr, C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/loco_probe_xrv.json")
    args = ap.parse_args()
    X, meta = load_all("xrv")
    y, dom = meta["label"].values, meta["domain"].values
    print("n per domain/label:\n", meta.groupby(["domain", "label"]).size().unstack())

    results = {}
    variants = dict(raw=(False, False), dstd=(True, False), sbal=(False, True), dstd_sbal=(True, True))
    for vname, (dstd, sbal) in variants.items():
        for target in SOURCES + ["cidrz"]:
            tr = np.isin(dom, [s for s in SOURCES if s != target])
            te = dom == target
            s, thr, C = fit_score(X[tr], y[tr], dom[tr], X[te], dom[te], dstd, sbal)
            yt = y[te]
            r = dict(auc=roc_auc_score(yt, s), auc_ci=boot_auc_ci(yt, s), C=C, **op_points(yt, s, thr))
            if target == "tbx11k":
                cat = meta["category"].values[te]
                for other in ("healthy", "sick"):
                    k = (cat == other) | (yt == 1)
                    r[f"auc_vs_{other}"] = roc_auc_score(yt[k], s[k])
            results[f"{vname}/{target}"] = r
            print(f"{vname:10s} -> {target:9s} AUC {r['auc']:.3f} [{r['auc_ci'][0]:.3f},{r['auc_ci'][1]:.3f}] "
                  f"sens@spec70 {r['sens_at_spec70']:.2f} spec@sens90 {r['spec_at_sens90']:.2f} | "
                  f"transfer sens {r['transfer_sens']:.2f} spec {r['transfer_spec']:.2f}"
                  + (f" | vs healthy {r['auc_vs_healthy']:.3f} vs sick {r['auc_vs_sick']:.3f}" if target == "tbx11k" else ""),
                  flush=True)

    # --- audits ---
    corp = meta["corpus"].values
    ids = LogisticRegression(C=0.1, max_iter=3000)
    acc = (cross_val_predict(ids, StandardScaler().fit_transform(X), corp, cv=StratifiedKFold(5, shuffle=True, random_state=0)) == corp).mean()
    results["audit/corpus_identity_acc_from_embedding"] = acc
    print(f"\ncorpus-identity probe (6 corpora, xrv embedding, 5-fold): accuracy {acc:.3f} (chance {max(pd.Series(corp).value_counts(normalize=True)):.3f})")

    M = np.c_[meta["orig_h"], meta["orig_w"], meta["orig_w"] / meta["orig_h"], np.log(meta["orig_h"] * meta["orig_w"]),
              (meta["ext"] == ".jpg").astype(float)]
    # Pooled-training shortcut potency: how well do corpus identity / file metadata alone predict the label
    # when the pool is randomly split (what a model trained on a pooled, unbalanced corpus can exploit)?
    onehot = pd.get_dummies(corp).values.astype(float)
    for name, F in (("corpus_id_only", onehot), ("corpus_id+size/aspect/ext", np.c_[onehot, M])):
        p = cross_val_predict(LogisticRegression(C=1.0, max_iter=3000), StandardScaler().fit_transform(F), y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0), method="decision_function")
        results[f"audit/pooled_label_auc/{name}"] = roc_auc_score(y, p)
        print(f"pooled 5-fold label AUC from {name}: {roc_auc_score(y, p):.3f}")
    json.dump(results, open(args.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
