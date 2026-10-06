"""Shared loading/metrics for the cross-corpus TB experiments (after `standardize` + `dedup`)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from datasets.tb_multi.standardize import OUT_DIR

CORPORA = ["qatar", "montgomery", "shenzhen", "tbx11k", "pakistan", "cidrz"]
# LOCO domains: Montgomery (138 images) is merged with Shenzhen as the NLM release ("nlm") so its fold
# has usable CIs; per-corpus numbers inside the fold can still be split via `corpus`.
DOMAIN_OF = dict(qatar="qatar", montgomery="nlm", shenzhen="nlm", tbx11k="tbx11k", pakistan="pakistan", cidrz="cidrz")
SOURCES = ["qatar", "nlm", "tbx11k", "pakistan"]


def load_all(suffix="xrv", kind="npy"):
    """Concatenate corpora (minus the dedup drop list, TBX11K latent/uncertain) -> (X, meta DataFrame)."""
    drop = pd.read_csv("results/tb_multi_dedup_drop.csv")
    metas, feats = [], []
    for c in CORPORA:
        m = pd.read_csv(f"{OUT_DIR}/{c}_meta.csv")
        x = np.load(f"{OUT_DIR}/{c}_{suffix}.npy", mmap_mode="r") if suffix else None
        keep = ~m["id"].isin(set(drop[drop.corpus == c]["id"]))
        if "use_for_binary" in m:
            keep &= m["use_for_binary"].fillna(True).astype(bool)
        m = m.assign(domain=DOMAIN_OF[c], row=np.arange(len(m)))[keep.values]
        metas.append(m)
        feats.append(np.asarray(x[m["row"].values]) if x is not None else None)
    meta = pd.concat(metas, ignore_index=True)
    return (np.concatenate(feats) if feats[0] is not None else None), meta


def boot_auc_ci(y, s, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], s[i]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def op_points(y, s, thr_from_source=None):
    """Sens@70% spec and spec@90% sens on the target itself, plus (if `thr_from_source`) the target
    sens/spec at a threshold fixed on the held-in data for 90% sensitivity (threshold transfer)."""
    pos, neg = s[y == 1], s[y == 0]
    t70 = np.quantile(neg, 0.70)
    sens70 = float((pos > t70).mean())
    t90 = np.quantile(pos, 0.10)
    spec90 = float((neg < t90).mean())
    out = dict(sens_at_spec70=sens70, spec_at_sens90=spec90)
    if thr_from_source is not None:
        out["transfer_sens"] = float((pos >= thr_from_source).mean())
        out["transfer_spec"] = float((neg < thr_from_source).mean())
    return out
