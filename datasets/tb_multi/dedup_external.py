"""Near-duplicate check of ONE external corpus against every other standardized corpus and against itself.

Same pHash + 64x64 correlation test as `dedup.py`, but (i) only pairs involving the external corpus are searched,
(ii) the source corpora are never touched: a duplicate is resolved by dropping the EXTERNAL copy, and
(iii) a positive control plants perturbed copies of random source images into the external hash set and reports
how many the detector recovers (an empty result is only meaningful if the control passes).

Usage:
    python -m datasets.tb_multi.dedup_external --corpus vindr [--append-drop]
Writes results/<corpus>_dedup_pairs.csv and, with --append-drop, adds the external-corpus drops to
results/tb_multi_dedup_drop.csv (rows for other corpora are left unchanged).
"""
import argparse

import numpy as np
import pandas as pd
from PIL import Image

from datasets.tb_multi.dedup import _thumb, find_pairs, phash
from datasets.tb_multi.standardize import load_corpus

SOURCES_ALL = ["qatar", "montgomery", "shenzhen", "tbx11k", "pakistan", "cidrz", "india"]


def perturb(img, rng):
    """Resize round-trip + brightness/contrast shift + noise: what re-encoding a copy of a radiograph might do."""
    s = int(rng.integers(150, 200))
    im = Image.fromarray(img).resize((s, s), Image.BILINEAR).resize(img.shape[::-1], Image.BILINEAR)
    a = np.asarray(im, np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12) + rng.normal(0, 2, img.shape)
    return np.clip(a, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="vindr")
    ap.add_argument("--max-hamming", type=int, default=8)
    ap.add_argument("--min-corr", type=float, default=0.97)
    ap.add_argument("--n-control", type=int, default=100)
    ap.add_argument("--append-drop", action="store_true")
    args = ap.parse_args()

    ext_img, ext_meta = load_corpus(args.corpus)
    eh, et = [phash(x) for x in ext_img], [_thumb(x) for x in ext_img]
    rows = []
    rng = np.random.default_rng(0)
    ctrl_found = ctrl_total = 0
    for c in SOURCES_ALL:
        img, meta = load_corpus(c)
        h, t = [phash(x) for x in img], [_thumb(x) for x in img]
        pairs = find_pairs(eh, et, h, t, args.max_hamming, args.min_corr)
        for i, j, d, corr in pairs:
            rows.append(dict(corpus_a=args.corpus, id_a=ext_meta.iloc[i]["id"], label_a=ext_meta.iloc[i]["label"], corpus_b=c,
                             id_b=meta.iloc[j]["id"], label_b=meta.iloc[j]["label"], hamming=d, corr=corr))
        print(f"{args.corpus} x {c}: {len(pairs)} near-duplicate pairs", flush=True)
        # positive control: perturbed copies of random images of this corpus must be found against it
        k = min(args.n_control // len(SOURCES_ALL) + 1, len(img))
        pick = rng.choice(len(img), k, replace=False)
        pert = [perturb(img[p], rng) for p in pick]
        ch, ct = [phash(x) for x in pert], [_thumb(x) for x in pert]
        found = {(i, j) for i, j, *_ in find_pairs(ch, ct, h, t, args.max_hamming, args.min_corr)}
        ctrl_found += sum((n, int(p)) in found for n, p in enumerate(pick))
        ctrl_total += k
    same = find_pairs(eh, et, eh, et, args.max_hamming, args.min_corr, same=True)
    for i, j, d, corr in same:
        rows.append(dict(corpus_a=args.corpus, id_a=ext_meta.iloc[i]["id"], label_a=ext_meta.iloc[i]["label"], corpus_b=args.corpus,
                         id_b=ext_meta.iloc[j]["id"], label_b=ext_meta.iloc[j]["label"], hamming=d, corr=corr))
    print(f"{args.corpus} x {args.corpus}: {len(same)} within-corpus pairs")
    print(f"positive control: {ctrl_found}/{ctrl_total} perturbed copies recovered")
    df = pd.DataFrame(rows, columns=["corpus_a", "id_a", "label_a", "corpus_b", "id_b", "label_b", "hamming", "corr"])
    df.to_csv(f"results/{args.corpus}_dedup_pairs.csv", index=False)
    drops = sorted({(args.corpus, r["id_a"] if r["corpus_b"] != args.corpus else r["id_b"]) for r in rows})
    print(f"external images to drop: {len(drops)}; label conflicts among pairs: {int((df.label_a != df.label_b).sum())}")
    if args.append_drop and drops:
        old = pd.read_csv("results/tb_multi_dedup_drop.csv")
        old = old[old.corpus != args.corpus]
        pd.concat([old, pd.DataFrame(drops, columns=["corpus", "id"])]).to_csv("results/tb_multi_dedup_drop.csv", index=False)
        print("appended to results/tb_multi_dedup_drop.csv")


if __name__ == "__main__":
    main()
