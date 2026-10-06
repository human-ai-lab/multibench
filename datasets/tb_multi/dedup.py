"""Near-duplicate detection across (and within) the standardized TB CXR corpora.

Qatar's "Tuberculosis" class is documented (datasets/tb_cxr_mont_shen/download.py) to be a pool
that includes NLM's Montgomery+Shenzhen images, so naive "cross-corpus" evaluation can score
the same radiograph in train and test. This finds those pairs.

Method: 64-bit DCT perceptual hash (pHash) of the 224x224 standardized image, candidate pairs at
Hamming distance <= `--max-hamming`, confirmed by Pearson correlation of 64x64 downsamples
>= `--min-corr` (pHash alone fires on structurally similar normal chests). Resizing/JPEG are
tolerated; heavy cropping/flipping are not (reported as a limitation).

Usage:
    python -m datasets.tb_multi.dedup --corpora qatar montgomery shenzhen tbx11k pakistan cidrz
"""
import argparse
import itertools

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

from datasets.tb_multi.standardize import load_corpus


def phash(img_u8: np.ndarray) -> int:
    small = np.asarray(Image.fromarray(img_u8).resize((32, 32), Image.LANCZOS), dtype=np.float32)
    low = dctn(small, norm="ortho")[:8, :8]
    bits = (low > np.median(low.flatten()[1:])).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _thumb(img_u8: np.ndarray) -> np.ndarray:
    t = np.asarray(Image.fromarray(img_u8).resize((64, 64), Image.BILINEAR), dtype=np.float32).flatten()
    t -= t.mean()
    return t / (np.linalg.norm(t) + 1e-8)


def _popcount_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(a[:, None], b[None, :])
    return np.unpackbits(x.view(np.uint8).reshape(len(a), len(b), 8), axis=2).sum(2)


def find_pairs(hashes_a, thumbs_a, hashes_b, thumbs_b, max_hamming, min_corr, same=False):
    ha, hb = np.array(hashes_a, dtype=np.uint64), np.array(hashes_b, dtype=np.uint64)
    out = []
    for s in range(0, len(ha), 500):
        d = _popcount_matrix(ha[s:s + 500], hb)
        ii, jj = np.nonzero(d <= max_hamming)
        for i, j in zip(ii + s, jj):
            if same and j <= i:
                continue
            c = float(thumbs_a[i] @ thumbs_b[j])
            if c >= min_corr:
                out.append((int(i), int(j), int(d[i - s, j]), c))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="+", default=["qatar", "montgomery", "shenzhen", "tbx11k", "pakistan", "cidrz"])
    ap.add_argument("--max-hamming", type=int, default=8)
    ap.add_argument("--min-corr", type=float, default=0.97)
    ap.add_argument("--out", default="results/tb_multi_dedup_pairs.csv")
    args = ap.parse_args()

    info = {}
    for c in args.corpora:
        img, meta = load_corpus(c)
        info[c] = (meta, [phash(x) for x in img], [_thumb(x) for x in img])
        print(f"{c}: hashed {len(img)}", flush=True)

    rows = []
    for a, b in itertools.combinations_with_replacement(args.corpora, 2):
        pairs = find_pairs(info[a][1], info[a][2], info[b][1], info[b][2], args.max_hamming, args.min_corr, same=(a == b))
        for i, j, d, corr in pairs:
            ma, mb = info[a][0].iloc[i], info[b][0].iloc[j]
            rows.append(dict(corpus_a=a, id_a=ma["id"], label_a=ma["label"], corpus_b=b, id_b=mb["id"],
                             label_b=mb["label"], hamming=d, corr=corr))
        print(f"  {a} x {b}: {len(pairs)} near-duplicate pairs", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    # Drop list: within a corpus drop the later copy; across corpora drop the copy in the corpus listed
    # later in `--corpora` (so earlier = higher priority; CIDRZ, the external target, is never dropped).
    drops = set()
    for r in rows:
        drops.add((r["corpus_a"], r["id_a"]) if r["corpus_b"] == "cidrz" else (r["corpus_b"], r["id_b"]))
    pd.DataFrame(sorted(drops), columns=["corpus", "id"]).to_csv(args.out.replace("_pairs", "_drop"), index=False)
    print(f"drop list: {len(drops)} images -> {args.out.replace('_pairs', '_drop')}")
    print(f"wrote {args.out} ({len(df)} pairs)")
    if len(df):
        print(df.groupby(["corpus_a", "corpus_b"]).agg(pairs=("corr", "size"),
              label_conflicts=("label_a", lambda s: int((s.values != df.loc[s.index, "label_b"].values).sum()))))


if __name__ == "__main__":
    main()
