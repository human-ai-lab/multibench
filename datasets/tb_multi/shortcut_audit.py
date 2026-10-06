"""Leave-one-corpus-out AUROC of label-free-of-pathology shortcut features: lung-mask area, 28x28 lung
silhouette, and outer-border pixels only (corners/annotation tags/collimation; central 144x144 zeroed).
If these approach a real model's AUROC on a held-out corpus, that corpus's cross-corpus score is not
evidence of pathology detection."""
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from datasets.tb_multi.common import CORPORA, SOURCES, load_all
from datasets.tb_multi.standardize import OUT_DIR


def _stack(meta, suffix):
    return np.concatenate([np.load(f"{OUT_DIR}/{c}_{suffix}.npy", mmap_mode="r")[meta[meta.corpus == c].row.values]
                           for c in CORPORA])


def _small(a, v=255):
    return np.asarray(Image.fromarray(a.astype(np.uint8)).resize((28, 28), Image.BILINEAR), dtype=np.float32) / 255


def main():
    _, meta = load_all(None)
    y, dom = meta.label.values, meta.domain.values
    mask, gray = _stack(meta, "mask"), np.concatenate([np.load(f"{OUT_DIR}/{c}.npy", mmap_mode="r")[meta[meta.corpus == c].row.values] for c in CORPORA])
    area = mask.mean((1, 2))[:, None]
    sil = np.stack([_small(m * 255) for m in mask]).reshape(len(mask), -1)
    ring = np.ones((224, 224), bool)
    ring[40:184, 40:184] = False
    border = np.stack([_small(np.where(ring, g, 0)) for g in gray]).reshape(len(gray), -1)
    for name, F in (("mask-area only", area), ("silhouette 28x28", sil), ("outer-border pixels only", border)):
        res = []
        for t in SOURCES + ["cidrz"]:
            tr, te = np.isin(dom, [s for s in SOURCES if s != t]), dom == t
            sc = StandardScaler().fit(F[tr])
            m = LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced").fit(sc.transform(F[tr]), y[tr])
            res.append(roc_auc_score(y[te], m.decision_function(sc.transform(F[te]))))
        print(f"{name:28s} " + "  ".join(f"{t} {a:.3f}" for t, a in zip(SOURCES + ["cidrz"], res)))


if __name__ == "__main__":
    main()
