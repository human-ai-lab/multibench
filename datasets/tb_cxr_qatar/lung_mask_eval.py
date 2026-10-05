"""Does masking out everything but the lungs narrow the Qatar -> CIDRZ cross-dataset gap?

Compares two arms of Qatar-trained VGG11Slim image models, several training seeds each:
- whole: trained and scored on whole images (configs/tb_cxr_qatar.yaml)
- lungmask: trained and scored on lung-masked images (configs/tb_cxr_qatar_lungmask.yaml)
Each arm is scored on its own matching preprocessing, so the comparison isolates masking.

Per model it reports:
- Qatar test AUROC (in-domain). A drop here together with a gain on CIDRZ is the signature of
  a removed shortcut; a drop on both points at bad segmentation instead.
- CIDRZ AUROC on all 364 participants (none used for training), zero-shot: eval-mode logit
  margin, no adaptation.
- CIDRZ AUROC with label-free adaptation: out-of-fold BN-only AdaBN + 4-view TTA logit margin,
  the same protocol as multimodal_late_fusion.image_expert_scores.
Arms are compared on the seed-averaged z-scored margins with a paired participant bootstrap.

Usage:
    python -m datasets.tb_cxr_qatar.lung_mask_eval \
        --whole results/models/tb_cxr_qatar_whole_s0.pt ... \
        --lungmask results/models/tb_cxr_qatar_lungmask_s0.pt ...
"""
import argparse
import pickle

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from datasets.tb_cxr_qatar.multimodal_late_fusion import image_expert_scores
from utils.device import get_device

ARM_DATA = {
    "whole": ("data/tb_cxr_qatar/tb_cxr_qatar.pkl", "data/google_health/tb_dataset.pkl"),
    "lungmask": ("data/tb_cxr_qatar/tb_cxr_qatar_lungmask.pkl", "data/google_health/tb_dataset_lungmask.pkl"),
}


def _load(path: str, splits):
    with open(path, "rb") as f:
        data = pickle.load(f)
    images = np.concatenate([data[s]["image"] for s in splits]).astype(np.float32)
    labels = np.concatenate([data[s]["label"] for s in splits]).astype(np.int64)
    return images, labels


def plain_margin(model, images: np.ndarray, device, batch_size: int = 16) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            logits = model([torch.from_numpy(images[i:i + batch_size]).to(device)])
            out.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    return np.concatenate(out)


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


def _paired_bootstrap(y, a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return np.percentile(diffs, [2.5, 97.5])


def score_arm(arm: str, model_paths, device):
    qatar_path, cidrz_path = ARM_DATA[arm]
    q_img, q_y = _load(qatar_path, ("test",))
    c_img, c_y = _load(cidrz_path, ("train", "valid", "test"))
    rows, zero_shot, adapted = [], [], []
    for path in model_paths:
        model = torch.load(path, weights_only=False).to(device)
        q = plain_margin(model, q_img, device)
        c0 = plain_margin(model, c_img, device)
        print(f"  [{arm}] {path}: adapting...", flush=True)
        c1 = image_expert_scores(path, c_img, device)
        rows.append({
            "model": path,
            "qatar_auc": roc_auc_score(q_y, q),
            "qatar_uar": balanced_accuracy_score(q_y, q > 0),
            "cidrz_auc": roc_auc_score(c_y, c0),
            "cidrz_uar": balanced_accuracy_score(c_y, c0 > 0),
            "cidrz_adapted_auc": roc_auc_score(c_y, c1),
        })
        zero_shot.append(_z(c0))
        adapted.append(_z(c1))
    return rows, c_y, np.mean(zero_shot, axis=0), np.mean(adapted, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whole", nargs="+", required=True, help="Model paths for the whole-image arm.")
    parser.add_argument("--lungmask", nargs="+", required=True, help="Model paths for the lung-masked arm.")
    args = parser.parse_args()

    device = get_device()
    results = {}
    for arm, paths in (("whole", args.whole), ("lungmask", args.lungmask)):
        results[arm] = score_arm(arm, paths, device)

    cols = ("qatar_auc", "qatar_uar", "cidrz_auc", "cidrz_uar", "cidrz_adapted_auc")
    print(f"\n{'model':58s} " + " ".join(f"{c:>17s}" for c in cols))
    for arm, (rows, *_rest) in results.items():
        for r in rows:
            print(f"{r['model']:58s} " + " ".join(f"{r[c]:17.3f}" for c in cols))
        means = [np.mean([r[c] for r in rows]) for c in cols]
        sds = [np.std([r[c] for r in rows]) for c in cols]
        print(f"{arm + ' mean +/- sd (n=' + str(len(rows)) + ')':58s} "
              + " ".join(f"{m:10.3f}+/-{s:.3f}" for m, s in zip(means, sds)))

    y = results["whole"][1]
    assert (y == results["lungmask"][1]).all(), "CIDRZ label order must match across arms"
    print("\nSeed-averaged CIDRZ scores, lungmask minus whole (paired participant bootstrap, n=364):")
    for name, k in (("zero-shot", 2), ("AdaBN + TTA", 3)):
        a, b = results["lungmask"][k], results["whole"][k]
        lo, hi = _paired_bootstrap(y, a, b)
        print(f"  {name:12s} AUROC whole={roc_auc_score(y, b):.3f}  lungmask={roc_auc_score(y, a):.3f}  "
              f"diff={roc_auc_score(y, a) - roc_auc_score(y, b):+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
