"""Rebuild the Qatar and CIDRZ image pickles with everything outside the lungs blacked out.

A whole-image classifier can latch onto non-lung cues that differ between hospitals - burnt-in
text and side markers, collimation edges, borders, patient positioning - which help within
Qatar but do not transfer to CIDRZ. Masking the input removes those cues regardless of the
encoder architecture.

Lungs are segmented with torchxrayvision's ChestX-Det PSPNet (14 anatomical structures). The
mask is the union of left/right lung and left/right hilus (hilar lymphadenopathy is a TB
finding a lungs-only mask would cut away), morphologically closed, hole-filled, then dilated
by `--dilate` pixels at 224x224 so pleural margins (effusion) stay visible. Outside the mask
each channel is set to the ImageNet-normalized value of raw black, (0 - mean) / std - a
literal 0 in normalized space would be mid-gray. Images whose mask covers less than
`--min-area` of the frame (a segmentation failure) are kept unmasked and counted.

Output pickles keep every other field (text, audio, label) and the sample order unchanged, so
CIDRZ's masked pickle is a drop-in replacement for downstream scripts.

Usage:
    python -m datasets.tb_cxr_qatar.lung_mask_preprocess \
        --qatar-out data/tb_cxr_qatar/tb_cxr_qatar_lungmask.pkl \
        --cidrz-out data/google_health/tb_dataset_lungmask.pkl \
        --overlay-dir results/images/lung_mask
"""
import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from datasets.google_health.features import IMAGENET_MEAN, IMAGENET_STD
from utils.device import get_device

MASK_STRUCTURES = ("Left Lung", "Right Lung", "Left Hilus Pulmonis", "Right Hilus Pulmonis")


def _to_gray01(images: np.ndarray) -> np.ndarray:
    """(N, 3, H, W) ImageNet-normalized, channel-replicated -> (N, H, W) in [0, 1]."""
    raw = images[:, 0] * IMAGENET_STD[0] + IMAGENET_MEAN[0]
    return np.clip(raw, 0.0, 1.0).astype(np.float32)


def segment_lungs(gray01: np.ndarray, device, batch_size: int = 8) -> np.ndarray:
    """(N, H, W) in [0, 1] -> (N, H, W) bool union of lung + hilus masks at input resolution."""
    import torchxrayvision as xrv

    model = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
    channels = [model.targets.index(name) for name in MASK_STRUCTURES]
    h, w = gray01.shape[1:]
    masks = np.zeros(gray01.shape, dtype=bool)
    with torch.no_grad():
        for i in range(0, len(gray01), batch_size):
            x = torch.from_numpy(gray01[i:i + batch_size])[:, None].to(device) * 2048 - 1024
            logits = model(x)[:, channels]  # (B, 4, 512, 512)
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            masks[i:i + batch_size] = (torch.sigmoid(logits) > 0.5).any(dim=1).cpu().numpy()
    return masks


def refine_mask(mask: np.ndarray, dilate: int) -> np.ndarray:
    mask = ndimage.binary_closing(mask, iterations=3)
    mask = ndimage.binary_fill_holes(mask)
    if dilate > 0:
        mask = ndimage.binary_dilation(mask, iterations=dilate)
    return mask


def apply_mask(images: np.ndarray, masks: np.ndarray) -> np.ndarray:
    black = ((0.0 - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)[None, :, None, None]
    return np.where(masks[:, None], images, black).astype(np.float32)


def _save_overlays(gray01: np.ndarray, masks: np.ndarray, labels: np.ndarray, path: str, n: int = 16):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    idx = np.concatenate([rng.choice(pos, min(n // 2, len(pos)), replace=False),
                          rng.choice(neg, min(n // 2, len(neg)), replace=False)])
    cols = 8
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.2 * rows))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(gray01[i], cmap="gray")
        ax.contour(masks[i], levels=[0.5], colors="lime", linewidths=0.8)
        ax.set_title(f"#{i} y={labels[i]} {masks[i].mean():.2f}", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _rebuild(in_path: str, out_path: str, name: str, args, device) -> None:
    with open(in_path, "rb") as f:
        data = pickle.load(f)
    rebuilt, all_gray, all_masks, all_labels = {}, [], [], []
    for split in ("train", "valid", "test"):
        images = data[split]["image"].astype(np.float32)
        gray01 = _to_gray01(images)
        raw = segment_lungs(gray01, device)
        masks = np.stack([refine_mask(m, args.dilate) for m in raw])
        failed = masks.mean(axis=(1, 2)) < args.min_area
        masks[failed] = True
        area = masks[~failed].mean(axis=(1, 2))
        print(f"  {name} [{split}]: n={len(images)}, mask area mean={area.mean():.3f} "
              f"(p5={np.percentile(area, 5):.3f}, p95={np.percentile(area, 95):.3f}), "
              f"segmentation failures kept unmasked: {int(failed.sum())}", flush=True)
        rebuilt[split] = {**data[split], "image": apply_mask(images, masks)}
        all_gray.append(gray01)
        all_masks.append(masks)
        all_labels.append(np.asarray(data[split]["label"]))
    with open(out_path, "wb") as f:
        pickle.dump(rebuilt, f)
    print(f"  saved -> {out_path}")
    if args.overlay_dir:
        os.makedirs(args.overlay_dir, exist_ok=True)
        path = os.path.join(args.overlay_dir, f"{name}_overlay.png")
        _save_overlays(np.concatenate(all_gray), np.concatenate(all_masks), np.concatenate(all_labels), path)
        print(f"  overlay grid -> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qatar-in", default="data/tb_cxr_qatar/tb_cxr_qatar.pkl")
    parser.add_argument("--qatar-out", default="data/tb_cxr_qatar/tb_cxr_qatar_lungmask.pkl")
    parser.add_argument("--cidrz-in", default="data/google_health/tb_dataset.pkl")
    parser.add_argument("--cidrz-out", default="data/google_health/tb_dataset_lungmask.pkl")
    parser.add_argument("--dilate", type=int, default=6, help="Mask dilation in pixels at 224x224.")
    parser.add_argument("--min-area", type=float, default=0.05,
                        help="Masks covering less than this fraction of the image count as failures.")
    parser.add_argument("--overlay-dir", default="results/images/lung_mask")
    args = parser.parse_args()

    device = get_device()
    print("Rebuilding CIDRZ pickle with lung masks...")
    _rebuild(args.cidrz_in, args.cidrz_out, "cidrz", args, device)
    print("Rebuilding Qatar pickle with lung masks...")
    _rebuild(args.qatar_in, args.qatar_out, "qatar", args, device)


if __name__ == "__main__":
    main()
