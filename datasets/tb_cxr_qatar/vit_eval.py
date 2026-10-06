"""Vision Transformer (ImageNet ViT-B/16) counterpart to the VGG11Slim Qatar -> CIDRZ experiments.

Same protocol as `lung_mask_eval.py`: train on Qatar (whole or lung-masked images), several seeds,
select the checkpoint on the Qatar valid split, then score all 364 CIDRZ participants zero-shot
(none used for training). The ViT has no BatchNorm, so the AdaBN adaptation arm does not apply.
Optionally compares against existing VGG11Slim checkpoints with a paired participant bootstrap.

Backbone: torchvision vit_b_16 (ImageNet-1k), frozen except the last `--unfreeze-blocks`
transformer blocks + final LayerNorm (low LR) and a new linear head (higher LR).

Usage:
    python -m datasets.tb_cxr_qatar.vit_eval --seeds 0 1 2 \
        --vgg-whole results/models/tb_cxr_qatar_whole_s{0,1,2}.pt \
        --vgg-lungmask results/models/tb_cxr_qatar_lungmask_s{0,1,2}.pt
"""
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torchvision.models import ViT_B_16_Weights, vit_b_16

from datasets.tb_cxr_qatar.lung_mask_eval import _load, _paired_bootstrap, _z, plain_margin
from utils.device import get_device

ARM_DATA = {
    "whole": ("data/tb_cxr_qatar/tb_cxr_qatar.pkl", "data/google_health/tb_dataset.pkl"),
    "lungmask": ("data/tb_cxr_qatar/tb_cxr_qatar_lungmask.pkl", "data/google_health/tb_dataset_lungmask.pkl"),
}


class ViTClassifier(nn.Module):
    """ViT-B/16 + linear head; takes `[images]` like the other models in this directory."""

    def __init__(self, num_classes: int = 2, unfreeze_blocks: int = 2, pretrained: bool = True):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.vit.heads = nn.Linear(self.vit.hidden_dim, num_classes)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.backbone_params = []
        for m in list(self.vit.encoder.layers)[len(self.vit.encoder.layers) - unfreeze_blocks:] + [self.vit.encoder.ln]:
            for p in m.parameters():
                p.requires_grad = True
                self.backbone_params.append(p)
        self.head_params = list(self.vit.heads.parameters())

    def forward(self, inputs):
        return self.vit(inputs[0])


def train_one(arm, seed, args, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    qpath, _ = ARM_DATA[arm]
    tr_x, tr_y = _load(qpath, ("train",))
    va_x, va_y = _load(qpath, ("valid",))
    model = ViTClassifier(unfreeze_blocks=args.unfreeze_blocks).to(device)
    opt = torch.optim.AdamW([
        {"params": model.backbone_params, "lr": args.backbone_lr},
        {"params": model.head_params, "lr": args.head_lr},
    ], weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.6, 3.0], device=device))
    save = f"results/models/tb_cxr_qatar_vit{args.tag}_{arm}_s{seed}.pt"
    best = -1.0
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(tr_x))
        tot = 0.0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            x = torch.from_numpy(tr_x[idx]).to(device); y = torch.from_numpy(tr_y[idx]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = loss_fn(model([x]).float(), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        va = plain_margin(model, va_x, device, 64)
        uar = balanced_accuracy_score(va_y, va > 0)
        print(f"  [{arm} s{seed}] epoch {ep} train_loss={tot / len(tr_x):.4f} valid_uar={uar:.4f}", flush=True)
        if uar > best:
            best = uar
            torch.save(model, save)
    return save


def score(arm, path, device):
    qpath, cpath = ARM_DATA[arm]
    q_x, q_y = _load(qpath, ("test",))
    c_x, c_y = _load(cpath, ("train", "valid", "test"))
    model = torch.load(path, weights_only=False).to(device)
    q = plain_margin(model, q_x, device, 32)
    c = plain_margin(model, c_x, device, 32)
    return dict(model=path, qatar_auc=roc_auc_score(q_y, q), qatar_uar=balanced_accuracy_score(q_y, q > 0),
                cidrz_auc=roc_auc_score(c_y, c), cidrz_uar=balanced_accuracy_score(c_y, c > 0)), c_y, c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--unfreeze-blocks", type=int, default=2)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--vgg-whole", nargs="*", default=[])
    ap.add_argument("--vgg-lungmask", nargs="*", default=[])
    ap.add_argument("--tag", default="", help="Suffix for checkpoint/score filenames, e.g. _u6_lr5e-5.")
    ap.add_argument("--skip-train", action="store_true", help="Reuse existing results/models/tb_cxr_qatar_vit_*.pt")
    args = ap.parse_args()
    device = get_device()

    out = {}  # (family, arm) -> (rows, y, seed-avg z-scored zero-shot margin)
    for arm in ("whole", "lungmask"):
        paths = [f"results/models/tb_cxr_qatar_vit{args.tag}_{arm}_s{s}.pt" if args.skip_train else train_one(arm, s, args, device)
                 for s in args.seeds]
        res = [score(arm, p, device) for p in paths]
        np.save(f"results/cidrz_image_scores_vit{args.tag}_{arm}_zeroshot_seedavg.npy",
                np.mean([_z(r[2]) for r in res], axis=0))
        out[("vit", arm)] = ([r[0] for r in res], res[0][1], np.mean([_z(r[2]) for r in res], axis=0))
    for arm, paths in (("whole", args.vgg_whole), ("lungmask", args.vgg_lungmask)):
        if paths:
            res = [score(arm, p, device) for p in paths]
            out[("vgg", arm)] = ([r[0] for r in res], res[0][1], np.mean([_z(r[2]) for r in res], axis=0))

    cols = ("qatar_auc", "qatar_uar", "cidrz_auc", "cidrz_uar")
    print(f"\n{'model':58s} " + " ".join(f"{c:>15s}" for c in cols))
    for (fam, arm), (rows, _, _) in out.items():
        for r in rows:
            print(f"{r['model']:58s} " + " ".join(f"{r[c]:15.3f}" for c in cols))
        print(f"{fam + ' ' + arm + ' mean +/- sd (n=' + str(len(rows)) + ')':58s} "
              + " ".join(f"{np.mean([r[c] for r in rows]):8.3f}+/-{np.std([r[c] for r in rows]):.3f}" for c in cols))

    print("\nSeed-averaged zero-shot CIDRZ AUROC (n=364), paired participant bootstrap:")
    pairs = [(("vit", "lungmask"), ("vit", "whole"))]
    if ("vgg", "whole") in out:
        pairs += [(("vit", "whole"), ("vgg", "whole")), (("vit", "lungmask"), ("vgg", "lungmask")),
                  (("vit", "lungmask"), ("vgg", "lungmask"))][:2]
    for a, b in pairs:
        y = out[a][1]
        assert (y == out[b][1]).all()
        lo, hi = _paired_bootstrap(y, out[a][2], out[b][2])
        aa, bb = roc_auc_score(y, out[a][2]), roc_auc_score(y, out[b][2])
        print(f"  {a[0]} {a[1]} ({aa:.3f}) minus {b[0]} {b[1]} ({bb:.3f}): {aa - bb:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
