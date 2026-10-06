"""Fine-tuned leave-one-corpus-out (LOCO) TB classifier with domain-generalization options.

All images live on the GPU as uint8 (standardize.py output), augmentation is done on-GPU. Backbone:
torchvision ViT-B/16 (ImageNet), last `--unfreeze-blocks` blocks + final LN trainable. Domains: qatar, nlm,
tbx11k (active TB vs healthy + sick-non-TB), pakistan; `--target` is held out entirely (`cidrz` = train on
all four, i.e. the external real-world target). 10% of every source is held in for checkpoint selection and
for the 90%-sensitivity threshold; the target is never used for selection.

Method knobs (compose freely):
  --sampler random|sbal    sbal: every (source, class) cell equally likely (source can't predict label)
  --loss ce|groupdro|vrex|coral|dann
  --aug none|strong        strong: affine, intensity/gamma, resolution/blur jitter, border/corner erasing
  --input whole|lung       lung: outside-lung zeroed (masks.py)
  --dstd                   per-domain feature standardization (train: within source in batch; test: whole target)
Writes results/loco/<name>_<target>_s<seed>.json and raw target scores .npy.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torchvision.models import ViT_B_16_Weights, vit_b_16

from datasets.tb_multi.common import CORPORA, SOURCES, boot_auc_ci, load_all, op_points
from datasets.tb_multi.standardize import OUT_DIR
from utils.device import get_device

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class Net(nn.Module):
    """Backbone (vit | xrv_densenet | resnet50 | raddino) + linear heads. Only the last stage(s) train; frozen
    parts are kept in eval mode so BatchNorm statistics do not drift. `features` returns (B, dim); ViT-style
    backbones can also return patch tokens (for the lesion auxiliary head)."""

    def __init__(self, backbone, unfreeze_blocks, n_dom):
        super().__init__()
        self.kind = backbone
        self.backbone_params, self.train_mods = [], []
        if backbone == "vit":
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            self.vit.heads = nn.Identity()
            body, tail, self.dim = self.vit, list(self.vit.encoder.layers)[-unfreeze_blocks:] + [self.vit.encoder.ln], 768
        elif backbone == "raddino":
            from transformers import AutoModel
            self.vit = AutoModel.from_pretrained("microsoft/rad-dino")
            body, tail, self.dim = self.vit, list(self.vit.encoder.layer)[-unfreeze_blocks:] + [self.vit.layernorm], 768
        elif backbone == "resnet50":
            from torchvision.models import ResNet50_Weights, resnet50
            self.cnn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.cnn.fc = nn.Identity()
            body, tail, self.dim = self.cnn, [self.cnn.layer4], 2048
        elif backbone == "xrv_densenet":
            import torchxrayvision as xrv
            self.cnn = xrv.models.DenseNet(weights="densenet121-res224-all")
            body = self.cnn
            tail, self.dim = [self.cnn.features.denseblock4, self.cnn.features.norm5], 1024
        else:
            raise ValueError(backbone)
        for p in body.parameters():
            p.requires_grad = False
        for m in tail:
            for p in m.parameters():
                p.requires_grad = True
                self.backbone_params.append(p)
        self.tail_mods = tail
        self.head = nn.Linear(self.dim, 2)
        self.dom_head = nn.Linear(self.dim, n_dom)
        self.patch_head = nn.Linear(self.dim, 1)  # lesion auxiliary head (ViT-style backbones only)

    def train(self, mode=True):
        super().train(mode)
        if mode:  # frozen parts stay in eval mode; only the unfrozen tail trains/updates BN
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.Dropout)):
                    m.eval()
            for m in self.tail_mods:
                m.train()
        return self

    def features(self, x, tokens=False):  # x in [0,1], (B,1,H,W)
        x = x.expand(-1, 3, -1, -1)
        if self.kind == "vit":
            x = (x - MEAN.to(x.device)) / STD.to(x.device)
            v = self.vit
            t = v._process_input(x)
            t = torch.cat([v.class_token.expand(t.shape[0], -1, -1), t], 1)
            t = v.encoder(t)
            return (t[:, 0], t[:, 1:]) if tokens else t[:, 0]
        if self.kind == "raddino":
            x = (x - 0.5307) / 0.2583
            t = self.vit(pixel_values=x).last_hidden_state
            return (t[:, 0], t[:, 1:]) if tokens else t[:, 0]
        if self.kind == "resnet50":
            return self.cnn((x - MEAN.to(x.device)) / STD.to(x.device))
        f = self.cnn.features(((2 * x[:, :1] - 1) * 1024))  # xrv: single channel in [-1024, 1024]
        return F.adaptive_avg_pool2d(F.relu(f), 1).flatten(1)


def augment(x, strong, mask=None):
    """x: (B,1,224,224) float [0,1] on GPU. `mask` (B,1,224,224) lesion map gets the same geometric transform.
    Returns x, or (x, mask) when a mask is given."""
    if not strong:
        return x if mask is None else (x, mask)
    B = x.shape[0]
    dev = x.device
    ang = (torch.rand(B, device=dev) - 0.5) * 2 * np.deg2rad(8)
    sc = 1 + (torch.rand(B, device=dev) - 0.5) * 0.2
    tx, ty = (torch.rand(B, device=dev) - 0.5) * 0.12, (torch.rand(B, device=dev) - 0.5) * 0.12
    theta = torch.stack([torch.stack([torch.cos(ang) / sc, -torch.sin(ang) / sc, tx], 1),
                         torch.stack([torch.sin(ang) / sc, torch.cos(ang) / sc, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    if mask is not None:
        mask = F.grid_sample(mask, grid, padding_mode="zeros", align_corners=False)
    gamma = torch.exp((torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.8)
    x = x.clamp(1e-4, 1) ** gamma
    c = 1 + (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.6
    b = (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.2
    x = ((x - 0.5) * c + 0.5 + b).clamp(0, 1)
    # resolution jitter (harmonizes 256px-JPEG-like vs 512+ sources): random down-up for ~half the batch
    r = int(np.random.choice([112, 160, 224]))
    if r < 224:
        half = torch.rand(B, device=dev) < 0.5
        low = F.interpolate(F.interpolate(x, size=(r, r), mode="bilinear", align_corners=False), size=(224, 224),
                            mode="bilinear", align_corners=False)
        x = torch.where(half.view(B, 1, 1, 1), low, x)
    # erase a random border/corner patch (annotation tags, collimation, laterality markers)
    for i in range(B):
        if random.random() < 0.5:
            h, w = random.randint(12, 48), random.randint(12, 48)
            y0 = random.choice([0, 224 - h]) if random.random() < 0.7 else random.randint(0, 224 - h)
            x0 = random.choice([0, 224 - w]) if random.random() < 0.7 else random.randint(0, 224 - w)
            x[i, :, y0:y0 + h, x0:x0 + w] = random.random()
    return x if mask is None else (x, mask)


def coral(f, d, n_dom):
    cs = []
    for k in range(n_dom):
        fk = f[d == k]
        if len(fk) > 2:
            cs.append((fk.mean(0), torch.cov(fk.T)))
    loss = f.new_zeros(())
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            loss = loss + ((cs[i][0] - cs[j][0]) ** 2).mean() + ((cs[i][1] - cs[j][1]) ** 2).mean()
    return loss


def dom_standardize(f, d, n_dom):
    out = f.clone()
    for k in range(n_dom):
        i = d == k
        if i.sum() > 3:
            out[i] = (f[i] - f[i].mean(0)) / (f[i].std(0) + 1e-5)
    return out


@torch.no_grad()
def embed_all(net, X, bs=128):
    net.eval()
    out = []
    for i in range(0, len(X), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(net.features(X[i:i + bs].float().div(255).unsqueeze(1)).float())
    return torch.cat(out)


def margins(net, feats, dstd):
    if dstd:
        feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-5)
    lg = net.head(feats)
    return (lg[:, 1] - lg[:, 0]).detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, choices=SOURCES + ["cidrz"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sampler", default="sbal", choices=["random", "sbal"])
    ap.add_argument("--loss", default="ce", choices=["ce", "groupdro", "vrex", "coral", "dann"])
    ap.add_argument("--aug", default="none", choices=["none", "strong"])
    ap.add_argument("--input", default="whole", choices=["whole", "lung"])
    ap.add_argument("--dstd", action="store_true")
    ap.add_argument("--backbone", default="vit", choices=["vit", "raddino", "resnet50", "xrv_densenet"])
    ap.add_argument("--lesion", type=float, default=0.0,
                    help="weight of the TBX11K lesion-box auxiliary patch loss (ViT-style backbones; 0 = off)")
    ap.add_argument("--unfreeze-blocks", type=int, default=6)
    ap.add_argument("--backbone-lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--reg", type=float, default=1.0, help="vrex beta / coral & dann weight")
    ap.add_argument("--outdir", default="results/loco")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = get_device()

    _, meta = load_all(None)
    suffix = "lung" if args.input == "lung" else ""
    arrs = [np.load(os.path.join(OUT_DIR, f"{c}{'_' + suffix if suffix else ''}.npy"), mmap_mode="r")
            [meta[meta.corpus == c].row.values] for c in CORPORA]
    X = torch.from_numpy(np.concatenate(arrs)).to(dev)
    y = torch.from_numpy(meta.label.values).long().to(dev)
    dom_names = meta.domain.values
    srcs = [s for s in SOURCES if s != args.target]
    dom = torch.tensor([srcs.index(d) if d in srcs else -1 for d in dom_names], device=dev)
    n_dom = len(srcs)

    # held-in train / val split (10% of every source, stratified, fixed by seed 0 so all methods share it)
    rng = np.random.default_rng(0)
    is_tr = np.zeros(len(meta), bool); is_va = np.zeros(len(meta), bool)
    for s in srcs:
        for c in (0, 1):
            idx = np.flatnonzero((dom_names == s) & (meta.label.values == c))
            rng.shuffle(idx)
            k = max(1, int(0.1 * len(idx)))
            is_va[idx[:k]] = True; is_tr[idx[k:]] = True
    tr_idx = torch.from_numpy(np.flatnonzero(is_tr)).to(dev)
    va_idx = np.flatnonzero(is_va)
    te_idx = np.flatnonzero(dom_names == args.target)

    row2box, box_masks = None, None
    if args.lesion > 0 and args.target != "tbx11k" and "tbx11k" in srcs:
        tb_rows = np.flatnonzero(dom_names == "tbx11k")
        tb_meta = meta.iloc[tb_rows]
        # boxes are stored as fractions of the original image: [class, x0, y0, x1, y1]
        bm = np.zeros((len(tb_rows), 224, 224), np.uint8)
        for k, bx in enumerate(tb_meta["boxes"].values):
            for b in json.loads(bx):
                if b[0] != "ActiveTuberculosis":
                    continue
                x0, y0, x1, y1 = [int(round(v * 224)) for v in b[1:]]
                bm[k, max(y0, 0):y1, max(x0, 0):x1] = 1
        box_masks = torch.from_numpy(bm).to(dev)
        row2box = torch.full((len(meta),), -1, dtype=torch.long, device=dev)
        row2box[torch.from_numpy(tb_rows).to(dev)] = torch.arange(len(tb_rows), device=dev)
        print(f"lesion supervision: {int((bm.sum((1, 2)) > 0).sum())} TBX11K images with active-TB boxes", flush=True)

    if args.sampler == "sbal":
        w = torch.zeros(len(tr_idx), device=dev)
        for k in range(n_dom):
            for c in (0, 1):
                m = (dom[tr_idx] == k) & (y[tr_idx] == c)
                w[m] = 1.0 / m.sum()
    else:
        w = torch.ones(len(tr_idx), device=dev)
    cls_w = torch.tensor([1.0, 1.0], device=dev) if args.sampler == "sbal" else torch.tensor(
        [1.0, float((y[tr_idx] == 0).sum() / (y[tr_idx] == 1).sum())], device=dev).clamp(max=6)

    net = Net(args.backbone, args.unfreeze_blocks, n_dom).to(dev)
    params = [{"params": net.backbone_params, "lr": args.backbone_lr},
              {"params": list(net.head.parameters()) + list(net.dom_head.parameters())
                         + list(net.patch_head.parameters()), "lr": args.head_lr}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    n_cell = n_dom * 2
    q = torch.ones(n_cell, device=dev) / n_cell  # groupdro weights
    best, best_state, t0 = -1, None, time.time()
    total = args.epochs * args.steps_per_epoch
    step = 0
    for ep in range(args.epochs):
        net.train()
        for _ in range(args.steps_per_epoch):
            pick = tr_idx[torch.multinomial(w, args.batch_size, replacement=True)]
            xb = X[pick].float().div(255).unsqueeze(1)
            yb, db = y[pick], dom[pick]
            lesion_loss = None
            if row2box is not None:
                bi = row2box[pick]
                has = bi >= 0
                mb = torch.where(has.view(-1, 1, 1), box_masks[bi.clamp(min=0)].float(), torch.zeros_like(xb[:, 0])).unsqueeze(1)
                xb, mb = augment(xb, args.aug == "strong", mb)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f, tok = net.features(xb, tokens=True)
                f = f.float()
                if has.any():
                    g = int(tok.shape[1] ** 0.5)
                    tgt = F.adaptive_avg_pool2d(mb, g).flatten(1)
                    lp = net.patch_head(tok.float()).squeeze(-1)
                    lesion_loss = F.binary_cross_entropy_with_logits(lp[has], tgt[has], pos_weight=torch.tensor(5.0, device=dev))
            else:
                xb = augment(xb, args.aug == "strong")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f = net.features(xb).float()
            fz = dom_standardize(f, db, n_dom) if args.dstd else f
            logits = net.head(fz)
            ce = F.cross_entropy(logits, yb, weight=cls_w, reduction="none")
            if args.loss == "groupdro":
                cell = db * 2 + yb
                risks = torch.stack([ce[cell == k].mean() if (cell == k).any() else ce.new_zeros(()) for k in range(n_cell)])
                present = torch.stack([(cell == k).any() for k in range(n_cell)])
                q = q * torch.exp(0.01 * risks.detach() * present)
                q = q / q.sum()
                loss = (q * risks).sum()
            elif args.loss == "vrex":
                risks = torch.stack([ce[db == k].mean() for k in range(n_dom) if (db == k).any()])
                beta = args.reg * 10 if step > args.steps_per_epoch else 1.0
                loss = risks.mean() + beta * risks.var() if len(risks) > 1 else risks.mean()
            elif args.loss == "coral":
                loss = ce.mean() + args.reg * coral(f, db, n_dom)
            elif args.loss == "dann":
                lam = args.reg * (2 / (1 + np.exp(-10 * step / total)) - 1)
                loss = ce.mean() + F.cross_entropy(net.dom_head(GRL.apply(f, lam)), db)
            else:
                loss = ce.mean()
            if lesion_loss is not None:
                loss = loss + args.lesion * lesion_loss
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
        # held-in validation: mean per-source AUROC (never touches the target)
        fv = embed_all(net, X[va_idx])
        dv = dom_names[va_idx]; yv = meta.label.values[va_idx]
        sv = np.zeros(len(va_idx))
        for s in srcs:
            i = dv == s
            sv[i] = margins(net, fv[torch.from_numpy(i).to(dev)], args.dstd)
        val = np.mean([roc_auc_score(yv[dv == s], sv[dv == s]) for s in srcs])
        print(f"[{args.name} {args.target} s{args.seed}] epoch {ep} loss {loss.item():.3f} val_mean_auc {val:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if val > best:
            best = val
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items() if "vit.encoder" in k or "head" in k}
    net.load_state_dict(best_state, strict=False)

    fv = embed_all(net, X[va_idx])
    dv = dom_names[va_idx]; yv = meta.label.values[va_idx]
    sv = np.zeros(len(va_idx))
    for s in srcs:
        i = dv == s
        sv[i] = margins(net, fv[torch.from_numpy(i).to(dev)], args.dstd)
    thr = float(np.quantile(sv[yv == 1], 0.10))  # pooled held-in val: 90% sensitivity
    ft = embed_all(net, X[te_idx])
    s = margins(net, ft, args.dstd)
    yt = meta.label.values[te_idx]
    extra = {}
    if args.lesion > 0 and args.target != "tbx11k":
        pm = []
        with torch.no_grad():
            for i in range(0, len(te_idx), 128):
                xb = X[te_idx[i:i + 128]].float().div(255).unsqueeze(1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, tok = net.features(xb, tokens=True)
                pm.append(net.patch_head(tok.float()).squeeze(-1).max(1).values.cpu().numpy())
        pm = np.concatenate(pm)
        zc = lambda v: (v - v.mean()) / (v.std() + 1e-8)
        extra = dict(auc_patchmax=float(roc_auc_score(yt, pm)), auc_comb=float(roc_auc_score(yt, zc(s) + zc(pm))))
        np.save(os.path.join(args.outdir, f"{args.name}_{args.target}_s{args.seed}_patchmax.npy"), pm.astype(np.float32))
    r = dict(extra, args=vars(args), val_mean_auc=best, auc=float(roc_auc_score(yt, s)), auc_ci=boot_auc_ci(yt, s),
             **op_points(yt, s, thr), n=int(len(yt)), pos=int(yt.sum()))
    if args.target == "tbx11k":
        cat = meta["category"].values[te_idx]
        for other in ("healthy", "sick"):
            k = (cat == other) | (yt == 1)
            r[f"auc_vs_{other}"] = float(roc_auc_score(yt[k], s[k]))
    tag = f"{args.name}_{args.target}_s{args.seed}"
    json.dump(r, open(os.path.join(args.outdir, tag + ".json"), "w"), indent=1, default=float)
    np.save(os.path.join(args.outdir, tag + "_scores.npy"), s.astype(np.float32))
    print(f"RESULT {tag} AUC {r['auc']:.3f} [{r['auc_ci'][0]:.3f},{r['auc_ci'][1]:.3f}] "
          f"sens@spec70 {r['sens_at_spec70']:.2f} spec@sens90 {r['spec_at_sens90']:.2f} "
          f"transfer sens {r['transfer_sens']:.2f} spec {r['transfer_spec']:.2f}", flush=True)


if __name__ == "__main__":
    main()
