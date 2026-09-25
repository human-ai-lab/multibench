"""Train Qatar TB CXR with the last VGG11-BN stage unfrozen, evaluate cross-dataset on CIDRZ.

Custom (not `training_structures.Supervised_Learning.train`-based) training loop: the
unfrozen backbone stage needs a much lower learning rate than the randomly-initialized
classifier head, which the shared `train()` helper's single global `lr` doesn't support.
Two parameter groups: `backbone_lr` for `VGG11SlimPartialUnfreeze.unfrozen_params`,
`head_lr` for everything else trainable (the MLP classifier). Checkpoint selection by
best validation accuracy (same convention as the rest of this project), early stopping
after `patience` epochs without improvement.

Usage:
    python -m datasets.tb_cxr_qatar.train_partial_unfreeze \
        --qatar-pkl data/tb_cxr_qatar/tb_cxr_qatar.pkl \
        --unfreeze-stages 1 --backbone-lr 1e-5 --head-lr 1e-3 --epochs 20 \
        --save results/models/tb_cxr_qatar_partial_unfreeze.pt
"""
import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from datasets.tb_cxr_qatar.cross_dataset_analysis import evaluate, print_result
from datasets.tb_cxr_qatar.partial_unfreeze import VGG11SlimPartialUnfreeze
from unimodals.common_models import MLP
from utils.device import get_device


class _Wrapped(nn.Module):
    """`torch.save` pickles by reference to a module-level class - a class defined inside
    `main()` fails with `AttributeError: Can't pickle local object 'main.<locals>._Wrapped'`."""

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, inputs):
        return self.head(self.encoder(inputs[0]))


def _load_split(path: str, split: str) -> TensorDataset:
    with open(path, "rb") as f:
        data = pickle.load(f)
    images = data[split]["image"].astype(np.float32)
    labels = data[split]["label"].astype(np.int64)
    return TensorDataset(torch.from_numpy(images), torch.from_numpy(labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qatar-pkl", default="data/tb_cxr_qatar/tb_cxr_qatar.pkl")
    parser.add_argument("--cidrz-path", default="data/google_health/tb_dataset.pkl")
    parser.add_argument("--save", default="results/models/tb_cxr_qatar_partial_unfreeze.pt")
    parser.add_argument("--unfreeze-stages", type=int, default=1)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pos-weight", type=float, default=3.0)
    args = parser.parse_args()

    device = get_device()
    encoder = VGG11SlimPartialUnfreeze(128, unfreeze_last_n_stages=args.unfreeze_stages).to(device)
    head = MLP(128, 64, 2).to(device)
    objective = nn.CrossEntropyLoss(weight=torch.tensor([0.6, args.pos_weight], device=device))

    backbone_params = list(encoder.unfrozen_params)
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in list(encoder.parameters()) + list(head.parameters())
                   if p.requires_grad and id(p) not in backbone_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

    train_loader = DataLoader(_load_split(args.qatar_pkl, "train"), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(_load_split(args.qatar_pkl, "valid"), batch_size=args.batch_size, shuffle=False)

    best_valid_acc, epochs_since_best, best_state = -1.0, 0, None
    for epoch in range(args.epochs):
        encoder.train()
        head.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = head(encoder(x))
            loss = objective(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(backbone_params + head_params, 8.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        encoder.eval()
        head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                pred = torch.argmax(head(encoder(x)), dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        valid_acc = correct / total
        print(f"epoch {epoch}: train_loss={total_loss / len(train_loader.dataset):.4f} valid_acc={valid_acc:.4f}")

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            epochs_since_best = 0
            best_state = {"encoder": encoder.state_dict(), "head": head.state_dict()}
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"early stopping at epoch {epoch} (best valid_acc={best_valid_acc:.4f})")
                break

    encoder.load_state_dict(best_state["encoder"])
    head.load_state_dict(best_state["head"])

    wrapped = _Wrapped(encoder, head).to(device)
    torch.save(wrapped, args.save)
    print(f"saved best checkpoint (valid_acc={best_valid_acc:.4f}) to {args.save}")

    r = evaluate(args.save, cidrz_path=args.cidrz_path, device=device)
    print_result(f"{args.save} (partial unfreeze, {args.unfreeze_stages} stage(s))", r)
    r_bn = evaluate(args.save, cidrz_path=args.cidrz_path, adapt_bn=True, device=device)
    print_result(f"{args.save} + AdaBN (partial unfreeze, {args.unfreeze_stages} stage(s))", r_bn)


if __name__ == "__main__":
    main()
