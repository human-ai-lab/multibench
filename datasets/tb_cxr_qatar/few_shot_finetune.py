"""Few-shot supervised domain adaptation: fine-tune the classifier head on a handful of
labeled CIDRZ examples, on top of the Qatar-trained (optionally AdaBN-adapted) backbone.

Unlike every other technique in this project so far, this one legitimately uses CIDRZ
labels - flagged clearly here and in every result this script prints. To keep the
evaluation honest:
  - Shots are sampled ONLY from CIDRZ's "train" split (254 images, 42 positive).
  - Scoring is ALWAYS on CIDRZ's "test" split (55 images, disjoint from the shot pool,
    never touched during fine-tuning).
  - A **control** condition is included at every (k, seed): a fresh head trained from
    random initialization on the identical k shots, on features from an UNTOUCHED
    ImageNet-pretrained VGG11 (no Qatar training anywhere in the pipeline - not even the
    Qatar-trained `Linear(25088->128)` projection the treatment's backbone uses) -
    isolating how much of any gain is genuine cross-dataset transfer vs. just having a
    few real CIDRZ labels plus generic ImageNet features.
  - k is swept (5/10/20 per class) over 5 random shot draws each; mean +/- std reported,
    since single draws at this sample size are noisy.
  - The backbone is frozen throughout (only the small MLP head is fine-tuned) and its
    features are cached once - fine-tuning a linear/MLP head on ~2000-dim cached features
    for 50 epochs takes seconds, not the ~13 min/epoch a full backbone pass costs.

Usage:
    python -m datasets.tb_cxr_qatar.few_shot_finetune \
        --model results/models/tb_cxr_qatar_augmented_to_cidrz.pt --adapt-bn
"""
import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from datasets.tb_cxr_qatar.cross_dataset_analysis import (
    _load_split, _qatar_source_prior, adapt_batchnorm, calibrate_threshold_to_prior,
)
from eval_scripts.performance import compute_metrics
from unimodals.common_models import MLP
from utils.device import get_device


def extract_features(encoder, images, device, batch_size=32):
    encoder.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            feats.append(encoder(images[i:i + batch_size].to(device)).cpu())
    return torch.cat(feats, dim=0)


def sample_k_shot(labels: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    chosen = np.concatenate([rng.choice(pos_idx, k, replace=False), rng.choice(neg_idx, k, replace=False)])
    rng.shuffle(chosen)
    return chosen


def train_head(head: nn.Module, features: torch.Tensor, labels: torch.Tensor, device,
                epochs: int = 50, lr: float = 1e-3, weight_decay: float = 0.01,
                class_weight=(1.0, 1.0)) -> nn.Module:
    """`class_weight` defaults to balanced (1, 1): the k-shot pool is itself class-balanced
    (k per class), unlike Qatar's ~17% positive rate the (0.6, 3.0) weighting elsewhere in
    this project was tuned for - reusing that weighting here would bias every prediction
    toward positive regardless of what the shots show."""
    head = head.to(device)
    features, labels = features.to(device), labels.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    obj = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, device=device))
    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = obj(head(features), labels)
        loss.backward()
        opt.step()
    head.eval()
    return head


def score_head(head: nn.Module, features: torch.Tensor, labels: np.ndarray, device, qatar_pkl: str):
    with torch.no_grad():
        probs = torch.softmax(head(features.to(device)), dim=1)[:, 1].cpu().numpy()
    default_preds = (probs > 0.5).astype(np.int64)
    default_metrics = compute_metrics(torch.tensor(labels), torch.tensor(default_preds), ["UA", "WA", "F1"])
    prior = _qatar_source_prior(qatar_pkl)
    t_star = calibrate_threshold_to_prior(probs, prior)
    calibrated_preds = (probs > t_star).astype(np.int64)
    calibrated_metrics = compute_metrics(torch.tensor(labels), torch.tensor(calibrated_preds), ["UA", "WA", "F1"])
    auroc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    return {"auroc": auroc, "default": default_metrics, "calibrated": calibrated_metrics}


def _fresh_imagenet_encoder(device):
    """A VGG11-BN encoder with NO Qatar training anywhere - not even the
    `Linear(25088->128)` projection `VGG11Slim` normally trains, which the treatment
    condition's `model.encoders[0]` does carry (that layer was fit on Qatar). Returns
    features straight from ImageNet-pretrained `avgpool` (25088-dim, flattened)."""
    import torchvision.models as tmodels

    backbone = tmodels.vgg11_bn(pretrained=True).to(device)
    backbone.eval()

    class _FeatureExtractor(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.features = backbone.features
            self.avgpool = backbone.avgpool

        def forward(self, x):
            # `adapt_batchnorm` (shared with the full MMDL model) calls `model([tensor])`,
            # matching MMDL's list-of-modalities convention; `extract_features` here calls
            # `encoder(tensor)` directly. Accept both.
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = self.features(x)
            x = self.avgpool(x)
            return torch.flatten(x, 1)

    return _FeatureExtractor(backbone).to(device)


def run_sweep(model_path: str, cidrz_path: str, qatar_pkl: str, adapt_bn: bool,
              k_values=(5, 10, 20), n_seeds: int = 5, epochs: int = 50):
    device = get_device()
    model = torch.load(model_path, weights_only=False).to(device)
    encoder = model.encoders[0]
    control_encoder = _fresh_imagenet_encoder(device)

    if adapt_bn:
        adapt_loader = DataLoader(_load_split(cidrz_path, ("train", "valid")), batch_size=16, shuffle=False)
        adapt_batchnorm(model, adapt_loader, device)
        # Same unlabeled adaptation applied to the untouched control encoder, for parity -
        # otherwise a treatment-vs-control gap could just reflect one having AdaBN and the
        # other not, rather than Qatar pretraining.
        adapt_loader2 = DataLoader(_load_split(cidrz_path, ("train", "valid")), batch_size=16, shuffle=False)
        adapt_batchnorm(control_encoder, adapt_loader2, device)

    train_ds = _load_split(cidrz_path, ("train",))
    test_ds = _load_split(cidrz_path, ("test",))
    train_images, train_labels = train_ds.tensors[0], train_ds.tensors[1].numpy()
    test_images, test_labels = test_ds.tensors[0], test_ds.tensors[1].numpy()

    train_feats = extract_features(encoder, train_images, device)
    test_feats = extract_features(encoder, test_images, device)
    feat_dim = train_feats.shape[1]
    base_head_state = copy.deepcopy(model.head.state_dict())

    control_train_feats = extract_features(control_encoder, train_images, device)
    control_test_feats = extract_features(control_encoder, test_images, device)
    control_feat_dim = control_train_feats.shape[1]

    results = {}
    for k in k_values:
        treat_rows, control_rows = [], []
        for seed in range(n_seeds):
            shot_idx = sample_k_shot(train_labels, k, seed)
            shot_labels = torch.tensor(train_labels[shot_idx])

            treat_shot_feats = train_feats[shot_idx]
            treat_head = MLP(feat_dim, 64, 2)
            treat_head.load_state_dict(base_head_state)
            treat_head = train_head(treat_head, treat_shot_feats, shot_labels, device, epochs=epochs)
            treat_rows.append(score_head(treat_head, test_feats, test_labels, device, qatar_pkl))

            control_shot_feats = control_train_feats[shot_idx]
            control_head = MLP(control_feat_dim, 64, 2)
            # Heavier weight decay than the treatment: 25088-dim features vs. as few as
            # 10 shots is a huge params-to-samples ratio, prone to memorizing rather than
            # generalizing without stronger regularization.
            control_head = train_head(control_head, control_shot_feats, shot_labels, device,
                                       epochs=epochs, weight_decay=0.1)
            control_rows.append(score_head(control_head, control_test_feats, test_labels, device, qatar_pkl))

        results[k] = {"treatment": treat_rows, "control": control_rows}
    return results


def _summarize(rows, key_path):
    vals = [r[key_path[0]][key_path[1]] if len(key_path) > 1 else r[key_path[0]] for r in rows]
    return float(np.mean(vals)), float(np.std(vals))


def print_sweep(results, label):
    print(f"\n=== Few-shot fine-tuning sweep: {label} ===")
    print(f"{'k':>4} {'cond':>10} {'AUROC':>16} {'ACC(default)':>16} {'UA(default)':>14} "
          f"{'F1(default)':>14} {'ACC(calib)':>16} {'UA(calib)':>14} {'F1(calib)':>14}")
    for k, cond_results in results.items():
        for cond in ("treatment", "control"):
            rows = cond_results[cond]
            auroc_m, auroc_s = _summarize(rows, ("auroc",))
            acc_m, acc_s = _summarize(rows, ("default", "WA"))
            ua_m, ua_s = _summarize(rows, ("default", "UA"))
            f1_m, f1_s = _summarize(rows, ("default", "F1"))
            cacc_m, cacc_s = _summarize(rows, ("calibrated", "WA"))
            cua_m, cua_s = _summarize(rows, ("calibrated", "UA"))
            cf1_m, cf1_s = _summarize(rows, ("calibrated", "F1"))
            print(f"{k:>4} {cond:>10} {auroc_m:.3f}+/-{auroc_s:.3f}   "
                  f"{acc_m:.3f}+/-{acc_s:.3f}   {ua_m:.3f}+/-{ua_s:.3f}   {f1_m:.3f}+/-{f1_s:.3f}   "
                  f"{cacc_m:.3f}+/-{cacc_s:.3f}   {cua_m:.3f}+/-{cua_s:.3f}   {cf1_m:.3f}+/-{cf1_s:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="results/models/tb_cxr_qatar_to_cidrz.pt")
    parser.add_argument("--cidrz-path", default="data/google_health/tb_dataset.pkl")
    parser.add_argument("--qatar-pkl", default="data/tb_cxr_qatar/tb_cxr_qatar.pkl")
    parser.add_argument("--adapt-bn", action="store_true")
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    results = run_sweep(args.model, args.cidrz_path, args.qatar_pkl, args.adapt_bn,
                         tuple(args.k_values), args.n_seeds, args.epochs)
    label = args.model + (" + AdaBN" if args.adapt_bn else "")
    print_sweep(results, label)


if __name__ == "__main__":
    main()
