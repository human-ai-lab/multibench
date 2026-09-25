"""Cross-dataset external validation: train on Qatar TB CXR, test on CIDRZ (google_health).

Both datasets' images were preprocessed identically (`features.extract_image_features_pretrained`:
224x224, 3-channel, ImageNet-normalized - verified byte-identical value range), so a model
trained on one is directly evaluable on the other's raw image tensors with no reprocessing.

This is a much harder, more realistic test than an in-distribution held-out split: Qatar's
images come from one hospital/population/scanner, CIDRZ's from three different Lusaka,
Zambia facilities - a real domain-shift check, in the spirit of the "why ML-based cough
models do not generalize" cross-dataset finding this project's literature research turned
up for TB acoustic models (the same concern plausibly applies to imaging).

Trains on Qatar's train split (its own valid split used for early-stopping/checkpoint
selection, same as normal), then evaluates the trained model on ALL 364 CIDRZ images
(train+valid+test pooled, since none of them were used for training here - the whole CIDRZ
dataset is legitimately held-out).

Usage:
    python -m datasets.tb_cxr_qatar.cross_dataset_eval \
        --qatar-config configs/tb_cxr_qatar.yaml \
        --cidrz-path data/google_health/tb_dataset.pkl
"""
import argparse
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from eval_scripts.performance import compute_metrics
from training_structures.Supervised_Learning import train as _train
from utils.config import (
    build_classifier, build_dataloaders, build_encoders, build_fusion, build_objective,
    build_optimizer_type, load_config,
)
from utils.device import get_device


def _pool_images_labels(path: str) -> TensorDataset:
    with open(path, "rb") as f:
        data = pickle.load(f)
    images = np.concatenate([data[split]["image"] for split in ("train", "valid", "test")], axis=0)
    labels = np.concatenate([data[split]["label"] for split in ("train", "valid", "test")], axis=0)
    return TensorDataset(
        torch.from_numpy(images.astype(np.float32)), torch.from_numpy(labels.astype(np.int64))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qatar-config", default="configs/tb_cxr_qatar.yaml")
    parser.add_argument("--cidrz-path", default="data/google_health/tb_dataset.pkl")
    parser.add_argument("--save", default="results/models/tb_cxr_qatar_to_cidrz.pt")
    args = parser.parse_args()

    config = load_config(args.qatar_config)
    device = get_device()

    encoders = build_encoders(config["model"], device)
    fusion = build_fusion(config["model"], device)
    classifier = build_classifier(config["model"], device)
    objective = build_objective(config["training"], device)
    optimizer_type = build_optimizer_type(config["training"])

    print("Training on Qatar TB CXR (train split; valid split for checkpoint selection)...")
    traindata, validdata, _ = build_dataloaders(config["dataset"])
    _train(
        encoders, fusion, classifier, traindata, validdata,
        total_epochs=config["training"].get("epochs", 10), task="classification",
        optimtype=optimizer_type, lr=config["training"].get("lr", 0.001),
        weight_decay=config["training"].get("weight_decay", 0.0), objective=objective,
        is_packed=False, early_stop=config["training"].get("early_stop", False), save=args.save,
    )

    print(f"\nEvaluating on ALL of CIDRZ (n=364, none used for training) - {args.cidrz_path} ...")
    model = torch.load(args.save, weights_only=False).to(device)
    model.eval()
    cidrz_loader = DataLoader(_pool_images_labels(args.cidrz_path), batch_size=16, shuffle=False)

    truths, preds = [], []
    with torch.no_grad():
        for batch in cidrz_loader:
            inputs = [batch[0].float().to(device)]
            labels = batch[1]
            outs = model(inputs)
            preds.append(torch.argmax(outs, dim=1).cpu())
            truths.append(labels)
    truths = torch.cat(truths)
    preds = torch.cat(preds)
    results = compute_metrics(truths, preds, config.get("evaluation", ["UA", "WA", "F1"]))
    print("Cross-dataset evaluation results (trained on Qatar, tested on CIDRZ):")
    for name, value in results.items():
        print(f"  {name}: {value}")
    return results


if __name__ == "__main__":
    main()
