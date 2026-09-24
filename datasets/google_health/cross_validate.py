"""Stratified k-fold cross-validation for a google_health experiment config.

`run_experiment.py`/`utils/config.py` are built around one fixed train/valid/test split.
With only ~364 usable participants here (a 55-sample test split has as few as 9
positives), that single split's metrics carry real luck-of-the-draw variance - this
session's own repeated-seed harnesses kept surfacing exactly that. This script instead
pools every split back together, runs stratified k-fold CV over the full dataset (each
fold trains fresh encoders/fusion/classifier from scratch, with an inner stratified
train/valid split carved out of the fold's training portion for early-stopping/checkpoint
selection), and reports mean +/- std metrics across folds - using every sample for both
training and evaluation at some point, rather than permanently holding ~30% out.

Usage:
    python -m datasets.google_health.cross_validate --config configs/google_health_tb.yaml --k 5
"""
import argparse
import pickle
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from training_structures.Supervised_Learning import test as _test
from training_structures.Supervised_Learning import train as _train
from utils.config import build_classifier, build_encoders, build_fusion, build_objective, build_optimizer_type, load_config
from utils.device import get_device


def _pool_dataset(path: str) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return {
        key: np.concatenate([data[split][key] for split in ("train", "valid", "test")], axis=0)
        for key in ("text", "audio", "image", "label")
    }


def _make_loader(pooled: Dict[str, np.ndarray], idx: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(pooled["text"][idx].astype(np.float32)),
        torch.from_numpy(pooled["audio"][idx].astype(np.float32)),
        torch.from_numpy(pooled["image"][idx].astype(np.float32)),
        torch.from_numpy(pooled["label"][idx].astype(np.int64)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _stratified_kfold_indices(labels: np.ndarray, k: int, seed: int) -> List[np.ndarray]:
    """A dependency-free stratified k-fold split (avoids requiring scikit-learn's exact
    StratifiedKFold API surface): shuffles each class's indices, then divides each class's
    shuffled indices into k roughly-equal chunks so every fold gets a proportional share of
    each class."""
    rng = np.random.default_rng(seed)
    fold_indices: List[List[int]] = [[] for _ in range(k)]
    for class_label in np.unique(labels):
        class_idx = np.where(labels == class_label)[0]
        rng.shuffle(class_idx)
        for fold, chunk in enumerate(np.array_split(class_idx, k)):
            fold_indices[fold].extend(chunk.tolist())
    return [np.array(sorted(idx)) for idx in fold_indices]


def cross_validate(config_path: str, k: int = 5, seed: int = 42, val_frac: float = 0.15) -> Dict[str, List[float]]:
    config = load_config(config_path)
    pooled = _pool_dataset(config["dataset"]["path"])
    labels = pooled["label"]
    n = len(labels)
    batch_size = config["dataset"].get("kwargs", {}).get("batch_size", 8)
    training_cfg = config["training"]
    task = training_cfg.get("task", "classification")

    folds = _stratified_kfold_indices(labels, k, seed)
    metrics: Dict[str, List[float]] = {}

    for fold in range(k):
        test_idx = folds[fold]
        train_val_idx = np.concatenate([folds[i] for i in range(k) if i != fold])

        # Carve a stratified validation slice out of this fold's training portion, for
        # early-stopping/best-checkpoint selection - same mechanism as build_dataset's
        # main train/valid/test split, just nested one level deeper.
        inner_val_folds = _stratified_kfold_indices(labels[train_val_idx], max(2, round(1 / val_frac)), seed)
        val_idx = train_val_idx[inner_val_folds[0]]
        train_idx = np.setdiff1d(train_val_idx, val_idx)

        print(f"\n=== Fold {fold + 1}/{k}: train={len(train_idx)} valid={len(val_idx)} test={len(test_idx)} "
              f"(test pos={int(labels[test_idx].sum())}/{len(test_idx)}) ===")

        device = get_device()
        torch.manual_seed(seed + fold)
        encoders = build_encoders(config["model"], device)
        fusion = build_fusion(config["model"], device)
        classifier = build_classifier(config["model"], device)
        objective = build_objective(training_cfg, device)
        optimizer_type = build_optimizer_type(training_cfg)

        traindata = _make_loader(pooled, train_idx, batch_size, True)
        validdata = _make_loader(pooled, val_idx, batch_size, False)
        testdata = _make_loader(pooled, test_idx, batch_size, False)

        save_path = training_cfg.get("save", "results/models/from_config.pt") + f".cv_fold{fold}"
        _train(
            encoders, fusion, classifier, traindata, validdata,
            total_epochs=training_cfg.get("epochs", 10),
            task=task,
            optimtype=optimizer_type,
            lr=training_cfg.get("lr", 0.001),
            weight_decay=training_cfg.get("weight_decay", 0.0),
            objective=objective,
            is_packed=False,
            early_stop=training_cfg.get("early_stop", False),
            save=save_path,
        )
        model = torch.load(save_path, weights_only=False).to(device)
        result = _test(
            model=model, test_dataloaders_all=testdata, is_packed=False,
            criterion=objective, task=task, no_robust=True, metrics=config.get("evaluation"),
        )
        print(f"Fold {fold + 1} result: {result}")
        for name, value in result.items():
            metrics.setdefault(name, []).append(value)

    print(f"\n=== {k}-fold cross-validation summary ({n} total samples) ===")
    for name, values in metrics.items():
        arr = np.array(values)
        print(f"{name}: mean={arr.mean():.3f} std={arr.std():.3f}  values={np.round(arr, 3).tolist()}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cross_validate(args.config, k=args.k, seed=args.seed)


if __name__ == "__main__":
    main()
