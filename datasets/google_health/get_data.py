"""Dataloader for the preprocessed Google + CIDRZ Health AI TB dataset.

Assumes `download.py` (or `download.build_dataset`) has already been run to produce a
pickle at `path` with the shape `{"train": split, "valid": split, "test": split}`, each
split a dict of `{"text": [N, TEXT_FEATURE_DIM], "audio": [N, audio_bins], "image": [N, 1,
size, size], "label": [N]}` float32/int64 numpy arrays - see `download.py`'s docstring for
how to build one from the raw Kaggle dataset.
"""
import pickle
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _split_to_dataset(split: dict) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(np.asarray(split["text"], dtype=np.float32)),
        torch.from_numpy(np.asarray(split["audio"], dtype=np.float32)),
        torch.from_numpy(np.asarray(split["image"], dtype=np.float32)),
        torch.from_numpy(np.asarray(split["label"], dtype=np.int64)),
    )


def get_dataloader(
    path: str,
    batch_size: int = 8,
    num_workers: int = 0,
    train_shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load the preprocessed TB dataset pickle and wrap each split in a DataLoader.

    Each batch is `[text, audio, image, label]`, matching what
    `training_structures.Supervised_Learning` expects for `is_packed=False`.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    train = DataLoader(
        _split_to_dataset(data["train"]), batch_size=batch_size,
        shuffle=train_shuffle, num_workers=num_workers,
    )
    valid = DataLoader(
        _split_to_dataset(data["valid"]), batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )
    test = DataLoader(
        _split_to_dataset(data["test"]), batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )
    return train, valid, test
