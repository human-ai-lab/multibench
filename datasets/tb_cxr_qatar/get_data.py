"""Dataloader for the preprocessed Qatar TB Chest X-ray dataset.

Single modality (image + label) - assumes `download.py` has already been run to produce a
pickle at `path` with `{"train": split, "valid": split, "test": split}`, each split a dict
of `{"image": [N, C, size, size], "label": [N]}` float32/int64 numpy arrays.
"""
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _split_to_dataset(split: dict) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(np.asarray(split["image"], dtype=np.float32)),
        torch.from_numpy(np.asarray(split["label"], dtype=np.int64)),
    )


def get_dataloader(
    path: str,
    batch_size: int = 16,
    num_workers: int = 0,
    train_shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load the preprocessed TB CXR pickle and wrap each split in a DataLoader.

    Each batch is `[image, label]`, matching what `training_structures.Supervised_Learning`
    expects for `is_packed=False` with a single-modality model.
    """
    import pickle

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
