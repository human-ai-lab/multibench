"""On-the-fly, randomly augmented PyTorch Dataset for Qatar TB CXR training images.

Domain-randomizing pixel-level augmentation (random resized crop, flip, small affine
jitter, brightness/contrast jitter, Gaussian noise) applied fresh on every access - unlike
the static `extract_image_features_pretrained`-preprocessed pickle `download.py` produces,
which has no per-epoch randomness at all. Only the TRAIN split is affected: valid/test
still come from the existing unaugmented pickle, keeping evaluation comparable across
experiments in this project.

Motivated by MixStyle's failure on this project's frozen-backbone architecture (see
`mixstyle.py`'s docstring for the full result: UA collapsed to chance-level 0.502).
MixStyle perturbs intermediate feature-map statistics, which only the model's single
trainable `Linear` layer can try to compensate for - not enough capacity. Augmenting the
INPUT before the frozen backbone ever sees it instead lets the (already fixed, generic)
backbone extract features from many synthetic "styles" of the same real image, and the
trainable classifier learns from that variety directly - no reliance on backbone
plasticity, so it should not hit the same failure mode.
"""
import glob
import os
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from datasets.google_health.download import _stratified_split
from datasets.google_health.features import IMAGENET_MEAN, IMAGENET_STD
from datasets.tb_cxr_qatar.download import DATASET_REF


def qatar_train_paths_and_labels(
    seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15
) -> Tuple[list, np.ndarray]:
    """Re-derive the same train split `download.py`'s `build_dataset` produces (identical
    glob order + `_stratified_split` call, same seed) - returns only the TRAIN portion's
    file paths and labels, for on-the-fly loading rather than the static pickle.
    """
    import kagglehub

    local_dir = kagglehub.dataset_download(DATASET_REF)
    base = os.path.join(local_dir, "TB_Chest_Radiography_Database")
    normal_paths = sorted(glob.glob(os.path.join(base, "Normal", "*.png")))
    tb_paths = sorted(glob.glob(os.path.join(base, "Tuberculosis", "*.png")))
    paths = normal_paths + tb_paths
    labels = np.array([0] * len(normal_paths) + [1] * len(tb_paths), dtype=np.int64)

    train_idx, _, _ = _stratified_split(labels, val_frac, test_frac, seed)
    return [paths[i] for i in train_idx], labels[train_idx]


class AugmentedQatarTrainDataset(Dataset):
    """Randomly augments each Qatar training image fresh on every `__getitem__` call."""

    def __init__(self, image_size: int = 224, seed: int = 42, noise_std: float = 0.02):
        self.paths, self.labels = qatar_train_paths_and_labels(seed=seed)
        self.noise_std = noise_std
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.ToTensor(),  # -> (1, H, W) in [0, 1], source is grayscale PIL 'L'
        ])
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        from PIL import Image

        img = Image.open(self.paths[idx]).convert("L")
        tensor = self.transform(img)
        tensor = (tensor + torch.randn_like(tensor) * self.noise_std).clamp(0.0, 1.0)
        tensor = tensor.repeat(3, 1, 1)  # channel-replicate, matching extract_image_features_pretrained
        tensor = (tensor - self.mean) / self.std
        return tensor, int(self.labels[idx])
