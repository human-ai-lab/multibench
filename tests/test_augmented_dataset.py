import numpy as np
import torch

from datasets.tb_cxr_qatar.augmented_dataset import AugmentedQatarTrainDataset


def _make_synthetic_dataset(tmp_path, n=4):
    from PIL import Image

    paths = []
    for i in range(n):
        p = tmp_path / f"img{i}.png"
        arr = np.random.default_rng(i).integers(0, 255, size=(64, 64), dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(p)
        paths.append(str(p))
    labels = np.array([0, 1] * (n // 2), dtype=np.int64)
    return paths, labels


def test_augmented_dataset_shape_and_range(tmp_path, monkeypatch):
    paths, labels = _make_synthetic_dataset(tmp_path)
    monkeypatch.setattr(
        "datasets.tb_cxr_qatar.augmented_dataset.qatar_train_paths_and_labels",
        lambda **kwargs: (paths, labels),
    )

    ds = AugmentedQatarTrainDataset(image_size=32)

    assert len(ds) == len(paths)
    x, y = ds[0]
    assert x.shape == (3, 32, 32)
    assert x.dtype == torch.float32
    assert y == int(labels[0])


def test_augmented_dataset_is_stochastic(tmp_path, monkeypatch):
    paths, labels = _make_synthetic_dataset(tmp_path)
    monkeypatch.setattr(
        "datasets.tb_cxr_qatar.augmented_dataset.qatar_train_paths_and_labels",
        lambda **kwargs: (paths, labels),
    )

    ds = AugmentedQatarTrainDataset(image_size=32)
    x1, _ = ds[0]
    x2, _ = ds[0]

    assert not torch.allclose(x1, x2)
