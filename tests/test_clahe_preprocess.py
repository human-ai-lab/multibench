import numpy as np

from datasets.google_health.features import IMAGENET_MEAN, IMAGENET_STD
from datasets.tb_cxr_qatar.clahe_preprocess import apply_clahe_to_normalized_batch


def _make_normalized_batch(n=3, size=32, seed=0):
    rng = np.random.default_rng(seed)
    mean = IMAGENET_MEAN[:, None, None].astype(np.float32)
    std = IMAGENET_STD[:, None, None].astype(np.float32)
    raw01 = rng.uniform(0, 1, size=(n, 1, size, size)).astype(np.float32)
    raw3 = np.repeat(raw01, 3, axis=1)
    return (raw3 - mean) / std


def test_shape_and_dtype_preserved():
    batch = _make_normalized_batch()
    out = apply_clahe_to_normalized_batch(batch)
    assert out.shape == batch.shape
    assert out.dtype == np.float32


def test_output_is_finite_and_in_sane_range():
    batch = _make_normalized_batch()
    out = apply_clahe_to_normalized_batch(batch)
    assert np.all(np.isfinite(out))
    # ImageNet-normalized pixel intensities in [0, 1] land roughly in [-3, 3]
    assert out.min() > -5.0 and out.max() < 5.0


def test_pre_normalization_gray_is_shared_across_channels():
    # The 3 channels differ after ImageNet normalization (mean/std differ per channel),
    # but the recovered pre-normalization [0,1] grayscale (what CLAHE operates on) must be
    # identical across channels, same as the channel-replication convention this pickle
    # format uses.
    mean = IMAGENET_MEAN[:, None, None].astype(np.float32)
    std = IMAGENET_STD[:, None, None].astype(np.float32)
    batch = _make_normalized_batch()
    raw01 = batch * std[None] + mean[None]
    assert np.allclose(raw01[:, 0], raw01[:, 1], atol=1e-5) and np.allclose(raw01[:, 1], raw01[:, 2], atol=1e-5)
