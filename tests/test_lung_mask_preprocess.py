import numpy as np

from datasets.google_health.features import IMAGENET_MEAN, IMAGENET_STD
from datasets.tb_cxr_qatar.lung_mask_preprocess import _to_gray01, apply_mask, refine_mask


def _normalized(gray01):
    rgb = np.repeat(gray01[:, None], 3, axis=1)
    return ((rgb - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]).astype(np.float32)


def test_gray_roundtrip():
    gray = np.random.default_rng(0).random((2, 16, 16)).astype(np.float32)
    assert np.allclose(_to_gray01(_normalized(gray)), gray, atol=1e-5)


def test_outside_mask_is_raw_black_and_inside_unchanged():
    gray = np.full((1, 16, 16), 0.7, dtype=np.float32)
    images = _normalized(gray)
    mask = np.zeros((1, 16, 16), dtype=bool)
    mask[0, 4:12, 4:12] = True
    out = apply_mask(images, mask)
    assert np.allclose(out[:, :, 4:12, 4:12], images[:, :, 4:12, 4:12])
    assert np.allclose(_to_gray01(out)[0, 0, 0], 0.0, atol=1e-5)
    assert np.allclose(out[0, :, 0, 0], -IMAGENET_MEAN / IMAGENET_STD, atol=1e-5)


def test_refine_fills_holes_and_dilates():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    mask[14:18, 14:18] = False
    refined = refine_mask(mask, dilate=2)
    assert refined[15, 15]
    assert refined[6, 16] and not refined[4, 16]
