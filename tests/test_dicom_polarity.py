from types import SimpleNamespace

import numpy as np

from datasets.google_health.download import dicom_pixels


def _ds(pi):
    return SimpleNamespace(pixel_array=np.array([[0, 100], [300, 400]], dtype=np.uint16), PhotometricInterpretation=pi)


def test_monochrome1_is_inverted():
    assert np.array_equal(dicom_pixels(_ds("MONOCHROME1")), [[400, 300], [100, 0]])


def test_monochrome2_is_unchanged():
    assert np.array_equal(dicom_pixels(_ds("MONOCHROME2")), [[0, 100], [300, 400]])
