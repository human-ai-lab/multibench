import numpy as np
import pytest

from datasets.google_health.features import (
    TEXT_FEATURE_DIM,
    encode_label,
    extract_audio_features,
    extract_image_features,
    extract_text_features,
)


def test_extract_text_features_shape_and_known_values():
    row = {"sex": "f", "hiv_status": "pos", "age": 31, "bmi": 24.0}

    features = extract_text_features(row)

    assert features.shape == (TEXT_FEATURE_DIM,)
    assert features.dtype == np.float32


def test_extract_text_features_fills_missing_with_sentinel():
    features = extract_text_features({})

    assert (features == -1.0).all()


def test_extract_audio_features_shape_and_finite():
    waveform = np.sin(np.linspace(0, 100, 4000)).astype(np.float32)

    features = extract_audio_features(waveform, n_bins=64)

    assert features.shape == (64,)
    assert np.isfinite(features).all()


def test_extract_audio_features_handles_multichannel():
    waveform = np.random.default_rng(0).normal(size=(2000, 2)).astype(np.float32)

    features = extract_audio_features(waveform, n_bins=32)

    assert features.shape == (32,)


def test_extract_image_features_shape_and_range():
    pixel_array = np.random.default_rng(0).integers(0, 4096, size=(128, 128)).astype(np.uint16)

    features = extract_image_features(pixel_array, size=64)

    assert features.shape == (1, 64, 64)
    assert features.min() >= 0.0 and features.max() <= 1.0


def test_encode_label():
    assert encode_label("neg") == 0
    assert encode_label("pos") == 1
    assert encode_label(" POS ") == 1


def test_encode_label_rejects_unknown_value():
    with pytest.raises(ValueError):
        encode_label("unknown")
