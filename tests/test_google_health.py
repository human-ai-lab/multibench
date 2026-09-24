import numpy as np
import pytest

from datasets.google_health.features import (
    TEXT_FEATURE_DIM,
    encode_label,
    extract_audio_features,
    extract_image_features,
    extract_image_features_pretrained,
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

    features = extract_audio_features(waveform, sr=16000, n_bins=64)

    assert features.shape == (64,)
    assert np.isfinite(features).all()


def test_extract_audio_features_handles_multichannel():
    waveform = np.random.default_rng(0).normal(size=(2000, 2)).astype(np.float32)

    features = extract_audio_features(waveform, sr=16000, n_bins=32)

    assert features.shape == (32,)


def test_extract_audio_features_handles_very_short_clip():
    waveform = np.zeros(4, dtype=np.float32)

    features = extract_audio_features(waveform, sr=16000, n_bins=64)

    assert features.shape == (64,)
    assert np.isfinite(features).all()


def test_extract_audio_features_differs_by_sample_rate():
    rng = np.random.default_rng(0)
    waveform = rng.normal(size=16000).astype(np.float32)

    low_sr = extract_audio_features(waveform, sr=8000, n_bins=64)
    high_sr = extract_audio_features(waveform, sr=44100, n_bins=64)

    assert not np.allclose(low_sr, high_sr)


def test_extract_image_features_shape_and_range():
    pixel_array = np.random.default_rng(0).integers(0, 4096, size=(128, 128)).astype(np.uint16)

    features = extract_image_features(pixel_array, size=64)

    assert features.shape == (1, 64, 64)
    assert features.min() >= 0.0 and features.max() <= 1.0


def test_extract_image_features_pretrained_shape_and_channels():
    from datasets.google_health.features import IMAGENET_MEAN, IMAGENET_STD

    pixel_array = np.random.default_rng(0).integers(0, 4096, size=(128, 128)).astype(np.uint16)

    features = extract_image_features_pretrained(pixel_array, size=224)

    assert features.shape == (3, 224, 224)
    # each channel is the same grayscale image, so undoing the (per-channel) ImageNet
    # normalization should make all 3 channels identical
    denormalized = features * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
    assert np.allclose(denormalized[0], denormalized[1]) and np.allclose(denormalized[1], denormalized[2])


def test_encode_label():
    assert encode_label("neg") == 0
    assert encode_label("pos") == 1
    assert encode_label(" POS ") == 1


def test_encode_label_rejects_unknown_value():
    with pytest.raises(ValueError):
        encode_label("unknown")


def test_select_loudest_window_picks_the_high_energy_segment():
    from datasets.google_health.hear_features import _select_loudest_window

    rng = np.random.default_rng(0)
    quiet = rng.normal(scale=0.01, size=40000).astype(np.float32)
    loud = rng.normal(scale=1.0, size=32000).astype(np.float32)
    signal = np.concatenate([quiet[:20000], loud, quiet[20000:]])

    window = _select_loudest_window(signal, window=32000, hop=8000)

    assert window.shape == (32000,)
    assert np.abs(window).mean() > np.abs(quiet).mean() * 5


def test_select_loudest_window_short_clip_returned_as_is():
    from datasets.google_health.hear_features import _select_loudest_window

    clip = np.ones(1000, dtype=np.float32)

    window = _select_loudest_window(clip, window=32000, hop=8000)

    assert window.shape == (1000,)
