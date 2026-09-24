"""Pure feature-extraction helpers for the Google + CIDRZ Health AI TB dataset.

Kept dependency-light (numpy + scipy, both already base MultiBench dependencies) and
separate from `download.py` so they're testable without pydicom/pandas/kagglehub or any
network access.
"""
from typing import Any, Dict

import numpy as np
from scipy.signal import stft

# Ordinal/binary encodings for the clinical symptom & history fields in
# `GHAI_Final_Data_2023.csv` (see `Google_Health_AI_Final_Codebook.csv`). AI-model
# outputs (abn_*, tb_decision*, tb_predictions*) and the `ground_truth_tb` label itself
# are intentionally excluded - only participant-reported/clinical fields are features.
CATEGORY_MAPS: Dict[str, Dict[str, float]] = {
    "sex": {"m": 0.0, "f": 1.0},
    "coughdur": {"no cough": 0.0, "<1wk": 1.0, "1-2wks": 2.0, "3-4wks": 3.0, ">4weeks": 4.0},
    "cough_productive": {"no": 0.0, "yes": 1.0},
    "haemoptysis": {"no": 0.0, "yes": 1.0},
    "chestpain": {"no": 0.0, "yes": 1.0},
    "shortbreath": {"no": 0.0, "yes": 1.0},
    "fever": {"no": 0.0, "yes": 1.0},
    "ngtsweats": {"no": 0.0, "yes": 1.0},
    "weightloss": {"no": 0.0, "yes": 1.0},
    "hiv_status": {"neg": 0.0, "pos": 1.0},
    "prev_tb": {"no": 0.0, "yes": 1.0},
    "tobacco_use": {"never": 0.0, "stopped": 1.0, "current": 2.0, "not disclosed": -1.0},
}
NUMERIC_COLUMNS = ["age", "bmi", "body_wt", "height"]
MISSING_VALUE = -1.0

TEXT_FEATURE_DIM = len(CATEGORY_MAPS) + len(NUMERIC_COLUMNS)


def extract_text_features(row: Dict[str, Any]) -> np.ndarray:
    """Encode one participant's clinical/demographic fields into a fixed-length vector."""
    values = []
    for column, mapping in CATEGORY_MAPS.items():
        raw = row.get(column)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            values.append(MISSING_VALUE)
        else:
            values.append(mapping.get(str(raw).strip().lower(), MISSING_VALUE))
    for column in NUMERIC_COLUMNS:
        raw = row.get(column)
        try:
            values.append(float(raw) if raw is not None and not (isinstance(raw, float) and np.isnan(raw)) else MISSING_VALUE)
        except (TypeError, ValueError):
            values.append(MISSING_VALUE)
    return np.asarray(values, dtype=np.float32)


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_freqs: int, sr: float, n_mels: int, fmax: float) -> np.ndarray:
    """Build a (n_mels, n_freqs) triangular mel filterbank for an `n_freqs`-bin rFFT of a
    signal sampled at `sr` Hz. Concentrates resolution in the low frequencies where cough
    energy lives, unlike linear FFT-bin binning (which spreads most bins across frequencies
    a cough barely touches)."""
    mel_edges = np.linspace(_hz_to_mel(np.array(0.0)), _hz_to_mel(np.array(fmax)), n_mels + 2)
    bin_edges = np.floor(_mel_to_hz(mel_edges) / fmax * (n_freqs - 1)).astype(int)
    bin_edges = np.clip(bin_edges, 0, n_freqs - 1)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = bin_edges[m], bin_edges[m + 1], bin_edges[m + 2]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for k in range(left, min(center, n_freqs)):
            fb[m, k] = (k - left) / (center - left)
        for k in range(center, min(right, n_freqs)):
            fb[m, k] = (right - k) / (right - center)
    return fb


def extract_audio_features(waveform: np.ndarray, sr: float, n_bins: int = 64) -> np.ndarray:
    """Reduce a raw (possibly multi-channel) cough waveform to a fixed-length log-mel vector.

    STFT -> mel filterbank -> log-compress -> mean+std pooled over time, giving `n_bins`
    values total (n_bins // 2 mel bands x {mean, std}). Needs `sr` (the actual sample rate)
    to place the mel bands correctly - the dataset's 4 recording devices differ in sample
    rate, so a fixed bin-index scheme (as opposed to a frequency-based one) would silently
    mean something different per device.
    """
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)
    waveform = waveform.astype(np.float32)
    peak = np.abs(waveform).max()
    if peak > 0:
        waveform = waveform / peak

    n_mels = max(1, n_bins // 2)
    nperseg = min(1024, len(waveform))
    if nperseg < 8:
        return np.zeros(2 * n_mels, dtype=np.float32)
    noverlap = nperseg // 2
    _, _, Zxx = stft(waveform, fs=sr, nperseg=nperseg, noverlap=noverlap)
    magnitude = np.abs(Zxx)  # (n_freqs, n_frames)

    fb = _mel_filterbank(magnitude.shape[0], sr, n_mels=n_mels, fmax=sr / 2)
    log_mel = np.log1p(fb @ magnitude)  # (n_mels, n_frames)
    return np.concatenate([log_mel.mean(axis=1), log_mel.std(axis=1)]).astype(np.float32)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def extract_image_features_pretrained(pixel_array: np.ndarray, size: int = 224) -> np.ndarray:
    """Prepare a chest X-ray for an ImageNet-pretrained encoder (e.g. `vgg11_slim`).

    Resizes to `size`, replicates the grayscale channel to 3 (the "channel replication"
    trick used in the CXR-classification literature to reuse RGB pretrained backbones on
    grayscale medical images - see e.g. RepViT-CXR, arXiv:2509.08234), and normalizes with
    ImageNet statistics, since a pretrained backbone's BatchNorm layers assume that input
    distribution.
    """
    from PIL import Image

    arr = pixel_array.astype(np.float32)
    arr -= arr.min()
    peak = arr.max()
    if peak > 0:
        arr /= peak
    img = Image.fromarray((arr * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    gray = np.asarray(img, dtype=np.float32) / 255.0
    rgb = np.stack([gray, gray, gray], axis=0)  # (3, size, size)
    return ((rgb - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]).astype(np.float32)


def extract_image_features(pixel_array: np.ndarray, size: int = 64) -> np.ndarray:
    """Normalize and downsize a chest X-ray pixel array to a fixed (1, size, size) tensor.

    For the `lenet` (trained-from-scratch) image encoder; see `extract_image_features_pretrained`
    for the `vgg11_slim` (ImageNet-pretrained) path.
    """
    from PIL import Image

    arr = pixel_array.astype(np.float32)
    arr -= arr.min()
    peak = arr.max()
    if peak > 0:
        arr /= peak
    img = Image.fromarray((arr * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    return (np.asarray(img, dtype=np.float32) / 255.0)[None, :, :]


def encode_label(ground_truth_tb: Any) -> int:
    """Map the `ground_truth_tb` column ('neg'/'pos') to a binary classification label."""
    value = str(ground_truth_tb).strip().lower()
    if value not in ("neg", "pos"):
        raise ValueError(f"Unexpected ground_truth_tb value: {ground_truth_tb!r}")
    return 1 if value == "pos" else 0
