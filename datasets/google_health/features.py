"""Pure feature-extraction helpers for the Google + CIDRZ Health AI TB dataset.

Kept dependency-light (numpy only) and separate from `download.py` so they're testable
without pydicom/pandas/kagglehub or any network access.
"""
from typing import Any, Dict

import numpy as np

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


def extract_audio_features(waveform: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """Reduce a raw (possibly multi-channel) cough waveform to a fixed-length log-spectral vector.

    Deliberately simple (numpy FFT + linear binning rather than a proper mel filterbank) to
    avoid pulling in an audio dependency - this is a demo-scale feature, not a research one.
    """
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)
    waveform = waveform.astype(np.float32)
    peak = np.abs(waveform).max()
    if peak > 0:
        waveform = waveform / peak
    spectrum = np.log1p(np.abs(np.fft.rfft(waveform)))
    chunks = np.array_split(spectrum, n_bins)
    return np.asarray([chunk.mean() if chunk.size else 0.0 for chunk in chunks], dtype=np.float32)


def extract_image_features(pixel_array: np.ndarray, size: int = 64) -> np.ndarray:
    """Normalize and downsize a chest X-ray pixel array to a fixed (1, size, size) tensor."""
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
