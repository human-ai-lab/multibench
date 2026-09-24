"""Audio embeddings from Google's HeAR (Health Acoustic Representations) model.

Kept separate from `features.py` (numpy/scipy only) since this needs `transformers` +
`torch` + a one-time ~1.2GB gated model download - a Hugging Face account that has clicked
"Acknowledge license" on https://huggingface.co/google/hear-pytorch (Health AI Developer
Foundations terms), with a token at `~/.cache/huggingface/token` (or `HF_TOKEN` env var).

HeAR (arXiv:2403.02522) is a ViT-L masked-autoencoder pretrained on ~174k hours of health
acoustic sounds; its own paper reports AUROC 0.739 for TB classification on this exact
CIDRZ cough dataset (audio-only) - see the model card's "How to use" section for the
canonical preprocessing steps this module follows.
"""
from typing import Optional

import numpy as np

from ._hear_audio_utils import preprocess_audio, resample_audio_and_convert_to_mono

HEAR_EMBEDDING_DIM = 512
HEAR_SAMPLE_RATE = 16000
_CLIP_SAMPLES = 2 * HEAR_SAMPLE_RATE  # HeAR's fixed input: exactly 2 seconds

_model = None


def _load_model():
    global _model
    if _model is None:
        from transformers import AutoModel

        _model = AutoModel.from_pretrained("google/hear-pytorch")
        _model.eval()
    return _model


def _select_loudest_window(mono16k: np.ndarray, window: int = _CLIP_SAMPLES, hop: int = 8000) -> np.ndarray:
    """HeAR takes a fixed 2-second clip; our recordings run 2-13s (often with a verbal
    prompt or silence before/after the cough), so pick the loudest 2-second window rather
    than an arbitrary fixed offset."""
    if len(mono16k) <= window:
        return mono16k
    best_start, best_energy = 0, -1.0
    for start in range(0, len(mono16k) - window + 1, hop):
        energy = float(np.mean(mono16k[start:start + window].astype(np.float64) ** 2))
        if energy > best_energy:
            best_energy = energy
            best_start = start
    return mono16k[best_start:best_start + window]


def _segment_into_windows(mono16k: np.ndarray, window: int = _CLIP_SAMPLES) -> np.ndarray:
    """Split the whole clip into non-overlapping `window`-sample segments (zero-padding the
    last one), rather than picking a single window - uses the full recording instead of
    discarding everything outside one 2-second slice. Returns (n_windows, window)."""
    n_windows = max(1, int(np.ceil(len(mono16k) / window)))
    padded_len = n_windows * window
    if len(mono16k) < padded_len:
        mono16k = np.pad(mono16k, (0, padded_len - len(mono16k)))
    return mono16k[:padded_len].reshape(n_windows, window)


def extract_hear_features(
    waveform: np.ndarray, sr: float, model: Optional[object] = None, pooling: str = "loudest",
) -> np.ndarray:
    """Embed a cough recording with HeAR and return its 512-dim embedding.

    `pooling`:
    - "loudest" (default, cheap: 1 forward pass): pick the single loudest 2-second window.
    - "mean_windows" (more thorough: ceil(duration/2s) forward passes): split the whole
      clip into non-overlapping 2-second windows, embed each, and mean-pool - uses the full
      recording rather than a 2-second slice of a clip that can run up to ~13s.
    """
    import torch

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)
    mono16k = resample_audio_and_convert_to_mono(waveform.astype(np.float32), sr, HEAR_SAMPLE_RATE)
    model = model or _load_model()

    if pooling == "mean_windows":
        windows = _segment_into_windows(mono16k)
        audio_t = torch.from_numpy(windows.astype(np.float32))
        spectrogram = preprocess_audio(audio_t)
        with torch.no_grad():
            out = model(spectrogram, return_dict=True)
        return out.pooler_output.mean(dim=0).numpy().astype(np.float32)

    clip = _select_loudest_window(mono16k)
    audio_t = torch.from_numpy(clip.astype(np.float32)).unsqueeze(0)
    spectrogram = preprocess_audio(audio_t)
    with torch.no_grad():
        out = model(spectrogram, return_dict=True)
    return out.pooler_output[0].numpy().astype(np.float32)
