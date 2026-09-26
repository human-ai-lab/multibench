import numpy as np
from sklearn.metrics import balanced_accuracy_score

from datasets.tb_cxr_qatar.multimodal_late_fusion import ARMS, AVG_ARMS, _nadeau_bengio, run_cv


def _synthetic(n=120, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.25).astype(np.int64)
    text = rng.normal(size=(n, 4)) + y[:, None] * 1.5  # informative
    audio = rng.normal(size=(n, 6))  # pure noise
    image = rng.normal(size=n) + y * 0.5  # weakly informative
    return {"text": text, "audio": audio, "image": image}, y


def test_run_cv_shapes_and_informative_text():
    feats, y = _synthetic()
    probs, fold_uar, fold_sizes = run_cv(feats, y, n_repeats=1)
    assert set(probs) == set(ARMS) | set(AVG_ARMS)
    assert probs["text"].shape == (1, len(y))
    assert len(fold_sizes) == 5 and len(fold_uar["text"]) == 5
    assert balanced_accuracy_score(y, probs["text"][0] > 0.5) > 0.75
    assert balanced_accuracy_score(y, probs["audio"][0] > 0.5) < balanced_accuracy_score(y, probs["text"][0] > 0.5)


def test_nadeau_bengio_zero_difference_is_not_significant():
    rng = np.random.default_rng(0)
    diffs = rng.normal(0, 0.05, size=50)
    assert _nadeau_bengio(diffs, [(290, 73)] * 50) > 0.05
    assert _nadeau_bengio(np.zeros(50), [(290, 73)] * 50) == 1.0
