import numpy as np
import pytest

# pandas is an optional extra (`.[google_health]`), not in requirements.txt, which CI installs.
pd = pytest.importorskip("pandas")

from datasets.tb_cxr_mont_shen.download import _load_pixel_array, _load_source


def test_load_pixel_array_grayscale(tmp_path):
    from PIL import Image

    path = tmp_path / "sample.png"
    Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)).save(path)

    arr = _load_pixel_array(str(path))

    assert arr.shape == (16, 16)
    assert arr.dtype == np.uint8


def test_load_source_label_from_findings(tmp_path):
    metadata = pd.DataFrame({
        "study_id": ["CHNCXR_0001_0.png", "CHNCXR_0002_1.png", "CHNCXR_0003_1.png"],
        "findings": ["normal", "bilateral PTB", "  Normal  "],
    })
    metadata.to_csv(tmp_path / "shenzhen_metadata.csv", index=False)
    (tmp_path / "images" / "images").mkdir(parents=True)

    paths, labels, sources = _load_source(str(tmp_path), "shenzhen_metadata.csv", "shenzhen")

    assert labels.tolist() == [0, 1, 0]
    assert sources == ["shenzhen", "shenzhen", "shenzhen"]
    assert paths[0].endswith("CHNCXR_0001_0.png")
