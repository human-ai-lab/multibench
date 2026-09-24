import numpy as np

from datasets.tb_cxr_qatar.download import _load_pixel_array


def test_load_pixel_array_grayscale(tmp_path):
    from PIL import Image

    path = tmp_path / "sample.png"
    Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)).save(path)

    arr = _load_pixel_array(str(path))

    assert arr.shape == (16, 16)
    assert arr.dtype == np.uint8
