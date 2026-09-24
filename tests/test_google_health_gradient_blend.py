import pickle

import torch

from datasets.google_health.train_gradient_blend import _FlattenWrapper


def test_flatten_wrapper_flattens_spatial_output():
    module = torch.nn.Identity()
    wrapper = _FlattenWrapper(module)

    out = wrapper(torch.zeros(4, 8, 1, 1))

    assert out.shape == (4, 8)


def test_flatten_wrapper_is_noop_on_already_flat_output():
    module = torch.nn.Identity()
    wrapper = _FlattenWrapper(module)

    out = wrapper(torch.zeros(4, 16))

    assert out.shape == (4, 16)


def test_flatten_wrapper_is_picklable():
    wrapper = _FlattenWrapper(torch.nn.Linear(4, 2))

    pickle.dumps(wrapper)
