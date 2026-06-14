import torch

from datasets.stocks.get_data import _quantize_timeseries


def test_quantize_timeseries_preserves_constant_windows_without_nans():
    x = torch.ones(2, 3)

    quantized = _quantize_timeseries(x)

    assert torch.allclose(quantized, x)
    assert not torch.isnan(quantized).any()


def test_quantize_timeseries_keeps_nonconstant_values_finite_and_bounded():
    x = torch.tensor([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]])

    quantized = _quantize_timeseries(x)

    assert not torch.isnan(quantized).any()
    assert torch.min(quantized) >= torch.min(x)
    assert torch.max(quantized) <= torch.max(x)
