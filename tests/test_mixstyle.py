import torch

from datasets.tb_cxr_qatar.mixstyle import MixStyle, VGG11SlimMixStyle


def test_mixstyle_is_noop_at_eval():
    ms = MixStyle(p=1.0)
    ms.eval()
    x = torch.randn(4, 8, 7, 7)

    assert torch.allclose(ms(x), x)


def test_mixstyle_changes_output_at_train_with_p1():
    torch.manual_seed(0)
    ms = MixStyle(p=1.0, alpha=0.1)
    ms.train()
    x = torch.randn(4, 8, 7, 7)

    out = ms(x)

    assert out.shape == x.shape
    assert not torch.allclose(out, x)


def test_mixstyle_preserves_shape_with_single_sample():
    ms = MixStyle(p=1.0)
    ms.train()
    x = torch.randn(1, 8, 7, 7)

    out = ms(x)

    assert out.shape == x.shape


def test_vgg11_slim_mixstyle_output_shape():
    enc = VGG11SlimMixStyle(128, pretrained=False)
    x = torch.zeros(3, 3, 224, 224)

    enc.train()
    out_train = enc(x)
    enc.eval()
    out_eval = enc(x)

    assert out_train.shape == (3, 128)
    assert out_eval.shape == (3, 128)


def test_vgg11_slim_mixstyle_eval_is_deterministic():
    torch.manual_seed(0)
    enc = VGG11SlimMixStyle(128, pretrained=False)
    x = torch.randn(2, 3, 224, 224)

    enc.eval()
    out1 = enc(x)
    out2 = enc(x)

    assert torch.allclose(out1, out2)
