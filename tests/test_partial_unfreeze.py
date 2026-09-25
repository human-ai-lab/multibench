import torch

from datasets.tb_cxr_qatar.partial_unfreeze import VGG11SlimPartialUnfreeze


def test_only_last_stage_unfrozen_by_default():
    enc = VGG11SlimPartialUnfreeze(64, pretrained=False, unfreeze_last_n_stages=1)
    trainable = [p for p in enc.model.features.parameters() if p.requires_grad]
    frozen = [p for p in enc.model.features.parameters() if not p.requires_grad]
    assert len(trainable) > 0
    assert len(frozen) > 0
    # classifier's own final Linear is always trainable regardless of features freezing
    assert enc.model.classifier.weight.requires_grad


def test_unfreezing_more_stages_unfreezes_more_params():
    enc1 = VGG11SlimPartialUnfreeze(64, pretrained=False, unfreeze_last_n_stages=1)
    enc2 = VGG11SlimPartialUnfreeze(64, pretrained=False, unfreeze_last_n_stages=2)
    n1 = sum(p.numel() for p in enc1.model.features.parameters() if p.requires_grad)
    n2 = sum(p.numel() for p in enc2.model.features.parameters() if p.requires_grad)
    assert n2 > n1


def test_batchnorm_stays_frozen_in_unfrozen_stage():
    enc = VGG11SlimPartialUnfreeze(64, pretrained=False, unfreeze_last_n_stages=1)
    import torch.nn as nn
    for m in enc.model.features.modules():
        if isinstance(m, nn.BatchNorm2d):
            for p in m.parameters():
                assert not p.requires_grad


def test_forward_shape():
    enc = VGG11SlimPartialUnfreeze(64, pretrained=False, unfreeze_last_n_stages=1)
    x = torch.randn(2, 3, 224, 224)
    out = enc(x)
    assert out.shape == (2, 64)
