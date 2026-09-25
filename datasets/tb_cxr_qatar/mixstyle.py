"""MixStyle domain generalization (Zhou, Yang, Hospedales, Xiang, ICLR 2021,
"Domain Generalization with MixStyle", https://arxiv.org/abs/2104.02008), applied to a
frozen VGG11-BN backbone for the tb_cxr_qatar -> CIDRZ cross-dataset gap (UA 0.973 in-
distribution, 0.527 zero-shot cross-dataset).

Mechanism: during training only, randomly mixes the per-instance channel-wise mean/std
("style" statistics) of intermediate conv feature maps across the batch, forcing whatever
sits downstream to be robust to style variation rather than memorizing it. A TB-specific
domain-generalization paper (MixStyle + multi-level augmentation) reports large cross-
dataset gains on Shenzhen (66%->89%) and a Pakistan TB dataset (71%->78%) - directly
analogous to this project's Qatar->CIDRZ gap.

Applicability to a FROZEN backbone: MixStyle is disabled at eval time and only perturbs
activations during training - it needs no gradient through the layers it's inserted after
to have an effect, only through whatever is downstream and trainable. Here that's
`VGG11Slim`'s final classifier `nn.Linear` (the encoder's only trainable part when
`freeze_features=True`): perturbing the conv feature maps it's built from during training
still forces that trainable layer to learn a style-robust representation, even though the
conv weights producing those feature maps never update. Inserted after the first 3 of
VGG11-BN's 5 pooling stages (indices 4, 9, 18 in `vgg11_bn.features`), per the original
paper's finding that mixing at earlier/shallower stages generalizes better than at the
deepest, most semantic layers.
"""
import random
from typing import Optional, Sequence

import torch
import torchvision.models as tmodels
from torch import nn


class MixStyle(nn.Module):
    """Randomly mixes per-instance channel-wise feature statistics across a batch."""

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        """
        Args:
            p (float): Probability of applying MixStyle to a given forward call (a no-op
                the rest of the time, and always at eval time).
            alpha (float): Beta(alpha, alpha) concentration for the mixing weight - low
                values (paper default 0.1) bias toward near-0/near-1 (mostly one instance's
                own style, occasionally mostly swapped) rather than a bland 50/50 average.
            eps (float): Numerical stability constant for the std.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or random.random() > self.p:
            return x
        batch_size = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        sigma = x.std(dim=[2, 3], keepdim=True) + self.eps
        x_normed = (x - mu) / sigma

        lmda = self.beta.sample((batch_size, 1, 1, 1)).to(x.device)
        perm = torch.randperm(batch_size, device=x.device)
        mu_mixed = mu * lmda + mu[perm] * (1 - lmda)
        sigma_mixed = sigma * lmda + sigma[perm] * (1 - lmda)
        return x_normed * sigma_mixed + mu_mixed


class VGG11SlimMixStyle(nn.Module):
    """`VGG11Slim` (see `unimodals/common_models.py`) with MixStyle inserted after the
    first 3 of its 5 max-pool stages. Same constructor/forward contract as `VGG11Slim` so
    it drops into `configs/*.yaml` via a dotted `type:` path in place of `type: vgg11_slim`.
    """

    def __init__(
        self,
        hiddim: int,
        dropout: bool = True,
        dropoutp: float = 0.2,
        pretrained: bool = True,
        freeze_features: bool = True,
        mixstyle_p: float = 0.5,
        mixstyle_alpha: float = 0.1,
        mixstyle_stages: Sequence[int] = (1, 2, 3),
    ):
        """
        Args:
            hiddim, dropout, dropoutp, pretrained, freeze_features: see `VGG11Slim`.
            mixstyle_p (float): MixStyle's per-call application probability.
            mixstyle_alpha (float): MixStyle's Beta concentration.
            mixstyle_stages (Sequence[int]): which of VGG11-BN's 5 stages (1-indexed) to
                insert MixStyle after. Defaults to the first 3, per the original paper.
        """
        super().__init__()
        self.model = tmodels.vgg11_bn(pretrained=pretrained)
        self.model.classifier = nn.Linear(512 * 7 * 7, hiddim)
        if dropout:
            feats_list = list(self.model.features)
            new_feats_list = []
            for feat in feats_list:
                new_feats_list.append(feat)
                if isinstance(feat, nn.ReLU):
                    new_feats_list.append(nn.Dropout(p=dropoutp))
            self.model.features = nn.Sequential(*new_feats_list)
        for p in self.model.features.parameters():
            p.requires_grad = not freeze_features

        self._mixstyle_stages = set(mixstyle_stages)
        self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply VGG11SlimMixStyle to layer input.

        Args:
            x (torch.Tensor): Layer input, (B, 3, 224, 224).

        Returns:
            torch.Tensor: (B, hiddim) embedding.
        """
        # `stage` counts MaxPool2d layers encountered so far (VGG11-BN has 5, marking the
        # end of each of its 5 stages) - insertion happens right after the pool marking the
        # end of a targeted stage, regardless of how Dropout insertion (in __init__, if
        # `dropout=True`) shifted the Sequential's raw indices.
        stage = 0
        for layer in self.model.features:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                stage += 1
                if stage in self._mixstyle_stages:
                    x = self.mixstyle(x)
        x = torch.flatten(x, 1)
        return self.model.classifier(x)
