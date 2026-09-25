"""VGG11Slim variant with the last N feature-extraction stages unfrozen.

Motivated by MixStyle's diagnosed failure mode on this project's Qatar->CIDRZ cross-dataset
task (see mixstyle.py): with `VGG11Slim`'s default `freeze_features=True`, only the final
`nn.Linear` classifier layer is trainable, which is not enough downstream capacity to
compensate for anything happening to the frozen backbone's features - whether that's
MixStyle's intentional perturbation or simply CIDRZ's different imaging distribution.

This unfreezes the last `unfreeze_last_n_stages` convolutional stages (a "stage" = the
layers between consecutive MaxPool2d layers) of VGG11-BN's `features`, at a low learning
rate, so the model has a chance to actually adapt low-level filters toward
domain-invariant features rather than only re-weighting whatever the fixed ImageNet
backbone happens to produce. All earlier stages, and BatchNorm2d layers within the
unfrozen stages, remain frozen (unfreezing BN there would let it re-learn Qatar-specific
running statistics, defeating AdaBN-style domain adaptation applied downstream).
"""
from typing import List

import torch.nn as nn

from unimodals.common_models import VGG11Slim


class VGG11SlimPartialUnfreeze(VGG11Slim):
    def __init__(self, hiddim: int, dropout: bool = True, dropoutp: float = 0.2,
                 pretrained: bool = True, unfreeze_last_n_stages: int = 1):
        super().__init__(hiddim, dropout=dropout, dropoutp=dropoutp, pretrained=pretrained,
                          freeze_features=True)
        self.unfrozen_params: List[nn.Parameter] = []
        feats = list(self.model.features)
        pool_indices = [i for i, m in enumerate(feats) if isinstance(m, nn.MaxPool2d)]
        n_stages = len(pool_indices)
        if unfreeze_last_n_stages >= n_stages:
            start_idx = -1
        else:
            start_idx = pool_indices[n_stages - unfreeze_last_n_stages - 1]

        for i, m in enumerate(feats):
            if i > start_idx and not isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = True
                    self.unfrozen_params.append(p)
