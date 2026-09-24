"""Domain-pretrained chest X-ray encoder via torchxrayvision.

Wraps a DenseNet121 pretrained on real chest X-ray datasets (NIH, CheXpert, MIMIC-CXR,
PadChest, RSNA, etc. - see torchxrayvision's `xrv.models.model_urls` for the exact weight
sets) rather than generic ImageNet, which `vgg11_slim` uses. Domain-matched pretraining is
the improvement the CXR-classification literature actually supports at small sample sizes
(see `download.py`'s `image_encoder` docstring) - not training a CNN-ViT hybrid from
scratch, and arguably a better match than ImageNet transfer for this exact modality.

Requires the `google_health_xrv` extra (`pip install -e '.[google_health_xrv]'`); the first
use downloads the pretrained weights (~30MB) to `~/.torchxrayvision/models_data/`.

Referenced from `configs/google_health_tb.yaml` via a dotted path (MultiBench's config
system resolves any `type: some.module.ClassName` - see `utils/config.py`'s
`_resolve_type`), not the unimodals registry, since it's specific to this one dataset.
"""
import torch
from torch import nn


class XRVDenseNetEncoder(nn.Module):
    """Chest-X-ray-domain-pretrained image encoder with a small trainable head.

    Expects input prepared by `features.extract_image_features_xrv`: (B, 1, 224, 224),
    single channel, scaled to ~[-1024, 1024].
    """

    def __init__(
        self,
        hiddim: int,
        weights: str = "densenet121-res224-all",
        freeze_features: bool = True,
        dropoutp: float = 0.2,
    ):
        """Initialize XRVDenseNetEncoder.

        Args:
            hiddim (int): Output dimension of the trainable head.
            weights (str, optional): Which torchxrayvision pretrained weight set to load.
                Defaults to "densenet121-res224-all" (trained across all its datasets combined).
            freeze_features (bool, optional): Whether to keep the pretrained backbone frozen.
                Defaults to True.
            dropoutp (float, optional): Dropout probability in the trainable head. Defaults to 0.2.
        """
        super().__init__()
        import torchxrayvision as xrv

        self.backbone = xrv.models.DenseNet(weights=weights)
        for p in self.backbone.parameters():
            p.requires_grad = not freeze_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropoutp), nn.Linear(1024, hiddim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply XRVDenseNetEncoder to layer input.

        Args:
            x (torch.Tensor): Layer input, (B, 1, 224, 224).

        Returns:
            torch.Tensor: (B, hiddim) embedding.
        """
        features = self.backbone.features(x)
        return self.head(self.pool(features))
