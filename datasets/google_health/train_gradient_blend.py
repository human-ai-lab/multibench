"""Train the google_health TB pipeline with gradient blending instead of naive joint training.

Gradient blending (Wang, Tran, Feiszli, CVPR 2020, "What Makes Training Multi-Modal
Classification Networks Hard?", https://arxiv.org/abs/1905.12681) trains a small
classification head per modality alongside the joint one, periodically re-estimating each
modality's contribution weight from its own overfitting-to-generalization ratio. This can
down-weight a modality that overfits without generalizing - this project's own audio
branch, per the session-long finding that neither hand-rolled mel features nor pretrained
HeAR embeddings showed a measurable, above-noise-floor correlation with the label - rather
than treating every modality equally, as the plain `concat` fusion in
`configs/google_health_tb.yaml` does.

Reuses the same architecture/dataset as a given YAML config (via `utils/config.py`'s
builders), adding one small classification head per modality and overriding
`training_structures.gradient_blend`'s hardcoded unweighted loss with the same class
weights used elsewhere in this project (the module's own example scripts, e.g.
`examples/affect/affect_gradient_blend.py`, show this exact `module.criterion = ...`
override as the supported way to customize its loss).

Cost note: gradient_blend's periodic re-estimation deep-copies and separately retrains
every encoder (once per modality, plus once more for a multimodal copy) each cycle - with
a heavy encoder like `vgg11_slim` this is several times slower than plain joint training.
Consider a lighter image encoder (e.g. `lenet`) for iterating on `--gb-epoch`/`--epochs`
before committing to a long `vgg11_slim` run.

RESULT (tested, not recommended as a default): at this dataset's scale, the per-modality
weight re-estimation is degenerate, not a fix. With `vgg11_slim`, three consecutive
re-estimates gave weights ~[0.87-0.9998, ~0.0-0.13, ~0, ~1e-8] for [text, audio, image,
joint] - the *joint* (fused) head got essentially zero weight every time, so the actual
multimodal model was barely trained (UA 0.565, WA 0.273 - a mirror-image collapse: nearly
all-positive instead of all-negative). The mechanism (Wang et al.'s per-modality
overfitting-to-generalization ratio) needs a large-enough held-out slice to estimate
reliably; `v_rate=0.08` on ~290 training samples is ~20-30 samples, which is not enough
here. A `lenet`-encoder run looked much better (UA 0.671+-0.070 vs. the `concat`+
`run_experiment.py` baseline's 0.520+-0.018) but that comparison is confounded - this
script's own evaluation loop, checkpoint selection, and `finetune_epoch` head-retraining
differ from `run_experiment.py`/`Supervised_Learning`'s, so some (or all) of that gap may
be from those differences rather than gradient blending itself. Kept as a working,
documented alternative training path - not validated as an improvement over `concat`.

Usage:
    python -m datasets.google_health.train_gradient_blend --config configs/google_health_tb.yaml
"""
import argparse

import torch
from torch import nn

from eval_scripts.performance import compute_metrics
from training_structures import gradient_blend
from unimodals.common_models import MLP
from utils.config import build_classifier, build_dataloaders, build_encoders, build_fusion, load_config
from utils.device import get_device


class _FlattenWrapper(nn.Module):
    """Flattens an encoder's output to (B, D) - gradient_blend applies a flat classification
    head directly to each encoder's raw output, but `lenet` (with `squeeze_output: false`,
    as this project's Concat-fusion configs use to survive batch-size-1) returns
    (B, C, 1, 1) unflattened for Concat's own internal flatten() to handle. Module-level
    (not a local class) so the trained model can still be pickled by `torch.save`.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.module(x), start_dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=20, help="Total epochs (across all gb_epoch cycles).")
    parser.add_argument("--gb-epoch", type=int, default=5, help="Epochs between gradient-blend weight re-estimation.")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    encoders = [_FlattenWrapper(e).to(device) for e in build_encoders(config["model"], device)]
    fusion = build_fusion(config["model"], device)
    classifier = build_classifier(config["model"], device)

    training_cfg = config["training"]
    weight = training_cfg.get("objective_kwargs", {}).get("weight")
    gradient_blend.criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weight, dtype=torch.float32).to(device) if weight else None
    )

    traindata, validdata, testdata = build_dataloaders(config["dataset"])

    # One small classification head per modality, sized to that encoder's actual output dim
    # (gradient_blend.train's per-modality loss term needs this - see its docstring).
    sample_batch = next(iter(traindata))
    with torch.no_grad():
        encoder_out_dims = [
            encoder(x.float().to(device)).shape[-1] for encoder, x in zip(encoders, sample_batch[:-1])
        ]
    n_classes = classifier(fusion([encoder(x.float().to(device)) for encoder, x in zip(encoders, sample_batch[:-1])])).shape[-1]
    unimodal_heads = [MLP(dim, max(8, dim // 2), n_classes).to(device) for dim in encoder_out_dims]

    save_path = training_cfg.get("save", "results/models/from_config.pt") + ".gradient_blend"
    gradient_blend.train(
        encoders, classifier, unimodal_heads, fusion, traindata, validdata,
        num_epoch=args.epochs, gb_epoch=args.gb_epoch, lr=training_cfg.get("lr", 0.001),
        weight_decay=training_cfg.get("weight_decay", 0.0), optimtype=torch.optim.AdamW,
        savedir=save_path,
    )

    model = torch.load(save_path, weights_only=False).to(device)
    model.eval()
    truths, preds = [], []
    with torch.no_grad():
        for batch in testdata:
            inputs = [x.float().to(device) for x in batch[:-1]]
            labels = batch[-1]
            outs = model(inputs)
            preds.append(torch.argmax(outs, dim=1).cpu())
            truths.append(labels)
    truths = torch.cat(truths)
    preds = torch.cat(preds)
    results = compute_metrics(truths, preds, config.get("evaluation", ["UA", "WA", "F1"]))
    print("Evaluation results:")
    for name, value in results.items():
        print(f"  {name}: {value}")
    return results


if __name__ == "__main__":
    main()
