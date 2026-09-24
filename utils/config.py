"""YAML-driven configuration for MultiBench experiments.

Wraps MultiBench's existing building blocks - dataset loaders, unimodal encoders
(unimodals/common_models.py), fusion modules (fusions/common_fusions.py) and the
Supervised_Learning training loop - behind a single YAML config, e.g.::

    dataset:
      name: affect
      path: data/affect/mosi_raw.pkl
      kwargs:
        data_type: mosi
        robust_test: false

    model:
      features:                # per-modality encoders; "encoders" also accepted
        - type: gru
          args: [35, 70]
          kwargs: {dropout: true, has_padding: true, batch_first: true}
        - type: gru
          args: [74, 200]
          kwargs: {dropout: true, has_padding: true, batch_first: true}
      fusion:
        type: concat
      classifier:               # the classification/regression head; "head" also accepted
        type: mlp
        args: [270, 270, 1]

    training:
      task: regression
      epochs: 40
      optimizer: adamw
      lr: 0.001
      objective: l1
      is_packed: true
      save: results/models/from_config.pt

    evaluation:
      - UA
      - WA
      - F1

Anything outside the curated registries can be reached with a dotted path instead of a
short name - datasets via ``dataset: {loader: some.module.get_dataloader, ...}``, and
encoders/fusion/classifier via ``type: some.module.SomeClass`` - so any class anywhere
in the repo can be wired in even if it isn't one of the common ones listed below.
`args`/`kwargs` entries can themselves be nested ``{type, args, kwargs}`` specs, which
is what lets ``type: sequential`` compose child modules, e.g.::

    - type: sequential
      args:
        - {type: transpose, args: [1, 2]}
        - {type: gru, args: [35, 70], kwargs: {batch_first: true}}
"""
import importlib
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from fusions import common_fusions
from unimodals import common_models
from training_structures.Supervised_Learning import train as _train, test as _test
from utils.device import get_device

ENCODER_REGISTRY = {
    "linear": common_models.Linear,
    "mlp": common_models.MLP,
    "gru": common_models.GRU,
    "gru_with_linear": common_models.GRUWithLinear,
    "lstm": common_models.LSTM,
    "two_layers_lstm": common_models.TwoLayersLSTM,
    "lenet": common_models.LeNet,
    "transformer": common_models.Transformer,
    "identity": common_models.Identity,
    "maxout_mlp": common_models.MaxOut_MLP,
    "maxout": common_models.Maxout,
    "sequential": common_models.Sequential,
    "squeeze": common_models.Squeeze,
    "reshape": common_models.Reshape,
    "transpose": common_models.Transpose,
    "constant": common_models.Constant,
    "global_pooling_2d": common_models.GlobalPooling2D,
    "dan": common_models.DAN,
    "resnet_lstm_enc": common_models.ResNetLSTMEnc,
    "vgg": common_models.VGG,
    "vgg16": common_models.VGG16,
    "vgg16_slim": common_models.VGG16Slim,
    "vgg11_slim": common_models.VGG11Slim,
    "vgg11_pruned": common_models.VGG11Pruned,
    "vgg16_pruned": common_models.VGG16Pruned,
}
# The classifier/head is built from the same set of building blocks as encoders.
CLASSIFIER_REGISTRY = ENCODER_REGISTRY

FUSION_REGISTRY = {
    "concat": common_fusions.Concat,
    "concat_early": common_fusions.ConcatEarly,
    "stack": common_fusions.Stack,
    "concat_with_linear": common_fusions.ConcatWithLinear,
    "tensor_fusion": common_fusions.TensorFusion,
    "low_rank_tensor_fusion": common_fusions.LowRankTensorFusion,
    "nlgate": common_fusions.NLgate,
    "early_fusion_transformer": common_fusions.EarlyFusionTransformer,
    "late_fusion_transformer": common_fusions.LateFusionTransformer,
    "multiplicative_interactions_2modal": common_fusions.MultiplicativeInteractions2Modal,
    "multiplicative_interactions_3modal": common_fusions.MultiplicativeInteractions3Modal,
}

OPTIMIZER_REGISTRY = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "rmsprop": torch.optim.RMSprop,
}

OBJECTIVE_REGISTRY = {
    "cross_entropy": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "bce_with_logits": nn.BCEWithLogitsLoss,
}

# Short names for dataset loaders whose get_dataloader(path, ...) returns a plain
# (train, valid, test) tuple, so they fit `dataset: {name: ..., path: ..., kwargs: ...}`
# directly. Anything else - a different call signature or return shape - can still be
# used with `dataset: {loader: "<dotted.path.to.get_dataloader>", args: [...], ...}`.
DATASET_REGISTRY = {
    "affect": "datasets.affect.get_data.get_dataloader",
    "avmnist": "datasets.avmnist.get_data.get_dataloader",
    "google_health": "datasets.google_health.get_data.get_dataloader",
    "tb_cxr_qatar": "datasets.tb_cxr_qatar.get_data.get_dataloader",
}
# Not registered: datasets.enrico.get_data.get_dataloader always builds its test split as
# an unconditional dict of per-noise-level dataloaders (see its `dl_test = dict()` /
# `dl_test['image'] = [...]`), in every branch regardless of `return_class_weights` - so
# it can never satisfy this config system's "test is a plain DataLoader" requirement and
# would always trip the robust_test dict guard below. Would need get_data.py changed
# upstream (e.g. an argument to opt out of the noise sweep) before it could be registered.


def load_config(path: str) -> Dict[str, Any]:
    """Load and parse a YAML experiment config from disk."""
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_dotted(path: str):
    if "." not in path:
        raise ValueError(
            f"'{path}' is not a dotted path (expected '<module.path>.<attr>', "
            "e.g. 'datasets.affect.get_data.get_dataloader')")
    module_path, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _resolve_type(type_name: str, registry: Dict[str, type], kind: str) -> type:
    """Resolve a `type` string to a class: short registry name, or a dotted path."""
    if "." in type_name:
        resolved = _resolve_dotted(type_name)
        if not isinstance(resolved, type):
            raise ValueError(f"'{type_name}' does not resolve to a class")
        return resolved
    cls = registry.get(type_name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown {kind} type '{type_name}'. Supported: {sorted(registry)} "
            "(or a dotted path, e.g. 'unimodals.common_models.MLP')")
    return cls


def _build_arg(value: Any, registry: Dict[str, type], kind: str) -> Any:
    """Recursively build a nested {type, args, kwargs} spec; pass any other value through."""
    if isinstance(value, dict) and "type" in value:
        return _build_module(value, registry, kind)
    if isinstance(value, list):
        return [_build_arg(v, registry, kind) for v in value]
    return value


def _build_module(spec: Dict[str, Any], registry: Dict[str, type], kind: str) -> nn.Module:
    cls = _resolve_type(str(spec["type"]), registry, kind)
    args = [_build_arg(a, registry, kind) for a in spec.get("args", [])]
    kwargs = {k: _build_arg(v, registry, kind) for k, v in spec.get("kwargs", {}).items()}
    return cls(*args, **kwargs)


def build_encoders(model_cfg: Dict[str, Any], device: torch.device) -> List[nn.Module]:
    """Build the list of per-modality encoders from `model.features` (or `model.encoders`)."""
    specs = model_cfg.get("features") or model_cfg.get("encoders")
    if not specs:
        raise ValueError("model config must define 'features' (or 'encoders')")
    return [_build_module(spec, ENCODER_REGISTRY, "encoder").to(device) for spec in specs]


def build_fusion(model_cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    """Build the fusion module from `model.fusion`."""
    spec = model_cfg.get("fusion")
    if not spec:
        raise ValueError("model config must define 'fusion'")
    return _build_module(spec, FUSION_REGISTRY, "fusion").to(device)


def build_classifier(model_cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    """Build the classification/regression head from `model.classifier` (or `model.head`)."""
    spec = model_cfg.get("classifier") or model_cfg.get("head")
    if not spec:
        raise ValueError("model config must define 'classifier' (or 'head')")
    return _build_module(spec, CLASSIFIER_REGISTRY, "classifier").to(device)


def build_dataloaders(dataset_cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, Any]:
    """Build (train, valid, test) dataloaders from the `dataset` section."""
    if "loader" in dataset_cfg:
        loader = _resolve_dotted(dataset_cfg["loader"])
    else:
        name = str(dataset_cfg.get("name", "")).lower()
        if name not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset '{dataset_cfg.get('name')}'. Supported: {sorted(DATASET_REGISTRY)}. "
                "Use 'loader: <dotted.path.to.get_dataloader>' for other datasets.")
        loader = _resolve_dotted(DATASET_REGISTRY[name])

    args = list(dataset_cfg.get("args", []))
    if "path" in dataset_cfg:
        args.insert(0, dataset_cfg["path"])
    kwargs = dataset_cfg.get("kwargs", {})
    result = loader(*args, **kwargs)

    if not (isinstance(result, (tuple, list)) and len(result) == 3):
        got = len(result) if isinstance(result, (tuple, list)) else type(result).__name__
        raise ValueError(
            "Dataset loader must return exactly (train, valid, test) dataloaders, got "
            f"{got} element(s) instead. Some loaders (e.g. enrico) return extra values by "
            "default - check its docstring for a kwarg to disable that.")
    train, valid, test = result
    if isinstance(test, dict):
        raise ValueError(
            "The test split came back as a dict of per-modality noise dataloaders "
            "(robust_test=True), which run_experiment.py's single-pass evaluation does not "
            "support - set robust_test: false (or omit it) in the dataset config.")
    return train, valid, test


def build_optimizer_type(training_cfg: Dict[str, Any]) -> type:
    name = str(training_cfg.get("optimizer", "rmsprop")).lower()
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer '{name}'. Supported: {sorted(OPTIMIZER_REGISTRY)}")
    return OPTIMIZER_REGISTRY[name]


def build_objective(training_cfg: Dict[str, Any], device: Optional[torch.device] = None) -> nn.Module:
    """Build the training objective from `training.objective` (+ optional `objective_kwargs`).

    `objective_kwargs.weight` (a list of per-class weights, e.g. inverse class frequency for
    an imbalanced dataset) is converted to a tensor and moved to `device` automatically -
    `nn.CrossEntropyLoss`/`nn.BCEWithLogitsLoss` otherwise reject a plain list.
    """
    name = str(training_cfg.get("objective", "cross_entropy")).lower()
    if name not in OBJECTIVE_REGISTRY:
        raise ValueError(f"Unknown objective '{name}'. Supported: {sorted(OBJECTIVE_REGISTRY)}")
    kwargs = dict(training_cfg.get("objective_kwargs", {}))
    if "weight" in kwargs:
        weight = torch.tensor(kwargs["weight"], dtype=torch.float32)
        kwargs["weight"] = weight.to(device) if device is not None else weight
    return OBJECTIVE_REGISTRY[name](**kwargs)


def run(config: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Build and run a full train + evaluate experiment from a parsed config dict.

    Trains with `training_structures.Supervised_Learning.train`, then evaluates the
    best checkpoint on the test dataloader with `..Supervised_Learning.test`
    (robustness sweep disabled), applying the metrics named in `evaluation`.

    :return: dict of computed evaluation metrics (always populated, since this always
        runs with the robustness sweep disabled - see `Supervised_Learning.test`).
    """
    device = get_device()

    # Resolve training config (optimizer/objective/etc.) and build the model up front,
    # before the (potentially slow) dataset load, so a config typo fails fast.
    training_cfg = config.get("training", {})
    task = training_cfg.get("task", "classification")
    is_packed = training_cfg.get("is_packed", False)
    save_path = training_cfg.get("save", "results/models/from_config.pt")
    optimizer_type = build_optimizer_type(training_cfg)
    objective = build_objective(training_cfg, device)

    model_cfg = config["model"]
    encoders = build_encoders(model_cfg, device)
    fusion = build_fusion(model_cfg, device)
    classifier = build_classifier(model_cfg, device)

    traindata, validdata, testdata = build_dataloaders(config["dataset"])

    _train(
        encoders, fusion, classifier, traindata, validdata,
        total_epochs=training_cfg.get("epochs", 10),
        task=task,
        optimtype=optimizer_type,
        lr=training_cfg.get("lr", 0.001),
        weight_decay=training_cfg.get("weight_decay", 0.0),
        objective=objective,
        is_packed=is_packed,
        early_stop=training_cfg.get("early_stop", False),
        save=save_path,
    )

    model = torch.load(save_path, weights_only=False).to(device)
    return _test(
        model=model,
        test_dataloaders_all=testdata,
        is_packed=is_packed,
        criterion=objective,
        task=task,
        no_robust=True,
        metrics=config.get("evaluation"),
    )


def run_from_file(path: str) -> Optional[Dict[str, float]]:
    """Load a YAML config from `path` and run it end to end."""
    return run(load_config(path))
