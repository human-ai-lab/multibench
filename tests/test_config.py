import pytest
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from eval_scripts.performance import compute_metrics
from utils.config import (
    build_classifier,
    build_dataloaders,
    build_encoders,
    build_fusion,
    build_objective,
    build_optimizer_type,
    load_config,
    run,
    run_from_file,
)


def test_load_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dataset:\n  name: affect\n  path: data/x.pkl\n")

    config = load_config(str(config_path))

    assert config == {"dataset": {"name": "affect", "path": "data/x.pkl"}}


def test_build_encoders_from_features_key():
    model_cfg = {"features": [{"type": "mlp", "args": [2, 8, 4]}]}

    encoders = build_encoders(model_cfg, torch.device("cpu"))

    assert len(encoders) == 1
    assert encoders[0](torch.zeros(1, 2)).shape == (1, 4)


def test_build_encoders_from_encoders_key():
    model_cfg = {"encoders": [{"type": "mlp", "args": [2, 8, 4]}]}

    encoders = build_encoders(model_cfg, torch.device("cpu"))

    assert len(encoders) == 1


def test_build_encoders_missing_raises():
    with pytest.raises(ValueError, match="features"):
        build_encoders({}, torch.device("cpu"))


def test_build_fusion():
    fusion = build_fusion({"fusion": {"type": "concat"}}, torch.device("cpu"))
    out = fusion([torch.ones(1, 2), torch.ones(1, 3)])
    assert out.shape == (1, 5)


def test_build_fusion_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown fusion type"):
        build_fusion({"fusion": {"type": "not_a_fusion"}}, torch.device("cpu"))


def test_build_classifier_from_classifier_key():
    classifier = build_classifier({"classifier": {"type": "mlp", "args": [4, 8, 2]}}, torch.device("cpu"))
    assert classifier(torch.zeros(1, 4)).shape == (1, 2)


def test_build_classifier_from_head_key():
    classifier = build_classifier({"head": {"type": "mlp", "args": [4, 8, 2]}}, torch.device("cpu"))
    assert classifier(torch.zeros(1, 4)).shape == (1, 2)


def test_build_optimizer_type_unknown_raises():
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer_type({"optimizer": "not_an_optimizer"})


def test_build_objective_unknown_raises():
    with pytest.raises(ValueError, match="Unknown objective"):
        build_objective({"objective": "not_an_objective"})


def test_build_objective_with_class_weight():
    objective = build_objective({"objective": "cross_entropy", "objective_kwargs": {"weight": [0.5, 2.0]}})

    assert isinstance(objective, torch.nn.CrossEntropyLoss)
    assert torch.allclose(objective.weight, torch.tensor([0.5, 2.0]))


def test_build_objective_with_class_weight_moves_to_device():
    objective = build_objective(
        {"objective": "cross_entropy", "objective_kwargs": {"weight": [0.5, 2.0]}},
        device=torch.device("cpu"),
    )

    assert objective.weight.device == torch.device("cpu")


def test_build_dataloaders_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        build_dataloaders({"name": "not_a_dataset"})


def test_build_dataloaders_malformed_loader_path_raises():
    with pytest.raises(ValueError, match="not a dotted path"):
        build_dataloaders({"loader": "not_a_dotted_path"})


def test_build_dataloaders_wrong_arity_raises():
    with pytest.raises(ValueError, match="exactly \\(train, valid, test\\)"):
        build_dataloaders({"loader": "tests.test_config.fake_get_dataloader_two_tuple"})


def test_build_dataloaders_robust_test_dict_raises():
    with pytest.raises(ValueError, match="robust_test=True"):
        build_dataloaders({"loader": "tests.test_config.fake_get_dataloader_robust"})


def test_build_encoders_via_dotted_path():
    model_cfg = {"features": [{"type": "unimodals.common_models.MLP", "args": [2, 8, 4]}]}

    encoders = build_encoders(model_cfg, torch.device("cpu"))

    assert isinstance(encoders[0], __import__("unimodals.common_models", fromlist=["MLP"]).MLP)
    assert encoders[0](torch.zeros(1, 2)).shape == (1, 4)


def test_build_fusion_via_dotted_path():
    fusion = build_fusion({"fusion": {"type": "fusions.common_fusions.Concat"}}, torch.device("cpu"))
    out = fusion([torch.ones(1, 2), torch.ones(1, 3)])
    assert out.shape == (1, 5)


def test_build_encoders_sequential_composition():
    model_cfg = {
        "features": [
            {
                "type": "sequential",
                "args": [
                    {"type": "mlp", "args": [2, 8, 4]},
                    {"type": "mlp", "args": [4, 8, 2]},
                ],
            }
        ]
    }

    encoders = build_encoders(model_cfg, torch.device("cpu"))

    assert encoders[0](torch.zeros(1, 2)).shape == (1, 2)


def test_compute_metrics_unknown_name_raises():
    truth = torch.tensor([0, 1])
    pred = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="Unknown evaluation metric"):
        compute_metrics(truth, pred, ["not_a_metric"])


class _MultimodalDataset(Dataset):
    def __init__(self, xs, y):
        self.xs = xs
        self.y = y

    def __getitem__(self, index):
        return [*[x[index] for x in self.xs], self.y[index]]

    def __len__(self):
        return len(self.y)


def fake_get_dataloader(batch_size=4):
    """Stand-in for a dataset's get_dataloader, resolved via dotted path in tests."""
    x1 = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]] * 2)
    x2 = torch.tensor([[0., 1.], [1., 0.], [1., 0.], [0., 1.]] * 2)
    y = torch.tensor([0, 1, 1, 0] * 2, dtype=torch.long)
    loader = DataLoader(_MultimodalDataset([x1, x2], y), batch_size=batch_size)
    return loader, loader, loader


def test_build_dataloaders_via_dotted_loader():
    train, valid, test = build_dataloaders(
        {"loader": "tests.test_config.fake_get_dataloader", "kwargs": {"batch_size": 4}})
    assert train is valid is test


def fake_get_dataloader_two_tuple():
    """Stand-in for a loader with an unsupported return arity, e.g. enrico's default."""
    return None, None


def fake_get_dataloader_robust():
    """Stand-in for get_dataloader(robust_test=True): test split comes back as a dict."""
    return None, None, {"vision": [None]}


def test_run_end_to_end_classification(tmp_path):
    torch.manual_seed(0)
    config = {
        "dataset": {
            "loader": "tests.test_config.fake_get_dataloader",
            "kwargs": {"batch_size": 4},
        },
        "model": {
            "features": [
                {"type": "mlp", "args": [2, 8, 4]},
                {"type": "mlp", "args": [2, 8, 4]},
            ],
            "fusion": {"type": "concat"},
            "classifier": {"type": "mlp", "args": [8, 8, 2]},
        },
        "training": {
            "task": "classification",
            "epochs": 5,
            "optimizer": "adam",
            "objective": "cross_entropy",
            "save": str(tmp_path / "model.pt"),
        },
        "evaluation": ["UA", "WA", "F1"],
    }

    results = run(config)

    assert set(results) >= {"Accuracy", "UA", "WA", "F1"}
    for value in results.values():
        assert 0.0 <= value <= 1.0


class _PackedMultimodalDataset(Dataset):
    """Variable-length sequences, like affect data - exercises `has_padding=True`."""

    def __init__(self, seqs1, seqs2, y):
        self.seqs1 = seqs1
        self.seqs2 = seqs2
        self.y = y

    def __getitem__(self, index):
        return self.seqs1[index], self.seqs2[index], self.y[index]

    def __len__(self):
        return len(self.y)


def _packed_collate(batch):
    """Mirrors datasets.affect.get_data._process_1's (inputs, lengths, ..., labels) shape."""
    seqs1 = [b[0] for b in batch]
    seqs2 = [b[1] for b in batch]
    lengths1 = torch.as_tensor([v.size(0) for v in seqs1])
    lengths2 = torch.as_tensor([v.size(0) for v in seqs2])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long).view(len(batch), 1)
    return (
        [pad_sequence(seqs1, batch_first=True), pad_sequence(seqs2, batch_first=True)],
        [lengths1, lengths2],
        labels,
    )


def fake_get_dataloader_packed(batch_size=4):
    """Stand-in for get_dataloader(..., max_pad=False) - the shape configs/affect_mosi_late_fusion.yaml relies on."""
    torch.manual_seed(0)
    seqs1 = [torch.randn(3, 2), torch.randn(5, 2), torch.randn(4, 2), torch.randn(3, 2)] * 2
    seqs2 = [torch.randn(3, 2), torch.randn(5, 2), torch.randn(4, 2), torch.randn(3, 2)] * 2
    y = [0, 1, 1, 0] * 2
    loader = DataLoader(_PackedMultimodalDataset(seqs1, seqs2, y),
                        batch_size=batch_size, collate_fn=_packed_collate)
    return loader, loader, loader


def test_run_end_to_end_packed_classification(tmp_path):
    """Exercises the exact GRU(has_padding=True) + Concat + MLP path configs/affect_mosi_late_fusion.yaml uses."""
    torch.manual_seed(0)
    config = {
        "dataset": {
            "loader": "tests.test_config.fake_get_dataloader_packed",
            "kwargs": {"batch_size": 4},
        },
        "model": {
            "features": [
                {"type": "gru", "args": [2, 4], "kwargs": {"has_padding": True, "batch_first": True}},
                {"type": "gru", "args": [2, 4], "kwargs": {"has_padding": True, "batch_first": True}},
            ],
            "fusion": {"type": "concat"},
            "classifier": {"type": "mlp", "args": [8, 8, 2]},
        },
        "training": {
            "task": "classification",
            "epochs": 3,
            "optimizer": "adam",
            "objective": "cross_entropy",
            "is_packed": True,
            "save": str(tmp_path / "packed_model.pt"),
        },
        "evaluation": ["UA", "WA", "UAR", "F1"],
    }

    results = run(config)

    assert set(results) >= {"Accuracy", "UA", "WA", "UAR", "F1"}
    for value in results.values():
        assert 0.0 <= value <= 1.0


def test_run_from_file(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
dataset:
  loader: tests.test_config.fake_get_dataloader
  kwargs: {{batch_size: 4}}
model:
  features:
    - {{type: mlp, args: [2, 8, 4]}}
    - {{type: mlp, args: [2, 8, 4]}}
  fusion: {{type: concat}}
  classifier: {{type: mlp, args: [8, 8, 2]}}
training:
  task: classification
  epochs: 2
  optimizer: adam
  objective: cross_entropy
  save: {tmp_path / "model.pt"}
evaluation: [UA, WA]
""")

    results = run_from_file(str(config_path))

    assert set(results) >= {"Accuracy", "UA", "WA"}


def test_single_test_metrics_require_classification_task():
    from training_structures.Supervised_Learning import single_test

    model = torch.nn.Linear(2, 1)
    loader = DataLoader(_MultimodalDataset([torch.zeros(1, 2)], torch.zeros(1, 1)), batch_size=1)
    with pytest.raises(ValueError, match="requires task='classification'"):
        single_test(model, loader, task="regression", metrics=["UA"])
