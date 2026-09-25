import numpy as np
import torch

from datasets.tb_cxr_qatar.few_shot_finetune import sample_k_shot, train_head
from unimodals.common_models import MLP


def test_sample_k_shot_is_balanced_and_correct_size():
    labels = np.array([0] * 20 + [1] * 10)
    idx = sample_k_shot(labels, k=5, seed=0)
    assert len(idx) == 10
    assert (labels[idx] == 0).sum() == 5
    assert (labels[idx] == 1).sum() == 5


def test_sample_k_shot_is_deterministic_per_seed():
    labels = np.array([0] * 20 + [1] * 10)
    idx1 = sample_k_shot(labels, k=5, seed=0)
    idx2 = sample_k_shot(labels, k=5, seed=0)
    idx3 = sample_k_shot(labels, k=5, seed=1)
    assert np.array_equal(idx1, idx2)
    assert not np.array_equal(idx1, idx3)


def test_train_head_default_weight_is_balanced():
    import inspect
    sig = inspect.signature(train_head)
    assert sig.parameters["class_weight"].default == (1.0, 1.0)


def test_train_head_reduces_loss_on_separable_data():
    device = torch.device("cpu")
    torch.manual_seed(0)
    features = torch.cat([torch.zeros(10, 4), torch.ones(10, 4)], dim=0)
    labels = torch.tensor([0] * 10 + [1] * 10)
    head = MLP(4, 8, 2)

    def _loss(h):
        with torch.no_grad():
            return torch.nn.functional.cross_entropy(h(features), labels).item()

    before = _loss(head)
    head = train_head(head, features, labels, device, epochs=100, lr=1e-2)
    after = _loss(head)
    assert after < before
