"""Small helpers for running lesson scripts with the local multibench checkout."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def _find_multibench_root() -> Path:
    env_root = os.getenv("MULTIBENCH_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root).expanduser())

    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parents[2],
            here.parents[3] / "multibench",
            Path.cwd(),
            Path.cwd().parent / "multibench",
        ]
    )

    for candidate in candidates:
        if (candidate / "training_structures" / "Supervised_Learning.py").exists():
            return candidate

    raise RuntimeError(
        "Could not find the multibench checkout. Set MULTIBENCH_ROOT to its path."
    )


MULTIBENCH_ROOT = _find_multibench_root()
if str(MULTIBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIBENCH_ROOT))

from fusions.common_fusions import Concat  # noqa: E402
from training_structures.Supervised_Learning import single_test, train  # noqa: E402
from unimodals.common_models import MLP  # noqa: E402
from utils.device import get_device  # noqa: E402


class ArrayDataset(Dataset):
    """Dataset that returns one modality tensor and one target tensor."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        task: str,
    ) -> None:
        self.x = torch.as_tensor(np.array(x, dtype=np.float32, copy=True))
        if task == "classification":
            self.y = torch.as_tensor(np.array(y, dtype=np.int64, copy=True))
        elif task == "regression":
            self.y = torch.as_tensor(
                np.array(y, dtype=np.float32, copy=True).reshape(-1, 1)
            )
        else:
            raise ValueError(f"Unsupported task: {task}")

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        return [self.x[index], self.y[index]]

    def __len__(self) -> int:
        return len(self.y)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    task: str,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ArrayDataset(x, y, task=task),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def count_parameters(modules: Iterable[nn.Module]) -> int:
    return sum(param.numel() for module in modules for param in module.parameters())


def train_and_load(
    encoders: list[nn.Module],
    fusion: nn.Module,
    head: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    *,
    epochs: int,
    task: str,
    objective: nn.Module,
    save_path: Path,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> tuple[nn.Module, float]:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    train(
        encoders,
        fusion,
        head,
        train_loader,
        valid_loader,
        epochs,
        task=task,
        optimtype=torch.optim.AdamW,
        lr=lr,
        weight_decay=weight_decay,
        objective=objective,
        save=str(save_path),
        track_complexity=False,
    )
    elapsed = time.perf_counter() - start
    model = torch.load(save_path, map_location=get_device(), weights_only=False)
    return model.to(get_device()), elapsed


def predict_numpy(model: nn.Module, x: np.ndarray, *, batch_size: int = 256) -> np.ndarray:
    device = next(model.parameters()).device
    x_tensor = torch.as_tensor(np.array(x, dtype=np.float32, copy=True))
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x_tensor), batch_size):
            batch = x_tensor[start : start + batch_size].to(device)
            outputs.append(model([batch]).detach().cpu())
    return torch.cat(outputs, dim=0).numpy()
