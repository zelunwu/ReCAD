"""Training callbacks: checkpointing, early stopping, history."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from recad.utils.logging import get_logger

_LOG = get_logger(__name__)


class History:
    """Lightweight per-epoch scalar history."""

    def __init__(self) -> None:
        self.train_loss: list[float] = []
        self.val_loss: list[float] = []
        self.val_rmse: list[float] = []
        self.val_r2: list[float] = []
        self.epochs: list[int] = []

    def log(self, epoch: int, train_loss: float, val: dict[str, float] | None) -> None:
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        if val is not None:
            self.val_loss.append(val.get("loss", float("nan")))
            self.val_rmse.append(val.get("rmse", float("nan")))
            self.val_r2.append(val.get("r2", float("nan")))

    def best_epoch(self, key: str = "val_rmse", minimize: bool = True) -> int:
        values = getattr(self, key)
        if not values:
            return 0
        return int(self.epochs[int(np_argmin_or_max(values, minimize))])

    def to_dict(self) -> dict[str, list]:
        return {
            "epoch": self.epochs,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_rmse": self.val_rmse,
            "val_r2": self.val_r2,
        }


def np_argmin_or_max(values, minimize: bool) -> int:
    if minimize:
        return min(range(len(values)), key=lambda i: values[i])
    return max(range(len(values)), key=lambda i: values[i])


class ModelCheckpoint:
    """Save the best state dict (by a monitored metric) to disk."""

    def __init__(self, path: str | Path, monitor: str = "val_rmse", minimize: bool = True):
        self.path = Path(path)
        self.monitor = monitor
        self.minimize = minimize
        self.best = float("inf") if minimize else float("-inf")

    def __call__(self, model: nn.Module, metrics: dict[str, float], epoch: int) -> None:
        value = metrics.get(self.monitor)
        if value is None or value != value:  # NaN guard
            return
        better = value < self.best if self.minimize else value > self.best
        if better:
            self.best = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, self.path)
            _LOG.info("checkpoint saved (%s=%.4f) -> %s", self.monitor, value, self.path)

    def load_best(self, model: nn.Module) -> dict[str, Any]:
        state = torch.load(self.path, map_location="cpu")
        model.load_state_dict(state["model_state"])
        return state


class EarlyStopping:
    """Stop training when the monitored metric stalls for ``patience`` epochs."""

    def __init__(self, patience: int = 10, monitor: str = "val_rmse", minimize: bool = True):
        self.patience = patience
        self.monitor = monitor
        self.minimize = minimize
        self.best = float("inf") if minimize else float("-inf")
        self.wait = 0
        self.stopped_epoch = 0

    def __call__(self, metrics: dict[str, float], epoch: int) -> bool:
        value = metrics.get(self.monitor)
        if value is None or value != value:
            return False
        improved = value < self.best if self.minimize else value > self.best
        if improved:
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            _LOG.info("early stopping at epoch %d", epoch)
            return True
        return False
