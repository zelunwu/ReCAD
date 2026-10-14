"""Finite model-family implementations for the repaired P2 model zoo."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TabularTransform:
    """Training-only median imputation and standardization."""

    features: tuple[str, ...]
    median: tuple[float, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(cls, frame: pd.DataFrame, features: tuple[str, ...]) -> TabularTransform:
        values = frame.loc[:, features].to_numpy(dtype=np.float64, copy=True)
        median = np.nanmedian(values, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        values = np.where(np.isfinite(values), values, median)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        return cls(
            features=features,
            median=tuple(float(value) for value in median),
            mean=tuple(float(value) for value in mean),
            scale=tuple(float(value) for value in scale),
        )

    def apply(self, frame: pd.DataFrame) -> np.ndarray:
        """Return finite standardized float32 features."""

        values = frame.loc[:, self.features].to_numpy(dtype=np.float64, copy=True)
        median = np.asarray(self.median)
        values = np.where(np.isfinite(values), values, median)
        return ((values - np.asarray(self.mean)) / np.asarray(self.scale)).astype("float32")


class ResidualBlock(nn.Module):
    """Small residual MLP block."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.Dropout(dropout),
        )
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(values + self.layers(values))


class ResidualMLP(nn.Module):
    """Residual tabular MLP."""

    def __init__(self, n_features: int, width: int, depth: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(n_features, width), nn.SiLU())
        self.blocks = nn.Sequential(*(ResidualBlock(width, dropout) for _ in range(depth)))
        self.output = nn.Linear(width, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(self.input(values))).squeeze(-1)


class FTTransformer(nn.Module):
    """Compact numerical-feature-token Transformer."""

    def __init__(
        self,
        n_features: int,
        width: int,
        depth: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, width))
        self.bias = nn.Parameter(torch.empty(n_features, width))
        self.cls = nn.Parameter(torch.empty(1, 1, width))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        tokens = values.unsqueeze(-1) * self.weight + self.bias
        cls = self.cls.expand(values.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.output(self.norm(encoded[:, 0])).squeeze(-1)


@dataclass
class FittedModel:
    """One fitted family model with transform and resource metadata."""

    family: str
    config: list[float | int | None]
    transform: TabularTransform
    estimator: Any
    target_mean: float
    target_scale: float
    parameter_count: int
    updates: int
    device: str


def stable_seed(value: str) -> int:
    """Return a stable positive 31-bit seed."""

    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def deterministic_cap(frame: pd.DataFrame, maximum: int, salt: str) -> pd.DataFrame:
    """Cap rows deterministically without changing group membership between partitions."""

    if len(frame) <= maximum:
        return frame.reset_index(drop=True)
    seed = stable_seed(salt)
    rng = np.random.default_rng(seed)
    index = np.sort(rng.choice(len(frame), size=maximum, replace=False))
    return frame.iloc[index].reset_index(drop=True)


def group_equal_rmse(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    """Return RMSE after each complete group receives equal weight."""

    error = np.asarray(prediction, dtype=float) - frame.truth.to_numpy(float)
    per_group = (
        pd.DataFrame({"group_key": frame.group_key.to_numpy(), "squared": error**2})
        .groupby("group_key", observed=True)
        .squared.mean()
    )
    return float(np.sqrt(per_group.mean()))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fit_torch(
    family: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    sample_weight: np.ndarray,
    config: list[float | int | None],
    seed: int,
) -> tuple[nn.Module, float, float, int, int, str]:
    """Fit one neural candidate with a fixed epoch budget."""

    _seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_mean = float(np.average(train_y, weights=sample_weight))
    target_scale = float(np.sqrt(np.average((train_y - target_mean) ** 2, weights=sample_weight)))
    target_scale = max(target_scale, 1e-6)
    if family == "residual_mlp":
        width, depth, dropout, learning_rate, epochs = config
        model: nn.Module = ResidualMLP(train_x.shape[1], int(width), int(depth), float(dropout))
    elif family == "ft_transformer":
        width, depth, heads, dropout, learning_rate, epochs = config
        model = FTTransformer(train_x.shape[1], int(width), int(depth), int(heads), float(dropout))
    else:
        raise ValueError(family)
    model = model.to(device)
    dataset = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(((train_y - target_mean) / target_scale).astype("float32")),
        torch.from_numpy(sample_weight.astype("float32")),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=4096,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    updates = 0
    model.train()
    for _ in range(int(epochs)):
        for features, target, weight in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            error = model(features) - target
            loss = torch.sum(weight * error.square()) / torch.sum(weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            updates += 1
    count = sum(parameter.numel() for parameter in model.parameters())
    return model, target_mean, target_scale, count, updates, device


def fit_model(
    family: str,
    config: list[float | int | None],
    train: pd.DataFrame,
    features: tuple[str, ...],
    target: np.ndarray,
    sample_weight: np.ndarray,
    seed: int,
) -> FittedModel:
    """Fit one model family under a common transformed numeric interface."""

    transform = TabularTransform.fit(train, features)
    train_x = transform.apply(train)
    train_y = np.asarray(target, dtype="float32")
    weights = np.asarray(sample_weight, dtype="float32")
    if family == "ridge":
        estimator = Ridge(alpha=float(config[0]))
        estimator.fit(train_x, train_y, sample_weight=weights)
        count = train_x.shape[1] + 1
        return FittedModel(family, config, transform, estimator, 0.0, 1.0, count, 1, "cpu")
    if family == "extra_trees":
        trees, depth, leaf, max_features = config
        estimator = ExtraTreesRegressor(
            n_estimators=int(trees),
            max_depth=None if depth is None else int(depth),
            min_samples_leaf=int(leaf),
            max_features=float(max_features),
            n_jobs=-1,
            random_state=seed,
        )
        estimator.fit(train_x, train_y, sample_weight=weights)
        return FittedModel(
            family, config, transform, estimator, 0.0, 1.0, len(estimator.estimators_), 1, "cpu"
        )
    if family == "catboost":
        iterations, depth, learning_rate, l2 = config
        task_type = "GPU" if torch.cuda.is_available() else "CPU"
        parameters: dict[str, object] = {
            "iterations": int(iterations),
            "depth": int(depth),
            "learning_rate": float(learning_rate),
            "l2_leaf_reg": float(l2),
            "loss_function": "RMSE",
            "random_seed": seed,
            "task_type": task_type,
            "allow_writing_files": False,
            "verbose": False,
        }
        if task_type == "GPU":
            parameters["devices"] = "0"
        estimator = CatBoostRegressor(
            **parameters,
        )
        estimator.fit(train_x, train_y, sample_weight=weights)
        return FittedModel(
            family,
            config,
            transform,
            estimator,
            0.0,
            1.0,
            int(iterations),
            int(iterations),
            task_type.lower(),
        )
    if family == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise RuntimeError(
                "XGBoost is required for this family; install ReCAD[model-zoo]"
            ) from error
        trees, depth, learning_rate, min_child_weight, subsample = config
        parameters = {
            "n_estimators": int(trees),
            "max_depth": int(depth),
            "learning_rate": float(learning_rate),
            "min_child_weight": float(min_child_weight),
            "subsample": float(subsample),
            "colsample_bytree": 0.8,
            "reg_lambda": 5.0,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": seed,
            "n_jobs": -1,
        }
        device = "cuda" if torch.cuda.is_available() else "cpu"
        estimator = XGBRegressor(**parameters, device=device)
        try:
            estimator.fit(train_x, train_y, sample_weight=weights)
        except Exception:
            if device == "cpu":
                raise
            device = "cpu"
            estimator = XGBRegressor(**parameters, device=device)
            estimator.fit(train_x, train_y, sample_weight=weights)
        return FittedModel(
            family, config, transform, estimator, 0.0, 1.0, int(trees), int(trees), device
        )
    if family in {"residual_mlp", "ft_transformer"}:
        estimator, mean, scale, count, updates, device = _fit_torch(
            family, train_x, train_y, weights, config, seed
        )
        return FittedModel(
            family, config, transform, estimator, mean, scale, count, updates, device
        )
    raise ValueError(f"unknown family: {family}")


def predict_model(model: FittedModel, frame: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Predict and return elapsed inference seconds."""

    started = time.perf_counter()
    features = model.transform.apply(frame)
    if model.family in {"residual_mlp", "ft_transformer"}:
        estimator: nn.Module = model.estimator
        estimator.eval()
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(features), 16384):
                values = torch.from_numpy(features[start : start + 16384]).to(model.device)
                chunks.append(estimator(values).cpu().numpy())
        prediction = np.concatenate(chunks) * model.target_scale + model.target_mean
    else:
        prediction = model.estimator.predict(features)
    return np.asarray(prediction, dtype="float32"), time.perf_counter() - started


def save_model(model: FittedModel, destination: Path) -> list[Path]:
    """Save a reproducible external checkpoint and metadata."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "family": model.family,
        "config": model.config,
        "transform": asdict(model.transform),
        "target_mean": model.target_mean,
        "target_scale": model.target_scale,
        "parameter_count": model.parameter_count,
        "updates": model.updates,
        "device": model.device,
    }
    metadata_path = destination.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if model.family in {"residual_mlp", "ft_transformer"}:
        model_path = destination.with_suffix(".pt")
        torch.save(model.estimator.state_dict(), model_path)
    elif model.family == "catboost":
        model_path = destination.with_suffix(".cbm")
        model.estimator.save_model(model_path)
    elif model.family == "xgboost":
        model_path = destination.with_suffix(".ubj")
        model.estimator.save_model(model_path)
    else:
        model_path = destination.with_suffix(".joblib")
        joblib.dump(model.estimator, model_path)
    return [metadata_path, model_path]


def parameter_count_for_config(family: str, config: list[Any], n_features: int) -> int:
    """Return exact neural parameter count or a transparent tree/linear proxy."""

    if family == "ridge":
        return n_features + 1
    if family in {"extra_trees", "catboost", "xgboost"}:
        return int(config[0])
    if family == "residual_mlp":
        width, depth, dropout, *_ = config
        model = ResidualMLP(n_features, int(width), int(depth), float(dropout))
    elif family == "ft_transformer":
        width, depth, heads, dropout, *_ = config
        model = FTTransformer(n_features, int(width), int(depth), int(heads), float(dropout))
    else:
        raise ValueError(family)
    return sum(parameter.numel() for parameter in model.parameters())
