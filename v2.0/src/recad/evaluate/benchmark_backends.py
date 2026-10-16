"""Uniform backend adapters for the Issue 89 model-family benchmark.

Optional third-party libraries are imported only when their family is fitted so
the core ReCAD package and CI remain usable without the benchmark environment.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from recad.evaluate.model_zoo import FTTransformer, ResidualMLP

TABULAR_FAMILIES = (
    "ridge",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
    "residual_mlp",
    "realmlp",
    "tabm",
    "ft_transformer",
    "tabpfn_3",
)
PATCH_FAMILIES = ("unet_convlstm", "vision_transformer")


@dataclass
class BenchmarkEstimator:
    """A fitted estimator plus auditable resource metadata."""

    family: str
    estimator: Any
    target_mean: float = 0.0
    target_scale: float = 1.0
    device: str = "cpu"
    parameter_count: int = 0
    updates: int = 0


class ConvLSTMCell(nn.Module):
    """A compact convolutional LSTM cell for monthly coastal patches."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.gates = nn.Conv2d(2 * channels, 4 * channels, kernel_size=3, padding=1)

    def forward(
        self, values: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = state
        input_gate, forget_gate, output_gate, candidate = self.gates(
            torch.cat([values, hidden], dim=1)
        ).chunk(4, dim=1)
        cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(candidate)
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        return hidden, cell


class UNetConvLSTMRegressor(nn.Module):
    """Point regressor with a U-Net spatial path and ConvLSTM temporal bottleneck."""

    def __init__(self, channels: int, base_channels: int, depth: int, dropout: float) -> None:
        super().__init__()
        widths = [base_channels * 2**index for index in range(depth)]
        self.encoders = nn.ModuleList()
        previous = channels
        for width in widths:
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(previous, width, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(width, width, kernel_size=3, padding=1),
                    nn.GELU(),
                )
            )
            previous = width
        self.recurrent = ConvLSTMCell(widths[-1])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(widths[-1], 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # values: sample, month, channel, latitude, longitude
        encoded_months = []
        for month in range(values.shape[1]):
            layer = values[:, month]
            for index, encoder in enumerate(self.encoders):
                layer = encoder(layer)
                if index + 1 < len(self.encoders):
                    layer = nn.functional.avg_pool2d(layer, 2)
            encoded_months.append(layer)
        hidden = torch.zeros_like(encoded_months[0])
        cell = torch.zeros_like(encoded_months[0])
        for encoded in encoded_months:
            hidden, cell = self.recurrent(encoded, (hidden, cell))
        pooled = nn.functional.adaptive_avg_pool2d(hidden, 1).flatten(1)
        return self.output(self.dropout(pooled)).squeeze(-1)


class VisionTransformerRegressor(nn.Module):
    """Spatio-temporal vision Transformer over one coastal patch sequence."""

    def __init__(
        self,
        channels: int,
        width: int,
        depth: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Conv2d(channels, width, kernel_size=2, stride=2)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=4 * width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 1)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, months, channels, height, width = values.shape
        embedded = self.embedding(values.reshape(batch * months, channels, height, width))
        tokens = embedded.flatten(2).transpose(1, 2).reshape(batch, -1, embedded.shape[1])
        cls = self.cls.expand(batch, -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.output(self.norm(encoded[:, 0])).squeeze(-1)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _torch_regression_fit(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    seed: int,
    batch_size: int = 4096,
) -> BenchmarkEstimator:
    _seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mean = float(np.mean(train_y))
    scale = max(float(np.std(train_y)), 1e-6)
    dataset = TensorDataset(
        torch.as_tensor(train_x, dtype=torch.float32),
        torch.as_tensor((train_y - mean) / scale, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device == "cuda",
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    updates = 0
    model.train()
    for _ in range(epochs):
        for features, target in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(features), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            updates += 1
    return BenchmarkEstimator(
        family="",
        estimator=model,
        target_mean=mean,
        target_scale=scale,
        device=device,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        updates=updates,
    )


def _fit_tabm(
    train_x: np.ndarray, train_y: np.ndarray, parameters: dict[str, Any], seed: int
) -> BenchmarkEstimator:
    try:
        from tabm import TabM
    except ImportError as error:
        raise RuntimeError("TabM benchmark backend is not installed") from error

    width = int(parameters["width"])
    depth = int(parameters["depth"])
    ensemble_size = int(parameters["ensemble_size"])
    dropout = float(parameters["dropout"])
    model = TabM.make(
        n_num_features=train_x.shape[1],
        cat_cardinalities=[],
        d_out=1,
        arch_type="tabm",
        k=ensemble_size,
        n_blocks=depth,
        d_block=width,
        dropout=dropout,
    )

    class MeanTabM(nn.Module):
        def __init__(self, backbone: nn.Module) -> None:
            super().__init__()
            self.backbone = backbone

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            output = self.backbone(values, None)
            return output.squeeze(-1).mean(dim=1)

    result = _torch_regression_fit(
        MeanTabM(model),
        train_x,
        train_y,
        learning_rate=float(parameters["learning_rate"]),
        weight_decay=float(parameters.get("weight_decay", 3e-4)),
        epochs=int(parameters["epochs"]),
        seed=seed,
    )
    result.family = "tabm"
    return result


def fit_tabular_estimator(
    family: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    parameters: dict[str, Any],
    *,
    seed: int,
) -> BenchmarkEstimator:
    """Fit one preregistered tabular family on exactly the supplied rows."""

    if family not in TABULAR_FAMILIES:
        raise ValueError(f"not a tabular benchmark family: {family}")
    _seed_all(seed)
    train_x = np.asarray(train_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.float32)
    if family == "ridge":
        estimator = Ridge(alpha=float(parameters["alpha"]))
    elif family == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=int(parameters["n_estimators"]),
            max_depth=parameters["max_depth"],
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            max_features=float(parameters["max_features"]),
            random_state=seed,
            n_jobs=-1,
        )
    elif family == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=int(parameters["n_estimators"]),
            max_depth=parameters["max_depth"],
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            max_features=float(parameters["max_features"]),
            random_state=seed,
            n_jobs=-1,
        )
    elif family == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            max_iter=int(parameters["max_iter"]),
            learning_rate=float(parameters["learning_rate"]),
            max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            l2_regularization=float(parameters["l2_regularization"]),
            random_state=seed,
        )
    elif family == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise RuntimeError("XGBoost benchmark backend is not installed") from error
        estimator = XGBRegressor(
            **parameters,
            objective="reg:squarederror",
            tree_method="hist",
            device="cuda" if torch.cuda.is_available() else "cpu",
            random_state=seed,
            n_jobs=-1,
        )
    elif family == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as error:
            raise RuntimeError("LightGBM benchmark backend is not installed") from error
        estimator = LGBMRegressor(
            **parameters,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    elif family == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as error:
            raise RuntimeError("CatBoost benchmark backend is not installed") from error
        estimator = CatBoostRegressor(
            **parameters,
            loss_function="RMSE",
            random_seed=seed,
            task_type="GPU" if torch.cuda.is_available() else "CPU",
            allow_writing_files=False,
            verbose=False,
        )
    elif family in {"residual_mlp", "ft_transformer"}:
        if family == "residual_mlp":
            network: nn.Module = ResidualMLP(
                train_x.shape[1],
                int(parameters["width"]),
                int(parameters["depth"]),
                float(parameters["dropout"]),
            )
        else:
            network = FTTransformer(
                train_x.shape[1],
                int(parameters["width"]),
                int(parameters["depth"]),
                int(parameters["heads"]),
                float(parameters["dropout"]),
            )
        result = _torch_regression_fit(
            network,
            train_x,
            train_y,
            learning_rate=float(parameters["learning_rate"]),
            weight_decay=float(parameters.get("weight_decay", 1e-4)),
            epochs=int(parameters["epochs"]),
            seed=seed,
        )
        result.family = family
        return result
    elif family == "realmlp":
        try:
            from pytabkit import RealMLP_TD_Regressor
        except ImportError as error:
            raise RuntimeError("RealMLP benchmark backend is not installed") from error
        estimator = RealMLP_TD_Regressor(
            device="cuda" if torch.cuda.is_available() else "cpu",
            random_state=seed,
            n_cv=1,
            n_refit=0,
            n_epochs=int(parameters["epochs"]),
            hidden_sizes=[int(value) for value in str(parameters["hidden_sizes"]).split("-")],
            p_drop=float(parameters["dropout"]),
            lr=float(parameters["learning_rate"]),
            wd=float(parameters["weight_decay"]),
            verbosity=0,
        )
    elif family == "tabm":
        return _fit_tabm(train_x, train_y, parameters, seed)
    else:
        try:
            from tabpfn import TabPFNRegressor
            from tabpfn.constants import ModelVersion
        except ImportError as error:
            raise RuntimeError("TabPFN benchmark backend is not installed") from error
        inference_precision: object = parameters["inference_precision"]
        if inference_precision == "float32":
            inference_precision = torch.float32
        overrides: dict[str, object] = {
            "n_estimators": int(parameters["n_estimators"]),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "fit_mode": str(parameters["fit_mode"]),
            "inference_precision": inference_precision,
            "random_state": seed,
        }
        if parameters["internal_subsample"] is not None:
            overrides["inference_config"] = {
                "SUBSAMPLE_SAMPLES": int(parameters["internal_subsample"])
            }
        estimator = TabPFNRegressor.create_default_for_version(ModelVersion.V3, **overrides)
    started = time.perf_counter()
    estimator.fit(train_x, train_y)
    elapsed = time.perf_counter() - started
    if family == "ridge":
        parameter_count = train_x.shape[1] + 1
    elif family in {"random_forest", "extra_trees", "xgboost", "lightgbm"}:
        parameter_count = int(parameters["n_estimators"])
    elif family == "hist_gradient_boosting":
        parameter_count = int(parameters["max_iter"])
    elif family == "catboost":
        parameter_count = int(parameters["iterations"])
    elif family == "realmlp":
        hidden = [int(value) for value in str(parameters["hidden_sizes"]).split("-")]
        widths = [train_x.shape[1], *hidden, 1]
        parameter_count = sum((left + 1) * right for left, right in pairwise(widths))
    else:
        parameter_count = int(parameters["n_estimators"])
    return BenchmarkEstimator(
        family=family,
        estimator=estimator,
        device="cuda" if torch.cuda.is_available() and family in {"xgboost", "catboost"} else "cpu",
        parameter_count=parameter_count,
        updates=max(1, round(elapsed)),
    )


def predict_tabular_estimator(model: BenchmarkEstimator, values: np.ndarray) -> np.ndarray:
    """Predict from a fitted tabular family through one common interface."""

    values = np.asarray(values, dtype=np.float32)
    if isinstance(model.estimator, nn.Module):
        model.estimator.eval()
        result = []
        with torch.inference_mode():
            for start in range(0, len(values), 16384):
                batch = torch.as_tensor(values[start : start + 16384], device=model.device)
                result.append(model.estimator(batch).detach().cpu().numpy())
        prediction = np.concatenate(result)
        prediction = prediction * model.target_scale + model.target_mean
    else:
        prediction = model.estimator.predict(values)
    return np.asarray(prediction, dtype=np.float32).reshape(-1)


def fit_patch_estimator(
    family: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    parameters: dict[str, Any],
    *,
    seed: int,
) -> BenchmarkEstimator:
    """Fit a preregistered patch-sequence model on a materialized tensor."""

    if family not in PATCH_FAMILIES:
        raise ValueError(f"not a patch benchmark family: {family}")
    if train_x.ndim != 5:
        raise ValueError("patch input must have shape [sample, month, channel, y, x]")
    channels = train_x.shape[2]
    if family == "unet_convlstm":
        network: nn.Module = UNetConvLSTMRegressor(
            channels,
            int(parameters["base_channels"]),
            int(parameters["depth"]),
            float(parameters["dropout"]),
        )
    else:
        network = VisionTransformerRegressor(
            channels,
            int(parameters["width"]),
            int(parameters["depth"]),
            int(parameters["heads"]),
            float(parameters["dropout"]),
        )
    result = _torch_regression_fit(
        network,
        np.asarray(train_x, dtype=np.float32),
        np.asarray(train_y, dtype=np.float32),
        learning_rate=float(parameters["learning_rate"]),
        weight_decay=float(parameters.get("weight_decay", 1e-4)),
        epochs=int(parameters["epochs"]),
        seed=seed,
        batch_size=int(parameters.get("batch_size", 64)),
    )
    result.family = family
    return result


def predict_patch_estimator(model: BenchmarkEstimator, values: np.ndarray) -> np.ndarray:
    """Predict a patch family in bounded GPU batches."""

    if model.family not in PATCH_FAMILIES or not isinstance(model.estimator, nn.Module):
        raise ValueError("model is not a fitted patch estimator")
    model.estimator.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(values), 256):
            batch = torch.as_tensor(
                values[start : start + 256], dtype=torch.float32, device=model.device
            )
            result.append(model.estimator(batch).detach().cpu().numpy())
    prediction = np.concatenate(result) * model.target_scale + model.target_mean
    return np.asarray(prediction, dtype=np.float32).reshape(-1)
