"""Leakage-safe reliability utilities for the coastal SSS product.

The functions in this module are deliberately independent of the locked-test
gateway.  They operate on predictions whose provenance has already been
checked by the experiment runner and never load labels themselves.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

NUMERIC_FEATURES = (
    "sss",
    "sst",
    "adt",
    "wspd",
    "latitude",
    "lon_sin",
    "lon_cos",
    "month_sin",
    "month_cos",
)
CATEGORICAL_FEATURES = ("lme_id", "basin_id", "regime_id")


def balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    """Return row probabilities for equal LME, cruise, month sampling."""

    required = {"lme_id", "group_key", "month"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"balanced weights missing {sorted(missing)}")
    keys = ["lme_id", "group_key", "month"]
    n_regions = frame.lme_id.nunique()
    cruises = frame.groupby("lme_id").group_key.transform("nunique")
    months = frame.groupby(["lme_id", "group_key"])["month"].transform("nunique")
    records = frame.groupby(keys)["month"].transform("size")
    weights = (1.0 / (n_regions * cruises * months * records)).to_numpy(float)
    return weights / weights.sum()


class TabularTransform:
    """Frozen numeric standardisation and categorical one-hot transform."""

    def fit(self, frame: pd.DataFrame) -> TabularTransform:
        numeric = frame[list(NUMERIC_FEATURES)].to_numpy(float)
        self.median = np.nanmedian(numeric, axis=0)
        numeric = np.where(np.isfinite(numeric), numeric, self.median)
        self.mean = numeric.mean(axis=0)
        self.std = numeric.std(axis=0)
        self.std[self.std < 1e-7] = 1.0
        self.categories = []
        for column in CATEGORICAL_FEATURES:
            values = sorted(frame[column].fillna(-999).astype(int).astype(str).unique())
            self.categories.append([*values, "__unknown__"])
        return self

    @classmethod
    def from_state(cls, state: dict[str, object]) -> TabularTransform:
        transform = cls()
        transform.median = np.asarray(state["median"])
        transform.mean = np.asarray(state["mean"])
        transform.std = np.asarray(state["std"])
        transform.categories = [list(values) for values in state["categories"]]
        return transform

    def state(self) -> dict[str, object]:
        return {
            "median": self.median,
            "mean": self.mean,
            "std": self.std,
            "categories": self.categories,
        }

    def apply(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = frame[list(NUMERIC_FEATURES)].to_numpy(float)
        numeric = np.where(np.isfinite(numeric), numeric, self.median)
        parts = [(numeric - self.mean) / self.std]
        for column, categories in zip(CATEGORICAL_FEATURES, self.categories, strict=True):
            mapping = {value: index for index, value in enumerate(categories)}
            unknown = len(categories) - 1
            values = frame[column].fillna(-999).astype(int).astype(str)
            codes = np.fromiter((mapping.get(value, unknown) for value in values), int)
            one_hot = np.zeros((len(frame), len(categories)), dtype=np.float32)
            one_hot[np.arange(len(frame)), codes] = 1.0
            parts.append(one_hot)
        return np.concatenate(parts, axis=1).astype(np.float32)


class SoftExperts(nn.Module):
    """Eight-expert top-two residual model frozen by Issue #7."""

    def __init__(self, n_features: int, experts: int = 8, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.trunk = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        self.experts = nn.Linear(128, experts)
        self.gate = nn.Sequential(nn.Linear(n_features, 64), nn.GELU(), nn.Linear(64, experts))

    def forward(self, values: torch.Tensor, return_experts: bool = False):
        expert_values = self.experts(self.trunk(values))
        logits = self.gate(values)
        top = torch.topk(logits, self.top_k, dim=-1)
        masked = torch.full_like(logits, -torch.inf).scatter(1, top.indices, top.values)
        weights = torch.softmax(masked, dim=-1)
        prediction = (weights * expert_values).sum(dim=-1)
        if return_experts:
            return prediction, expert_values, weights
        return prediction


def train_soft_expert(
    frame: pd.DataFrame,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    """Fit a fixed-step model without consulting an evaluation partition."""

    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    transform = TabularTransform().fit(frame)
    x_train = torch.as_tensor(transform.apply(frame), device=selected_device)
    response = torch.as_tensor(
        (frame.truth - frame.sss).to_numpy(np.float32), device=selected_device
    )
    probabilities = torch.as_tensor(
        balanced_weights(frame).astype(np.float32), device=selected_device
    )
    model = SoftExperts(x_train.shape[1]).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(steps):
        index = torch.multinomial(probabilities, batch_size, replacement=True)
        prediction, experts, weights = model(x_train[index], return_experts=True)
        loss = nn.functional.huber_loss(prediction, response[index], delta=1.0)
        loss = loss + 0.001 * experts.var(dim=1).mean() + 1e-5 * (weights**2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "transform": transform.state(),
        "family": "soft_experts_residual",
        "seed": int(seed),
        "steps": int(steps),
    }


def predict_soft_expert(
    state: dict[str, object],
    frame: pd.DataFrame,
    *,
    batch_size: int = 131_072,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Predict SSS in bounded batches from one frozen model state."""

    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    transform = TabularTransform.from_state(state["transform"])
    values = transform.apply(frame)
    model = SoftExperts(values.shape[1]).to(selected_device)
    model.load_state_dict(state["model"])
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(values[start : start + batch_size], device=selected_device)
            output.append(model(batch).cpu().numpy())
    residual = np.concatenate(output) if output else np.empty(0)
    return frame.sss.to_numpy(float) + residual


def shrinkage_weight(risk: Sequence[float], method: str) -> np.ndarray:
    """Return preregistered correction weights from raw environmental risk."""

    values = np.asarray(risk, dtype=float)
    if method == "none":
        result = np.ones(len(values))
    elif method == "linear_2_5":
        result = np.clip((5.0 - values) / 3.0, 0.0, 1.0)
    elif method == "quadratic_5":
        result = np.square(np.clip(1.0 - values / 5.0, 0.0, 1.0))
    elif method == "hard_4":
        result = (values <= 4.0).astype(float)
    else:
        raise ValueError(f"unknown shrinkage method: {method}")
    result[~np.isfinite(values)] = 0.0
    return result


def apply_shrinkage(
    background: Sequence[float],
    prediction: Sequence[float],
    risk: Sequence[float],
    method: str,
) -> np.ndarray:
    """Shrink the learned residual toward the physical product background."""

    background_values = np.asarray(background, dtype=float)
    prediction_values = np.asarray(prediction, dtype=float)
    weight = shrinkage_weight(risk, method)
    return background_values + weight * (prediction_values - background_values)


def finite_quantile(values: Sequence[float], coverage: float) -> float:
    """Split-conformal finite-sample quantile with the higher order statistic."""

    array = np.sort(np.asarray(values, dtype=float))
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan
    rank = min(len(array), math.ceil((len(array) + 1) * float(coverage)))
    return float(array[rank - 1])


def salinity_band(background: Sequence[float]) -> pd.Categorical:
    """Label-free salinity regime used by the interval calibrator."""

    return pd.cut(
        np.asarray(background, dtype=float),
        [-np.inf, 20.0, 30.0, 33.0, 36.0, np.inf],
        right=False,
        labels=["lt20", "20_30", "30_33", "33_36", "ge36"],
    )


@dataclass(frozen=True)
class BinnedIntervalModel:
    """Risk-decile and background-salinity split-conformal interval model."""

    risk_edges: tuple[float, ...]
    cells: dict[str, tuple[float, float, int]]
    global_widths: tuple[float, float]
    minimum_cell: int = 200

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        error_column: str = "absolute_error",
        risk_column: str = "risk_environment_k64",
        minimum_cell: int = 200,
    ) -> BinnedIntervalModel:
        required = {error_column, risk_column, "background"}
        missing = required - set(frame)
        if missing:
            raise ValueError(f"interval calibration missing {sorted(missing)}")
        risk = frame[risk_column].to_numpy(float)
        finite = risk[np.isfinite(risk)]
        if not len(finite):
            raise ValueError("interval calibration has no finite risk")
        edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 11)))
        if len(edges) < 2:
            edges = np.array([-np.inf, np.inf])
        else:
            edges[0], edges[-1] = -np.inf, np.inf
        work = frame[[error_column, risk_column, "background"]].copy()
        work["risk_bin"] = np.digitize(work[risk_column], edges[1:-1], right=True)
        work["salinity_band"] = salinity_band(work.background).astype(str)
        cells: dict[str, tuple[float, float, int]] = {}
        for (risk_bin, band), part in work.groupby(["risk_bin", "salinity_band"]):
            if len(part) >= minimum_cell:
                errors = part[error_column]
                cells[f"{int(risk_bin)}|{band}"] = (
                    finite_quantile(errors, 0.50),
                    finite_quantile(errors, 0.90),
                    len(part),
                )
        global_widths = (
            finite_quantile(work[error_column], 0.50),
            finite_quantile(work[error_column], 0.90),
        )
        return cls(tuple(float(value) for value in edges), cells, global_widths, minimum_cell)

    def predict(
        self, frame: pd.DataFrame, *, risk_column: str = "risk_environment_k64"
    ) -> pd.DataFrame:
        edges = np.asarray(self.risk_edges)
        bins = np.digitize(frame[risk_column], edges[1:-1], right=True)
        bands = salinity_band(frame.background).astype(str)
        rows = []
        for risk_bin, band in zip(bins, bands, strict=True):
            key = f"{int(risk_bin)}|{band}"
            q50, q90, count = self.cells.get(key, (*self.global_widths, 0))
            rows.append((q50, q90, count, key in self.cells))
        return pd.DataFrame(rows, columns=["width50", "width90", "calibration_n", "cell_supported"])

    def to_dict(self) -> dict[str, object]:
        return {
            "risk_edges": list(self.risk_edges),
            "cells": {key: list(value) for key, value in self.cells.items()},
            "global_widths": list(self.global_widths),
            "minimum_cell": self.minimum_cell,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> BinnedIntervalModel:
        return cls(
            tuple(payload["risk_edges"]),
            {key: tuple(value) for key, value in payload["cells"].items()},
            tuple(payload["global_widths"]),
            int(payload["minimum_cell"]),
        )


def assign_reliability_grade(frame: pd.DataFrame) -> pd.Series:
    """Apply frozen target-specific SSS reliability rules.

    A/B/C encode intended absolute-error use thresholds.  D suppresses values
    without demonstrated environmental, regional, or calibration support.
    """

    required = {
        "width90",
        "risk_environment_k64",
        "unique_cruises",
        "effective_groups",
        "cell_supported",
        "background",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"grade assignment missing {sorted(missing)}")
    supported = (
        frame.cell_supported.astype(bool)
        & frame.risk_environment_k64.le(5.0)
        & frame.unique_cruises.ge(2)
        & frame.effective_groups.ge(1.5)
    )
    # Very fresh water is retained only when its calibrated error width earns
    # the grade; no geographic salinity cut is used.
    grade = np.select(
        [
            supported & frame.width90.le(0.50),
            supported & frame.width90.le(1.00),
            supported & frame.width90.le(2.00),
        ],
        ["A", "B", "C"],
        default="D",
    )
    return pd.Series(grade, index=frame.index, dtype="string")


def interval_coverage(frame: pd.DataFrame) -> dict[str, float]:
    """Return empirical 50/90 coverage and interval widths."""

    error = np.abs(frame.prediction.to_numpy(float) - frame.truth.to_numpy(float))
    return {
        "n": len(frame),
        "coverage50": float(np.mean(error <= frame.width50)),
        "coverage90": float(np.mean(error <= frame.width90)),
        "median_width50": float(np.median(frame.width50)),
        "median_width90": float(np.median(frame.width90)),
    }
