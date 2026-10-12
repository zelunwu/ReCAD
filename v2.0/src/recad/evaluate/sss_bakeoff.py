"""Leakage-safe model and scoring helpers for the global SSS bake-off."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from recad.evaluate.predictor_ablation import balanced_sample_weight


@dataclass(frozen=True)
class NumericTransform:
    """Training-fold median and standard-score transform."""

    columns: tuple[str, ...]
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: tuple[str, ...]) -> NumericTransform:
        values = frame.loc[:, columns].to_numpy(float)
        median = np.nanmedian(values, axis=0)
        values = np.where(np.isfinite(values), values, median)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-6] = 1.0
        return cls(columns, median, mean, scale)

    def apply(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, self.columns].to_numpy(float)
        values = np.where(np.isfinite(values), values, self.median)
        return ((values - self.mean) / self.scale).astype("float32")


class RobustRegionalCorrection:
    """Hierarchical Huber-like LME/month correction with finite-cell shrinkage."""

    def __init__(self, prior_rows: float = 100.0, clip_quantile: float = 0.99) -> None:
        self.prior_rows = prior_rows
        self.clip_quantile = clip_quantile

    def fit(self, frame: pd.DataFrame) -> RobustRegionalCorrection:
        residual = (frame.truth - frame.sss_bg).to_numpy(float)
        bound = float(np.quantile(np.abs(residual), self.clip_quantile))
        work = frame[["lme_id", "mon"]].copy()
        work["residual"] = np.clip(residual, -bound, bound)
        self.global_value = float(np.median(work.residual))
        self.month = work.groupby("mon", observed=True).residual.median().to_dict()
        cells = work.groupby(["lme_id", "mon"], observed=True).residual.agg(["median", "size"])
        estimates: dict[tuple[int, int], float] = {}
        for (lme_id, month), row in cells.iterrows():
            prior = self.month.get(int(month), self.global_value)
            weight = float(row["size"] / (row["size"] + self.prior_rows))
            estimates[(int(lme_id), int(month))] = (
                weight * float(row["median"]) + (1.0 - weight) * prior
            )
        self.cells = estimates
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        correction = np.fromiter(
            (
                self.cells.get(
                    (int(row.lme_id), int(row.mon)),
                    self.month.get(int(row.mon), self.global_value),
                )
                for row in frame.itertuples()
            ),
            dtype=float,
            count=len(frame),
        )
        return frame.sss_bg.to_numpy(float) + correction


class Top2ResidualExperts(nn.Module):
    """P1-style eight-expert top-two residual network on the frozen inputs."""

    def __init__(self, features: int, width: int = 128, experts: int = 8) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(features, width),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.experts = nn.Linear(width, experts)
        self.gate = nn.Sequential(nn.Linear(features, 64), nn.GELU(), nn.Linear(64, experts))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        expert_values = self.experts(self.trunk(values))
        logits = self.gate(values)
        top = torch.topk(logits, 2, dim=-1)
        masked = torch.full_like(logits, -torch.inf).scatter(1, top.indices, top.values)
        return (torch.softmax(masked, dim=-1) * expert_values).sum(dim=-1)


class GroupedTokenST96(nn.Module):
    """A 96-wide Transformer control over time, geometry, environment, and SSS tokens."""

    def __init__(self, group_sizes: tuple[int, ...], width: int = 96) -> None:
        super().__init__()
        self.group_sizes = group_sizes
        self.projections = nn.ModuleList(nn.Linear(size, width) for size in group_sizes)
        self.token_type = nn.Parameter(torch.randn(len(group_sizes), width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=2 * width,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        groups = torch.split(values, self.group_sizes, dim=1)
        tokens = torch.stack(
            [projection(group) for projection, group in zip(self.projections, groups, strict=True)],
            dim=1,
        )
        tokens = tokens + self.token_type.unsqueeze(0)
        return self.head(self.encoder(tokens).mean(dim=1)).squeeze(-1)


def train_neural_residual(
    family: str,
    fit: pd.DataFrame,
    features: tuple[str, ...],
    *,
    seed: int,
    steps: int,
    batch_size: int,
    group_sizes: tuple[int, ...] = (3, 8, 3, 2),
) -> dict[str, object]:
    """Train a fixed-budget neural residual model without held-label access."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = NumericTransform.fit(fit, features)
    values = torch.as_tensor(transform.apply(fit), device=device)
    response = torch.as_tensor(
        (fit.truth - fit.sss_bg).to_numpy(dtype="float32", copy=True), device=device
    )
    probability = torch.as_tensor(balanced_sample_weight(fit).astype("float32"), device=device)
    if family == "top2_experts":
        model: nn.Module = Top2ResidualExperts(len(features))
    elif family == "st96":
        if sum(group_sizes) != len(features):
            raise ValueError("ST96 group sizes do not match features")
        model = GroupedTokenST96(group_sizes)
    else:
        raise ValueError(f"unknown neural family: {family}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    model.train()
    for _ in range(steps):
        index = torch.multinomial(probability, batch_size, replacement=True)
        prediction = model(values[index])
        loss = nn.functional.huber_loss(prediction, response[index], delta=1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return {
        "family": family,
        "features": features,
        "transform": transform,
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "group_sizes": group_sizes,
        "seed": seed,
        "steps": steps,
    }


def predict_neural_residual(
    state: dict[str, object], frame: pd.DataFrame, *, batch_size: int = 65_536
) -> np.ndarray:
    """Apply one frozen neural residual model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform: NumericTransform = state["transform"]
    values = transform.apply(frame)
    if state["family"] == "top2_experts":
        model: nn.Module = Top2ResidualExperts(len(transform.columns))
    else:
        model = GroupedTokenST96(tuple(state["group_sizes"]))
    model.load_state_dict(state["model"])
    model = model.to(device).eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(values[start : start + batch_size], device=device)
            output.append(model(batch).cpu().numpy())
    residual = np.concatenate(output) if output else np.empty(0)
    return frame.sss_bg.to_numpy(float) + residual


def finite_quantile(values: np.ndarray, probability: float) -> float:
    """Finite-sample conformal absolute-error quantile."""

    values = np.sort(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    rank = min(len(values), math.ceil((len(values) + 1) * probability))
    return float(values[rank - 1]) if rank else math.nan


def interval_scores(
    truth: np.ndarray, prediction: np.ndarray, width50: np.ndarray, width90: np.ndarray
) -> dict[str, float]:
    """Return interval coverage, width, WIS, and Gaussian-approximation CRPS."""

    error = np.asarray(truth, float) - np.asarray(prediction, float)
    absolute = np.abs(error)
    width50 = np.asarray(width50, float)
    width90 = np.asarray(width90, float)
    score50 = 2 * width50 + 4 * np.maximum(absolute - width50, 0)
    score90 = 2 * width90 + 20 * np.maximum(absolute - width90, 0)
    wis = (0.5 * absolute + 0.25 * score50 + 0.05 * score90) / 0.8
    sigma = np.maximum(width90 / 1.6448536269514722, 1e-6)
    z = error / sigma
    normal_pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    normal_cdf = 0.5 * (1 + torch.erf(torch.as_tensor(z / np.sqrt(2))).numpy())
    crps = sigma * (z * (2 * normal_cdf - 1) + 2 * normal_pdf - 1 / np.sqrt(np.pi))
    return {
        "coverage50": float(np.mean(absolute <= width50)),
        "coverage90": float(np.mean(absolute <= width90)),
        "mean_width50": float(np.mean(2 * width50)),
        "mean_width90": float(np.mean(2 * width90)),
        "wis": float(np.mean(wis)),
        "crps": float(np.mean(crps)),
    }
