"""Temporal-context and water-connected models for the P2.6 fCO2 experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def water_neighbors(
    grid_flat: np.ndarray,
    *,
    grid_rows: int = 1297,
    grid_columns: int = 2880,
) -> np.ndarray:
    """Return four-neighbour coastal-node indices without crossing land.

    Longitude wraps at the dateline. Missing or land neighbours are encoded as
    ``-1``. The returned node order is the input row-major coastal-node order.
    """

    flat = np.asarray(grid_flat, dtype=np.int64)
    if flat.ndim != 1:
        raise ValueError("grid_flat must be one-dimensional")
    if np.any((flat < 0) | (flat >= grid_rows * grid_columns)):
        raise ValueError("grid_flat contains an out-of-grid index")
    lookup = np.full(grid_rows * grid_columns, -1, dtype=np.int32)
    lookup[flat] = np.arange(len(flat), dtype=np.int32)
    row = flat // grid_columns
    column = flat % grid_columns
    result = np.full((len(flat), 4), -1, dtype=np.int32)
    for slot, (delta_row, delta_column) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
        neighbour_row = row + delta_row
        neighbour_column = (column + delta_column) % grid_columns
        valid = (neighbour_row >= 0) & (neighbour_row < grid_rows)
        result[valid, slot] = lookup[neighbour_row[valid] * grid_columns + neighbour_column[valid]]
    return result


def adjacent_residual_signal(
    residual: np.ndarray,
    neighbour_residual: np.ndarray,
) -> dict[str, float | int]:
    """Summarize residual structure on observed water-connected pairs."""

    left = np.asarray(residual, dtype=float)
    right = np.asarray(neighbour_residual, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3:
        return {"pairs": int(finite.sum()), "correlation": np.nan, "same_sign": np.nan}
    return {
        "pairs": int(finite.sum()),
        "correlation": float(np.corrcoef(left[finite], right[finite])[0, 1]),
        "same_sign": float(np.mean(np.sign(left[finite]) == np.sign(right[finite]))),
    }


@dataclass
class ArrayStandardizer:
    """Training-only mean/scale transformation preserving missingness flags."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> ArrayStandardizer:
        array = np.asarray(values, dtype=np.float64)
        axes = tuple(range(array.ndim - 1))
        mean = np.nanmean(array, axis=axes)
        scale = np.nanstd(array, axis=axes)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        return cls(mean.astype("float32"), scale.astype("float32"))

    def apply(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        array = np.asarray(values, dtype=np.float32)
        available = np.isfinite(array).astype("float32")
        standardized = (array - self.mean) / self.scale
        return np.where(np.isfinite(standardized), standardized, 0.0), available


class ContextResidualNet(nn.Module):
    """Small GRU or ST96 residual model with an optional graph context stream."""

    def __init__(
        self,
        *,
        sequence_features: int,
        static_features: int,
        mode: str,
        sequence_length: int = 6,
    ) -> None:
        super().__init__()
        if mode not in {"gru", "st96_local", "st96_graph"}:
            raise ValueError(f"unknown context mode: {mode}")
        self.mode = mode
        self.sequence_length = sequence_length
        stream_multiplier = 4 if mode == "st96_graph" else 2
        token_features = sequence_features * stream_multiplier
        if mode == "gru":
            self.encoder = nn.GRU(token_features, 64, num_layers=2, batch_first=True)
            encoded_features = 64
        else:
            self.token_projection = nn.Linear(token_features, 96)
            self.position = nn.Parameter(torch.zeros(1, sequence_length, 96))
            layer = nn.TransformerEncoderLayer(
                d_model=96,
                nhead=4,
                dim_feedforward=192,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.normalization = nn.LayerNorm(96)
            encoded_features = 96
        self.head = nn.Sequential(
            nn.Linear(encoded_features + static_features, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        own: torch.Tensor,
        own_available: torch.Tensor,
        static: torch.Tensor,
        neighbour: torch.Tensor | None = None,
        neighbour_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        streams = [own, own_available]
        if self.mode == "st96_graph":
            if neighbour is None or neighbour_available is None:
                raise ValueError("graph mode requires neighbour inputs")
            streams.extend([neighbour, neighbour_available])
        tokens = torch.cat(streams, dim=-1)
        if self.mode == "gru":
            encoded, _ = self.encoder(tokens)
            context = encoded[:, -1]
        else:
            projected = self.token_projection(tokens) + self.position[:, : tokens.shape[1]]
            context = self.normalization(self.encoder(projected).mean(dim=1))
        return self.head(torch.cat([context, static], dim=-1)).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """Return trainable parameter count."""

    return int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
