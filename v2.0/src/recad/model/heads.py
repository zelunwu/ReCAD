"""Output heads of the transformer decoder."""

from __future__ import annotations

import torch
from torch import nn

LOGVAR_MIN = -8.0
LOGVAR_MAX = 5.0


class GaussianHead(nn.Module):
    """Heteroscedastic Gaussian head: outputs (mean, clamped log-variance)."""

    def __init__(self, d_in: int, hidden: int | None = None) -> None:
        super().__init__()
        h = hidden or d_in
        self.net = nn.Sequential(
            nn.Linear(d_in, h),
            nn.GELU(),
            nn.Linear(h, 2),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.net(x)
        mean, logvar = out.chunk(2, dim=-1)
        logvar = logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)
        return {"mean": mean[..., 0], "logvar": logvar[..., 0]}


class MeanHead(nn.Module):
    """Point-prediction head (MSE training)."""

    def __init__(self, d_in: int, hidden: int | None = None) -> None:
        super().__init__()
        h = hidden or d_in
        self.net = nn.Sequential(
            nn.Linear(d_in, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"mean": self.net(x)[..., 0]}


class QuantileHead(nn.Module):
    """Quantile head for q in (0.1, 0.5, 0.9) (optional, quantile loss)."""

    QUANTILES = (0.1, 0.5, 0.9)

    def __init__(self, d_in: int, hidden: int | None = None) -> None:
        super().__init__()
        h = hidden or d_in
        self.net = nn.Sequential(
            nn.Linear(d_in, h),
            nn.GELU(),
            nn.Linear(h, len(self.QUANTILES)),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.net(x)  # [..., 3] ordered by self.QUANTILES
        return {
            "quantiles": out,
            "mean": out[..., 1],
            "low": out[..., 0],
            "high": out[..., 2],
        }


def build_head(head_name: str, d_in: int, hidden: int | None = None) -> nn.Module:
    """Construct the head selected by ``ModelConfig.head``."""
    if head_name == "gaussian":
        return GaussianHead(d_in, hidden)
    if head_name == "mean":
        return MeanHead(d_in, hidden)
    if head_name == "quantile":
        return QuantileHead(d_in, hidden)
    raise ValueError(f"unknown head: {head_name}")
