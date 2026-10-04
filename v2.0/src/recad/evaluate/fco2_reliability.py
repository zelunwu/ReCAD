"""Leakage-safe utilities for the selective coastal fCO2 product."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from recad.evaluate.sss_reliability import shrinkage_weight


def apply_shrinkage(
    background: Sequence[float],
    prediction: Sequence[float],
    risk: Sequence[float],
    method: str,
) -> np.ndarray:
    """Shrink a learned residual toward the training-only background."""

    background_values = np.asarray(background, dtype=float)
    prediction_values = np.asarray(prediction, dtype=float)
    weight = shrinkage_weight(risk, method)
    return background_values + weight * (prediction_values - background_values)


def finite_quantile(values: Sequence[float], coverage: float) -> float:
    """Return the finite-sample split-conformal quantile."""

    array = np.sort(np.asarray(values, dtype=float))
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan
    rank = min(len(array), math.ceil((len(array) + 1) * float(coverage)))
    return float(array[rank - 1])


def fco2_band(background: Sequence[float]) -> pd.Categorical:
    """Label-free background-fCO2 regime used for calibration."""

    return pd.cut(
        np.asarray(background, dtype=float),
        [-np.inf, 250.0, 350.0, 450.0, 550.0, np.inf],
        right=False,
        labels=["lt250", "250_350", "350_450", "450_550", "ge550"],
    )


@dataclass(frozen=True)
class BinnedIntervalModel:
    """Risk-decile and background-band split-conformal interval model."""

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
        minimum_cell: int = 200,
    ) -> BinnedIntervalModel:
        risk = frame.risk_environment_k64.to_numpy(float)
        finite = risk[np.isfinite(risk)]
        edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 11)))
        if len(edges) < 2:
            edges = np.array([-np.inf, np.inf])
        else:
            edges[0], edges[-1] = -np.inf, np.inf
        work = frame.copy()
        work["risk_bin"] = pd.cut(risk, edges, include_lowest=True).astype(str)
        work["background_band"] = fco2_band(work.background).astype(str)
        global_widths = (
            finite_quantile(work[error_column], 0.50),
            finite_quantile(work[error_column], 0.90),
        )
        cells: dict[str, tuple[float, float, int]] = {}
        for (risk_bin, band), part in work.groupby(["risk_bin", "background_band"]):
            if len(part) >= minimum_cell:
                key = f"{risk_bin}|{band}"
                cells[key] = (
                    finite_quantile(part[error_column], 0.50),
                    finite_quantile(part[error_column], 0.90),
                    len(part),
                )
        return cls(tuple(edges), cells, global_widths, minimum_cell)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        risk_bin = pd.cut(
            frame.risk_environment_k64.to_numpy(float),
            np.asarray(self.risk_edges),
            include_lowest=True,
        ).astype(str)
        band = fco2_band(frame.background).astype(str)
        rows = []
        for left, right in zip(risk_bin, band, strict=True):
            cell = self.cells.get(f"{left}|{right}")
            if cell is None:
                rows.append((*self.global_widths, False))
            else:
                rows.append((cell[0], cell[1], True))
        return pd.DataFrame(rows, columns=["width50", "width90", "cell_supported"])


def assign_reliability_grade(frame: pd.DataFrame) -> pd.Categorical:
    """Assign the frozen fCO2 A-D grade from interval and support fields."""

    q90 = frame.width90.to_numpy(float)
    supported = (
        frame.cell_supported.to_numpy(bool)
        & (frame.risk_environment_k64.to_numpy(float) <= 5.0)
        & (frame.unique_cruises.to_numpy(float) >= 2.0)
        & (frame.effective_groups.to_numpy(float) >= 1.5)
    )
    values = np.full(len(frame), "D", dtype="U1")
    values[supported & (q90 <= 60.0)] = "C"
    values[supported & (q90 <= 35.0)] = "B"
    values[supported & (q90 <= 20.0)] = "A"
    return pd.Categorical(values, categories=list("ABCD"), ordered=True)


def interval_coverage(frame: pd.DataFrame, width: str) -> float:
    """Return symmetric interval coverage around ``prediction``."""

    return float(
        (
            np.abs(frame.truth.to_numpy(float) - frame.prediction.to_numpy(float))
            <= frame[width].to_numpy(float)
        ).mean()
    )
