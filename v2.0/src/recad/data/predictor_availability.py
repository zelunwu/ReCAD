"""Utilities for auditing global-coastal predictor availability."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

TIME_BLOCKS = (
    ("1993-1997", 1993, 1997),
    ("1998-2001", 1998, 2001),
    ("2002-2020", 2002, 2020),
    ("2021-2025", 2021, 2025),
)


@dataclass(frozen=True)
class PredictorSpec:
    """A standardized predictor and its physically plausible audit range."""

    name: str
    variable: str
    minimum: float
    maximum: float

    def valid(self, values: np.ndarray) -> np.ndarray:
        """Return finite values inside the inclusive physical audit range."""

        array = np.asarray(values)
        return np.isfinite(array) & (array >= self.minimum) & (array <= self.maximum)


def month_index(year: int, month: int, *, start_year: int = 1993) -> int:
    """Map a calendar year/month onto the frozen monthly axis."""

    if month < 1 or month > 12:
        raise ValueError("month must be in 1..12")
    if year < start_year:
        raise ValueError("year precedes the frozen time axis")
    return (year - start_year) * 12 + month - 1


def time_block(year: int) -> str:
    """Return the pre-registered audit block for a core-product year."""

    for label, first, last in TIME_BLOCKS:
        if first <= year <= last:
            return label
    raise ValueError(f"year {year} is outside the 1993-2025 core period")


def weighted_fraction(valid: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> float:
    """Return the area-weighted valid fraction inside a two-dimensional mask."""

    valid = np.asarray(valid, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    if valid.shape != mask.shape:
        raise ValueError("valid and mask must have identical shapes")
    if weights.ndim == 1:
        if weights.size != mask.shape[0]:
            raise ValueError("row weights do not match the latitude dimension")
        weights = np.broadcast_to(weights[:, None], mask.shape)
    if weights.shape != mask.shape:
        raise ValueError("weights must be row weights or match the mask")
    denominator = weights[mask].sum()
    if denominator <= 0:
        return float("nan")
    return float(weights[valid & mask].sum() / denominator)


def aggregate_time_blocks(monthly: pd.DataFrame, value_columns: Iterable[str]) -> pd.DataFrame:
    """Aggregate monthly metrics over the four frozen calendar blocks."""

    required = {"predictor", "year", *value_columns}
    missing = required.difference(monthly.columns)
    if missing:
        raise ValueError(f"missing monthly columns: {sorted(missing)}")
    frame = monthly.loc[monthly["year"].between(1993, 2025)].copy()
    frame["time_block"] = frame["year"].map(time_block)
    return (
        frame.groupby(["predictor", "time_block"], observed=True)[list(value_columns)]
        .agg(["mean", "min"])
        .reset_index()
    )


def coverage_classification(minimum_area_fraction: float) -> str:
    """Classify a standardized field using the frozen coverage thresholds."""

    if not np.isfinite(minimum_area_fraction):
        return "unavailable"
    if minimum_area_fraction >= 0.99:
        return "deployable"
    if minimum_area_fraction >= 0.90:
        return "deployable_with_missingness"
    return "repair_required"
