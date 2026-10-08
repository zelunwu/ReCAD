"""Small deterministic utilities for the MAB TA reliability experiment."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


def finite_sample_quantile(values: Sequence[float], coverage: float) -> float:
    """Return the split-conformal higher order statistic."""

    array = np.sort(np.asarray(values, dtype=float))
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan
    rank = min(math.ceil((len(array) + 1) * coverage), len(array))
    return float(array[rank - 1])


def calibration_groups(groups: Sequence[object], fraction: float = 0.20) -> set[str]:
    """Select a deterministic group-level calibration subset."""

    unique = sorted({str(value) for value in groups})
    if len(unique) < 2:
        return set()
    target = max(1, min(len(unique) - 1, round(len(unique) * fraction)))
    ranked = sorted(
        unique,
        key=lambda value: hashlib.sha256(value.encode("utf-8")).digest(),
    )
    return set(ranked[:target])


def assign_grade(
    q90: Sequence[float],
    support_km: Sequence[float],
    training_cruises: Sequence[int],
    environmental_ood: Sequence[bool],
) -> np.ndarray:
    """Assign preregistered MAB TA A-D grades."""

    width = np.asarray(q90, dtype=float)
    distance = np.asarray(support_km, dtype=float)
    cruises = np.asarray(training_cruises, dtype=int)
    ood = np.asarray(environmental_ood, dtype=bool)
    grade = np.full(len(width), "D", dtype="U1")
    c = (width <= 120) & (distance <= 500) & (cruises >= 3) & ~ood
    b = (width <= 80) & (distance <= 200) & (cruises >= 5) & ~ood
    a = (width <= 50) & (distance <= 100) & (cruises >= 8) & ~ood
    grade[c] = "C"
    grade[b] = "B"
    grade[a] = "A"
    return grade


def regression_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    """Compute pooled point and interval metrics for a prediction frame."""

    truth = frame.truth.to_numpy(float)
    prediction = frame.prediction.to_numpy(float)
    error = prediction - truth
    denominator = np.sum((truth - truth.mean()) ** 2)
    return {
        "n": len(frame),
        "cruises": int(frame.group_key.nunique()),
        "years": int(frame.year.nunique()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1 - np.sum(error**2) / denominator) if denominator > 0 else math.nan,
        "coverage50": float(np.mean(np.abs(error) <= frame.q50.to_numpy(float))),
        "coverage90": float(np.mean(np.abs(error) <= frame.q90.to_numpy(float))),
        "median_width50": float(2 * frame.q50.median()),
        "median_width90": float(2 * frame.q90.median()),
        "grade_ab_fraction": float(frame.grade.isin(["A", "B"]).mean()),
    }
