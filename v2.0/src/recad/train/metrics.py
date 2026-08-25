"""Evaluation metrics with the exact v1.1 definitions.

``calculateR2RMSE.m`` in v1.1 defined:
    R2   = corrcoef(x, y)[0,1] ** 2      (squared Pearson correlation)
    RMSE = sqrt(mean((x - y) ** 2))

These definitions are intentionally preserved in v2.0 so the new model can be
benchmarked one-to-one against the published random-forest product. All
metrics are computed on masked (valid) pairs; NaN handling is explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Squared Pearson correlation (v1.1 definition)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if valid.sum() < 2:
        return float("nan")
    corr = np.corrcoef(y_true[valid], y_pred[valid])[0, 1]
    return float(corr**2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error (v1.1 definition)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if valid.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[valid] - y_pred[valid]) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[valid] - y_pred[valid])))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean predicted-minus-observed bias (µatm)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(y_pred[valid] - y_true[valid]))


def metrics_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Full metric dict for one (possibly masked) pair of arrays."""
    return {
        "r2": r2(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "n": int(np.count_nonzero(~np.isnan(y_true) & ~np.isnan(y_pred))),
    }


def per_year_metrics(
    y_true_4d: np.ndarray,
    y_pred_4d: np.ndarray,
    years: np.ndarray,
) -> pd.DataFrame:
    """Per-year R2/RMSE/MAE table (mirrors the v1.1 leave-year-out table).

    The input arrays are 4-D ``[n_year, 12, n_lat, n_lon]``; the table has one
    row per year with at least one valid pair.
    """
    rows = []
    for y in range(y_true_4d.shape[0]):
        t = y_true_4d[y]
        p = y_pred_4d[y]
        n = int(np.count_nonzero(~np.isnan(t) & ~np.isnan(p)))
        if n == 0:
            continue
        row = metrics_summary(t, p)
        row["year"] = int(years[y])
        rows.append(row)
    return pd.DataFrame(rows)[["year", "n", "r2", "rmse", "mae", "bias"]]
