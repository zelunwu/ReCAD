"""Utilities for the ReCAD v1.1 fCO2 parity experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd

DISPLAY_TO_KEY = {
    "GStL & GB": "GStL",
    "SS": "SS",
    "GoME": "GoMe",
    "MAB": "MAB",
    "SAB": "SAB",
    "GoMX": "GoMx",
    "NAACOM": "Atlantic",
}
KEY_TO_DISPLAY = {value: key for key, value in DISPLAY_TO_KEY.items()}


def published_reference(path: str) -> pd.DataFrame:
    """Load and validate the immutable published metric table."""

    frame = pd.read_csv(path)
    required = {"Region", "Type", "R2", "RMSE", "MAE", "MBE"}
    if set(frame) != required:
        raise ValueError(f"unexpected v1.1 columns: {list(frame)}")
    if not required - set(frame) and len(frame) != 28:
        raise ValueError("v1.1 table must contain seven regions by four split types")
    frame["region_key"] = frame.Region.map(DISPLAY_TO_KEY)
    if frame.region_key.isna().any():
        raise ValueError("unknown v1.1 region")
    return frame


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    """Return metrics using the published v1.1 definitions."""

    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - truth
    denominator = np.sum((truth - truth.mean()) ** 2)
    return {
        "n": len(truth),
        "R2": float(1.0 - np.sum(error**2) / denominator) if denominator else np.nan,
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "MBE": float(np.mean(error)),
    }


def parity_gate(metrics: pd.DataFrame, config: dict[str, object]) -> dict[str, object]:
    """Evaluate the frozen NAACOM minimum and strong historical gates."""

    lookup = metrics.set_index(["Region", "Type"])
    test = float(lookup.loc[("NAACOM", "Test"), "RMSE"])
    validation = float(lookup.loc[("NAACOM", "Validation"), "RMSE"])
    minimum = config["minimum_parity"]
    strong = config["strong_product_target"]
    checks = {
        "test_better_than_v11": test < minimum["naacom_test_rmse_lt"],
        "validation_better_than_v11": (validation < minimum["naacom_validation_rmse_lt"]),
        "test_ten_percent_better": test <= strong["naacom_test_rmse_lte"],
        "validation_ten_percent_better": (validation <= strong["naacom_validation_rmse_lte"]),
    }
    return {
        "minimum_parity_passed": bool(
            checks["test_better_than_v11"] and checks["validation_better_than_v11"]
        ),
        "strong_product_target_passed": bool(all(checks.values())),
        "checks": checks,
        "naacom_test_rmse": test,
        "naacom_validation_rmse": validation,
    }
