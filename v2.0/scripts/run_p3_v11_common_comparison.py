"""Compare the frozen v2 fCO2 result with strict v1.x on identical held cruises."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_p2_process_predictors_v2_4 import load_augmented_frame
from run_p2_validation_repair_v2_4 import (
    STRICT_V1_FEATURES,
    add_pco2air,
    fit_rf,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/frozen/p3_cruise_holdout_manifest_v2.4.csv"
P3 = Path(r"C:\backup\phd\data\processed\recad_v2_4\p3_cruise_holdout_v2.4")
OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_4\recad_v2_release_v2.4")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(frame: pd.DataFrame, column: str) -> dict[str, float | int]:
    selected = np.isfinite(frame[column]) & np.isfinite(frame.truth)
    part = frame.loc[selected]
    error = part[column].to_numpy(float) - part.truth.to_numpy(float)
    per_cruise = part.assign(squared=error**2).groupby("group_key").squared.mean()
    return {
        "rows": len(part),
        "cruises": int(part.group_key.nunique()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1 - np.sum(error**2) / np.sum((part.truth - part.truth.mean()) ** 2)),
        "cruise_equal_rmse": float(np.sqrt(per_cruise.mean())),
    }


def main() -> None:
    tables = OUTPUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    holdout = pd.read_csv(MANIFEST)
    groups = set(holdout.group_key.astype(str))
    frame = load_augmented_frame("fCO2")
    train = frame.loc[~frame.group_key.astype(str).isin(groups)].reset_index(drop=True)
    held = frame.loc[frame.group_key.astype(str).isin(groups)].reset_index(drop=True)
    if set(train.group_key.astype(str)) & groups:
        raise RuntimeError("holdout leakage in strict v1 comparison")
    train = add_pco2air(train)
    held = add_pco2air(held)
    prediction, eligible = fit_rf(
        train,
        held,
        features=STRICT_V1_FEATURES,
        strict=True,
        seed=66024,
    )
    held["v1_strict_prediction"] = prediction
    held["v1_eligible"] = eligible
    v2 = pd.read_parquet(P3 / "predictions/fCO2_held_predictions.parquet")
    v2 = v2[["record_id", "prediction", "grade"]].rename(
        columns={"prediction": "v2_prediction", "grade": "v2_grade"}
    )
    held = held.merge(v2, on="record_id", how="inner", validate="one_to_one")
    held.to_parquet(OUTPUT / "fco2_v11_common_predictions.parquet", index=False)

    common = held.loc[
        held.v1_eligible & np.isfinite(held.v1_strict_prediction) & np.isfinite(held.v2_prediction)
    ].reset_index(drop=True)

    native = []
    for model in ("v1_strict_prediction", "v2_prediction"):
        native.append(
            {"resolution": "native_1over8_common", "model": model, **metrics(common, model)}
        )
    common["lat_025"] = np.round(common.latitude.to_numpy(float) * 4) / 4
    common["lon_025"] = np.round(common.longitude.to_numpy(float) * 4) / 4
    quarter = (
        common.groupby(["group_key", "yr", "mon", "lat_025", "lon_025"], as_index=False)
        .agg(
            truth=("truth", "mean"),
            v1_strict_prediction=("v1_strict_prediction", "mean"),
            v2_prediction=("v2_prediction", "mean"),
        )
        .reset_index(drop=True)
    )
    for model in ("v1_strict_prediction", "v2_prediction"):
        native.append(
            {"resolution": "common_0.25_cruise", "model": model, **metrics(quarter, model)}
        )
    comparison = pd.DataFrame(native)
    comparison.to_csv(tables / "v11_common_metrics.csv", index=False)
    quarter.to_csv(tables / "v11_common_025_predictions.csv", index=False)
    v1_native = comparison.loc[
        comparison.resolution.eq("native_1over8_common")
        & comparison.model.eq("v1_strict_prediction")
    ].iloc[0]
    v2_native = comparison.loc[
        comparison.resolution.eq("native_1over8_common") & comparison.model.eq("v2_prediction")
    ].iloc[0]
    result = {
        "manifest_sha256": sha256(MANIFEST),
        "same_held_rows": True,
        "common_rows": len(common),
        "same_complete_cruise_exclusion": True,
        "strict_v1_uses_published_feature_family_without_full_period_local_calibration": True,
        "native_rmse_improvement_fraction": float(
            (v1_native.rmse - v2_native.rmse) / v1_native.rmse
        ),
        "passes_five_percent_gate": bool(v2_native.rmse <= 0.95 * v1_native.rmse),
        "cross_dataset_claim": False,
    }
    (OUTPUT / "v11_common_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
