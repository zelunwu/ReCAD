"""Run the post-Stage-A exact-feature parity audit for Issue #32."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

from recad.evaluate.v11_parity import KEY_TO_DISPLAY, regression_metrics
from recad.viz.regions import v11_region_keys
from run_p1_fco2_v11_parity import legacy_frame

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/experiments/p1_fco2_v11_parity_v2.2"
PREPARED = Path("D:/proj_personal/PhD/ReCAD/v2.0/outputs/prepared_naccom.nc")
MASKS = Path("D:/proj_personal/PhD/ReCAD/v2.0/outputs/masks_naccom.nc")
FEATURES = ["longitude", "latitude", "month", "sss", "sst", "adt", "pco2air"]


def metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, candidate_part in predictions.groupby("candidate"):
        for split, split_part in candidate_part.groupby("Type"):
            for key in [*v11_region_keys()[:-1], "Atlantic"]:
                part = (
                    split_part
                    if key == "Atlantic"
                    else split_part.loc[split_part.region_key.eq(key)]
                )
                rows.append(
                    {
                        "candidate": candidate,
                        "Region": KEY_TO_DISPLAY[key],
                        "Type": split,
                        **regression_metrics(part.truth, part.prediction),
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = legacy_frame(PREPARED, MASKS, 2021)
    frame = frame.loc[np.isfinite(frame[FEATURES]).all(axis=1)].reset_index(drop=True)
    train = frame.Type.eq("Train")
    x_train = frame.loc[train, FEATURES]
    y_train = frame.loc[train, "truth"]
    outputs = []

    print("PHASE_B v1_rf_replica seed=100", flush=True)
    forest = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=1,
        max_features=1.0,
        bootstrap=True,
        random_state=100,
        n_jobs=-1,
    )
    forest.fit(x_train, y_train)
    rf_prediction = forest.predict(frame[FEATURES])
    outputs.append(frame.assign(candidate="v1_rf_replica", prediction=rf_prediction))

    seed_predictions = []
    for seed in (100, 101, 102):
        print(f"PHASE_B direct_catboost seed={seed}", flush=True)
        model = CatBoostRegressor(
            iterations=2500,
            depth=10,
            learning_rate=0.04,
            loss_function="RMSE",
            random_seed=seed,
            random_strength=0.5,
            l2_leaf_reg=5.0,
            allow_writing_files=False,
            verbose=False,
            task_type="GPU",
            devices="0",
        )
        model.fit(x_train, y_train, verbose=False)
        model.save_model(OUTPUT / f"checkpoints/direct_catboost_seed{seed}.cbm")
        seed_predictions.append(model.predict(frame[FEATURES]))
    ensemble = np.vstack(seed_predictions)
    outputs.append(
        frame.assign(
            candidate="direct_catboost",
            prediction=ensemble.mean(axis=0),
            epistemic_sd=ensemble.std(axis=0, ddof=1),
        )
    )

    predictions = pd.concat(outputs, ignore_index=True)
    keep = [
        "candidate",
        "year",
        "month",
        "latitude",
        "longitude",
        "region_key",
        "Type",
        "truth",
        "prediction",
    ]
    predictions[keep].to_parquet(OUTPUT / "phase_b_predictions.parquet", index=False)
    metrics = metric_table(predictions)
    metrics.to_csv(OUTPUT / "phase_b_metrics.csv", index=False)
    aggregate = metrics.loc[metrics.Region.eq("NAACOM")].to_dict(orient="records")
    (OUTPUT / "phase_b_decision.json").write_text(
        json.dumps(
            {
                "aggregate_metrics": aggregate,
                "post_stage_a_amendment": True,
                "label_informed_calibration_used": False,
                "locked_fco2_opened": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
