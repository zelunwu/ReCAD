"""Train final Issue #26 regional ensemble and build the 2025-2026 atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from catboost import CatBoostRegressor

from recad.evaluate.applicability import SupportIndex
from recad.evaluate.fco2_reliability import apply_shrinkage, finite_quantile
from recad.evaluate.p1_framework import FrozenManifest, P1DataGateway, Purpose
from run_p1_fco2_reliability import (
    CATEGORICAL,
    MODEL_COLUMNS,
    NUMERIC,
    ROOT,
    SeasonalTrendClimatology,
    add_crossfit_sss,
    balanced_weights,
    prepare,
)
from run_p1_sss_reliability import ENVIRONMENT_COLUMNS, grid_query

EXPERIMENT_ID = "p1_fco2_reliability_v2.2"
MODEL_VERSION = "p1_fco2_regional_lme12_v2.2"
CALIBRATION_VERSION = "nested_pseudo_forward_v2.2"


def train_model(frame: pd.DataFrame, seed: int, iterations: int) -> CatBoostRegressor:
    columns = NUMERIC + CATEGORICAL
    values = frame[columns].copy()
    for column in NUMERIC:
        values[column] = values[column].fillna(float(values[column].median()))
    for column in CATEGORICAL:
        values[column] = values[column].fillna(-999).astype(int).astype(str)
    model = CatBoostRegressor(
        iterations=iterations,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=seed,
        random_strength=0.5,
        l2_leaf_reg=5.0,
        allow_writing_files=False,
        verbose=False,
        task_type="GPU",
        devices="0",
    )
    model.fit(
        values,
        frame.residual,
        cat_features=[columns.index(column) for column in CATEGORICAL],
        sample_weight=balanced_weights(frame) * len(frame),
        verbose=False,
    )
    model._recad_numeric_medians = {column: float(frame[column].median()) for column in NUMERIC}
    return model


def predict_model(model: CatBoostRegressor, frame: pd.DataFrame) -> np.ndarray:
    columns = NUMERIC + CATEGORICAL
    values = frame[columns].copy()
    for column in NUMERIC:
        values[column] = values[column].fillna(model._recad_numeric_medians[column])
    for column in CATEGORICAL:
        values[column] = values[column].fillna(-999).astype(int).astype(str)
    return model.predict(values)


def final_evidence(gateway: P1DataGateway, config: dict[str, object]) -> pd.DataFrame:
    train = prepare(gateway.load_labels("fco2", Purpose.TRAIN, columns=MODEL_COLUMNS))
    development = prepare(gateway.load_labels("fco2", Purpose.SELECTION, columns=MODEL_COLUMNS))
    sss_path = Path(config["sss_reliability_output"]) / "development_oof_predictions.parquet"
    train = add_crossfit_sss(train, sss_path, "cruise")
    development = add_crossfit_sss(development, sss_path, "forward")
    return pd.concat([train, development], ignore_index=True)


def support_ready(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["latitude_sin"] = np.sin(np.deg2rad(result.latitude))
    result["longitude_sin"] = np.sin(np.deg2rad(result.longitude))
    result["longitude_cos"] = np.cos(np.deg2rad(result.longitude))
    return result


def sss_for_month(root: Path, year: int, month: int) -> pd.DataFrame:
    path = root / f"year={year}" / f"month={month:02d}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["coastal_node", "sss_prediction", "sss_grade"])
    result = pd.read_parquet(path, columns=["coastal_node", "prediction", "grade"])
    return result.rename(columns={"prediction": "sss_prediction", "grade": "sss_grade"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/p1_fco2_reliability_v2.2.yaml"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / f"outputs/experiments/{EXPERIMENT_ID}"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    decision = json.loads((args.output / "regional_decision.json").read_text(encoding="utf-8"))
    if decision["status"] != "pass_regional" or decision["confirmed_lmes"] != [12]:
        raise RuntimeError("atlas requires the frozen pass_regional decision for LME 12")
    manifest = FrozenManifest.load(ROOT / "configs/frozen/data_manifest_v2.2.json")
    gateway = P1DataGateway(manifest)
    evidence = final_evidence(gateway, config)
    background = SeasonalTrendClimatology().fit(evidence)
    region = evidence.loc[evidence.lme_id.eq(12)].copy()
    region["background"] = background.predict(region)
    region["residual"] = region.truth - region.background
    checkpoints = args.output / "final_regional_models"
    checkpoints.mkdir(parents=True, exist_ok=True)
    models = []
    for seed in config["models"]["final_seeds"]:
        print(f"FINAL LME=12 seed={seed}", flush=True)
        model = train_model(region, int(seed), int(config["regional_expert_stage"]["iterations"]))
        model.save_model(checkpoints / f"lme12_seed{seed}.cbm")
        models.append(model)
    oof = pd.read_parquet(args.output / "regional_expert_predictions.parquet")
    cruise = oof.loc[oof.outer_scheme.eq("cruise") & oof.lme_id.eq(12)]
    absolute = np.abs(cruise.prediction - cruise.truth)
    inflation = pd.read_csv(args.output / "temporal_interval_inflation.csv").set_index("lme_id")
    width50 = finite_quantile(absolute, 0.50) * inflation.loc[12, "q50_inflation"]
    width90 = finite_quantile(absolute, 0.90) * inflation.loc[12, "q90_inflation"]
    support_index = SupportIndex(environmental_columns=ENVIRONMENT_COLUMNS).fit(
        support_ready(evidence)
    )
    spatial_path = manifest.data_dir / "spatial_support_v2.2.nc"
    availability_path = manifest.data_dir / "inference_availability_v2.2.nc"
    sss_root = Path(config["sss_reliability_output"]) / "sss_reliability_atlas"
    atlas_root = args.output / "fco2_reliability_atlas"
    counts = []
    with (
        xr.open_dataset(config["prepared_global"]) as prepared,
        xr.open_dataset(spatial_path) as spatial,
        xr.open_dataset(availability_path) as availability,
    ):
        for year in (2025, 2026):
            for month in range(1, 13):
                ready = (
                    availability.strict_all_inputs_ready.sel(year=year, month=month)
                    .to_numpy()
                    .astype(bool)
                )
                nodes = np.flatnonzero(ready)
                if not len(nodes):
                    continue
                print(f"ATLAS {year}-{month:02d}: {len(nodes):,}", flush=True)
                query = grid_query(prepared, spatial, year, month, nodes)
                query["xco2air"] = prepared.xco2air.sel(year=year, month=month).to_numpy()[
                    spatial.lat_index.to_numpy()[nodes], spatial.lon_index.to_numpy()[nodes]
                ]
                query = query.merge(
                    sss_for_month(sss_root, year, month),
                    on="coastal_node",
                    how="left",
                    validate="one_to_one",
                )
                query["sss_input"] = query.sss_prediction.where(query.sss_grade.ne("D"), query.sss)
                support = support_index.query(support_ready(query))
                query["risk_environment_k64"] = support.environment_k64.to_numpy()
                query["unique_cruises"] = support.unique_cruises.to_numpy()
                query["effective_groups"] = support.effective_groups.to_numpy()
                query["background"] = background.predict(query)
                query["raw_prediction"] = query.background
                query["epistemic_sd"] = np.nan
                confirmed = query.lme_id.eq(12)
                if confirmed.any():
                    model_predictions = np.vstack(
                        [predict_model(model, query.loc[confirmed]) for model in models]
                    )
                    raw = query.loc[confirmed, "background"].to_numpy() + model_predictions.mean(0)
                    query.loc[confirmed, "raw_prediction"] = raw
                    query.loc[confirmed, "epistemic_sd"] = model_predictions.std(0, ddof=1)
                query["prediction"] = query.background
                query.loc[confirmed, "prediction"] = apply_shrinkage(
                    query.loc[confirmed, "background"],
                    query.loc[confirmed, "raw_prediction"],
                    query.loc[confirmed, "risk_environment_k64"],
                    "linear_2_5",
                )
                supported = (
                    confirmed
                    & query.risk_environment_k64.le(5.0)
                    & query.unique_cruises.ge(2)
                    & query.effective_groups.ge(1.5)
                    & query.sss_grade.ne("D")
                )
                query["width50"], query["width90"] = np.nan, np.nan
                query.loc[supported, "width50"] = width50
                query.loc[supported, "width90"] = width90
                query["grade"] = "D"
                query.loc[supported & query.width90.le(60.0), "grade"] = "C"
                query.loc[supported & query.width90.le(35.0), "grade"] = "B"
                query.loc[supported & query.width90.le(20.0), "grade"] = "A"
                query["expected_absolute_error"] = query.width50
                for coverage in (50, 90):
                    query[f"lower{coverage}"] = query.prediction - query[f"width{coverage}"]
                    query[f"upper{coverage}"] = query.prediction + query[f"width{coverage}"]
                query["reason_bits"] = 0
                query.loc[~confirmed, "reason_bits"] |= 1
                query.loc[query.risk_environment_k64.gt(5.0), "reason_bits"] |= 2
                query.loc[query.unique_cruises.lt(2), "reason_bits"] |= 4
                query.loc[query.sss_grade.eq("D") | query.sss_grade.isna(), "reason_bits"] |= 8
                query.loc[query.grade.eq("D"), "reason_bits"] |= 16
                if year == 2026:
                    query["reason_bits"] |= 32
                query["fallback_status"] = np.where(
                    query.grade.eq("D"), "background_only_suppressed", "regional_model"
                )
                query["prediction_status"] = "provisional" if year == 2026 else "core"
                query["model_version"] = MODEL_VERSION
                query["calibration_version"] = CALIBRATION_VERSION
                columns = [
                    "year",
                    "month",
                    "coastal_node",
                    "latitude",
                    "longitude",
                    "lme_id",
                    "background",
                    "prediction",
                    "epistemic_sd",
                    "expected_absolute_error",
                    "lower50",
                    "upper50",
                    "lower90",
                    "upper90",
                    "width50",
                    "width90",
                    "risk_environment_k64",
                    "unique_cruises",
                    "effective_groups",
                    "sss_grade",
                    "grade",
                    "reason_bits",
                    "fallback_status",
                    "prediction_status",
                    "model_version",
                    "calibration_version",
                ]
                partition = atlas_root / f"year={year}"
                partition.mkdir(parents=True, exist_ok=True)
                query[columns].to_parquet(partition / f"month={month:02d}.parquet", index=False)
                for grade, count in query.grade.value_counts().items():
                    counts.append(
                        {
                            "year": year,
                            "month": month,
                            "grade": grade,
                            "grid_months": int(count),
                            "fraction": float(count / len(query)),
                        }
                    )
    pd.DataFrame(counts).to_csv(args.output / "atlas_grade_counts.csv", index=False)
    metadata = {
        "confirmed_lmes": [12],
        "lme_12_name": "Caribbean Sea",
        "width50_uatm": width50,
        "width90_uatm": width90,
        "model_version": MODEL_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "locked_test_opened": False,
        "external_independent_opened": False,
    }
    (args.output / "atlas_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
