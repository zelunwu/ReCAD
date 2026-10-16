"""Run strict reverse-time validation for the Issue 81 residual bank."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from recad.evaluate.early_era import complete_group_indices, load_early_era_contract
from recad.evaluate.model_zoo import fit_model, predict_model
from recad.evaluate.predictor_ablation import balanced_sample_weight
from recad.evaluate.reliability_contract import DeployableSupportIndex, stable_group_fold
from recad.evaluate.validation_repair import conformal_widths
from run_p2_process_predictors_v2_4 import fit_variant, load_augmented_frame
from run_p2_validation_repair_v2_4 import MODERN_RF_FEATURES, fit_predict_sss

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/frozen/early_era_validation_v2.5.in"
ENVIRONMENT = (
    "sst",
    "wspd",
    "xco2air",
    "sss_bg",
    "distance_to_significant_land_km",
    "bathymetry_m",
)
KEEP = (
    "record_id",
    "group_key",
    "yr",
    "mon",
    "coastal_node",
    "latitude",
    "longitude",
    "truth",
    "prediction",
    "model_disagreement",
    "background_disagreement",
    "geographic_km",
    "environment_k16",
    "unique_cruises",
    "effective_groups",
    "support_entropy",
    "time_extrapolation_years",
    "missing_mandatory_input",
    "predicted_error_scale",
    "width50",
    "width90",
    "selected_interval_method",
    "target",
    "scheme",
    "fold",
)


def stable_seed(*parts: object) -> int:
    value = ":".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "little") & 0x7FFFFFFF


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def support_features(train: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    return DeployableSupportIndex(ENVIRONMENT).fit(train).query(query)


def split_later_training(
    frame: pd.DataFrame, *, folds: int, calibration_fold: int, salt: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignment = stable_group_fold(frame.group_key, salt=salt, folds=folds)
    calibration = assignment == calibration_fold
    model_fit = frame.loc[~calibration].reset_index(drop=True)
    calibration_frame = frame.loc[calibration].reset_index(drop=True)
    if set(model_fit.group_key.astype(str)) & set(calibration_frame.group_key.astype(str)):
        raise AssertionError("model-fit and calibration cruises overlap")
    return model_fit, calibration_frame


def sss_candidate(
    model_fit: pd.DataFrame,
    query: pd.DataFrame,
    *,
    configuration: list[float | int],
    seeds: tuple[int, ...],
    fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    members = [
        fit_variant(
            "SSS",
            "S0_base",
            configuration,
            model_fit,
            query,
            stable_seed("early-era", "SSS", fold, seed),
        )
        for seed in seeds
    ]
    array = np.column_stack(members)
    return array.mean(axis=1).astype("float32"), array.std(axis=1).astype("float32")


def nested_later_sss(
    sss_frame: pd.DataFrame,
    model_fit: pd.DataFrame,
    calibration: pd.DataFrame,
    held: pd.DataFrame,
    *,
    fold: int,
    inner_folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict SSS for fCO2 without exposing early or queried-cruise SSS labels."""

    assignment = stable_group_fold(
        model_fit.group_key, salt=f"issue81-early-era-upstream-{fold}", folds=inner_folds
    )
    fit_prediction = np.full(len(model_fit), np.nan, dtype="float32")
    fit_groups = model_fit.group_key.astype(str)
    sss_groups = sss_frame.group_key.astype(str)
    for inner in range(inner_folds):
        query = assignment == inner
        allowed_groups = set(fit_groups.loc[~query])
        upstream_train = sss_frame.loc[sss_groups.isin(allowed_groups)].reset_index(drop=True)
        fit_prediction[query] = fit_predict_sss(
            upstream_train,
            model_fit.loc[query],
            seed=stable_seed("early-era", "upstream-sss", fold, inner),
        )
    allowed_groups = set(fit_groups)
    upstream_train = sss_frame.loc[sss_groups.isin(allowed_groups)].reset_index(drop=True)
    downstream = pd.concat([calibration, held], ignore_index=True)
    downstream_prediction = fit_predict_sss(
        upstream_train,
        downstream,
        seed=stable_seed("early-era", "upstream-sss", fold, "downstream"),
    )
    if not np.isfinite(fit_prediction).all() or not np.isfinite(downstream_prediction).all():
        raise RuntimeError("nested later-only SSS contains non-finite values")
    return (
        fit_prediction,
        downstream_prediction[: len(calibration)],
        downstream_prediction[len(calibration) :],
    )


def fco2_candidate(
    model_fit: pd.DataFrame,
    query: pd.DataFrame,
    *,
    background_config: list[float | int],
    catboost_config: list[float | int],
    catboost_weight: float,
    fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    background = fit_variant(
        "fCO2",
        "F4_continuous_background_residual",
        background_config,
        model_fit,
        query,
        stable_seed("early-era", "fCO2", fold, "background"),
    )
    catboost = fit_model(
        "catboost",
        catboost_config,
        model_fit,
        MODERN_RF_FEATURES,
        model_fit.truth.to_numpy(float),
        balanced_sample_weight(model_fit),
        stable_seed("early-era", "fCO2", fold, "catboost"),
    )
    catboost_prediction, _ = predict_model(catboost, query)
    prediction = background + catboost_weight * (catboost_prediction - background)
    return prediction.astype("float32"), np.abs(catboost_prediction - background).astype("float32")


def attach_output(
    target: str,
    model_fit: pd.DataFrame,
    calibration: pd.DataFrame,
    held: pd.DataFrame,
    prediction: np.ndarray,
    disagreement: np.ndarray,
    *,
    fold: int,
    interval_method: str,
) -> pd.DataFrame:
    query = pd.concat([calibration, held], ignore_index=True)
    support = support_features(model_fit, query)
    result = pd.concat([query.reset_index(drop=True), support], axis=1)
    result["prediction"] = prediction
    result["model_disagreement"] = disagreement
    if target == "SSS":
        result["background_disagreement"] = np.abs(prediction - result.sss_bg.to_numpy(float))
    else:
        result["background_disagreement"] = disagreement
    n_calibration = len(calibration)
    calibration_output = result.iloc[:n_calibration]
    held_output = result.iloc[n_calibration:].reset_index(drop=True)
    width50, width90, _ = conformal_widths(
        calibration_output.truth.to_numpy(float),
        calibration_output.prediction.to_numpy(float),
        held_output.prediction.to_numpy(float),
        target=target,
    )
    held_output["predicted_error_scale"] = np.nan
    held_output["width50"] = width50
    held_output["width90"] = width90
    held_output["selected_interval_method"] = interval_method
    held_output["target"] = target
    held_output["scheme"] = "early_era"
    held_output["fold"] = fold
    return held_output.loc[:, KEEP]


def run_target(target: str, config_path: Path) -> None:
    contract = load_early_era_contract(config_path)
    output = contract.output_root
    partitions = output / "partitions"
    tables = output / "tables"
    upstream = output / "upstream_sss"
    frame = load_augmented_frame(target)
    frame = frame.loc[frame.yr.le(contract.maximum_label_year)].reset_index(drop=True)
    sss_frame = load_augmented_frame("SSS") if target == "fCO2" else None
    if sss_frame is not None:
        sss_frame = sss_frame.loc[sss_frame.yr.le(contract.maximum_label_year)].reset_index(
            drop=True
        )
    audits = []
    summaries = []
    for fold_spec in contract.folds:
        destination = partitions / f"{target}_early_era_fold{fold_spec.fold}.parquet"
        summary_path = tables / f"{target}_early_era_fold{fold_spec.fold}.json"
        train_index, held_index, excluded_index = complete_group_indices(frame, fold_spec)
        later = frame.iloc[train_index].reset_index(drop=True)
        held = frame.iloc[held_index].reset_index(drop=True)
        model_fit, calibration = split_later_training(
            later,
            folds=contract.calibration_folds,
            calibration_fold=contract.calibration_fold,
            salt=f"issue81-early-era-calibration-{target}-{fold_spec.fold}",
        )
        audit = {
            "target": target,
            "fold": fold_spec.fold,
            "held_start": fold_spec.held_start,
            "held_end": fold_spec.held_end,
            "train_start": fold_spec.train_start,
            "model_fit_rows": len(model_fit),
            "model_fit_groups": int(model_fit.group_key.nunique()),
            "calibration_rows": len(calibration),
            "calibration_groups": int(calibration.group_key.nunique()),
            "held_rows": len(held),
            "held_groups": int(held.group_key.nunique()),
            "excluded_rows": len(excluded_index),
            "latest_held_year": int(held.yr.max()),
            "earliest_model_fit_year": int(model_fit.yr.min()),
            "group_overlap": len(
                set(later.group_key.astype(str)) & set(held.group_key.astype(str))
            ),
        }
        audits.append(audit)
        if destination.is_file() and summary_path.is_file():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        combined = pd.concat([calibration, held], ignore_index=True)
        if target == "SSS":
            prediction, disagreement = sss_candidate(
                model_fit,
                combined,
                configuration=list(contract.sss_config),
                seeds=contract.sss_seeds,
                fold=fold_spec.fold,
            )
        else:
            assert sss_frame is not None
            later_sss = sss_frame.loc[sss_frame.yr.ge(fold_spec.train_start)].reset_index(drop=True)
            fit_sss, calibration_sss, held_sss = nested_later_sss(
                later_sss,
                model_fit,
                calibration,
                held,
                fold=fold_spec.fold,
                inner_folds=contract.upstream_sss_folds,
            )
            model_fit = model_fit.copy()
            calibration = calibration.copy()
            held = held.copy()
            model_fit["sss_crossfit"] = fit_sss
            calibration["sss_crossfit"] = calibration_sss
            held["sss_crossfit"] = held_sss
            upstream.mkdir(parents=True, exist_ok=True)
            pd.concat(
                [
                    pd.DataFrame(
                        {
                            "role": "model_fit",
                            "record_id": model_fit.record_id,
                            "sss_crossfit": fit_sss,
                        }
                    ),
                    pd.DataFrame(
                        {
                            "role": "calibration",
                            "record_id": calibration.record_id,
                            "sss_crossfit": calibration_sss,
                        }
                    ),
                    pd.DataFrame(
                        {"role": "held", "record_id": held.record_id, "sss_crossfit": held_sss}
                    ),
                ],
                ignore_index=True,
            ).to_parquet(upstream / f"fCO2_early_era_fold{fold_spec.fold}.parquet", index=False)
            combined = pd.concat([calibration, held], ignore_index=True)
            prediction, disagreement = fco2_candidate(
                model_fit,
                combined,
                background_config=list(contract.fco2_background_config),
                catboost_config=list(contract.fco2_catboost_config),
                catboost_weight=contract.fco2_catboost_weight,
                fold=fold_spec.fold,
            )
        result = attach_output(
            target,
            model_fit,
            calibration,
            held,
            prediction,
            disagreement,
            fold=fold_spec.fold,
            interval_method=contract.interval_method,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(destination, index=False, compression="zstd")
        error = result.prediction.to_numpy(float) - result.truth.to_numpy(float)
        absolute = np.abs(error)
        summary = {
            **audit,
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(absolute)),
            "bias": float(np.mean(error)),
            "record_q90_absolute_error": float(np.quantile(absolute, 0.9, method="higher")),
            "coverage90": float(np.mean(absolute <= result.width90.to_numpy(float))),
        }
        write_json(summary_path, summary)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        gc.collect()
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audits).to_csv(tables / f"{target}_split_audit.csv", index=False)
    pd.DataFrame(summaries).to_csv(tables / f"{target}_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--target", choices=("SSS", "fCO2", "all"), default="all")
    args = parser.parse_args()
    targets = ("SSS", "fCO2") if args.target == "all" else (args.target,)
    for target in targets:
        run_target(target, args.config)
    if args.target == "all":
        contract = load_early_era_contract(args.config)
        (contract.output_root / "early_era.complete").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
