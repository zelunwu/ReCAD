"""Run nested calibration and applicability evaluation for Issue 44."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from recad.evaluate.model_zoo import fit_model, predict_model
from recad.evaluate.predictor_ablation import balanced_sample_weight
from recad.evaluate.reliability_contract import (
    DeployableSupportIndex,
    conformal_quantile,
    interval_wis,
    scaled_widths,
    stable_group_fold,
)
from recad.evaluate.validation_repair import conformal_widths
from run_p2_model_zoo_v2_4 import prepare_split, split_seed
from run_p2_process_predictors_v2_4 import fit_variant, load_augmented_frame
from run_p2_validation_repair_v2_4 import MODERN_RF_FEATURES, cached_splits

OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_4\p2_global_reliability_contract_v2.4")
PARTITIONS = OUTPUT / "partitions"
SELECTION = OUTPUT / "selection"
TABLES = OUTPUT / "tables"
ISSUE62 = Path(r"C:\backup\phd\data\processed\recad_v2_4\p2_model_zoo_v2.4")
ISSUE63 = Path(r"C:\backup\phd\data\processed\recad_v2_4\p2_process_predictors_v2.4")
ISSUE64 = Path(r"C:\backup\phd\data\processed\recad_v2_4\p2_conditional_combination_v2.4")
SEEDS = (62024, 62025, 62026)
RISK_CONFIG = [192, 24, 20, 0.7]
ENVIRONMENT = (
    "sst",
    "wspd",
    "xco2air",
    "sss_bg",
    "distance_to_significant_land_km",
    "bathymetry_m",
)
COMMON_RISK = (
    "log1p_geographic_km",
    "log1p_environment_k16",
    "effective_groups",
    "unique_cruises",
    "support_entropy",
    "time_extrapolation_years",
    "missing_mandatory_input",
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def selected(path: Path) -> list[float | int | None]:
    return json.loads(path.read_text(encoding="utf-8"))["selected_config"]


def support_features(train: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    support = DeployableSupportIndex(ENVIRONMENT).fit(train).query(query)
    support["log1p_geographic_km"] = np.log1p(support.geographic_km)
    support["log1p_environment_k16"] = np.log1p(support.environment_k16)
    return support


def candidate_prediction(
    target: str,
    split,
    train: pd.DataFrame,
    query: pd.DataFrame,
    *,
    stage: str,
) -> tuple[np.ndarray, np.ndarray]:
    if target == "SSS":
        config = selected(
            ISSUE62 / "selection" / f"SSS_{split.scheme}_fold{split.fold}_catboost.json"
        )
        members = []
        for seed in SEEDS:
            members.append(
                fit_variant(
                    "SSS",
                    "S0_base",
                    config,
                    train,
                    query,
                    split_seed("SSS", split, f"issue44-{stage}", seed),
                )
            )
        array = np.column_stack(members)
        return array.mean(axis=1).astype("float32"), array.std(axis=1).astype("float32")
    f4_config = selected(
        ISSUE63
        / "selection"
        / f"fCO2_{split.scheme}_fold{split.fold}_F4_continuous_background_residual.json"
    )
    f4 = fit_variant(
        "fCO2",
        "F4_continuous_background_residual",
        f4_config,
        train,
        query,
        split_seed("fCO2", split, f"issue44-{stage}-f4", 44024),
    )
    cat_config = selected(
        ISSUE62 / "selection" / f"fCO2_{split.scheme}_fold{split.fold}_catboost.json"
    )
    cat_model = fit_model(
        "catboost",
        cat_config,
        train,
        MODERN_RF_FEATURES,
        train.truth.to_numpy(float),
        balanced_sample_weight(train),
        split_seed("fCO2", split, f"issue44-{stage}-cat", 44024),
    )
    cat, _ = predict_model(cat_model, query)
    audit = pd.read_csv(ISSUE64 / "tables/fCO2_inner_ensemble_trigger.csv")
    weight = float(
        audit.loc[
            (audit.scheme == split.scheme) & (audit.fold == split.fold), "oracle_weight_catboost"
        ].iloc[0]
    )
    prediction = f4 + weight * (cat - f4)
    return prediction.astype("float32"), np.abs(cat - f4).astype("float32")


def attach_risk_features(
    target: str,
    query: pd.DataFrame,
    support: pd.DataFrame,
    prediction: np.ndarray,
    disagreement: np.ndarray,
) -> pd.DataFrame:
    result = query.reset_index(drop=True).copy()
    result = pd.concat([result, support.reset_index(drop=True)], axis=1)
    result["prediction"] = prediction
    result["model_disagreement"] = disagreement
    if target == "SSS":
        result["background_disagreement"] = np.abs(prediction - result.sss_bg.to_numpy(float))
    else:
        result["background_disagreement"] = disagreement
    return result


def risk_features() -> tuple[str, ...]:
    return (*COMMON_RISK, "model_disagreement", "background_disagreement")


def fit_risk(train: pd.DataFrame, seed: int):
    response = np.log1p(np.abs(train.truth.to_numpy(float) - train.prediction.to_numpy(float)))
    return fit_model(
        "extra_trees",
        RISK_CONFIG,
        train,
        risk_features(),
        response,
        balanced_sample_weight(train),
        seed,
    )


def risk_scale(model, frame: pd.DataFrame) -> np.ndarray:
    prediction, _ = predict_model(model, frame)
    return np.maximum(np.expm1(prediction), 1e-3)


def risk_bin_widths(
    truth: np.ndarray,
    prediction: np.ndarray,
    calibration_risk: np.ndarray,
    destination_risk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cuts = np.quantile(calibration_risk, [0.25, 0.5, 0.75])
    source_bin = np.digitize(calibration_risk, cuts)
    destination_bin = np.digitize(destination_risk, cuts)
    error = np.abs(truth - prediction)
    global_width = (conformal_quantile(error, 0.5), conformal_quantile(error, 0.9))
    width50 = np.full(len(destination_risk), global_width[0])
    width90 = np.full(len(destination_risk), global_width[1])
    for index in range(4):
        selected = source_bin == index
        if selected.sum() < 200:
            continue
        width = (
            conformal_quantile(error[selected], 0.5),
            conformal_quantile(error[selected], 0.9),
        )
        width50[destination_bin == index] = width[0]
        width90[destination_bin == index] = width[1]
    return width50, width90


def method_widths(
    method: str,
    target: str,
    calibration: pd.DataFrame,
    destination: pd.DataFrame,
    calibration_risk: np.ndarray,
    destination_risk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    truth = calibration.truth.to_numpy(float)
    prediction = calibration.prediction.to_numpy(float)
    if method == "prediction_stratified":
        width50, width90, _ = conformal_widths(
            truth,
            prediction,
            destination.prediction.to_numpy(float),
            target=target,
        )
        return width50, width90
    if method == "raw_disagreement_scaled":
        return scaled_widths(
            truth,
            prediction,
            calibration.model_disagreement.to_numpy(float),
            destination.model_disagreement.to_numpy(float),
        )
    if method == "support_risk_scaled":
        return scaled_widths(truth, prediction, calibration_risk, destination_risk)
    if method == "risk_bin_grouped":
        return risk_bin_widths(
            truth,
            prediction,
            calibration_risk,
            destination_risk,
        )
    raise ValueError(method)


def score_method(frame: pd.DataFrame, width50: np.ndarray, width90: np.ndarray) -> dict[str, float]:
    truth = frame.truth.to_numpy(float)
    prediction = frame.prediction.to_numpy(float)
    absolute = np.abs(truth - prediction)
    return {
        "coverage50": float(np.mean(absolute <= width50)),
        "coverage90": float(np.mean(absolute <= width90)),
        "mean_width50": float(np.mean(2 * width50)),
        "mean_width90": float(np.mean(2 * width90)),
        "wis": interval_wis(truth, prediction, width50, width90),
    }


def run_target(target: str) -> None:
    frame = load_augmented_frame(target)
    summaries = []
    methods = (
        "prediction_stratified",
        "raw_disagreement_scaled",
        "support_risk_scaled",
        "risk_bin_grouped",
    )
    for split in cached_splits(target, frame):
        destination = PARTITIONS / f"{target}_{split.scheme}_fold{split.fold}.parquet"
        selection_path = SELECTION / f"{target}_{split.scheme}_fold{split.fold}.json"
        if destination.is_file() and selection_path.is_file():
            summaries.append(json.loads(selection_path.read_text(encoding="utf-8"))["summary"])
            continue
        tune_train, tune_validation, model_fit, calibration, held = prepare_split(
            target, frame, split
        )
        tune_prediction, tune_disagreement = candidate_prediction(
            target, split, tune_train, tune_validation, stage="inner"
        )
        tune = attach_risk_features(
            target,
            tune_validation,
            support_features(tune_train, tune_validation),
            tune_prediction,
            tune_disagreement,
        )
        assignment = stable_group_fold(
            tune.group_key, salt=f"issue44-method-{target}-{split.scheme}-{split.fold}"
        )
        risk_train = tune.loc[assignment == 0].reset_index(drop=True)
        inner_calibration = tune.loc[assignment == 1].reset_index(drop=True)
        inner_evaluation = tune.loc[assignment == 2].reset_index(drop=True)
        risk_model = fit_risk(
            risk_train, split_seed(target, split, "issue44-risk-selection", 44024)
        )
        inner_calibration_risk = risk_scale(risk_model, inner_calibration)
        inner_evaluation_risk = risk_scale(risk_model, inner_evaluation)
        method_scores = []
        for method in methods:
            width50, width90 = method_widths(
                method,
                target,
                inner_calibration,
                inner_evaluation,
                inner_calibration_risk,
                inner_evaluation_risk,
            )
            method_scores.append(
                {"method": method, **score_method(inner_evaluation, width50, width90)}
            )
        eligible = [row for row in method_scores if row["coverage90"] >= 0.85]
        if eligible:
            choice = min(eligible, key=lambda row: (row["wis"], row["mean_width90"]))
        else:
            choice = min(method_scores, key=lambda row: (-row["coverage90"], row["wis"]))

        combined = pd.concat([calibration, held], ignore_index=True)
        final_prediction, final_disagreement = candidate_prediction(
            target, split, model_fit, combined, stage="outer"
        )
        final = attach_risk_features(
            target,
            combined,
            support_features(model_fit, combined),
            final_prediction,
            final_disagreement,
        )
        final_risk_model = fit_risk(tune, split_seed(target, split, "issue44-risk-final", 44024))
        final["predicted_error_scale"] = risk_scale(final_risk_model, final)
        n_cal = len(calibration)
        final_calibration = final.iloc[:n_cal].reset_index(drop=True)
        final_held = final.iloc[n_cal:].reset_index(drop=True)
        width50, width90 = method_widths(
            str(choice["method"]),
            target,
            final_calibration,
            final_held,
            final_calibration.predicted_error_scale.to_numpy(float),
            final_held.predicted_error_scale.to_numpy(float),
        )
        final_held["width50"] = width50
        final_held["width90"] = width90
        final_held["selected_interval_method"] = str(choice["method"])
        final_held["target"] = target
        final_held["scheme"] = split.scheme
        final_held["fold"] = split.fold
        keep = [
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
        ]
        destination.parent.mkdir(parents=True, exist_ok=True)
        final_held[keep].to_parquet(destination, index=False, compression="zstd")
        held_score = score_method(final_held, width50, width90)
        summary = {
            "target": target,
            "scheme": split.scheme,
            "fold": split.fold,
            "held_rows": len(final_held),
            "selected_interval_method": choice["method"],
            **held_score,
        }
        write_json(
            selection_path,
            {
                "target": target,
                "scheme": split.scheme,
                "fold": split.fold,
                "method_scores": method_scores,
                "selected": choice,
                "summary": summary,
            },
        )
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(TABLES / f"{target}_partition_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["SSS", "fCO2", "all"], default="all")
    args = parser.parse_args()
    targets = ("SSS", "fCO2") if args.target == "all" else (args.target,)
    for target in targets:
        run_target(target)
    if args.target == "all":
        (OUTPUT / "calibration.complete").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
