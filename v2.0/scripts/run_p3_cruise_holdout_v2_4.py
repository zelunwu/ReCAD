"""Run the frozen Issue #5 complete-cruise validation once."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import build_p2_global_reliability_product_v2_4 as product
from recad.evaluate.model_zoo import fit_model, predict_model
from recad.evaluate.predictor_ablation import balanced_sample_weight
from recad.evaluate.reliability_contract import (
    DeployableSupportIndex,
    assign_grade,
    interval_wis,
    scaled_widths,
    stable_group_fold,
)
from recad.evaluate.validation_repair import conformal_widths, grouped_fit_calibration
from run_p2_global_reliability_contract_v2_4 import (
    COMMON_RISK,
    ENVIRONMENT,
    RISK_CONFIG,
)
from run_p2_process_predictors_v2_4 import load_augmented_frame

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/frozen/p3_cruise_holdout_manifest_v2.4.csv"
LOCK = ROOT / "configs/frozen/p3_cruise_holdout_lock_v2.4.json"
OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_4\p3_cruise_holdout_v2.4")
CHECKPOINTS = OUTPUT / "checkpoints"
TABLES = OUTPUT / "tables"
PREDICTIONS = OUTPUT / "predictions"
RISK_FEATURES = (*COMMON_RISK, "model_disagreement", "background_disagreement")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def split_fit_calibration(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_index, calibration_index = grouped_fit_calibration(
        frame, salt=f"p3-v2.4-complete-cruise-{target}"
    )
    fit = frame.iloc[fit_index].reset_index(drop=True)
    calibration = frame.iloc[calibration_index].reset_index(drop=True)
    if set(fit.group_key) & set(calibration.group_key):
        raise RuntimeError(f"{target}: fit/calibration group leakage")
    return fit, calibration


def fit_sss_models(frame: pd.DataFrame, checkpoint_dir: Path):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    product.CHECKPOINTS = checkpoint_dir
    return product.fit_sss(frame)


def upstream_sss(
    sss_train: pd.DataFrame,
    fco2_fit: pd.DataFrame,
    fco2_calibration: pd.DataFrame,
    held: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assignment = stable_group_fold(fco2_fit.group_key, salt="p3-v2.4-fco2-upstream-sss", folds=2)
    fit_prediction = np.full(len(fco2_fit), np.nan, dtype="float32")
    for fold in range(2):
        query = fco2_fit.loc[assignment == fold]
        excluded = set(query.group_key.astype(str))
        train = sss_train.loc[~sss_train.group_key.astype(str).isin(excluded)].reset_index(
            drop=True
        )
        models = fit_sss_models(train, CHECKPOINTS / f"upstream_sss_fold{fold}")
        fit_prediction[assignment == fold] = product.predict_sss(models, query)[0]
    final_models = fit_sss_models(sss_train, CHECKPOINTS / "upstream_sss_final")
    calibration_prediction = product.predict_sss(final_models, fco2_calibration)[0]
    held_prediction = product.predict_sss(final_models, held)[0]
    if not all(
        np.isfinite(values).all()
        for values in (fit_prediction, calibration_prediction, held_prediction)
    ):
        raise RuntimeError("non-finite upstream SSS")
    return fit_prediction, calibration_prediction, held_prediction


def support_frame(train: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    support = DeployableSupportIndex(ENVIRONMENT).fit(train).query(query)
    support["log1p_geographic_km"] = np.log1p(support.geographic_km)
    support["log1p_environment_k16"] = np.log1p(support.environment_k16)
    return support


def attach_risk(
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
    result["background_disagreement"] = (
        np.abs(prediction - result.sss_bg.to_numpy(float)) if target == "SSS" else disagreement
    )
    return result


def fit_risk(calibration: pd.DataFrame, target: str):
    assignment = stable_group_fold(calibration.group_key, salt=f"p3-v2.4-risk-{target}", folds=2)
    risk_fit = calibration.loc[assignment == 0].reset_index(drop=True)
    interval_calibration = calibration.loc[assignment == 1].reset_index(drop=True)
    response = np.log1p(
        np.abs(risk_fit.truth.to_numpy(float) - risk_fit.prediction.to_numpy(float))
    )
    model = fit_model(
        "extra_trees",
        RISK_CONFIG,
        risk_fit,
        RISK_FEATURES,
        response,
        balanced_sample_weight(risk_fit),
        65024,
    )
    return model, interval_calibration


def risk_scale(model, frame: pd.DataFrame) -> np.ndarray:
    prediction, _ = predict_model(model, frame)
    return np.maximum(np.expm1(prediction), 1e-3)


def finalize_uncertainty(
    target: str,
    calibration: pd.DataFrame,
    held: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    risk_model, interval_calibration = fit_risk(calibration, target)
    calibration_risk = risk_scale(risk_model, calibration)
    interval_risk = risk_scale(risk_model, interval_calibration)
    held_risk = risk_scale(risk_model, held)
    if target == "SSS":
        width50, width90 = scaled_widths(
            interval_calibration.truth.to_numpy(float),
            interval_calibration.prediction.to_numpy(float),
            interval_risk,
            held_risk,
        )
        interval_method = "support_risk_scaled"
    else:
        width50, width90, _ = conformal_widths(
            interval_calibration.truth.to_numpy(float),
            interval_calibration.prediction.to_numpy(float),
            held.prediction.to_numpy(float),
            target="fCO2",
        )
        interval_method = "prediction_stratified"
    thresholds = tuple(float(value) for value in np.quantile(calibration_risk, [0.5, 0.75, 0.9]))
    geographic_max = float(np.quantile(calibration.geographic_km, 0.99))
    environment_max = float(np.quantile(calibration.environment_k16, 0.99))
    grade, reason = assign_grade(
        held_risk,
        thresholds=thresholds,
        geographic_km=held.geographic_km,
        environment_k16=held.environment_k16,
        effective_groups=held.effective_groups,
        missing=held.missing_mandatory_input,
        geographic_max=geographic_max,
        environment_max=environment_max,
    )
    result = held.copy()
    result["predicted_error_scale"] = held_risk
    result["width50"] = width50.astype("float32")
    result["width90"] = width90.astype("float32")
    result["grade"] = grade
    result["reason_bits"] = reason
    contract = {
        "target": target,
        "interval_method": interval_method,
        "risk_thresholds": thresholds,
        "geographic_max_km": geographic_max,
        "environment_max": environment_max,
        "risk_fit_groups": int(
            calibration.group_key.nunique() - interval_calibration.group_key.nunique()
        ),
        "interval_calibration_groups": int(interval_calibration.group_key.nunique()),
        "held_groups": int(held.group_key.nunique()),
    }
    return result, contract


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    truth = frame.truth.to_numpy(float)
    prediction = frame.prediction.to_numpy(float)
    error = prediction - truth
    absolute = np.abs(error)
    per_cruise = frame.assign(squared_error=error**2).groupby("group_key").squared_error.mean()
    background = frame.background_prediction.to_numpy(float)
    common = np.isfinite(background)
    background_rmse = float(np.sqrt(np.mean((background[common] - truth[common]) ** 2)))
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "rows": len(frame),
        "cruises": int(frame.group_key.nunique()),
        "rmse": rmse,
        "mae": float(absolute.mean()),
        "bias": float(error.mean()),
        "r2": float(r2_score(truth, prediction)),
        "cruise_equal_rmse": float(np.sqrt(per_cruise.mean())),
        "coverage50": float(np.mean(absolute <= frame.width50.to_numpy(float))),
        "coverage90": float(np.mean(absolute <= frame.width90.to_numpy(float))),
        "median_width50": float(np.median(2 * frame.width50.to_numpy(float))),
        "median_width90": float(np.median(2 * frame.width90.to_numpy(float))),
        "wis": interval_wis(truth, prediction, frame.width50, frame.width90),
        "background_rmse": background_rmse,
        "background_skill": float(1 - rmse**2 / background_rmse**2),
    }


def summarize(target: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows = [{"target": target, "view": "all", "level": "all", **metrics(frame)}]
    views = {
        "selection_reason": frame.selection_reason.astype(str),
        "basin": frame.basin_id.astype(str),
        "grade": frame.grade.astype(str),
        "support_distance": pd.cut(
            frame.geographic_km,
            [-np.inf, 25, 100, 200, np.inf],
            labels=["0-25", "25-100", "100-200", ">200"],
        ).astype(str),
        "time": pd.cut(
            frame.yr,
            [1992, 2010, 2021, 2025],
            labels=["1993-2010", "2011-2021", "2022-2025"],
        ).astype(str),
        "cruise": frame.group_key.astype(str),
    }
    for view, labels in views.items():
        for level in sorted(labels.unique()):
            part = frame.loc[labels.eq(level)]
            if len(part):
                rows.append({"target": target, "view": view, "level": level, **metrics(part)})
    return pd.DataFrame(rows)


def run_target(
    target: str,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    held: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    support_calibration = support_frame(train, calibration)
    support_held = support_frame(train, held)
    if target == "SSS":
        models = fit_sss_models(train, CHECKPOINTS / "SSS_final")
        calibration_prediction, calibration_disagreement = product.predict_sss(models, calibration)
        held_prediction, held_disagreement = product.predict_sss(models, held)
        calibration_background = calibration.sss_bg.to_numpy(float)
        held_background = held.sss_bg.to_numpy(float)
    else:
        product.CHECKPOINTS = CHECKPOINTS / "fCO2_final"
        product.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        models = product.fit_fco2(train)
        calibration_prediction, calibration_disagreement = product.predict_fco2(models, calibration)
        held_prediction, held_disagreement = product.predict_fco2(models, held)
        calibration_background = models[0].predict(calibration)
        held_background = models[0].predict(held)
    calibration_result = attach_risk(
        target,
        calibration,
        support_calibration,
        calibration_prediction,
        calibration_disagreement,
    )
    calibration_result["background_prediction"] = calibration_background
    held_result = attach_risk(target, held, support_held, held_prediction, held_disagreement)
    held_result["background_prediction"] = held_background
    return finalize_uncertainty(target, calibration_result, held_result)


def main() -> None:
    global OUTPUT, CHECKPOINTS, TABLES, PREDICTIONS
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    OUTPUT = args.output
    CHECKPOINTS = OUTPUT / "checkpoints"
    TABLES = OUTPUT / "tables"
    PREDICTIONS = OUTPUT / "predictions"
    for directory in (CHECKPOINTS, TABLES, PREDICTIONS):
        directory.mkdir(parents=True, exist_ok=True)

    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["manifest_sha256"] != sha256(MANIFEST):
        raise RuntimeError("frozen holdout manifest hash mismatch")
    holdout = pd.read_csv(MANIFEST)
    held_groups = set(holdout.group_key.astype(str))

    sss = load_augmented_frame("SSS")
    fco2 = load_augmented_frame("fCO2")
    sss_held = sss.loc[sss.group_key.astype(str).isin(held_groups)].reset_index(drop=True)
    fco2_held = fco2.loc[fco2.group_key.astype(str).isin(held_groups)].reset_index(drop=True)
    sss_remaining = sss.loc[~sss.group_key.astype(str).isin(held_groups)].reset_index(drop=True)
    fco2_remaining = fco2.loc[~fco2.group_key.astype(str).isin(held_groups)].reset_index(drop=True)
    if set(sss_remaining.group_key.astype(str)) & held_groups:
        raise RuntimeError("SSS holdout leaked into remaining rows")
    if set(fco2_remaining.group_key.astype(str)) & held_groups:
        raise RuntimeError("fCO2 holdout leaked into remaining rows")

    sss_fit, sss_calibration = split_fit_calibration(sss_remaining, "SSS")
    fco2_fit, fco2_calibration = split_fit_calibration(fco2_remaining, "fCO2")

    sss_result, sss_contract = run_target("SSS", sss_fit, sss_calibration, sss_held)

    upstream_fit, upstream_calibration, upstream_held = upstream_sss(
        sss_remaining, fco2_fit, fco2_calibration, fco2_held
    )
    fco2_fit["sss_crossfit"] = upstream_fit
    fco2_calibration["sss_crossfit"] = upstream_calibration
    fco2_held["sss_crossfit"] = upstream_held
    fco2_result, fco2_contract = run_target("fCO2", fco2_fit, fco2_calibration, fco2_held)

    selection = holdout[["group_key", "selection_reason"]].drop_duplicates()
    results = []
    summaries = []
    for target, result, contract in (
        ("SSS", sss_result, sss_contract),
        ("fCO2", fco2_result, fco2_contract),
    ):
        result = result.merge(selection, on="group_key", how="left", validate="many_to_one")
        result["target"] = target
        result.to_parquet(PREDICTIONS / f"{target}_held_predictions.parquet", index=False)
        summary = summarize(target, result)
        summary.to_csv(TABLES / f"{target}_metrics.csv", index=False)
        summaries.append(summary)
        results.append(result)
        write_json(OUTPUT / f"{target}_calibration_contract.json", contract)
    pd.concat(summaries, ignore_index=True).to_csv(TABLES / "all_metrics.csv", index=False)

    exposure = {
        "manifest_sha256": sha256(MANIFEST),
        "label_access_count": 1,
        "target_labels_opened": True,
        "cross_dataset_claim": False,
        "held_groups": len(held_groups),
        "sss_held_rows": len(sss_held),
        "fco2_held_rows": len(fco2_held),
        "sss_remaining_group_overlap": 0,
        "fco2_remaining_group_overlap": 0,
    }
    write_json(OUTPUT / "label_access.json", exposure)
    write_json(
        OUTPUT / "pipeline.complete",
        {
            "status": "complete",
            "decision_gate": {
                target: bool(
                    metrics.loc[
                        (metrics.target == target) & (metrics.view == "all"), "coverage90"
                    ].iloc[0]
                    >= 0.85
                )
                for target, metrics in (
                    ("SSS", summaries[0]),
                    ("fCO2", summaries[1]),
                )
            },
        },
    )


if __name__ == "__main__":
    main()
