"""Cross-fit physical-unit conditional error estimators for Issue 81."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from recad.evaluate.conditional_error import (
    ConditionalErrorContract,
    calibration_offset,
    cruise_equal_weights,
    error_bound_metrics,
    load_conditional_error_contract,
    prepare_features,
    stable_group_folds,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/frozen/conditional_physical_error_v2.5.in"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def higher_quantile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(np.asarray(values, float), quantile, method="higher"))


def fit_transform(
    frame: pd.DataFrame, contract: ConditionalErrorContract
) -> tuple[np.ndarray, dict[str, object]]:
    prepared = prepare_features(frame, contract.features)
    values = prepared.to_numpy(float)
    median = np.nanmedian(values, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    values = np.where(np.isfinite(values), values, median)
    return values.astype("float32"), {
        "columns": list(prepared.columns),
        "median": [float(value) for value in median],
    }


def apply_transform(
    frame: pd.DataFrame, contract: ConditionalErrorContract, transform: dict[str, object]
) -> np.ndarray:
    prepared = prepare_features(frame, contract.features)
    if list(prepared.columns) != transform["columns"]:
        raise RuntimeError("conditional-error feature transform drift")
    values = prepared.to_numpy(float)
    median = np.asarray(transform["median"], float)
    return np.where(np.isfinite(values), values, median).astype("float32")


def catboost_model(contract: ConditionalErrorContract, seed: int) -> CatBoostRegressor:
    iterations, depth, learning_rate, l2 = contract.catboost_config
    parameters: dict[str, object] = {
        "iterations": int(iterations),
        "depth": int(depth),
        "learning_rate": float(learning_rate),
        "l2_leaf_reg": float(l2),
        "loss_function": f"Quantile:alpha={contract.quantile}",
        "random_seed": seed,
        "allow_writing_files": False,
        "verbose": False,
    }
    try:
        return CatBoostRegressor(task_type="GPU", devices="0", **parameters)
    except TypeError:
        return CatBoostRegressor(task_type="CPU", **parameters)


def fit_catboost(
    train: pd.DataFrame,
    contract: ConditionalErrorContract,
    seed: int,
) -> tuple[CatBoostRegressor, dict[str, object], str]:
    values, transform = fit_transform(train, contract)
    model = catboost_model(contract, seed)
    device = "gpu"
    try:
        model.fit(
            values,
            train.absolute_error.to_numpy(float),
            sample_weight=cruise_equal_weights(train.group_key),
        )
    except Exception as error:
        if "GPU" not in str(error).upper() and "CUDA" not in str(error).upper():
            raise
        iterations, depth, learning_rate, l2 = contract.catboost_config
        model = CatBoostRegressor(
            iterations=int(iterations),
            depth=int(depth),
            learning_rate=float(learning_rate),
            l2_leaf_reg=float(l2),
            loss_function=f"Quantile:alpha={contract.quantile}",
            random_seed=seed,
            task_type="CPU",
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(
            values,
            train.absolute_error.to_numpy(float),
            sample_weight=cruise_equal_weights(train.group_key),
        )
        device = "cpu"
    return model, transform, device


def fit_stratified(
    train: pd.DataFrame, target: str, contract: ConditionalErrorContract
) -> dict[str, object]:
    global_q90 = higher_quantile(train.absolute_error.to_numpy(float), contract.quantile)
    definitions = {"prediction": contract.prediction_edges[target], **contract.stratified_edges}
    factors: dict[str, dict[str, float]] = {}
    for field, edges in definitions.items():
        bins = pd.cut(train[field], edges, right=False, include_lowest=True)
        values = {}
        for label, cohort in train.groupby(bins, observed=True):
            if (
                len(cohort) >= contract.stratified_minimum_rows
                and cohort.group_key.nunique() >= contract.stratified_minimum_groups
            ):
                values[str(label)] = higher_quantile(
                    cohort.absolute_error.to_numpy(float), contract.quantile
                )
        factors[field] = values
    return {
        "global_q90": global_q90,
        "edges": {field: list(edges) for field, edges in definitions.items()},
        "factors": factors,
    }


def predict_stratified(frame: pd.DataFrame, model: dict[str, object]) -> np.ndarray:
    prediction = np.zeros(len(frame), dtype=float)
    available = np.zeros(len(frame), dtype=bool)
    factors = model["factors"]
    edges_by_field = model["edges"]
    for field, values in factors.items():
        labels = pd.cut(
            frame[field], edges_by_field[field], right=False, include_lowest=True
        ).astype(str)
        mapped = labels.map(values).to_numpy(float)
        finite = np.isfinite(mapped)
        prediction[finite] = np.maximum(prediction[finite], mapped[finite])
        available |= finite
    prediction[~available] = float(model["global_q90"])
    return prediction


def raw_prediction(
    candidate: str,
    train: pd.DataFrame,
    query: pd.DataFrame,
    target: str,
    contract: ConditionalErrorContract,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if candidate == "constant":
        value = higher_quantile(train.absolute_error.to_numpy(float), contract.quantile)
        return np.full(len(query), value), {"constant_q90": value}
    if candidate == "stratified":
        model = fit_stratified(train, target, contract)
        return predict_stratified(query, model), model
    model, transform, device = fit_catboost(train, contract, seed)
    prediction = model.predict(apply_transform(query, contract, transform))
    return np.maximum(np.asarray(prediction, float), 0), {
        "estimator": model,
        "transform": transform,
        "device": device,
    }


def save_final_model(
    candidate: str,
    model: dict[str, object],
    destination: Path,
    *,
    offset: float,
) -> list[Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if candidate == "catboost_quantile":
        estimator = model.pop("estimator")
        model_path = destination.with_suffix(".cbm")
        estimator.save_model(model_path)
        metadata_path = destination.with_suffix(".json")
        write_json(metadata_path, {**model, "calibration_offset": offset})
        return [metadata_path, model_path]
    model_path = destination.with_suffix(".json")
    write_json(model_path, {**model, "calibration_offset": offset})
    return [model_path]


def evaluate_candidate(
    target: str,
    protocol: str,
    candidate: str,
    frame: pd.DataFrame,
    assignment: np.ndarray,
    contract: ConditionalErrorContract,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    partitions = []
    fold_rows = []
    for fold in range(contract.group_folds):
        destination = (
            contract.output_root
            / "partitions"
            / f"{target}_{protocol}_{candidate}_fold{fold}.parquet"
        )
        if destination.is_file():
            result = pd.read_parquet(destination)
            partitions.append(result)
            fold_rows.append(
                {
                    "target": target,
                    "protocol": protocol,
                    "candidate": candidate,
                    "fold": fold,
                    **error_bound_metrics(result),
                    "calibration_offset": float(result.calibration_offset.iloc[0]),
                    "calibration_groups": int(result.calibration_groups.iloc[0]),
                }
            )
            continue
        test = assignment == fold
        calibration_fold = (fold + contract.calibration_lag) % contract.group_folds
        calibration = assignment == calibration_fold
        train = ~(test | calibration)
        training = frame.loc[train].reset_index(drop=True)
        calibration_frame = frame.loc[calibration].reset_index(drop=True)
        test_frame = frame.loc[test].reset_index(drop=True)
        combined = pd.concat([calibration_frame, test_frame], ignore_index=True)
        combined_raw, _ = raw_prediction(
            candidate,
            training,
            combined,
            target,
            contract,
            seed=81000 + fold,
        )
        calibration_raw = combined_raw[: len(calibration_frame)]
        test_raw = combined_raw[len(calibration_frame) :]
        offset, calibration_groups, rank = calibration_offset(
            calibration_frame.absolute_error.to_numpy(float),
            calibration_raw,
            calibration_frame.group_key.to_numpy(),
            quantile=contract.quantile,
            confidence=contract.confidence,
            minimum_groups=contract.minimum_calibration_groups,
        )
        if not np.isfinite(offset):
            raise RuntimeError(f"insufficient calibration groups for {target}/{protocol}/{fold}")
        result = test_frame.copy()
        result["raw_q90"] = test_raw
        result["calibration_offset"] = offset
        result["calibration_groups"] = calibration_groups
        result["calibration_rank"] = rank
        result["error_bound"] = test_raw + offset
        result["candidate"] = candidate
        result["estimator_fold"] = fold
        keep = [
            "evaluation_id",
            "record_id",
            "group_key",
            "target",
            "scheme",
            "fold",
            "ocean_region_id",
            "macro_basin_id",
            "yr",
            "mon",
            "coastal_node",
            "latitude",
            "longitude",
            "absolute_error",
            "raw_q90",
            "calibration_offset",
            "calibration_groups",
            "calibration_rank",
            "error_bound",
            "candidate",
            "estimator_fold",
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
        ]
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.loc[:, keep].to_parquet(destination, index=False, compression="zstd")
        partitions.append(result.loc[:, keep])
        fold_rows.append(
            {
                "target": target,
                "protocol": protocol,
                "candidate": candidate,
                "fold": fold,
                **error_bound_metrics(result),
                "calibration_offset": offset,
                "calibration_groups": calibration_groups,
            }
        )
        print(json.dumps(fold_rows[-1]), flush=True)
    return pd.concat(partitions, ignore_index=True), fold_rows


def run_protocol(
    target: str, protocol: str, contract: ConditionalErrorContract
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    source = contract.residual_bank_root / contract.residual_bank_pattern.format(
        target=target, protocol=protocol
    )
    frame = pd.read_parquet(source)
    assignment = stable_group_folds(
        frame.group_key,
        salt=f"{contract.split_salt}:{target}:{protocol}",
        folds=contract.group_folds,
    )
    summaries = []
    fold_rows = []
    predictions: dict[str, pd.DataFrame] = {}
    for candidate in contract.candidates:
        started = time.perf_counter()
        oof, rows = evaluate_candidate(target, protocol, candidate, frame, assignment, contract)
        predictions[candidate] = oof
        fold_rows.extend(rows)
        metrics = error_bound_metrics(oof)
        metrics.update(
            {
                "target": target,
                "protocol": protocol,
                "candidate": candidate,
                "minimum_fold_row_coverage": min(row["row_coverage"] for row in rows),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        metrics["eligible"] = bool(
            metrics["row_coverage"] >= contract.pooled_row_coverage_min
            and metrics["minimum_fold_row_coverage"] >= contract.minimum_fold_row_coverage
            and metrics["group_q90_coverage"] >= contract.pooled_group_q90_coverage_min
        )
        summaries.append(metrics)
    eligible = [row for row in summaries if row["eligible"]]
    selected = min(eligible, key=lambda row: row["mean_bound"]) if eligible else None
    decision: dict[str, object] = {
        "target": target,
        "protocol": protocol,
        "selected_candidate": selected["candidate"] if selected else "NONE",
        "calibration_passed": selected is not None,
        "fallback_tier": "U" if selected is None else "not_assigned_thresholds_unset",
    }
    if selected is not None:
        candidate = str(selected["candidate"])
        oof = predictions[candidate]
        final_offset, groups, rank = calibration_offset(
            oof.absolute_error.to_numpy(float),
            oof.raw_q90.to_numpy(float),
            oof.group_key.to_numpy(),
            quantile=contract.quantile,
            confidence=contract.confidence,
            minimum_groups=contract.minimum_calibration_groups,
        )
        _, final_model = raw_prediction(
            candidate,
            frame,
            frame.iloc[:1],
            target,
            contract,
            seed=81999,
        )
        files = save_final_model(
            candidate,
            final_model,
            contract.output_root / "models" / f"{target}_{protocol}_{candidate}",
            offset=final_offset,
        )
        decision.update(
            {
                "final_calibration_offset": final_offset,
                "final_calibration_groups": groups,
                "final_calibration_rank": rank,
                "model_files": [str(path).replace("\\", "/") for path in files],
            }
        )
    return summaries, fold_rows, decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--target", choices=("SSS", "fCO2", "all"), default="all")
    args = parser.parse_args()
    contract = load_conditional_error_contract(args.config)
    targets = contract.targets if args.target == "all" else (args.target,)
    all_summaries = []
    all_folds = []
    all_decisions = []
    for target in targets:
        for protocol in contract.protocols:
            summaries, folds, decision = run_protocol(target, protocol, contract)
            all_summaries.extend(summaries)
            all_folds.extend(folds)
            all_decisions.append(decision)
            table_dir = contract.output_root / "tables"
            table_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(all_summaries).to_csv(
                table_dir / f"{target}_candidate_summary.csv", index=False
            )
            pd.DataFrame(all_folds).to_csv(table_dir / f"{target}_fold_metrics.csv", index=False)
            write_json(table_dir / f"{target}_decisions.json", all_decisions)
    if args.target == "all":
        (contract.output_root / "conditional_error.complete").write_text(
            "complete\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
