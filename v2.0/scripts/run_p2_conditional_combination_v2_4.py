"""Run the preregistered Issue 64 graph and uncertainty controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from recad.evaluate.conditional_models import (
    SeasonalNearestResidual,
    finite_sample_quantile,
    water_graph,
)
from recad.evaluate.context_connectivity import water_neighbors
from recad.evaluate.model_zoo import fit_model, group_equal_rmse, predict_model
from recad.evaluate.predictor_ablation import balanced_sample_weight
from recad.evaluate.validation_repair import conformal_widths, grouped_fit_calibration
from run_p2_model_zoo_v2_4 import prepare_split, split_seed
from run_p2_process_predictors_v2_4 import (
    fit_variant,
    load_augmented_frame,
    variant_features,
)
from run_p2_validation_repair_v2_4 import cached_splits
from run_p2_validation_repair_v2_4 import MODERN_RF_FEATURES

OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_4\p2_conditional_combination_v2.4")
PARTITIONS = OUTPUT / "partitions"
SELECTION = OUTPUT / "selection"
TABLES = OUTPUT / "tables"
ISSUE63 = Path(r"C:\backup\phd\data\processed\recad_v2_4\p2_process_predictors_v2.4")
ISSUE62 = Path(r"C:\backup\phd\data\processed\recad_v2_4\p2_model_zoo_v2.4")
CUBE = Path(r"C:\backup\phd\data\processed\recad_v2_3\global_coastal_predictor_cube_v2.3.nc")
SEED = 64024
VARIANT = {"SSS": "S0_base", "fCO2": "F4_continuous_background_residual"}
SCALE_CONFIG = [192, 24, 20, 0.7]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def selected_config(target: str, scheme: str, fold: int) -> list[float | int | None]:
    variant = VARIANT[target]
    path = ISSUE63 / "selection" / f"{target}_{scheme}_fold{fold}_{variant}.json"
    return json.loads(path.read_text(encoding="utf-8"))["selected_config"]


def topology():
    with xr.open_dataset(CUBE, decode_times=False) as dataset:
        flat = dataset.grid_flat.values.astype(np.int64)
    return water_graph(water_neighbors(flat))


def standardized_widths(
    calibration_truth: np.ndarray,
    calibration_prediction: np.ndarray,
    calibration_scale: np.ndarray,
    held_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scale = np.maximum(np.asarray(calibration_scale, float), 1e-3)
    scores = np.abs(np.asarray(calibration_truth) - calibration_prediction) / scale
    q50 = finite_sample_quantile(scores, 0.5)
    q90 = finite_sample_quantile(scores, 0.9)
    return q50 * np.maximum(held_scale, 1e-3), q90 * np.maximum(held_scale, 1e-3)


def run_target(target: str, graph) -> None:
    frame = load_augmented_frame(target)
    summaries: list[dict[str, object]] = []
    for split in cached_splits(target, frame):
        destination = PARTITIONS / f"{target}_{split.scheme}_fold{split.fold}.parquet"
        selection_file = SELECTION / f"{target}_{split.scheme}_fold{split.fold}.json"
        if destination.is_file() and selection_file.is_file():
            summaries.append(json.loads(selection_file.read_text(encoding="utf-8"))["summary"])
            continue
        tune_train, tune_validation, model_fit, calibration, held = prepare_split(
            target, frame, split
        )
        configuration = selected_config(target, split.scheme, split.fold)
        provisional = fit_variant(
            target,
            VARIANT[target],
            configuration,
            tune_train,
            tune_validation,
            split_seed(target, split, "issue64-inner", SEED),
        )
        honest = tune_validation[["record_id", "group_key", "coastal_node", "mon"]].copy()
        honest["residual"] = tune_validation.truth.to_numpy(float) - provisional
        source_index, selection_index = grouped_fit_calibration(
            honest,
            calibration_fold=0,
            n_folds=2,
            salt=f"issue64-graph-{target}-{split.scheme}-{split.fold}",
        )
        adapter = SeasonalNearestResidual.fit(graph, honest.iloc[source_index], maximum_hops=64)
        candidates = []
        for decay in (8, 16, 32):
            for shrinkage in (0.25, 0.5):
                correction = adapter.predict(
                    honest.iloc[selection_index], decay_hops=decay, shrinkage=shrinkage
                )
                predicted = provisional[selection_index] + correction
                candidates.append(
                    {
                        "decay_hops": decay,
                        "shrinkage": shrinkage,
                        "group_equal_rmse": group_equal_rmse(
                            tune_validation.iloc[selection_index], predicted
                        ),
                    }
                )
        selected = min(candidates, key=lambda row: row["group_equal_rmse"])
        adapter = SeasonalNearestResidual.fit(graph, honest, maximum_hops=64)

        scale_target = np.log1p(np.abs(honest.residual.to_numpy(float)))
        scale_model = fit_model(
            "extra_trees",
            SCALE_CONFIG,
            tune_validation,
            variant_features(target, VARIANT[target]),
            scale_target,
            balanced_sample_weight(tune_validation),
            split_seed(target, split, "issue64-scale", SEED),
        )

        combined = pd.concat([calibration, held], ignore_index=True)
        base = fit_variant(
            target,
            VARIANT[target],
            configuration,
            model_fit,
            combined,
            split_seed(target, split, "issue64-final", SEED),
        )
        correction = adapter.predict(
            combined,
            decay_hops=float(selected["decay_hops"]),
            shrinkage=float(selected["shrinkage"]),
        )
        graph_prediction = base + correction
        n_cal = len(calibration)
        control50, control90, _ = conformal_widths(
            calibration.truth.to_numpy(float),
            base[:n_cal],
            base[n_cal:],
            target=target,
        )
        log_scale, _ = predict_model(scale_model, combined)
        scale = np.maximum(np.expm1(log_scale), 1e-3)
        scale50, scale90 = standardized_widths(
            calibration.truth.to_numpy(float),
            base[:n_cal],
            scale[:n_cal],
            scale[n_cal:],
        )
        held_output = held[["record_id", "group_key", "truth"]].copy()
        held_output["base_prediction"] = base[n_cal:]
        held_output["graph_prediction"] = graph_prediction[n_cal:]
        held_output["graph_correction"] = correction[n_cal:]
        held_output["control_width50"] = control50
        held_output["control_width90"] = control90
        held_output["scale_width50"] = scale50
        held_output["scale_width90"] = scale90
        held_output["predicted_scale"] = scale[n_cal:]
        held_output["target"] = target
        held_output["scheme"] = split.scheme
        held_output["fold"] = split.fold
        destination.parent.mkdir(parents=True, exist_ok=True)
        held_output.to_parquet(destination, index=False, compression="zstd")
        summary = {
            "target": target,
            "scheme": split.scheme,
            "fold": split.fold,
            "held_rows": len(held),
            "base_group_equal_rmse": group_equal_rmse(held, base[n_cal:]),
            "graph_group_equal_rmse": group_equal_rmse(held, graph_prediction[n_cal:]),
            "selected_decay_hops": selected["decay_hops"],
            "selected_shrinkage": selected["shrinkage"],
        }
        write_json(
            selection_file,
            {
                "target": target,
                "scheme": split.scheme,
                "fold": split.fold,
                "base_variant": VARIANT[target],
                "base_configuration": configuration,
                "source_groups": int(honest.iloc[source_index].group_key.nunique()),
                "selection_groups": int(honest.iloc[selection_index].group_key.nunique()),
                "candidates": candidates,
                "selected": selected,
                "summary": summary,
            },
        )
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(TABLES / f"{target}_partition_summary.csv", index=False)


def run_ensemble_trigger_audit() -> None:
    """Use honest inner-validation residuals to decide whether an ensemble is allowed."""

    destination = TABLES / "fCO2_inner_ensemble_trigger.csv"
    if destination.is_file():
        return
    frame = load_augmented_frame("fCO2")
    rows = []
    for split in cached_splits("fCO2", frame):
        tune_train, tune_validation, _model_fit, _calibration, _held = prepare_split(
            "fCO2", frame, split
        )
        f4 = fit_variant(
            "fCO2",
            VARIANT["fCO2"],
            selected_config("fCO2", split.scheme, split.fold),
            tune_train,
            tune_validation,
            split_seed("fCO2", split, "issue64-ensemble-f4", SEED),
        )
        cat_path = ISSUE62 / "selection" / f"fCO2_{split.scheme}_fold{split.fold}_catboost.json"
        cat_config = json.loads(cat_path.read_text(encoding="utf-8"))["selected_config"]
        cat_model = fit_model(
            "catboost",
            cat_config,
            tune_train,
            MODERN_RF_FEATURES,
            tune_train.truth.to_numpy(float),
            balanced_sample_weight(tune_train),
            split_seed("fCO2", split, "issue64-ensemble-catboost", SEED),
        )
        cat, _ = predict_model(cat_model, tune_validation)
        truth = tune_validation.truth.to_numpy(float)
        residual_f4 = truth - f4
        residual_cat = truth - cat
        delta = cat - f4
        denominator = float(np.dot(delta, delta))
        weight_cat = (
            0.0
            if denominator == 0
            else float(np.clip(np.dot(residual_f4, delta) / denominator, 0, 1))
        )
        blend = f4 + weight_cat * delta
        base_rmse = float(np.sqrt(np.mean(residual_f4**2)))
        blend_rmse = float(np.sqrt(np.mean((truth - blend) ** 2)))
        row = {
            "scheme": split.scheme,
            "fold": split.fold,
            "rows": len(tune_validation),
            "residual_correlation": float(np.corrcoef(residual_f4, residual_cat)[0, 1]),
            "oracle_weight_catboost": weight_cat,
            "f4_rmse": base_rmse,
            "blend_rmse": blend_rmse,
            "relative_improvement": (base_rmse - blend_rmse) / base_rmse,
        }
        rows.append(row)
        print(json.dumps({"ensemble_trigger": row}), flush=True)
    result = pd.DataFrame(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    summary = (
        result.groupby("scheme")
        .agg(
            median_correlation=("residual_correlation", "median"),
            median_improvement=("relative_improvement", "median"),
        )
        .reset_index()
    )
    triggered = bool(
        (result.residual_correlation.median() < 0.90)
        and ((summary.median_improvement >= 0.01).sum() >= 2)
        and (summary.median_improvement.min() >= 0)
    )
    write_json(
        OUTPUT / "ensemble_trigger_decision.json",
        {
            "triggered": triggered,
            "rule": "median residual correlation <0.90; at least two schemes median improvement >=1%; no scheme median regression",
            "overall_median_correlation": float(result.residual_correlation.median()),
            "schemes": summary.to_dict(orient="records"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["SSS", "fCO2", "all"], default="all")
    parser.add_argument("--ensemble-audit", action="store_true")
    args = parser.parse_args()
    graph = topology()
    targets = ("SSS", "fCO2") if args.target == "all" else (args.target,)
    for target in targets:
        run_target(target, graph)
    if args.ensemble_audit:
        run_ensemble_trigger_audit()
    if args.target == "all":
        (OUTPUT / "training.complete").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
