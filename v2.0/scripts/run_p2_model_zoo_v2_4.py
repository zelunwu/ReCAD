"""Run the finite fair model-family screen registered in GitHub Issue 62."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from recad.evaluate.fco2_bakeoff import interval_scores
from recad.evaluate.model_zoo import (
    deterministic_cap,
    fit_model,
    group_equal_rmse,
    predict_model,
    save_model,
)
from recad.evaluate.predictor_ablation import balanced_sample_weight
from recad.evaluate.validation_repair import (
    RepairedSplit,
    conformal_widths,
    grouped_fit_calibration,
    inner_group_folds,
)
from run_p2_validation_repair_v2_4 import (
    MODERN_RF_FEATURES,
    SSS_RESIDUAL_FEATURES,
    cached_splits,
    crossfit_sss,
    fit_predict_sss,
    load_frame,
)
from run_p2_validation_repair_v2_4 import (
    OUTPUT as REPAIR_OUTPUT,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/p2_model_zoo_v2.4.json"
OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_4\p2_model_zoo_v2.4")
PARTITIONS = OUTPUT / "partitions"
CHECKPOINTS = OUTPUT / "checkpoints"
SELECTION = OUTPUT / "selection"
RESOURCES = OUTPUT / "resources"
INNER_UPSTREAM = OUTPUT / "inner_upstream_sss"
TABLES = OUTPUT / "tables"
FAMILIES = (
    "ridge",
    "extra_trees",
    "catboost",
    "xgboost",
    "residual_mlp",
    "ft_transformer",
)


def sha256(path: Path) -> str:
    """Return a file SHA256."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    """Write deterministic JSON with repository-stable line endings."""

    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_bytes(content.encode("utf-8"))


def config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def family_configs(family: str) -> list[list[float | int | None]]:
    """Return the six frozen candidate configurations for a family."""

    specification = config()["families"][family]
    if family == "ridge":
        return [[value] for value in specification["alpha"]]
    return specification["configurations"]


def split_seed(target: str, split: RepairedSplit, family: str, seed: int) -> int:
    """Return a deterministic model seed unique to the experiment cell."""

    value = f"{target}:{split.scheme}:{split.fold}:{family}:{seed}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "little") & 0x7FFFFFFF


def nested_tuning_sss(
    split: RepairedSplit,
    model_fit: pd.DataFrame,
    tune_train_index: np.ndarray,
    tune_validation_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build upstream SSS without using any inner-selection SSS label."""

    destination = INNER_UPSTREAM / f"fCO2_{split.scheme}_fold{split.fold}.parquet"
    tune_train = model_fit.iloc[tune_train_index].reset_index(drop=True)
    tune_validation = model_fit.iloc[tune_validation_index].reset_index(drop=True)
    if destination.is_file():
        cached = pd.read_parquet(destination).set_index(["role", "record_id"])
        train_map = cached.loc["train", "sss_crossfit"]
        validation_map = cached.loc["validation", "sss_crossfit"]
        return (
            tune_train.record_id.map(train_map).to_numpy("float32"),
            tune_validation.record_id.map(validation_map).to_numpy("float32"),
        )
    INNER_UPSTREAM.mkdir(parents=True, exist_ok=True)
    assignment = inner_group_folds(
        tune_train,
        n_folds=2,
        salt=f"v2.4-model-zoo-inner-sss-{split.scheme}-{split.fold}",
    )
    train_prediction = np.full(len(tune_train), np.nan, dtype="float32")
    for inner in range(2):
        selected = assignment == inner
        train_prediction[selected] = fit_predict_sss(
            tune_train.loc[~selected],
            tune_train.loc[selected],
            seed=split_seed("fCO2", split, "inner_sss", 62024 + inner),
        )
    validation_prediction = fit_predict_sss(
        tune_train,
        tune_validation,
        seed=split_seed("fCO2", split, "inner_sss", 62026),
    )
    if not np.isfinite(train_prediction).all() or not np.isfinite(validation_prediction).all():
        raise RuntimeError("nested tuning SSS contains non-finite predictions")
    pd.concat(
        [
            pd.DataFrame(
                {
                    "role": "train",
                    "record_id": tune_train.record_id,
                    "sss_crossfit": train_prediction,
                }
            ),
            pd.DataFrame(
                {
                    "role": "validation",
                    "record_id": tune_validation.record_id,
                    "sss_crossfit": validation_prediction,
                }
            ),
        ],
        ignore_index=True,
    ).to_parquet(destination, index=False, compression="zstd")
    return train_prediction, validation_prediction


def prepare_split(
    target: str, frame: pd.DataFrame, split: RepairedSplit
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return inner train/selection, final model-fit, calibration, and held frames."""

    outer_fit = frame.iloc[split.fit_index].reset_index(drop=True)
    held = frame.iloc[split.held_index].reset_index(drop=True)
    model_fit_index, calibration_index = grouped_fit_calibration(
        outer_fit,
        salt=f"v2.4-calibration-{target}-{split.scheme}-{split.fold}",
    )
    model_fit = outer_fit.iloc[model_fit_index].reset_index(drop=True)
    calibration = outer_fit.iloc[calibration_index].reset_index(drop=True)
    tune_train_index, tune_validation_index = grouped_fit_calibration(
        model_fit,
        salt=f"v2.4-model-zoo-selection-{target}-{split.scheme}-{split.fold}",
    )
    tune_train = model_fit.iloc[tune_train_index].reset_index(drop=True)
    tune_validation = model_fit.iloc[tune_validation_index].reset_index(drop=True)
    if target == "fCO2":
        train_sss, validation_sss = nested_tuning_sss(
            split, model_fit, tune_train_index, tune_validation_index
        )
        tune_train["sss_crossfit"] = train_sss
        tune_validation["sss_crossfit"] = validation_sss
        predict_frame = pd.concat([calibration, held], ignore_index=True)
        fit_sss, predict_sss = crossfit_sss("fCO2", split, model_fit, predict_frame)
        model_fit["sss_crossfit"] = fit_sss
        predict_frame["sss_crossfit"] = predict_sss
        calibration = predict_frame.iloc[: len(calibration)].reset_index(drop=True)
        held = predict_frame.iloc[len(calibration) :].reset_index(drop=True)
    return tune_train, tune_validation, model_fit, calibration, held


def fit_family(
    target_name: str,
    family: str,
    configuration: list[float | int | None],
    train: pd.DataFrame,
    predict: pd.DataFrame,
    seed: int,
    checkpoint_prefix: Path | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit one family under the target-specific frozen parameterization."""

    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    resources: list[dict[str, object]] = []
    if target_name == "SSS":
        direct = fit_model(
            family,
            configuration,
            train,
            SSS_RESIDUAL_FEATURES,
            train.truth.to_numpy(float),
            balanced_sample_weight(train),
            seed,
        )
        prediction, direct_inference = predict_model(direct, predict)
        available_train = train.sss_bg.notna().to_numpy()
        residual_train = train.loc[available_train].reset_index(drop=True)
        residual = fit_model(
            family,
            configuration,
            residual_train,
            SSS_RESIDUAL_FEATURES,
            residual_train.truth.to_numpy(float) - residual_train.sss_bg.to_numpy(float),
            balanced_sample_weight(residual_train),
            seed + 1,
        )
        available_predict = predict.sss_bg.notna().to_numpy()
        residual_prediction, residual_inference = predict_model(
            residual, predict.loc[available_predict]
        )
        prediction[available_predict] = (
            predict.loc[available_predict, "sss_bg"].to_numpy(float) + residual_prediction
        )
        if checkpoint_prefix is not None:
            direct_paths = save_model(
                direct, checkpoint_prefix.with_name(checkpoint_prefix.name + "_direct")
            )
            residual_paths = save_model(
                residual, checkpoint_prefix.with_name(checkpoint_prefix.name + "_residual")
            )
            resources.extend(
                {"checkpoint": str(path), "sha256": sha256(path)}
                for path in [*direct_paths, *residual_paths]
            )
        parameter_count = direct.parameter_count + residual.parameter_count
        updates = direct.updates + residual.updates
        device = f"{direct.device}+{residual.device}"
        inference_seconds = direct_inference + residual_inference
        del direct, residual
    else:
        fitted = fit_model(
            family,
            configuration,
            train,
            MODERN_RF_FEATURES,
            train.truth.to_numpy(float),
            balanced_sample_weight(train),
            seed,
        )
        prediction, inference_seconds = predict_model(fitted, predict)
        if checkpoint_prefix is not None:
            paths = save_model(fitted, checkpoint_prefix)
            resources.extend({"checkpoint": str(path), "sha256": sha256(path)} for path in paths)
        parameter_count = fitted.parameter_count
        updates = fitted.updates
        device = fitted.device
        del fitted
    peak_gpu = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prediction.astype("float32"), {
        "elapsed_seconds": time.perf_counter() - started,
        "inference_seconds": inference_seconds,
        "parameter_count": parameter_count,
        "updates": updates,
        "device": device,
        "peak_gpu_bytes": peak_gpu,
        "checkpoint_files": resources,
    }


def select_configuration(
    target: str,
    split: RepairedSplit,
    family: str,
    tune_train: pd.DataFrame,
    tune_validation: pd.DataFrame,
) -> tuple[list[float | int | None], list[dict[str, object]]]:
    """Evaluate the six frozen configs on a group-disjoint inner holdout."""

    path = SELECTION / f"{target}_{split.scheme}_fold{split.fold}_{family}.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["selected_config"], payload["scores"]
    specification = config()["inner_selection"]
    train = deterministic_cap(
        tune_train,
        int(specification["maximum_training_rows"]),
        f"{target}:{split.scheme}:{split.fold}:{family}:train",
    )
    validation = deterministic_cap(
        tune_validation,
        int(specification["maximum_selection_rows"]),
        f"{target}:{split.scheme}:{split.fold}:{family}:validation",
    )
    scores = []
    for index, candidate in enumerate(family_configs(family)):
        started = time.perf_counter()
        prediction, resources = fit_family(
            target,
            family,
            candidate,
            train,
            validation,
            split_seed(target, split, family, 62024 + index),
        )
        scores.append(
            {
                "config_index": index,
                "config": candidate,
                "group_equal_rmse": group_equal_rmse(validation, prediction),
                "pooled_rmse": float(
                    np.sqrt(np.mean((prediction - validation.truth.to_numpy(float)) ** 2))
                ),
                "elapsed_seconds": time.perf_counter() - started,
                "parameter_count": resources["parameter_count"],
                "updates": resources["updates"],
                "device": resources["device"],
            }
        )
    selected = min(scores, key=lambda row: (row["group_equal_rmse"], row["config_index"]))
    payload = {
        "target": target,
        "scheme": split.scheme,
        "fold": split.fold,
        "family": family,
        "train_rows": len(train),
        "selection_rows": len(validation),
        "train_groups": train.group_key.nunique(),
        "selection_groups": validation.group_key.nunique(),
        "group_overlap": len(set(train.group_key) & set(validation.group_key)),
        "selected_config_index": selected["config_index"],
        "selected_config": selected["config"],
        "scores": scores,
    }
    write_json(path, payload)
    return selected["config"], scores


def write_partition(
    target: str,
    split: RepairedSplit,
    family: str,
    seed: int,
    calibration: pd.DataFrame,
    held: pd.DataFrame,
    calibration_prediction: np.ndarray,
    held_prediction: np.ndarray,
) -> Path:
    """Calibrate and persist one candidate outer-prediction partition."""

    destination = (
        PARTITIONS / f"{target}_{split.scheme}_fold{split.fold}_{family}_seed{seed}.parquet"
    )
    width50, width90, bands = conformal_widths(
        calibration.truth.to_numpy(float),
        calibration_prediction,
        held_prediction,
        target=target,
    )
    result = held[
        [
            "record_id",
            "group_key",
            "yr",
            "mon",
            "grid_flat",
            "latitude",
            "longitude",
            "lme_id",
            "distance_to_significant_land_km",
            "shelf_core_mask",
            "truth",
        ]
    ].copy()
    result["prediction"] = held_prediction
    result["eligible"] = np.isfinite(held_prediction)
    result["calibrated"] = np.isfinite(held_prediction)
    result["width50"] = width50.astype("float32")
    result["width90"] = width90.astype("float32")
    result["prediction_band"] = bands
    result["target"] = target
    result["scheme"] = split.scheme
    result["fold"] = split.fold
    result["model"] = family
    result["seed"] = seed
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(destination, index=False, compression="zstd")
    return destination


def run_cell(
    target: str,
    frame: pd.DataFrame,
    split: RepairedSplit,
    family: str,
    seed: int,
    *,
    tune: bool,
) -> None:
    """Run one target/split/family/seed cell with resumable output."""

    destination = (
        PARTITIONS / f"{target}_{split.scheme}_fold{split.fold}_{family}_seed{seed}.parquet"
    )
    if destination.is_file():
        return
    tune_train, tune_validation, model_fit, calibration, held = prepare_split(target, frame, split)
    if tune:
        selected, selection_scores = select_configuration(
            target, split, family, tune_train, tune_validation
        )
    else:
        selection_path = SELECTION / f"{target}_{split.scheme}_fold{split.fold}_{family}.json"
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        selected = payload["selected_config"]
        selection_scores = payload["scores"]
    predict_frame = pd.concat([calibration, held], ignore_index=True)
    checkpoint_prefix = (
        CHECKPOINTS / f"{target}_{split.scheme}_fold{split.fold}_{family}_seed{seed}"
    )
    prediction, resources = fit_family(
        target,
        family,
        selected,
        model_fit,
        predict_frame,
        split_seed(target, split, family, seed),
        checkpoint_prefix,
    )
    calibration_prediction = prediction[: len(calibration)]
    held_prediction = prediction[len(calibration) :]
    partition = write_partition(
        target,
        split,
        family,
        seed,
        calibration,
        held,
        calibration_prediction,
        held_prediction,
    )
    write_json(
        RESOURCES / f"{target}_{split.scheme}_fold{split.fold}_{family}_seed{seed}.json",
        {
            "target": target,
            "scheme": split.scheme,
            "fold": split.fold,
            "family": family,
            "seed": seed,
            "fit_rows": len(model_fit),
            "calibration_rows": len(calibration),
            "held_rows": len(held),
            "selected_config": selected,
            "selection_scores": selection_scores,
            "partition": str(partition),
            "partition_sha256": sha256(partition),
            **resources,
        },
    )


def metric_row(frame: pd.DataFrame, **keys: object) -> dict[str, object]:
    """Return pooled, group-equal, and interval metrics."""

    frame = frame.loc[frame.eligible & frame.prediction.notna()].copy()
    error = frame.prediction.to_numpy(float) - frame.truth.to_numpy(float)
    denominator = np.sum((frame.truth - frame.truth.mean()) ** 2)
    row: dict[str, object] = {
        **keys,
        "n": len(frame),
        "groups": frame.group_key.nunique(),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1 - np.sum(error**2) / denominator) if denominator else np.nan,
        "group_equal_rmse": group_equal_rmse(frame, frame.prediction.to_numpy(float)),
    }
    calibrated = frame.loc[frame.calibrated & frame.width90.notna()]
    if not calibrated.empty:
        row.update(
            interval_scores(
                calibrated.truth.to_numpy(float),
                calibrated.prediction.to_numpy(float),
                calibrated.width50.to_numpy(float),
                calibrated.width90.to_numpy(float),
            )
        )
    return row


def screen_metrics() -> pd.DataFrame:
    """Aggregate seed-62024 candidate metrics and repaired controls."""

    rows = []
    for path in sorted(PARTITIONS.glob("*_seed62024.parquet")):
        frame = pd.read_parquet(path)
        keys = frame[["target", "scheme", "model", "seed"]].iloc[0].to_dict()
        rows.append(metric_row(frame, **keys))
    candidates = pd.DataFrame(rows)
    controls = pd.read_csv(REPAIR_OUTPUT / "tables/corrected_native_metrics.csv")
    controls = controls.rename(columns={"cruise_equal_rmse": "group_equal_rmse"})
    controls["seed"] = -1
    TABLES.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(TABLES / "screen_candidate_metrics.csv", index=False)
    controls.to_csv(TABLES / "issue61_control_metrics.csv", index=False)
    return candidates


def select_finalists() -> dict[str, object]:
    """Apply the frozen improvement/Pareto rule after the complete screen."""

    metrics = screen_metrics()
    controls = pd.read_csv(TABLES / "issue61_control_metrics.csv")
    decisions: dict[str, object] = {}
    for target in ("SSS", "fCO2"):
        baseline_model = "catboost_background_residual" if target == "SSS" else "modernized_rf"
        baseline = (
            controls.loc[(controls.target == target) & (controls.model == baseline_model)]
            .set_index("scheme")
            .rmse
        )
        table = metrics.loc[metrics.target == target].pivot(
            index="model", columns="scheme", values="rmse"
        )
        improvement = table.copy()
        for scheme in table.columns:
            improvement[scheme] = (baseline[scheme] - table[scheme]) / baseline[scheme]
        qualified = []
        for family, row in improvement.iterrows():
            passes_two = int((row >= 0.05).sum()) >= 2
            no_regression = bool((row >= -0.03).all())
            if passes_two and no_regression:
                qualified.append(family)
        ranks = table.rank(axis=0, method="average").mean(axis=1).sort_values()
        selected = qualified[:2]
        status = "promotion_finalist"
        if not selected:
            selected = list(ranks.index[:2])
            status = "stability_finalist_only"
        decisions[target] = {
            "baseline": baseline_model,
            "selected": selected,
            "status": status,
            "qualified": qualified,
            "mean_rank": {name: float(value) for name, value in ranks.items()},
            "relative_improvement": improvement.to_dict("index"),
        }
    payload = {
        "screen_seed": 62024,
        "decisions": decisions,
        "conditional_context": config()["conditional_context"],
    }
    write_json(OUTPUT / "finalists.json", payload)
    return payload


def run_models(
    stage: str,
    only_target: str | None,
    only_scheme: str | None,
    only_fold: int | None,
    only_family: str | None,
) -> None:
    """Run screen or additional finalist seeds."""

    targets = [only_target] if only_target else ["SSS", "fCO2"]
    finalist_payload = None
    if stage == "finalists":
        finalist_payload = json.loads((OUTPUT / "finalists.json").read_text(encoding="utf-8"))
    for target in targets:
        frame = load_frame(target)
        splits = cached_splits(target, frame)
        families = [only_family] if only_family else list(FAMILIES)
        if finalist_payload is not None:
            selected = finalist_payload["decisions"][target]["selected"]
            families = [family for family in families if family in selected]
        for split in splits:
            if only_scheme and split.scheme != only_scheme:
                continue
            if only_fold is not None and split.fold != only_fold:
                continue
            for family in families:
                seeds = [62024] if stage == "screen" else [62025, 62026]
                for seed in seeds:
                    print(
                        f"{stage} {target} {split.scheme}/{split.fold} {family} seed={seed}",
                        flush=True,
                    )
                    run_cell(
                        target,
                        frame,
                        split,
                        family,
                        seed,
                        tune=stage == "screen",
                    )


def analyze_final() -> None:
    """Aggregate all seeds, stability, risk, compute, and context triggers."""

    rows = []
    risk_rows = []
    for path in sorted(PARTITIONS.glob("*.parquet")):
        frame = pd.read_parquet(path)
        keys = frame[["target", "scheme", "model", "seed"]].iloc[0].to_dict()
        rows.append(metric_row(frame, **keys))
        selected = frame.loc[frame.eligible & frame.width90.notna()].copy()
        selected["squared_error"] = (selected.prediction - selected.truth) ** 2
        selected = selected.sort_values("width90")
        for retained in (0.25, 0.5, 0.75, 1.0):
            subset = selected.iloc[: max(1, int(len(selected) * retained))]
            risk_rows.append(
                {
                    **keys,
                    "retained_fraction": retained,
                    "n": len(subset),
                    "rmse": float(np.sqrt(subset.squared_error.mean())),
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(TABLES / "all_seed_metrics.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(TABLES / "risk_coverage.csv", index=False)
    stability = metrics.groupby(["target", "scheme", "model"], as_index=False).agg(
        seeds=("seed", "nunique"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        group_equal_rmse_mean=("group_equal_rmse", "mean"),
        coverage90_mean=("coverage90", "mean"),
    )
    stability.to_csv(TABLES / "stability_metrics.csv", index=False)
    resource_rows = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(RESOURCES.glob("*.json"))
    ]
    pd.DataFrame(resource_rows).drop(columns=["selection_scores", "checkpoint_files"]).to_csv(
        TABLES / "compute_metrics.csv", index=False
    )
    checkpoint_rows = []
    for row in resource_rows:
        for checkpoint in row["checkpoint_files"]:
            checkpoint_rows.append(
                {
                    "target": row["target"],
                    "scheme": row["scheme"],
                    "fold": row["fold"],
                    "family": row["family"],
                    "seed": row["seed"],
                    **checkpoint,
                }
            )
    pd.DataFrame(checkpoint_rows).to_csv(TABLES / "checkpoint_manifest.csv", index=False)
    selection_rows = []
    for path in sorted(SELECTION.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for score in payload["scores"]:
            selection_rows.append(
                {
                    "target": payload["target"],
                    "scheme": payload["scheme"],
                    "fold": payload["fold"],
                    "family": payload["family"],
                    "selected": score["config_index"] == payload["selected_config_index"],
                    **score,
                }
            )
    pd.DataFrame(selection_rows).to_csv(TABLES / "inner_selection_metrics.csv", index=False)
    finalists = json.loads((OUTPUT / "finalists.json").read_text(encoding="utf-8"))
    triggers = {}
    for target, decision in finalists["decisions"].items():
        target_metrics = stability.loc[
            (stability.target == target) & stability.model.isin(decision["selected"])
        ]
        cruise = target_metrics.loc[target_metrics.scheme == "cruise", "rmse_mean"].min()
        geographic = target_metrics.loc[target_metrics.scheme == "geographic", "rmse_mean"].min()
        forward = target_metrics.loc[target_metrics.scheme == "forward", "rmse_mean"].min()
        ceilings = config()["promotion"][f"{target}_rmse_ceiling"]
        temporal = bool(
            cruise <= ceilings["cruise"]
            and geographic <= ceilings["geographic"]
            and forward > ceilings["forward"] * 1.05
        )
        triggers[target] = {
            "temporal_triggered": temporal,
            "graph_triggered": False,
            "graph_reason": "requires corrected water-connected residual analysis; no automatic family-wide graph claim",
        }
    write_json(
        OUTPUT / "decision.json",
        {
            "status": "development_model_zoo_complete",
            "finalists": finalists,
            "context_triggers": triggers,
            "new_independent_labels_opened": False,
            "production_model_promoted": False,
        },
    )


def write_manifest() -> None:
    """Write external artifact counts and source hashes."""

    sources = [
        CONFIG_PATH,
        Path(config()["data_contract"]["matchup"]),
        REPAIR_OUTPUT / "splits/SSS_split_indices.npz",
        REPAIR_OUTPUT / "splits/fCO2_split_indices.npz",
    ]
    write_json(
        OUTPUT / "data_manifest.json",
        {
            "sources": {
                str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in sources
            },
            "prediction_partitions": len(list(PARTITIONS.glob("*.parquet"))),
            "selection_records": len(list(SELECTION.glob("*.json"))),
            "checkpoint_files": len([path for path in CHECKPOINTS.glob("*") if path.is_file()]),
            "resource_records": len(list(RESOURCES.glob("*.json"))),
            "new_independent_labels_opened": False,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("screen", "select-finalists", "finalists", "analyze"),
        required=True,
    )
    parser.add_argument("--target", choices=("SSS", "fCO2"))
    parser.add_argument("--scheme", choices=("cruise", "geographic", "whole_region", "forward"))
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--family", choices=FAMILIES)
    args = parser.parse_args()
    for directory in (OUTPUT, PARTITIONS, CHECKPOINTS, SELECTION, RESOURCES, TABLES):
        directory.mkdir(parents=True, exist_ok=True)
    if args.stage in {"screen", "finalists"}:
        run_models(args.stage, args.target, args.scheme, args.fold, args.family)
    elif args.stage == "select-finalists":
        select_finalists()
    else:
        analyze_final()
    write_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
