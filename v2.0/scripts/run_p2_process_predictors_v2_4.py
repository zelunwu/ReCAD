"""Run the preregistered Issue 63 single-factor process ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from recad.evaluate.fco2_bakeoff import interval_scores
from recad.evaluate.model_zoo import (
    deterministic_cap,
    fit_model,
    group_equal_rmse,
    predict_model,
    save_model,
)
from recad.evaluate.predictor_ablation import balanced_sample_weight
from recad.evaluate.process_predictors import ContinuousBackground
from recad.evaluate.validation_repair import RepairedSplit, conformal_widths
from run_p2_model_zoo_v2_4 import family_configs, prepare_split, split_seed
from run_p2_validation_repair_v2_4 import (
    MODERN_RF_FEATURES,
    SSS_RESIDUAL_FEATURES,
    cached_splits,
    load_frame,
)

OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_4\p2_process_predictors_v2.4")
FEATURE_CACHE = OUTPUT / "process_features.parquet"
PARTITIONS = OUTPUT / "partitions"
SELECTION = OUTPUT / "selection"
CHECKPOINTS = OUTPUT / "checkpoints"
TABLES = OUTPUT / "tables"
SCREEN_SEED = 63024
EXTRA_SEEDS = (63025, 63026)

FRONT = (
    "sst_front_mad",
    "sst_front_range",
    "sst_front_neighbour_count",
    "wspd_front_mad",
    "wspd_front_range",
    "wspd_front_neighbour_count",
)
FRESHWATER = (
    "freshwater_deficit",
    "sss_local_mad",
    "sss_local_range",
    "sss_local_neighbour_count",
    "sss_process_available",
)
CIRCULATION = (
    "adt_local_mad",
    "adt_local_range",
    "adt_local_neighbour_count",
    "adt_process_available",
)
VARIANTS = {
    "SSS": {
        "S0_base": (),
        "S1_dense_front": FRONT,
        "S2_freshwater_proxy": FRESHWATER,
        "S3_optional_circulation": CIRCULATION,
    },
    "fCO2": {
        "F0_base": (),
        "F1_dense_front": FRONT,
        "F2_freshwater_proxy": FRESHWATER,
        "F3_optional_circulation": CIRCULATION,
        "F4_continuous_background_residual": (),
    },
}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def variant_features(target: str, variant: str) -> tuple[str, ...]:
    base = SSS_RESIDUAL_FEATURES if target == "SSS" else MODERN_RF_FEATURES
    return (*base, *VARIANTS[target][variant])


def load_augmented_frame(target: str) -> pd.DataFrame:
    frame = load_frame(target)
    features = pd.read_parquet(FEATURE_CACHE).drop(columns=["time_index"])
    merged = frame.merge(features, on="record_id", how="left", validate="one_to_one")
    if merged["sst_front_neighbour_count"].isna().any():
        raise RuntimeError("process feature cache is incomplete")
    return merged


def fit_variant(
    target: str,
    variant: str,
    configuration: list[float | int | None],
    train: pd.DataFrame,
    predict: pd.DataFrame,
    seed: int,
    checkpoint: Path | None = None,
) -> np.ndarray:
    family = "catboost" if target == "SSS" else "extra_trees"
    features = variant_features(target, variant)
    weights = balanced_sample_weight(train)
    paths: list[Path] = []
    if target == "SSS":
        direct = fit_model(
            family, configuration, train, features, train.truth.to_numpy(float), weights, seed
        )
        prediction, _ = predict_model(direct, predict)
        available_train = train.sss_bg.notna().to_numpy()
        residual_train = train.loc[available_train].reset_index(drop=True)
        residual = fit_model(
            family,
            configuration,
            residual_train,
            features,
            residual_train.truth.to_numpy(float) - residual_train.sss_bg.to_numpy(float),
            balanced_sample_weight(residual_train),
            seed + 1,
        )
        available_predict = predict.sss_bg.notna().to_numpy()
        residual_prediction, _ = predict_model(residual, predict.loc[available_predict])
        prediction[available_predict] = (
            predict.loc[available_predict, "sss_bg"].to_numpy(float) + residual_prediction
        )
        if checkpoint is not None:
            paths += save_model(direct, checkpoint.with_name(checkpoint.name + "_direct"))
            paths += save_model(residual, checkpoint.with_name(checkpoint.name + "_residual"))
    elif variant == "F4_continuous_background_residual":
        background = ContinuousBackground.fit(
            train, train.truth.to_numpy(float), weights, alpha=10.0
        )
        train_background = background.predict(train)
        fitted = fit_model(
            family,
            configuration,
            train,
            features,
            train.truth.to_numpy(float) - train_background,
            weights,
            seed,
        )
        residual_prediction, _ = predict_model(fitted, predict)
        prediction = background.predict(predict) + residual_prediction
        if checkpoint is not None:
            paths += save_model(fitted, checkpoint.with_name(checkpoint.name + "_residual"))
            background_path = checkpoint.with_name(checkpoint.name + "_background.joblib")
            background_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(background.estimator, background_path)
            paths.append(background_path)
    else:
        fitted = fit_model(
            family,
            configuration,
            train,
            features,
            train.truth.to_numpy(float),
            weights,
            seed,
        )
        prediction, _ = predict_model(fitted, predict)
        if checkpoint is not None:
            paths += save_model(fitted, checkpoint)
    if checkpoint is not None:
        write_json(
            checkpoint.with_suffix(".manifest.json"),
            {"files": [str(path) for path in paths]},
        )
    return prediction.astype("float32")


def selection_path(target: str, split: RepairedSplit, variant: str) -> Path:
    return SELECTION / f"{target}_{split.scheme}_fold{split.fold}_{variant}.json"


def select_configuration(
    target: str,
    split: RepairedSplit,
    variant: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> list[float | int | None]:
    path = selection_path(target, split, variant)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))["selected_config"]
    family = "catboost" if target == "SSS" else "extra_trees"
    train = deterministic_cap(
        train, 120000, f"63:{target}:{split.scheme}:{split.fold}:{variant}:train"
    )
    validation = deterministic_cap(
        validation, 60000, f"63:{target}:{split.scheme}:{split.fold}:{variant}:validation"
    )
    scores = []
    for index, configuration in enumerate(family_configs(family)):
        prediction = fit_variant(
            target,
            variant,
            configuration,
            train,
            validation,
            split_seed(target, split, variant, SCREEN_SEED + index),
        )
        scores.append(
            {
                "config_index": index,
                "config": configuration,
                "group_equal_rmse": group_equal_rmse(validation, prediction),
                "rmse": float(np.sqrt(np.mean((prediction - validation.truth) ** 2))),
            }
        )
    selected = min(scores, key=lambda row: (row["group_equal_rmse"], row["config_index"]))
    write_json(
        path,
        {
            "target": target,
            "scheme": split.scheme,
            "fold": split.fold,
            "variant": variant,
            "family": family,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "group_overlap": len(set(train.group_key) & set(validation.group_key)),
            "selected_config": selected["config"],
            "scores": scores,
        },
    )
    return selected["config"]


def run_cell(
    target: str, frame: pd.DataFrame, split: RepairedSplit, variant: str, seed: int
) -> None:
    destination = (
        PARTITIONS / f"{target}_{split.scheme}_fold{split.fold}_{variant}_seed{seed}.parquet"
    )
    if destination.is_file():
        return
    tune_train, tune_validation, model_fit, calibration, held = prepare_split(target, frame, split)
    configuration = select_configuration(target, split, variant, tune_train, tune_validation)
    combined = pd.concat([calibration, held], ignore_index=True)
    checkpoint = CHECKPOINTS / f"{target}_{split.scheme}_fold{split.fold}_{variant}_seed{seed}"
    prediction = fit_variant(
        target,
        variant,
        configuration,
        model_fit,
        combined,
        split_seed(target, split, variant, seed),
        checkpoint,
    )
    calibration_prediction = prediction[: len(calibration)]
    held_prediction = prediction[len(calibration) :]
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
            "truth",
            *FRONT,
            *FRESHWATER,
            *CIRCULATION,
        ]
    ].copy()
    result["prediction"] = held_prediction
    result["width50"] = width50
    result["width90"] = width90
    result["prediction_band"] = bands
    result["eligible"] = np.isfinite(held_prediction)
    result["calibrated"] = np.isfinite(held_prediction)
    result["target"] = target
    result["scheme"] = split.scheme
    result["fold"] = split.fold
    result["variant"] = variant
    result["seed"] = seed
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(destination, index=False, compression="zstd")


def metrics(frame: pd.DataFrame, **keys: object) -> dict[str, object]:
    error = frame.prediction.to_numpy(float) - frame.truth.to_numpy(float)
    denominator = np.sum((frame.truth - frame.truth.mean()) ** 2)
    result: dict[str, object] = {
        **keys,
        "n": len(frame),
        "groups": frame.group_key.nunique(),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "group_equal_rmse": group_equal_rmse(frame, frame.prediction.to_numpy(float)),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1 - np.sum(error**2) / denominator) if denominator else np.nan,
    }
    result.update(
        interval_scores(
            frame.truth.to_numpy(float),
            frame.prediction.to_numpy(float),
            frame.width50.to_numpy(float),
            frame.width90.to_numpy(float),
        )
    )
    return result


def pooled_metrics(seed: int) -> pd.DataFrame:
    rows = []
    for target, variants in VARIANTS.items():
        for variant in variants:
            for scheme in ("cruise", "geographic", "whole_region", "forward"):
                paths = sorted(
                    PARTITIONS.glob(f"{target}_{scheme}_fold*_{variant}_seed{seed}.parquet")
                )
                if not paths:
                    continue
                frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
                rows.append(
                    metrics(frame, target=target, variant=variant, scheme=scheme, seed=seed)
                )
    return pd.DataFrame(rows)


def bootstrap_upper(base: pd.DataFrame, candidate: pd.DataFrame, seed: int) -> float:
    joined = base[["record_id", "group_key", "truth", "prediction"]].merge(
        candidate[["record_id", "prediction"]],
        on="record_id",
        suffixes=("_base", "_candidate"),
        validate="one_to_one",
    )
    joined["mse_difference"] = (joined.prediction_candidate - joined.truth) ** 2 - (
        joined.prediction_base - joined.truth
    ) ** 2
    group_difference = joined.groupby("group_key").mse_difference.mean().to_numpy()
    rng = np.random.default_rng(seed)
    samples = rng.choice(group_difference, size=(2000, len(group_difference)), replace=True).mean(
        axis=1
    )
    return float(np.quantile(samples, 0.975))


def select_retained() -> dict[str, object]:
    table = pooled_metrics(SCREEN_SEED)
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "screen_metrics.csv", index=False)
    decisions: dict[str, object] = {}
    for target, variants in VARIANTS.items():
        base_name = "S0_base" if target == "SSS" else "F0_base"
        baseline = table.loc[(table.target == target) & (table.variant == base_name)].set_index(
            "scheme"
        )
        rows = []
        for variant in variants:
            if variant == base_name:
                continue
            candidate = table.loc[(table.target == target) & (table.variant == variant)].set_index(
                "scheme"
            )
            scheme_rows = []
            for scheme in baseline.index:
                base_paths = sorted(
                    PARTITIONS.glob(
                        f"{target}_{scheme}_fold*_{base_name}_seed{SCREEN_SEED}.parquet"
                    )
                )
                candidate_paths = sorted(
                    PARTITIONS.glob(f"{target}_{scheme}_fold*_{variant}_seed{SCREEN_SEED}.parquet")
                )
                base_frame = pd.concat(
                    [pd.read_parquet(path) for path in base_paths], ignore_index=True
                )
                candidate_frame = pd.concat(
                    [pd.read_parquet(path) for path in candidate_paths], ignore_index=True
                )
                relative = float(
                    (baseline.loc[scheme, "rmse"] - candidate.loc[scheme, "rmse"])
                    / baseline.loc[scheme, "rmse"]
                )
                upper = bootstrap_upper(base_frame, candidate_frame, SCREEN_SEED)
                coverage_drop = float(
                    baseline.loc[scheme, "coverage90"] - candidate.loc[scheme, "coverage90"]
                )
                scheme_rows.append(
                    {
                        "scheme": scheme,
                        "relative_rmse_improvement": relative,
                        "bootstrap_upper_mse_difference": upper,
                        "coverage90_drop": coverage_drop,
                    }
                )
            pass_count = sum(
                row["relative_rmse_improvement"] >= 0.01
                and row["bootstrap_upper_mse_difference"] < 0
                for row in scheme_rows
            )
            retained = pass_count >= 2 and all(
                row["relative_rmse_improvement"] >= -0.02 and row["coverage90_drop"] <= 0.02
                for row in scheme_rows
            )
            rows.append({"variant": variant, "retained": retained, "schemes": scheme_rows})
        decisions[target] = {
            "baseline": base_name,
            "factors": rows,
            "retained": [row["variant"] for row in rows if row["retained"]],
        }
    payload = {"screen_seed": SCREEN_SEED, "decisions": decisions}
    write_json(OUTPUT / "retained_factors.json", payload)
    return payload


def run(stage: str) -> None:
    decisions = None
    if stage == "retained":
        decisions = json.loads((OUTPUT / "retained_factors.json").read_text(encoding="utf-8"))
    for target in ("SSS", "fCO2"):
        frame = load_augmented_frame(target)
        splits = cached_splits(target, frame)
        variants = list(VARIANTS[target])
        seeds = (SCREEN_SEED,)
        if decisions is not None:
            variants = [
                decisions["decisions"][target]["baseline"],
                *decisions["decisions"][target]["retained"],
            ]
            seeds = EXTRA_SEEDS
        for split in splits:
            for variant in variants:
                for seed in seeds:
                    print(
                        f"{stage} {target} {split.scheme}/{split.fold} {variant} seed={seed}",
                        flush=True,
                    )
                    run_cell(target, frame, split, variant, seed)


def analyze() -> None:
    rows = []
    for seed in (SCREEN_SEED, *EXTRA_SEEDS):
        rows.append(pooled_metrics(seed))
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(TABLES / "all_seed_metrics.csv", index=False)
    summary = result.groupby(["target", "variant", "scheme"], as_index=False).agg(
        seeds=("seed", "nunique"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        coverage90_mean=("coverage90", "mean"),
    )
    summary.to_csv(TABLES / "stability_metrics.csv", index=False)
    write_json(OUTPUT / "analysis.json", {"rows": len(result), "complete": True})
    (OUTPUT / "pipeline.complete").write_text("complete\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["screen", "select", "retained", "analyze"])
    args = parser.parse_args()
    if args.phase in {"screen", "retained"}:
        run(args.phase)
    elif args.phase == "select":
        select_retained()
    else:
        analyze()


if __name__ == "__main__":
    main()
