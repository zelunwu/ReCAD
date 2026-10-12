"""Run the full Issue #40 global-coastal SSS model-family bake-off."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from sklearn.mixture import GaussianMixture

from recad.evaluate.predictor_ablation import balanced_sample_weight, frozen_splits
from recad.evaluate.sss_bakeoff import (
    NumericTransform,
    RobustRegionalCorrection,
    predict_neural_residual,
    train_neural_residual,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"C:\backup\phd\data\processed\recad_v2_3\p2_predictor_ablation_v2.3")
MATCHUP = SOURCE / "socat_predictor_matchup_v2.3.parquet"
SOURCE_PREDICTIONS = SOURCE / "fold_predictions"
OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_3\p2_sss_global_bakeoff_v2.3")
PARTITIONS = OUTPUT / "partitions"
CHECKPOINTS = OUTPUT / "checkpoints"
CONTRACT = ROOT / "configs/frozen/sss_global_bakeoff_contract_v2.3.json"

FEATURES = (
    "year_centered",
    "month_sin",
    "month_cos",
    "latitude_sin",
    "latitude_cos",
    "longitude_sin",
    "longitude_cos",
    "distance_to_significant_land_km",
    "bathymetry_m",
    "shelf_core_mask",
    "slope_deep_mask",
    "sst",
    "wspd",
    "xco2air",
    "sss_bg",
    "sss_bg_available",
)
MODELS = (
    "raw_glorys",
    "robust_regional",
    "catboost_global",
    "gmm_experts",
    "top2_experts",
    "st96",
)


def load_frame() -> pd.DataFrame:
    """Load the exact Issue #54 SSS rows and add diagnostic strata."""

    frame = pd.read_parquet(MATCHUP)
    frame = frame.loc[frame.salinity.notna()].reset_index(drop=True)
    frame["truth"] = frame.salinity.astype("float32")
    frame["salinity_band"] = pd.cut(
        frame.truth,
        [-np.inf, 20, 30, 33, 36, np.inf],
        labels=["lt20", "20_30", "30_33", "33_36", "ge36"],
        right=False,
    ).astype(str)
    frame["coast_band"] = pd.cut(
        frame.distance_to_significant_land_km,
        [-np.inf, 25, 100, 200, np.inf],
        labels=["0_25", "25_100", "100_200", "200_400"],
        right=False,
    ).astype(str)
    return frame


def source_prediction(
    target: str, scheme: str, fold: int, variant: str, held: pd.DataFrame
) -> np.ndarray:
    path = SOURCE_PREDICTIONS / f"{target}_{scheme}_fold{fold}_{variant}.parquet"
    values = pd.read_parquet(path, columns=["record_id", "prediction"]).set_index("record_id")
    prediction = held.record_id.map(values.prediction)
    if prediction.isna().any():
        raise RuntimeError(f"Issue 54 prediction join failed: {path.name}")
    return prediction.to_numpy(float)


def write_prediction(
    scheme: str,
    fold: int,
    model: str,
    held: pd.DataFrame,
    prediction: np.ndarray,
) -> None:
    path = PARTITIONS / f"SSS_{scheme}_fold{fold}_{model}.parquet"
    result = held[
        [
            "record_id",
            "group_key",
            "yr",
            "mon",
            "lme_id",
            "truth",
            "sss_bg",
            "sss_bg_available",
            "distance_to_significant_land_km",
            "shelf_core_mask",
            "salinity_band",
            "coast_band",
        ]
    ].copy()
    result["prediction"] = np.asarray(prediction, dtype="float32")
    result["scheme"] = scheme
    result["fold"] = fold
    result["model"] = model
    result.to_parquet(path, index=False, compression="zstd")


def train_gmm_experts(
    fit: pd.DataFrame,
    held: pd.DataFrame,
    *,
    seed: int,
    global_prediction: np.ndarray,
    components: int,
    iterations: int,
    shrinkage: float,
) -> np.ndarray:
    """Fit probabilistic environmental residual experts and shrink globally."""

    available_fit = fit.loc[fit.sss_bg.notna()].reset_index(drop=True)
    available_held = held.sss_bg.notna().to_numpy()
    transform = NumericTransform.fit(available_fit, FEATURES)
    fit_values = transform.apply(available_fit)
    held_values = transform.apply(held.loc[available_held])
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(fit_values), size=min(100_000, len(fit_values)), replace=False)
    mixture = GaussianMixture(
        n_components=components,
        covariance_type="diag",
        reg_covar=1e-4,
        max_iter=100,
        random_state=seed,
    ).fit(fit_values[sample])
    fit_responsibility = mixture.predict_proba(fit_values).astype("float32")
    held_responsibility = mixture.predict_proba(held_values).astype("float32")
    residual = (available_fit.truth - available_fit.sss_bg).to_numpy(float)
    base_weight = balanced_sample_weight(available_fit)
    expert_prediction = np.empty((len(held_values), components), dtype="float32")
    for component in range(components):
        model = CatBoostRegressor(
            iterations=iterations,
            depth=7,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=seed + component,
            task_type="GPU",
            devices="0",
            random_strength=0.5,
            l2_leaf_reg=5,
            allow_writing_files=False,
            verbose=False,
        )
        weights = base_weight * np.maximum(fit_responsibility[:, component], 1e-4)
        model.fit(fit_values, residual, sample_weight=weights)
        expert_prediction[:, component] = model.predict(held_values)
    mixture_prediction = held.loc[available_held, "sss_bg"].to_numpy(float) + np.sum(
        held_responsibility * expert_prediction, axis=1
    )
    prediction = np.asarray(global_prediction, dtype=float).copy()
    prediction[available_held] = prediction[available_held] + shrinkage * (
        mixture_prediction - prediction[available_held]
    )
    return prediction


def run(*, only_scheme: str | None = None) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PARTITIONS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    log_path = OUTPUT / "training_log.csv"
    log = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []
    for scheme_index, scheme in enumerate(contract["outer_schemes"]):
        if only_scheme and scheme != only_scheme:
            continue
        for split in frozen_splits(frame, scheme):
            fit = frame.iloc[split.fit_index].reset_index(drop=True)
            held = frame.iloc[split.held_index].reset_index(drop=True)
            a2 = source_prediction("SSS", scheme, split.fold, "A2_environment", held)
            catboost = source_prediction("SSS", scheme, split.fold, "A3_background_sss", held)
            start_fold = time.perf_counter()
            print(f"{scheme}/{split.fold}: fit={len(fit):,} held={len(held):,}", flush=True)
            for model_index, model_name in enumerate(MODELS):
                destination = PARTITIONS / f"SSS_{scheme}_fold{split.fold}_{model_name}.parquet"
                if destination.exists():
                    print(f"reuse {destination.name}", flush=True)
                    continue
                start = time.perf_counter()
                if model_name == "raw_glorys":
                    prediction = np.where(held.sss_bg.notna(), held.sss_bg, a2)
                elif model_name == "catboost_global":
                    prediction = catboost
                elif model_name == "robust_regional":
                    regional = RobustRegionalCorrection().fit(
                        fit.loc[fit.sss_bg.notna()].reset_index(drop=True)
                    )
                    prediction = a2.copy()
                    available = held.sss_bg.notna().to_numpy()
                    prediction[available] = regional.predict(held.loc[available])
                elif model_name == "gmm_experts":
                    gmm = contract["gmm_expert_budget"]
                    prediction = train_gmm_experts(
                        fit,
                        held,
                        seed=64023 + scheme_index * 100 + split.fold,
                        global_prediction=catboost,
                        components=int(gmm["components"]),
                        iterations=int(gmm["iterations_per_expert"]),
                        shrinkage=float(gmm["global_shrinkage"]),
                    )
                else:
                    available_fit = fit.loc[fit.sss_bg.notna()].reset_index(drop=True)
                    available_held = held.sss_bg.notna().to_numpy()
                    family = "top2_experts" if model_name == "top2_experts" else "st96"
                    budget = contract["neural_budget"]
                    seed = int(budget["seed"]) + scheme_index * 100 + split.fold * 10 + model_index
                    state_path = CHECKPOINTS / f"{scheme}_fold{split.fold}_{model_name}.pt"
                    if state_path.exists():
                        state = torch.load(state_path, map_location="cpu", weights_only=False)
                    else:
                        state = train_neural_residual(
                            family,
                            available_fit,
                            FEATURES,
                            seed=seed,
                            steps=int(budget["steps"]),
                            batch_size=int(budget["batch_size"]),
                        )
                        torch.save(state, state_path)
                    prediction = a2.copy()
                    prediction[available_held] = predict_neural_residual(
                        state, held.loc[available_held]
                    )
                write_prediction(scheme, split.fold, model_name, held, prediction)
                elapsed = time.perf_counter() - start
                log = [
                    row
                    for row in log
                    if not (
                        row["scheme"] == scheme
                        and int(row["fold"]) == split.fold
                        and row["model"] == model_name
                    )
                ]
                log.append(
                    {
                        "scheme": scheme,
                        "fold": split.fold,
                        "model": model_name,
                        "fit_rows": len(fit),
                        "held_rows": len(held),
                        "seconds": elapsed,
                    }
                )
                pd.DataFrame(log).to_csv(log_path, index=False)
                print(f"  {model_name}: {elapsed:.1f}s", flush=True)
            print(f"fold elapsed {(time.perf_counter() - start_fold) / 60:.1f} min", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=("cruise", "spatial", "whole_lme", "forward"))
    args = parser.parse_args()
    run(only_scheme=args.scheme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
