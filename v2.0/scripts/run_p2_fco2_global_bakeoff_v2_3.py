"""Run the full Issue #41 global-coastal fCO2 model-family bake-off."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture

from recad.evaluate.fco2_bakeoff import (
    LMEResidualAdapter,
    SeasonalTrendBackground,
    fco2_band,
)
from recad.evaluate.predictor_ablation import balanced_sample_weight, frozen_splits
from recad.evaluate.sss_bakeoff import NumericTransform

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"C:\backup\phd\data\processed\recad_v2_3\p2_predictor_ablation_v2.3")
MATCHUP = SOURCE / "socat_predictor_matchup_v2.3.parquet"
SOURCE_PREDICTIONS = SOURCE / "fold_predictions"
OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_3\p2_fco2_global_bakeoff_v2.3")
PARTITIONS = OUTPUT / "partitions"
CHECKPOINTS = OUTPUT / "checkpoints"
SSS_FEATURES = OUTPUT / "crossfit_sss"
CONTRACT = ROOT / "configs/frozen/fco2_global_bakeoff_contract_v2.3.json"

BASE_FEATURES = (
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
)
STACK_FEATURES = (*BASE_FEATURES, "sss_crossfit")
SSS_RESIDUAL_FEATURES = (*BASE_FEATURES, "sss_bg", "sss_bg_available")
MODELS = (
    "seasonal_trend",
    "v11_random_forest",
    "catboost_direct",
    "catboost_residual",
    "catboost_residual_sss",
    "lme_adapters",
    "gmm_experts",
)


def load_frame() -> pd.DataFrame:
    """Load the exact Issue #54 fCO2 rows and fixed diagnostic strata."""

    frame = pd.read_parquet(MATCHUP)
    frame = frame.loc[frame.fco2.notna()].reset_index(drop=True)
    frame["truth"] = frame.fco2.astype("float32")
    frame["fco2_band"] = fco2_band(frame.truth).astype(str)
    frame["coast_band"] = pd.cut(
        frame.distance_to_significant_land_km,
        [-np.inf, 25, 100, 200, np.inf],
        labels=["0_25", "25_100", "100_200", "200_400"],
        right=False,
    ).astype(str)
    return frame


def catboost_model(*, iterations: int, depth: int, seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=seed,
        task_type="GPU",
        devices="0",
        random_strength=0.5,
        l2_leaf_reg=5.0,
        allow_writing_files=False,
        verbose=False,
    )


def fit_predict_sss(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    *,
    seed: int,
    budget: dict[str, object],
) -> np.ndarray:
    """Fit the frozen residual SSS branch with a direct missing-background fallback."""

    labelled = train.loc[train.salinity.notna()].reset_index(drop=True)
    if len(labelled) < 100:
        raise RuntimeError("insufficient SSS labels in an outer training partition")
    weights = balanced_sample_weight(labelled)
    direct = catboost_model(
        iterations=int(budget["iterations"]), depth=int(budget["depth"]), seed=seed
    )
    direct.fit(labelled.loc[:, BASE_FEATURES], labelled.salinity, sample_weight=weights)
    prediction = direct.predict(predict.loc[:, BASE_FEATURES]).astype("float32")
    available_train = labelled.sss_bg.notna()
    residual = catboost_model(
        iterations=int(budget["iterations"]), depth=int(budget["depth"]), seed=seed + 1
    )
    residual.fit(
        labelled.loc[available_train, SSS_RESIDUAL_FEATURES],
        labelled.loc[available_train, "salinity"] - labelled.loc[available_train, "sss_bg"],
        sample_weight=weights[available_train.to_numpy()],
    )
    available_predict = predict.sss_bg.notna().to_numpy()
    prediction[available_predict] = predict.loc[available_predict, "sss_bg"].to_numpy(
        "float32"
    ) + residual.predict(predict.loc[available_predict, SSS_RESIDUAL_FEATURES]).astype("float32")
    return prediction


def crossfit_sss(
    scheme: str,
    fold: int,
    fit: pd.DataFrame,
    held: pd.DataFrame,
    *,
    seed: int,
    budget: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Build inner-OOF SSS for fit rows and all-outer-fit SSS for held rows."""

    path = SSS_FEATURES / f"{scheme}_fold{fold}.parquet"
    if path.exists():
        cached = pd.read_parquet(path).set_index(["role", "record_id"])
        fit_values = pd.MultiIndex.from_arrays(
            [np.repeat("fit", len(fit)), fit.record_id], names=["role", "record_id"]
        )
        held_values = pd.MultiIndex.from_arrays(
            [np.repeat("held", len(held)), held.record_id], names=["role", "record_id"]
        )
        return (
            cached.sss_crossfit.reindex(fit_values).to_numpy("float32"),
            cached.sss_crossfit.reindex(held_values).to_numpy("float32"),
        )

    labelled = fit.salinity.notna().to_numpy()
    inner_fold = pd.util.hash_pandas_object(fit.group_key, index=False).to_numpy(np.uint64) % 2
    fit_prediction = np.full(len(fit), np.nan, dtype="float32")
    for inner in range(2):
        train = fit.loc[labelled & (inner_fold != inner)]
        selected = labelled & (inner_fold == inner)
        fit_prediction[selected] = fit_predict_sss(
            train, fit.loc[selected], seed=seed + inner * 10, budget=budget
        )
    outer = fit.loc[labelled]
    missing_fit = ~labelled
    if missing_fit.any():
        fit_prediction[missing_fit] = fit_predict_sss(
            outer, fit.loc[missing_fit], seed=seed + 30, budget=budget
        )
    held_prediction = fit_predict_sss(outer, held, seed=seed + 40, budget=budget)
    if not np.isfinite(fit_prediction).all() or not np.isfinite(held_prediction).all():
        raise RuntimeError("non-finite cross-fitted SSS")
    saved = pd.concat(
        [
            pd.DataFrame(
                {"role": "fit", "record_id": fit.record_id, "sss_crossfit": fit_prediction}
            ),
            pd.DataFrame(
                {"role": "held", "record_id": held.record_id, "sss_crossfit": held_prediction}
            ),
        ],
        ignore_index=True,
    )
    saved.to_parquet(path, index=False, compression="zstd")
    return fit_prediction, held_prediction


def source_direct_prediction(scheme: str, fold: int, held: pd.DataFrame) -> np.ndarray:
    path = SOURCE_PREDICTIONS / f"fCO2_{scheme}_fold{fold}_A2_environment.parquet"
    values = pd.read_parquet(path, columns=["record_id", "prediction"]).set_index("record_id")
    prediction = held.record_id.map(values.prediction)
    if prediction.isna().any():
        raise RuntimeError(f"Issue #54 direct prediction join failed: {path.name}")
    return prediction.to_numpy("float32", copy=True)


def fit_residual_catboost(
    fit: pd.DataFrame,
    held: pd.DataFrame,
    features: tuple[str, ...],
    background_fit: np.ndarray,
    background_held: np.ndarray,
    *,
    seed: int,
    budget: dict[str, object],
    checkpoint: Path,
) -> tuple[np.ndarray, np.ndarray]:
    model = catboost_model(
        iterations=int(budget["iterations"]), depth=int(budget["depth"]), seed=seed
    )
    model.fit(
        fit.loc[:, features],
        fit.truth.to_numpy(float) - background_fit,
        sample_weight=balanced_sample_weight(fit),
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(checkpoint)
    return (
        background_fit + model.predict(fit.loc[:, features]),
        background_held + model.predict(held.loc[:, features]),
    )


def fit_random_forest(
    fit: pd.DataFrame,
    held: pd.DataFrame,
    *,
    seed: int,
    budget: dict[str, object],
    checkpoint: Path,
) -> np.ndarray:
    transform = NumericTransform.fit(fit, STACK_FEATURES)
    model = RandomForestRegressor(
        n_estimators=int(budget["trees"]),
        max_depth=int(budget["max_depth"]),
        min_samples_leaf=int(budget["minimum_leaf"]),
        max_features=0.7,
        max_samples=float(budget["max_samples"]),
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(
        transform.apply(fit), fit.truth.to_numpy(float), sample_weight=balanced_sample_weight(fit)
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "transform": transform}, checkpoint, compress=3)
    return model.predict(transform.apply(held))


def fit_gmm_experts(
    fit: pd.DataFrame,
    held: pd.DataFrame,
    *,
    background_fit: np.ndarray,
    background_held: np.ndarray,
    global_held: np.ndarray,
    seed: int,
    budget: dict[str, object],
    checkpoint_prefix: Path,
) -> np.ndarray:
    transform = NumericTransform.fit(fit, STACK_FEATURES)
    fit_values = transform.apply(fit)
    held_values = transform.apply(held)
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(fit), size=min(100_000, len(fit)), replace=False)
    mixture = GaussianMixture(
        n_components=int(budget["components"]),
        covariance_type="diag",
        reg_covar=1e-4,
        max_iter=100,
        random_state=seed,
    ).fit(fit_values[sample])
    fit_responsibility = mixture.predict_proba(fit_values).astype("float32")
    held_responsibility = mixture.predict_proba(held_values).astype("float32")
    target = fit.truth.to_numpy(float) - background_fit
    base_weight = balanced_sample_weight(fit)
    expert_prediction = np.empty((len(held), int(budget["components"])), dtype="float32")
    checkpoint_prefix.parent.mkdir(parents=True, exist_ok=True)
    for component in range(int(budget["components"])):
        model = catboost_model(
            iterations=int(budget["iterations_per_expert"]),
            depth=int(budget["depth"]),
            seed=seed + component,
        )
        model.fit(
            fit.loc[:, STACK_FEATURES],
            target,
            sample_weight=base_weight * np.maximum(fit_responsibility[:, component], 1e-4),
        )
        model.save_model(Path(f"{checkpoint_prefix}_expert{component}.cbm"))
        expert_prediction[:, component] = model.predict(held.loc[:, STACK_FEATURES])
    mixture_prediction = background_held + np.sum(held_responsibility * expert_prediction, axis=1)
    shrinkage = float(budget["global_shrinkage"])
    return global_held + shrinkage * (mixture_prediction - global_held)


def write_prediction(
    scheme: str,
    fold: int,
    model: str,
    held: pd.DataFrame,
    prediction: np.ndarray,
    background: np.ndarray,
) -> None:
    result = held[
        [
            "record_id",
            "group_key",
            "yr",
            "mon",
            "latitude",
            "longitude",
            "lme_id",
            "truth",
            "distance_to_significant_land_km",
            "shelf_core_mask",
            "fco2_band",
            "coast_band",
            "sss_crossfit",
        ]
    ].copy()
    result["background"] = np.asarray(background, dtype="float32")
    result["prediction"] = np.asarray(prediction, dtype="float32")
    result["scheme"] = scheme
    result["fold"] = fold
    result["model"] = model
    result.to_parquet(
        PARTITIONS / f"fCO2_{scheme}_fold{fold}_{model}.parquet",
        index=False,
        compression="zstd",
    )


def run(*, only_scheme: str | None = None) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for path in (OUTPUT, PARTITIONS, CHECKPOINTS, SSS_FEATURES):
        path.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    log_path = OUTPUT / "training_log.csv"
    log = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []
    for scheme_index, scheme in enumerate(contract["outer_schemes"]):
        if only_scheme and scheme != only_scheme:
            continue
        for split in frozen_splits(frame, scheme):
            expected = [
                PARTITIONS / f"fCO2_{scheme}_fold{split.fold}_{model}.parquet" for model in MODELS
            ]
            if all(path.exists() for path in expected):
                print(f"reuse complete {scheme}/{split.fold}", flush=True)
                continue
            fit = frame.iloc[split.fit_index].reset_index(drop=True)
            held = frame.iloc[split.held_index].reset_index(drop=True)
            seed = 65023 + scheme_index * 100 + split.fold * 10
            start_fold = time.perf_counter()
            print(f"{scheme}/{split.fold}: fit={len(fit):,} held={len(held):,}", flush=True)
            fit_sss, held_sss = crossfit_sss(
                scheme,
                split.fold,
                fit,
                held,
                seed=seed,
                budget=contract["sss_budget"],
            )
            fit["sss_crossfit"] = fit_sss
            held["sss_crossfit"] = held_sss
            background_model = SeasonalTrendBackground().fit(fit)
            background_fit = background_model.predict(fit)
            background_held = background_model.predict(held)
            direct = source_direct_prediction(scheme, split.fold, held)
            _, residual_held = fit_residual_catboost(
                fit,
                held,
                BASE_FEATURES,
                background_fit,
                background_held,
                seed=seed + 1,
                budget=contract["catboost_budget"],
                checkpoint=CHECKPOINTS / f"{scheme}_fold{split.fold}_residual.cbm",
            )
            stack_fit, stack_held = fit_residual_catboost(
                fit,
                held,
                STACK_FEATURES,
                background_fit,
                background_held,
                seed=seed + 2,
                budget=contract["catboost_budget"],
                checkpoint=CHECKPOINTS / f"{scheme}_fold{split.fold}_residual_sss.cbm",
            )
            adapter = LMEResidualAdapter().fit(fit, stack_fit)
            adapted = stack_held + adapter.predict_correction(held)
            candidates: list[tuple[str, np.ndarray]] = [
                ("seasonal_trend", background_held),
                ("catboost_direct", direct),
                ("catboost_residual", residual_held),
                ("catboost_residual_sss", stack_held),
                ("lme_adapters", adapted),
            ]
            rf_path = PARTITIONS / f"fCO2_{scheme}_fold{split.fold}_v11_random_forest.parquet"
            if not rf_path.exists():
                start = time.perf_counter()
                rf_prediction = fit_random_forest(
                    fit,
                    held,
                    seed=seed + 3,
                    budget=contract["random_forest_budget"],
                    checkpoint=CHECKPOINTS / f"{scheme}_fold{split.fold}_v11_rf.joblib",
                )
                candidates.append(("v11_random_forest", rf_prediction))
                log.append(
                    {
                        "scheme": scheme,
                        "fold": split.fold,
                        "model": "v11_random_forest",
                        "fit_rows": len(fit),
                        "held_rows": len(held),
                        "seconds": time.perf_counter() - start,
                    }
                )
            gmm_path = PARTITIONS / f"fCO2_{scheme}_fold{split.fold}_gmm_experts.parquet"
            if not gmm_path.exists():
                start = time.perf_counter()
                gmm_prediction = fit_gmm_experts(
                    fit,
                    held,
                    background_fit=background_fit,
                    background_held=background_held,
                    global_held=stack_held,
                    seed=seed + 4,
                    budget=contract["gmm_expert_budget"],
                    checkpoint_prefix=CHECKPOINTS / f"{scheme}_fold{split.fold}_gmm",
                )
                candidates.append(("gmm_experts", gmm_prediction))
                log.append(
                    {
                        "scheme": scheme,
                        "fold": split.fold,
                        "model": "gmm_experts",
                        "fit_rows": len(fit),
                        "held_rows": len(held),
                        "seconds": time.perf_counter() - start,
                    }
                )
            for model, prediction in candidates:
                destination = PARTITIONS / f"fCO2_{scheme}_fold{split.fold}_{model}.parquet"
                if destination.exists():
                    continue
                start = time.perf_counter()
                write_prediction(scheme, split.fold, model, held, prediction, background_held)
                log.append(
                    {
                        "scheme": scheme,
                        "fold": split.fold,
                        "model": model,
                        "fit_rows": len(fit),
                        "held_rows": len(held),
                        "seconds": time.perf_counter() - start,
                    }
                )
            pd.DataFrame(log).drop_duplicates(["scheme", "fold", "model"], keep="last").to_csv(
                log_path, index=False
            )
            print(f"fold elapsed {(time.perf_counter() - start_fold) / 60:.1f} min", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=("cruise", "spatial", "whole_lme", "forward"))
    args = parser.parse_args()
    run(only_scheme=args.scheme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
