"""Run the full P1.2 coastal-fCO2 viability experiment without sealed labels."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from recad.evaluate.p1_framework import (
    FrozenManifest,
    P1DataGateway,
    Purpose,
    evaluate_predictions,
    sha256,
    standardize_predictions,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "sss",
    "sst",
    "adt",
    "wspd",
    "pco2air",
    "xco2air",
    "year",
    "latitude",
    "longitude",
    "month",
    "lme_id",
    "basin_id",
    "regime_id",
]
NUMERIC = [
    "sss",
    "sst",
    "adt",
    "wspd",
    "pco2air",
    "xco2air",
    "year",
    "latitude",
    "lon_sin",
    "lon_cos",
    "month_sin",
    "month_cos",
]
CATEGORICAL = ["lme_id", "basin_id", "regime_id"]
MODELS = [
    "seasonal_climatology",
    "ridge_residual",
    "catboost_residual",
    "point_mlp_residual",
    "soft_experts_residual",
]


def stable_id(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.group_key.astype(str)
        + ":"
        + frame.year.astype(str)
        + ":"
        + frame.month.astype(str)
        + ":"
        + frame.latitude.round(4).astype(str)
        + ":"
        + frame.longitude.round(4).astype(str)
    )


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.loc[frame.truth.notna()].copy().reset_index(drop=True)
    radians = np.deg2rad(frame.longitude.to_numpy(float))
    phase = 2 * np.pi * (frame.month.to_numpy(float) - 1) / 12
    frame["lon_sin"] = np.sin(radians)
    frame["lon_cos"] = np.cos(radians)
    frame["month_sin"] = np.sin(phase)
    frame["month_cos"] = np.cos(phase)
    frame["record_id"] = stable_id(frame)
    for column in CATEGORICAL:
        frame[column] = frame[column].fillna(-999).astype(int).astype(str)
    return frame


def balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    """Exact row probability of region -> cruise -> month -> record sampling."""
    keys = ["lme_id", "group_key", "month"]
    work = frame[keys].copy()
    n_regions = work.lme_id.nunique()
    cruises = work.groupby("lme_id").group_key.transform("nunique")
    months = work.groupby(["lme_id", "group_key"])["month"].transform("nunique")
    records = work.groupby(keys)["month"].transform("size")
    weights = 1.0 / (n_regions * cruises * months * records)
    weights = weights.to_numpy(float)
    return weights / weights.sum()


def metric_summary(
    frame: pd.DataFrame, prediction: np.ndarray
) -> tuple[pd.DataFrame, dict[str, float]]:
    scored = frame[["truth", "group_key", "lme_id", "regime_id", "nearest_carbon_km"]].copy()
    scored["prediction"] = prediction
    details = evaluate_predictions(scored)
    lookup = {row.aggregation: row for row in details.itertuples() if row.group == "all"}
    pooled = lookup["pooled"]
    cruise = lookup["cruise_equal"]
    lme = lookup["lme_macro"]
    regime = lookup["regime_macro"]
    worst = details.loc[details.aggregation.eq("worst_lme")].iloc[0]
    compact = {
        "n": int(pooled.n),
        "pooled_rmse": float(pooled.rmse),
        "pooled_mae": float(pooled.mae),
        "pooled_bias": float(pooled.bias),
        "pooled_r2": float(pooled.r2),
        "cruise_equal_rmse": float(cruise.rmse),
        "lme_macro_rmse": float(lme.rmse),
        "regime_macro_rmse": float(regime.rmse),
        "worst_lme_rmse": float(worst.rmse),
        "worst_lme": str(worst.group),
    }
    return details, compact


class SeasonalTrendClimatology:
    """Training-only LME-month climatology with a shared linear trend."""

    def fit(self, train: pd.DataFrame) -> SeasonalTrendClimatology:
        self.center_year = float(train.year.median())
        time = train.year.to_numpy(float) - self.center_year
        truth = train.truth.to_numpy(float)
        self.slope = float(np.cov(time, truth, ddof=0)[0, 1] / max(np.var(time), 1e-12))
        work = train.assign(detrended=truth - self.slope * time)
        self.global_mean = float(work.detrended.mean())
        self.month = work.groupby("month").detrended.mean().to_dict()
        grouped = work.groupby(["lme_id", "month"]).detrended.agg(["mean", "count"])
        # Empirical-Bayes shrinkage prevents tiny LME-month cells from exploding.
        prior = np.array([self.month.get(idx[1], self.global_mean) for idx in grouped.index])
        weight = grouped["count"].to_numpy() / (grouped["count"].to_numpy() + 50.0)
        grouped["estimate"] = weight * grouped["mean"].to_numpy() + (1 - weight) * prior
        self.group = grouped.estimate.to_dict()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        seasonal = [
            self.group.get((r.lme_id, r.month), self.month.get(r.month, self.global_mean))
            for r in frame.itertuples()
        ]
        trend = self.slope * (frame.year.to_numpy(float) - self.center_year)
        return np.asarray(seasonal) + trend


class TabularTransform:
    def fit(self, train: pd.DataFrame) -> TabularTransform:
        x = train[NUMERIC].to_numpy(float)
        self.median = np.nanmedian(x, axis=0)
        x = np.where(np.isfinite(x), x, self.median)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std < 1e-7] = 1
        categories = []
        for column in CATEGORICAL:
            values = sorted(train[column].astype(str).unique().tolist())
            categories.append([*values, "__unknown__"])
        self.categories = categories
        return self

    def apply(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[NUMERIC].to_numpy(float)
        x = np.where(np.isfinite(x), x, self.median)
        parts = [(x - self.mean) / self.std]
        for column, categories in zip(CATEGORICAL, self.categories, strict=True):
            mapping = {value: i for i, value in enumerate(categories)}
            unknown = len(categories) - 1
            codes = np.array(
                [mapping.get(value, unknown) for value in frame[column].astype(str)], dtype=int
            )
            onehot = np.zeros((len(frame), len(categories)), dtype=np.float32)
            onehot[np.arange(len(frame)), codes] = 1
            parts.append(onehot)
        return np.concatenate(parts, axis=1).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class SoftExperts(nn.Module):
    def __init__(self, n_features: int, experts: int = 8, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.trunk = nn.Sequential(
            nn.Linear(n_features, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 128), nn.GELU()
        )
        self.experts = nn.Linear(128, experts)
        self.gate = nn.Sequential(nn.Linear(n_features, 64), nn.GELU(), nn.Linear(64, experts))

    def forward(self, x: torch.Tensor, return_experts: bool = False):
        expert_values = self.experts(self.trunk(x))
        logits = self.gate(x)
        top = torch.topk(logits, self.top_k, dim=-1)
        masked = torch.full_like(logits, -torch.inf).scatter(1, top.indices, top.values)
        weights = torch.softmax(masked, dim=-1)
        prediction = (weights * expert_values).sum(dim=-1)
        return (prediction, expert_values, weights) if return_experts else prediction


@dataclass
class NeuralResult:
    prediction: np.ndarray
    best_step: int
    best_score: float
    state: dict[str, torch.Tensor]


def lme_macro_rmse_tensor(y: torch.Tensor, p: torch.Tensor, lme: np.ndarray) -> float:
    values = []
    for group in np.unique(lme):
        mask = torch.as_tensor(lme == group, device=y.device)
        values.append(torch.sqrt(torch.mean((p[mask] - y[mask]) ** 2)))
    return float(torch.stack(values).mean().item())


def train_neural(
    family: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    checkpoint_every: int,
) -> NeuralResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    transform = TabularTransform().fit(train)
    x_train_np, x_val_np = transform.apply(train), transform.apply(validation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.as_tensor(x_train_np, device=device)
    y_train = torch.as_tensor(train.residual.to_numpy(np.float32), device=device)
    x_val = torch.as_tensor(x_val_np, device=device)
    y_val = torch.as_tensor(validation.truth.to_numpy(np.float32), device=device)
    bg_val = torch.as_tensor(validation.baseline.to_numpy(np.float32, copy=True), device=device)
    model = (
        MLP(x_train.shape[1]) if family == "point_mlp_residual" else SoftExperts(x_train.shape[1])
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    probabilities = torch.as_tensor(balanced_weights(train).astype(np.float32), device=device)
    best_score, best_step, best_state = math.inf, 0, None
    model.train()
    for step in range(1, steps + 1):
        index = torch.multinomial(probabilities, batch_size, replacement=True)
        if family == "soft_experts_residual":
            pred, expert_values, weights = model(x_train[index], return_experts=True)
            loss = nn.functional.huber_loss(pred, y_train[index], delta=1.0)
            loss = loss + 0.001 * expert_values.var(dim=1).mean() + 1e-5 * (weights**2).mean()
        else:
            pred = model(x_train[index])
            loss = nn.functional.huber_loss(pred, y_train[index], delta=1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % checkpoint_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                val_prediction = bg_val + model(x_val)
                score = lme_macro_rmse_tensor(y_val, val_prediction, validation.lme_id.to_numpy())
            if score < best_score:
                best_score, best_step = score, step
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
            model.train()
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = (bg_val + model(x_val)).cpu().numpy()
    state = {
        "model": best_state,
        "transform": transform.__dict__,
        "family": family,
        "features": NUMERIC + CATEGORICAL,
        "seed": seed,
        "best_step": best_step,
    }
    return NeuralResult(prediction, best_step, best_score, state)


def ridge_prediction(
    train: pd.DataFrame, validation: pd.DataFrame, alpha: float, weights: np.ndarray
) -> tuple[np.ndarray, object]:
    numeric = NUMERIC
    categorical = CATEGORICAL
    transformer = ColumnTransformer(
        [
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    model = make_pipeline(transformer, Ridge(alpha=alpha, solver="lsqr"))
    model.fit(
        train[numeric + categorical], train.residual, ridge__sample_weight=weights * len(weights)
    )
    return validation.baseline.to_numpy() + model.predict(validation[numeric + categorical]), model


def catboost_prediction(
    train: pd.DataFrame, validation: pd.DataFrame, seed: int, weights: np.ndarray, iterations: int
) -> tuple[np.ndarray, CatBoostRegressor]:
    columns = NUMERIC + CATEGORICAL
    x_train, x_val = train[columns].copy(), validation[columns].copy()
    for column in NUMERIC:
        median = float(x_train[column].median())
        x_train[column] = x_train[column].fillna(median)
        x_val[column] = x_val[column].fillna(median)
    cat_indices = [columns.index(column) for column in CATEGORICAL]
    kwargs = {
        "iterations": iterations,
        "depth": 8,
        "learning_rate": 0.05,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": seed,
        "verbose": False,
        "random_strength": 0.5,
        "l2_leaf_reg": 5.0,
        "allow_writing_files": False,
    }
    try:
        model = CatBoostRegressor(**kwargs, task_type="GPU", devices="0")
        model.fit(
            x_train,
            train.residual,
            cat_features=cat_indices,
            sample_weight=weights * len(weights),
            eval_set=(x_val, validation.residual),
            early_stopping_rounds=150,
            verbose=False,
        )
    except Exception as exc:
        print(f"CatBoost GPU unavailable ({exc}); falling back to CPU", flush=True)
        model = CatBoostRegressor(**kwargs, task_type="CPU", thread_count=-1)
        model.fit(
            x_train,
            train.residual,
            cat_features=cat_indices,
            sample_weight=weights * len(weights),
            eval_set=(x_val, validation.residual),
            early_stopping_rounds=150,
            verbose=False,
        )
    return validation.baseline.to_numpy() + model.predict(x_val), model


def run_model(
    name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
    args,
    model_dir: Path,
    run_id: str,
):
    baseline_model = SeasonalTrendClimatology().fit(train)
    train = train.copy()
    validation = validation.copy()
    train["baseline"] = baseline_model.predict(train)
    validation["baseline"] = baseline_model.predict(validation)
    train["residual"] = train.truth - train.baseline
    validation["residual"] = validation.truth - validation.baseline
    weights = balanced_weights(train)
    metadata: dict[str, object] = {}
    if name == "seasonal_climatology":
        prediction = validation.baseline.to_numpy(float)
        metadata["trend_uatm_per_year"] = baseline_model.slope
    elif name == "ridge_residual":
        best = None
        for alpha in (0.1, 1.0, 10.0, 100.0):
            candidate, model = ridge_prediction(train, validation, alpha, weights)
            _, compact = metric_summary(validation, candidate)
            if best is None or compact["lme_macro_rmse"] < best[0]:
                best = compact["lme_macro_rmse"], alpha, candidate, model
        _, alpha, prediction, model = best
        metadata["alpha"] = alpha
        import joblib

        joblib.dump(model, model_dir / f"{run_id}_{name}_seed{seed}.joblib")
    elif name == "catboost_residual":
        prediction, model = catboost_prediction(
            train, validation, seed, weights, args.catboost_iterations
        )
        metadata["best_iteration"] = int(model.get_best_iteration())
        model.save_model(model_dir / f"{run_id}_{name}_seed{seed}.cbm")
    else:
        result = train_neural(
            name,
            train,
            validation,
            seed=seed,
            steps=args.steps,
            batch_size=args.batch_size,
            checkpoint_every=args.checkpoint_every,
        )
        prediction = result.prediction
        metadata.update(best_step=result.best_step, selection_lme_macro_rmse=result.best_score)
        torch.save(result.state, model_dir / f"{run_id}_{name}_seed{seed}.pt")
    return np.asarray(prediction), metadata


def add_stratified_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> list[dict[str, object]]:
    result = []
    error = prediction - frame.truth.to_numpy()
    background_error = frame.baseline.to_numpy() - frame.truth.to_numpy()
    bands = pd.cut(frame.truth, [-np.inf, 250, 350, 450, 550, np.inf], right=False)
    for kind, groups in (("fco2_band", bands), ("lme", frame.lme_id), ("regime", frame.regime_id)):
        for group in sorted(pd.Series(groups).dropna().unique(), key=str):
            mask = np.asarray(groups == group)
            model_rmse = float(np.sqrt(np.mean(error[mask] ** 2)))
            background_rmse = float(np.sqrt(np.mean(background_error[mask] ** 2)))
            result.append(
                {
                    "stratum": kind,
                    "group": str(group),
                    "n": int(mask.sum()),
                    "model_rmse": model_rmse,
                    "background_rmse": background_rmse,
                    "skill_vs_background": 1 - model_rmse**2 / background_rmse**2,
                }
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/experiments/p1_fco2_viability_v2.2"
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--catboost-iterations", type=int, default=1500)
    parser.add_argument("--phase", choices=("primary", "cv", "all"), default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "checkpoints"
    model_dir.mkdir(exist_ok=True)
    manifest = FrozenManifest.load(ROOT / "configs/frozen/data_manifest_v2.2.json")
    validation = manifest.validate(hash_mode="full")
    gateway = P1DataGateway(manifest)
    columns = ["sst", "sss", "adt", "wspd", "pco2air", "xco2air"]
    train = prepare(gateway.load_labels("fco2", Purpose.TRAIN, columns=columns))
    development = prepare(gateway.load_labels("fco2", Purpose.SELECTION, columns=columns))
    seeds = [100, 101, 102]
    metrics_rows, stratified_rows, prediction_rows, cv_rows, forward_rows = [], [], [], [], []
    start = time.time()

    if args.phase in {"primary", "all"}:
        for name in MODELS:
            run_seeds = (
                seeds
                if name in {"catboost_residual", "point_mlp_residual", "soft_experts_residual"}
                else [100]
            )
            for seed in run_seeds:
                print(f"PRIMARY {name} seed={seed}", flush=True)
                prediction, metadata = run_model(
                    name, train, development, seed, args, model_dir, "primary"
                )
                details, compact = metric_summary(development, prediction)
                baseline = SeasonalTrendClimatology().fit(train).predict(development)
                development_with_baseline = development.assign(baseline=baseline)
                bg_rmse = float(np.sqrt(np.mean((baseline - development.truth) ** 2)))
                compact.update(
                    model=name,
                    seed=seed,
                    split_scheme="primary",
                    stage="development",
                    skill_vs_background=1 - compact["pooled_rmse"] ** 2 / bg_rmse**2,
                    **metadata,
                )
                metrics_rows.append(compact)
                for row in add_stratified_metrics(development_with_baseline, prediction):
                    stratified_rows.append(
                        {"model": name, "seed": seed, "split_scheme": "primary", **row}
                    )
                prediction_rows.append(
                    pd.DataFrame(
                        {
                            "record_id": development.record_id,
                            "model": name,
                            "seed": seed,
                            "truth": development.truth,
                            "background": baseline,
                            "prediction": prediction,
                        }
                    )
                )
                details.to_parquet(
                    args.output / f"metrics_detail_{name}_seed{seed}.parquet", index=False
                )
                pd.DataFrame(metrics_rows).to_csv(args.output / "metrics_by_seed.csv", index=False)

    if args.phase in {"cv", "all"}:
        # Full five-fold grouped CV. Deterministic models need one run; stochastic
        # candidates use the preregistered seed 100 for fold comparison.
        all_train = train
        for fold in range(5):
            fit = all_train.loc[all_train.cv_fold.ne(fold)].reset_index(drop=True)
            held = all_train.loc[all_train.cv_fold.eq(fold)].reset_index(drop=True)
            assert set(fit.group_key).isdisjoint(set(held.group_key))
            for name in MODELS:
                print(f"CV fold={fold} {name}", flush=True)
                prediction, metadata = run_model(name, fit, held, 100, args, model_dir, f"cv{fold}")
                _, compact = metric_summary(held, prediction)
                baseline = SeasonalTrendClimatology().fit(fit).predict(held)
                bg_rmse = float(np.sqrt(np.mean((baseline - held.truth) ** 2)))
                compact.update(
                    model=name,
                    seed=100,
                    fold=fold,
                    split_scheme="cruise_cv",
                    skill_vs_background=1 - compact["pooled_rmse"] ** 2 / bg_rmse**2,
                    **metadata,
                )
                cv_rows.append(compact)
                pd.DataFrame(cv_rows).to_csv(args.output / "cv_metrics.csv", index=False)

    metrics = pd.DataFrame(metrics_rows)
    cv = pd.DataFrame(cv_rows)
    predictions = (
        pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    )
    if len(predictions):
        predictions.to_parquet(args.output / "candidate_predictions.parquet", index=False)
        pd.DataFrame(stratified_rows).to_csv(args.output / "stratified_metrics.csv", index=False)
        summary = (
            metrics.groupby("model")
            .agg(
                seeds=("seed", "nunique"),
                pooled_rmse_mean=("pooled_rmse", "mean"),
                pooled_rmse_std=("pooled_rmse", "std"),
                cruise_equal_rmse_mean=("cruise_equal_rmse", "mean"),
                lme_macro_rmse_mean=("lme_macro_rmse", "mean"),
                worst_lme_rmse_mean=("worst_lme_rmse", "mean"),
                skill_vs_background_mean=("skill_vs_background", "mean"),
            )
            .reset_index()
        )
        if len(cv):
            cv_summary = (
                cv.groupby("model")
                .agg(
                    cv_lme_macro_mean=("lme_macro_rmse", "mean"),
                    cv_lme_macro_std=("lme_macro_rmse", "std"),
                    cv_skill_mean=("skill_vs_background", "mean"),
                )
                .reset_index()
            )
            summary = summary.merge(cv_summary, on="model", how="left")
        summary.to_csv(args.output / "summary.csv", index=False)
        eligible = summary.loc[summary.model.ne("seasonal_climatology")].sort_values(
            "lme_macro_rmse_mean"
        )
        selected = str(eligible.iloc[0].model)
        forward_train = prepare(
            gateway.load_labels("fco2", Purpose.TRAIN, columns=columns, split_scheme="forward")
        )
        forward_dev = prepare(
            gateway.load_labels("fco2", Purpose.SELECTION, columns=columns, split_scheme="forward")
        )
        for name in ["seasonal_climatology", selected]:
            run_seeds = (
                seeds
                if name in {"catboost_residual", "point_mlp_residual", "soft_experts_residual"}
                else [100]
            )
            for seed in run_seeds:
                print(f"FORWARD {name} seed={seed}", flush=True)
                prediction, metadata = run_model(
                    name, forward_train, forward_dev, seed, args, model_dir, "forward"
                )
                _, compact = metric_summary(forward_dev, prediction)
                baseline = SeasonalTrendClimatology().fit(forward_train).predict(forward_dev)
                bg_rmse = float(np.sqrt(np.mean((baseline - forward_dev.truth) ** 2)))
                compact.update(
                    model=name,
                    seed=seed,
                    split_scheme="forward",
                    stage="development",
                    skill_vs_background=1 - compact["pooled_rmse"] ** 2 / bg_rmse**2,
                    **metadata,
                )
                forward_rows.append(compact)
        forward = pd.DataFrame(forward_rows)
        forward.to_csv(args.output / "forward_metrics.csv", index=False)
        selected_predictions = predictions.loc[predictions.model.eq(selected)]
        ensemble = (
            selected_predictions.groupby("record_id")
            .agg(
                truth=("truth", "first"),
                prediction=("prediction", "mean"),
                uncertainty=("prediction", "std"),
                seed_count=("seed", "nunique"),
            )
            .reset_index()
        )
        ensemble["uncertainty"] = ensemble.uncertainty.fillna(0.0)
        meta = development.drop_duplicates("record_id").set_index("record_id")
        ensemble = ensemble.join(meta, on="record_id", rsuffix="_meta")
        q90 = float(np.quantile(np.abs(ensemble.truth - ensemble.prediction), 0.9))
        ensemble["lower"], ensemble["upper"] = ensemble.prediction - q90, ensemble.prediction + q90
        ensemble["target"] = "fco2"
        ensemble["target_provenance"] = "SOCAT_fCO2rec"
        ensemble["model_provenance"] = selected
        ensemble["ood"] = (ensemble.nearest_carbon_km > 500) | ensemble.lme_id.astype(str).eq("-1")
        ensemble["prediction_status"] = "development"
        ensemble["seed"] = "ensemble_100_101_102"
        standard = standardize_predictions(ensemble)
        standard.to_parquet(args.output / "selected_development_predictions.parquet", index=False)
        coverage = float(
            ((ensemble.truth >= ensemble.lower) & (ensemble.truth <= ensemble.upper)).mean()
        )
        selection = {
            "selected_model": selected,
            "criterion": "development LME-macro RMSE",
            "conformal_absolute_error_q90": q90,
            "development_coverage_90": coverage,
            "locked_test_opened": False,
            "external_opened": False,
        }
        (args.output / "selection.json").write_text(
            json.dumps(selection, indent=2), encoding="utf-8"
        )
        make_figures(summary, pd.DataFrame(stratified_rows), args.output)
        write_report(summary, cv, forward, pd.DataFrame(stratified_rows), selection, args.output)

    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_head(),
        "manifest_validation": validation,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "catboost_iterations": args.catboost_iterations,
        "seeds": seeds,
        "models": MODELS,
        "train_rows": len(train),
        "development_rows": len(development),
        "train_cruises": int(train.group_key.nunique()),
        "development_cruises": int(development.group_key.nunique()),
        "locked_test_opened": False,
        "external_opened": False,
        "elapsed_seconds": time.time() - start,
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    hashes = {
        str(path.relative_to(args.output)): sha256(path)
        for path in args.output.rglob("*")
        if path.is_file()
    }
    (args.output / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8"
    )
    return 0


def git_head() -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def make_figures(summary: pd.DataFrame, strata: pd.DataFrame, output: Path) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    ordered = summary.sort_values("lme_macro_rmse_mean")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ordered.model, ordered.lme_macro_rmse_mean, color="#2878B5")
    ax.set_ylabel("Development LME-macro RMSE (µatm)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(figure_dir / "model_lme_macro_rmse.png", dpi=180)
    plt.close(fig)
    lme = (
        strata.loc[strata.stratum.eq("lme")]
        .groupby(["model", "group"])
        .skill_vs_background.mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    for model, part in lme.groupby("model"):
        if model != "seasonal_climatology":
            ax.plot(part.group, part.skill_vs_background, marker="o", ms=3, label=model)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Skill versus seasonal-trend climatology")
    ax.set_xlabel("LME id")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "lme_skill.png", dpi=180)
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    cv: pd.DataFrame,
    forward: pd.DataFrame,
    strata: pd.DataFrame,
    selection: dict,
    output: Path,
) -> None:
    selected = selection["selected_model"]
    row = summary.loc[summary.model.eq(selected)].iloc[0]
    lme = (
        strata.loc[(strata.model.eq(selected)) & strata.stratum.eq("lme")]
        .groupby("group")
        .agg(n=("n", "max"), skill=("skill_vs_background", "mean"))
        .reset_index()
    )
    eligible = lme.loc[lme.n >= 100]
    positive_fraction = float((eligible.skill > 0).mean()) if len(eligible) else float("nan")
    low = (
        strata.loc[(strata.model.eq(selected)) & strata.stratum.eq("fco2_band")]
        .groupby("group")
        .agg(n=("n", "max"), skill=("skill_vs_background", "mean"))
        .reset_index()
    )
    report = f"""# P1.2 coastal fCO2 viability development report

This report uses only frozen `train` and `development` labels. Locked-test and
external-independent labels were not opened.

## Selection

- Selected candidate: `{selected}`
- Development pooled RMSE: {row.pooled_rmse_mean:.4f} µatm
- Development cruise-equal RMSE: {row.cruise_equal_rmse_mean:.4f} µatm
- Development LME-macro RMSE: {row.lme_macro_rmse_mean:.4f} µatm
- Development worst-LME RMSE: {row.worst_lme_rmse_mean:.4f} µatm
- Pooled skill versus seasonal-trend climatology: {row.skill_vs_background_mean:.3f}
- Fraction of LMEs with N>=100 and positive skill: {positive_fraction:.3f}
- Development-calibrated nominal 90% interval coverage: {selection["development_coverage_90"]:.3f}

## Scope

The frozen cache spans North-American-adjacent coastal waters (0-70.125 N,
180-315 E). These results can nominate a regional candidate for the one-time
locked gate in Issue #11; they cannot establish global coastal skill or final
product status. The final status remains pending the locked evaluation.

## Five-fold cruise CV

{table_text(cv.groupby("model")[["pooled_rmse", "cruise_equal_rmse", "lme_macro_rmse", "skill_vs_background"]].agg(["mean", "std"]).round(4)) if len(cv) else "Not run in this phase."}

## Development model comparison

{table_text(summary.round(4), index=False)}

## Forward-chain development

{table_text(forward.round(4), index=False)}

## fCO2-band diagnostics

{table_text(low.round(4), index=False)}
"""
    (output / "report.md").write_text(report, encoding="utf-8")


def table_text(frame: pd.DataFrame, *, index: bool = True) -> str:
    return "```text\n" + frame.to_string(index=index) + "\n```"


if __name__ == "__main__":
    raise SystemExit(main())
