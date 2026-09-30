"""Run the frozen P1.3 North-American coastal-TA viability experiment."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.neighbors import BallTree
from torch import nn

from recad.chem.esper_ta import ESPER_LIR_TA
from recad.evaluate.p1_framework import (
    FrozenManifest,
    P1DataGateway,
    Purpose,
    evaluate_predictions,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_ta_viability_v2.2"
PREREGISTRATION_COMMIT = "dfeb2f0a2855b3b84630c8dd1a466eff7af2738f"
DETERMINISTIC_MODELS = [
    "carter_esper_prior",
    "carter_region_season_corrected",
    "global_ta_sss",
    "lme_ta_sss",
    "hierarchical_ta_sss",
]
ORACLE_MODEL = "carter_esper_oracle_insitu_sss"
NEURAL_MODEL = "hierarchical_residual"
EARTH_RADIUS_KM = 6371.0088
BIGHT_PREREGISTRATION_COMMIT = "3936cc4658cf1d3912fc6611ff6ad4412ad7e44a"
BIGHT_DEFINITIONS = {
    "sab": {
        "source_lme_id": 6,
        "latitude_edges": [28.45, 30.50, 33.00, 35.30],
        "subregion_names": ["south", "central", "north"],
    },
    "mab": {
        "source_lme_id": 7,
        "latitude_edges": [35.20, 38.00, 40.00, 41.75],
        "subregion_names": ["south", "central", "north"],
    },
}


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def stable_id(frame: pd.DataFrame) -> pd.Series:
    if "obs_id" in frame:
        identifier = frame.obs_id.astype(str)
        if identifier.notna().all() and identifier.is_unique:
            return identifier
    return (
        frame.group_key.astype(str)
        + ":"
        + frame.year.astype(str)
        + ":"
        + frame.month.astype(str)
        + ":"
        + frame.latitude.round(5).astype(str)
        + ":"
        + frame.longitude.round(5).astype(str)
    )


def balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    """Exact probability of LME -> cruise -> month -> record sampling."""
    work = frame[["lme_id", "group_key", "month"]].copy()
    n_regions = max(work.lme_id.nunique(), 1)
    cruises = work.groupby("lme_id").group_key.transform("nunique")
    months = work.groupby(["lme_id", "group_key"])["month"].transform("nunique")
    records = work.groupby(["lme_id", "group_key", "month"])["month"].transform("size")
    weights = 1.0 / (n_regions * cruises * months * records)
    values = weights.to_numpy(float)
    return values / values.mean()


def nearest_support_km(fit: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Great-circle distance from target rows to the nearest fitted TA location."""
    fit_coords = np.deg2rad(fit[["latitude", "longitude"]].to_numpy(float))
    target_coords = np.deg2rad(target[["latitude", "longitude"]].to_numpy(float))
    tree = BallTree(fit_coords, metric="haversine")
    distance, _ = tree.query(target_coords, k=1)
    return distance[:, 0] * EARTH_RADIUS_KM


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["truth", "sss", "latitude", "longitude", "month", "group_key"]
    valid = np.ones(len(frame), dtype=bool)
    for column in required:
        valid &= frame[column].notna().to_numpy()
    frame = frame.loc[valid].copy().reset_index(drop=True)
    frame["record_id"] = stable_id(frame)
    frame["lme_id"] = frame.lme_id.fillna(-1).astype(int)
    frame["regime_id"] = frame.regime_id.fillna(-1).astype(int)
    phase = 2 * np.pi * (frame.month.to_numpy(float) - 1) / 12
    frame["month_sin"] = np.sin(phase)
    frame["month_cos"] = np.cos(phase)
    longitude = np.deg2rad(frame.longitude.to_numpy(float))
    frame["lon_sin"] = np.sin(longitude)
    frame["lon_cos"] = np.cos(longitude)
    return frame


def apply_bight_definition(frame: pd.DataFrame, region: str) -> pd.DataFrame:
    """Apply the preregistered LME/end-point mask and encode bight subregions."""
    if region == "all":
        return frame
    definition = BIGHT_DEFINITIONS[region]
    edges = definition["latitude_edges"]
    mask = frame.lme_id.eq(definition["source_lme_id"])
    mask &= frame.latitude.ge(edges[0])
    upper_mask = frame.latitude.lt(edges[-1]) if region == "sab" else frame.latitude.le(edges[-1])
    mask &= upper_mask
    selected = frame.loc[mask].copy().reset_index(drop=True)
    selected["source_lme_id"] = selected.lme_id
    selected["bight"] = region.upper()
    selected["subregion"] = pd.cut(
        selected.latitude,
        edges,
        labels=definition["subregion_names"],
        include_lowest=True,
        right=region == "mab",
    ).astype(str)
    mapping = {name: index for index, name in enumerate(definition["subregion_names"])}
    selected["lme_id"] = selected.subregion.map(mapping).astype(int)
    if selected.subregion.eq("nan").any():
        raise RuntimeError(f"{region} mask produced unassigned subregion rows")
    return selected


def add_carter(frame: pd.DataFrame, estimator: ESPER_LIR_TA) -> pd.DataFrame:
    frame = frame.copy()
    lon = np.mod(frame.longitude.to_numpy(float), 360.0)
    lat = frame.latitude.to_numpy(float)
    frame["carter"] = estimator.estimate_eq16_fast(lon, lat, frame.sss.to_numpy(float))
    frame["carter_oracle"] = estimator.estimate_eq16_fast(lon, lat, frame.salinity.to_numpy(float))
    return frame


class HierarchicalDesign:
    """Dense design for robust partially pooled varying TA-SSS relations."""

    def __init__(self, groups: bool = True):
        self.groups = groups

    def fit(self, frame: pd.DataFrame) -> HierarchicalDesign:
        self.s_center = float(frame.sss.median())
        self.s_scale = max(float(frame.sss.std()), 0.25)
        self.lmes = sorted(frame.lme_id.unique().tolist()) if self.groups else []
        self.regimes = sorted(frame.regime_id.unique().tolist()) if self.groups else []
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        salinity = (frame.sss.to_numpy(float) - self.s_center) / self.s_scale
        phase_sin = frame.month_sin.to_numpy(float)
        phase_cos = frame.month_cos.to_numpy(float)
        columns = [
            np.ones(len(frame)),
            salinity,
            phase_sin,
            phase_cos,
            salinity * phase_sin,
            salinity * phase_cos,
        ]
        penalties = [0.0, 0.2, 0.2, 0.2, 0.5, 0.5]
        for values, source in ((self.lmes, frame.lme_id), (self.regimes, frame.regime_id)):
            for value in values:
                indicator = source.eq(value).to_numpy(float)
                columns.extend([indicator, indicator * salinity])
                penalties.extend([1.0, 1.0])
        return np.column_stack(columns), np.asarray(penalties)


@dataclass
class RobustPartialPooling:
    alpha: float = 100.0
    groups: bool = True
    iterations: int = 8

    def fit(
        self,
        frame: pd.DataFrame,
        target: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> RobustPartialPooling:
        self.design = HierarchicalDesign(self.groups).fit(frame)
        x, penalties = self.design.transform(frame)
        y = frame.truth.to_numpy(float) if target is None else np.asarray(target, dtype=float)
        base_weight = np.ones(len(frame)) if sample_weight is None else sample_weight.copy()
        robust = np.ones(len(frame))
        coefficient = np.zeros(x.shape[1])
        for _ in range(self.iterations):
            weight = np.clip(base_weight * robust, 1e-8, None)
            xtw = x.T * weight
            system = xtw @ x + np.diag(self.alpha * penalties)
            coefficient = np.linalg.solve(system + np.eye(x.shape[1]) * 1e-9, xtw @ y)
            residual = y - x @ coefficient
            scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 1.0)
            cutoff = 1.345 * scale
            robust = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1e-12))
        self.coefficient = coefficient
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x, _ = self.design.transform(frame)
        return x @ self.coefficient


class LMEModel:
    def __init__(self, alpha: float = 1.0, min_cruises: int = 5):
        self.alpha = alpha
        self.min_cruises = min_cruises

    def fit(self, frame: pd.DataFrame) -> LMEModel:
        weights = balanced_weights(frame)
        self.global_model = RobustPartialPooling(self.alpha, groups=False).fit(
            frame, sample_weight=weights
        )
        self.models: dict[int, RobustPartialPooling] = {}
        for lme, part in frame.groupby("lme_id"):
            if part.group_key.nunique() < self.min_cruises:
                continue
            local_weights = balanced_weights(part)
            self.models[int(lme)] = RobustPartialPooling(self.alpha, groups=False).fit(
                part, sample_weight=local_weights
            )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        prediction = self.global_model.predict(frame)
        for lme, model in self.models.items():
            mask = frame.lme_id.eq(lme).to_numpy()
            if mask.any():
                prediction[mask] = model.predict(frame.loc[mask])
        return prediction


class CarterCorrection:
    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, frame: pd.DataFrame) -> CarterCorrection:
        valid = frame.carter.notna().to_numpy()
        residual = frame.loc[valid, "truth"].to_numpy(float) - frame.loc[valid, "carter"].to_numpy(
            float
        )
        self.model = RobustPartialPooling(self.alpha, groups=True).fit(
            frame.loc[valid], residual, balanced_weights(frame.loc[valid])
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.carter.to_numpy(float) + self.model.predict(frame)


def metric_summary(frame: pd.DataFrame, prediction: np.ndarray) -> tuple[pd.DataFrame, dict]:
    scored = frame[["truth", "group_key", "lme_id", "regime_id", "nearest_carbon_km"]].copy()
    scored["prediction"] = prediction
    details = evaluate_predictions(scored)
    lookup = {row.aggregation: row for row in details.itertuples() if row.group == "all"}
    worst = details.loc[details.aggregation.eq("worst_lme")].iloc[0]
    pooled = lookup["pooled"]
    compact = {
        "n": int(pooled.n),
        "pooled_rmse": float(pooled.rmse),
        "pooled_mae": float(pooled.mae),
        "pooled_bias": float(pooled.bias),
        "pooled_r2": float(pooled.r2),
        "cruise_equal_rmse": float(lookup["cruise_equal"].rmse),
        "lme_macro_rmse": float(lookup["lme_macro"].rmse),
        "regime_macro_rmse": float(lookup["regime_macro"].rmse),
        "worst_lme_rmse": float(worst.rmse),
        "worst_lme": str(worst.group),
    }
    return details, compact


def skill(model_rmse: float, baseline_rmse: float) -> float:
    return 1.0 - model_rmse**2 / baseline_rmse**2 if baseline_rmse > 0 else math.nan


def all_populated_support_bins_positive(strata: pd.DataFrame) -> bool:
    support = strata.loc[strata.stratum.eq("support_distance")]
    return bool(len(support) and (support.skill_vs_carter > 0).all())


def select_alpha(train: pd.DataFrame, candidates: list[float]) -> tuple[float, pd.DataFrame]:
    rows = []
    for alpha in candidates:
        fold_scores = []
        for fold in range(5):
            fit = train.loc[train.cv_fold.ne(fold)].reset_index(drop=True)
            held = train.loc[train.cv_fold.eq(fold)].reset_index(drop=True)
            model = RobustPartialPooling(alpha, groups=True).fit(
                fit, sample_weight=balanced_weights(fit)
            )
            prediction = model.predict(held)
            _, compact = metric_summary(held, prediction)
            fold_scores.append(compact["lme_macro_rmse"])
            rows.append({"alpha": alpha, "fold": fold, **compact})
        print(f"alpha={alpha:g} CV LME-macro={np.mean(fold_scores):.3f}", flush=True)
    result = pd.DataFrame(rows)
    selected = float(result.groupby("alpha").lme_macro_rmse.mean().idxmin())
    return selected, result


def fit_predict(
    name: str,
    fit: pd.DataFrame,
    target: pd.DataFrame,
    *,
    alpha: float,
) -> tuple[np.ndarray, object | None]:
    if name == "carter_esper_prior":
        return target.carter.to_numpy(float), None
    if name == ORACLE_MODEL:
        return target.carter_oracle.to_numpy(float), None
    if name == "carter_region_season_corrected":
        model = CarterCorrection(alpha).fit(fit)
    elif name == "global_ta_sss":
        model = RobustPartialPooling(1.0, groups=False).fit(
            fit, sample_weight=balanced_weights(fit)
        )
    elif name == "lme_ta_sss":
        model = LMEModel().fit(fit)
    elif name == "hierarchical_ta_sss":
        model = RobustPartialPooling(alpha, groups=True).fit(
            fit, sample_weight=balanced_weights(fit)
        )
    else:
        raise ValueError(name)
    return model.predict(target), model


NUMERIC_NEURAL = [
    "sss",
    "sst",
    "adt",
    "wspd",
    "pco2air",
    "latitude",
    "lon_sin",
    "lon_cos",
    "month_sin",
    "month_cos",
    "carter",
    "hierarchical_base",
]


class NeuralTransform:
    def fit(self, frame: pd.DataFrame) -> NeuralTransform:
        values = frame[NUMERIC_NEURAL].to_numpy(float)
        self.median = np.nanmedian(values, axis=0)
        values = np.where(np.isfinite(values), values, self.median)
        self.mean = values.mean(axis=0)
        self.std = values.std(axis=0)
        self.std[self.std < 1e-7] = 1.0
        self.lmes = sorted(frame.lme_id.unique().tolist())
        self.regimes = sorted(frame.regime_id.unique().tolist())
        return self

    def apply(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[NUMERIC_NEURAL].to_numpy(float)
        values = np.where(np.isfinite(values), values, self.median)
        parts = [(values - self.mean) / self.std]
        for categories, source in ((self.lmes, frame.lme_id), (self.regimes, frame.regime_id)):
            mapping = {value: i for i, value in enumerate(categories)}
            onehot = np.zeros((len(frame), len(categories) + 1), dtype=np.float32)
            codes = np.array([mapping.get(value, len(categories)) for value in source], dtype=int)
            onehot[np.arange(len(frame)), codes] = 1.0
            parts.append(onehot)
        return np.concatenate(parts, axis=1).astype(np.float32)


class ResidualMLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


@dataclass
class NeuralResult:
    prediction: np.ndarray
    best_step: int
    state: dict


def train_neural(
    fit: pd.DataFrame,
    target: pd.DataFrame,
    *,
    alpha: float,
    seed: int,
    steps: int,
    batch_size: int,
    checkpoint_every: int,
    select_checkpoint: bool,
) -> NeuralResult:
    base_model = RobustPartialPooling(alpha, groups=True).fit(
        fit, sample_weight=balanced_weights(fit)
    )
    fit = fit.copy()
    target = target.copy()
    fit["hierarchical_base"] = base_model.predict(fit)
    target["hierarchical_base"] = base_model.predict(target)
    transform = NeuralTransform().fit(fit)
    x_fit_np = transform.apply(fit)
    x_target_np = transform.apply(target)
    residual = fit.truth.to_numpy(float) - fit.hierarchical_base.to_numpy(float)
    residual_scale = max(float(np.std(residual)), 1.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_fit = torch.as_tensor(x_fit_np, device=device)
    y_fit = torch.as_tensor((residual / residual_scale).astype(np.float32), device=device)
    x_target = torch.as_tensor(x_target_np, device=device)
    probabilities = torch.as_tensor(balanced_weights(fit).astype(np.float32), device=device)
    probabilities /= probabilities.sum()
    model = ResidualMLP(x_fit.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best_score, best_step, best_state = math.inf, 0, None
    for step in range(1, steps + 1):
        model.train()
        index = torch.multinomial(probabilities, batch_size, replacement=True)
        prediction = model(x_fit[index])
        loss = nn.functional.huber_loss(prediction, y_fit[index], delta=1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % checkpoint_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                residual_prediction = model(x_target).cpu().numpy() * residual_scale
            full_prediction = target.hierarchical_base.to_numpy(float) + residual_prediction
            _, compact = metric_summary(target, full_prediction)
            score = compact["lme_macro_rmse"] if select_checkpoint else -step
            if score < best_score:
                best_score, best_step = score, step
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        residual_prediction = model(x_target).cpu().numpy() * residual_scale
    full_prediction = target.hierarchical_base.to_numpy(float) + residual_prediction
    state = {
        "model": best_state,
        "transform": transform.__dict__,
        "residual_scale": residual_scale,
        "seed": seed,
        "best_step": best_step,
        "alpha": alpha,
    }
    return NeuralResult(full_prediction, best_step, state)


def add_strata(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    carter: np.ndarray,
) -> list[dict]:
    rows = []
    error = prediction - frame.truth.to_numpy(float)
    baseline_error = carter - frame.truth.to_numpy(float)
    salinity_band = pd.cut(
        frame.salinity,
        [-np.inf, 20, 30, 33, 36, np.inf],
        right=False,
    )
    support_band = pd.cut(
        frame.nearest_carbon_km,
        [0, 25, 50, 100, 200, 500, np.inf],
        right=False,
        include_lowest=True,
    )
    for kind, groups in (
        ("lme", frame.lme_id),
        ("regime", frame.regime_id),
        ("salinity_band", salinity_band),
        ("support_distance", support_band),
    ):
        for group in sorted(pd.Series(groups).dropna().unique(), key=str):
            mask = np.asarray(groups == group)
            model_rmse = float(np.sqrt(np.mean(error[mask] ** 2)))
            baseline_rmse = float(np.sqrt(np.mean(baseline_error[mask] ** 2)))
            rows.append(
                {
                    "stratum": kind,
                    "group": str(group),
                    "n": int(mask.sum()),
                    "cruises": int(frame.loc[mask, "group_key"].nunique()),
                    "model_rmse": model_rmse,
                    "carter_rmse": baseline_rmse,
                    "skill_vs_carter": skill(model_rmse, baseline_rmse),
                }
            )
    return rows


def safe_forward_frames(
    train: pd.DataFrame, development: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = train.loc[train.forward_split.eq("train")].copy().reset_index(drop=True)
    held = (
        development.loc[development.forward_split.eq("development")].copy().reset_index(drop=True)
    )
    if set(fit.group_key) & set(held.group_key):
        raise RuntimeError("cruise crosses safe forward boundary")
    return fit, held


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / f"outputs/experiments/{EXPERIMENT_ID}"
    )
    parser.add_argument(
        "--esper-mat",
        type=Path,
        default=Path(r"D:\proj_personal\PhD\ESPER\ESPER_LIR_Files\LIR_files_TA_v3.mat"),
    )
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--region", choices=["all", "sab", "mab"], default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    start = time.time()

    config_path = ROOT / (
        "configs/p1_ta_baselines_v2.2.yaml"
        if args.region == "all"
        else "configs/p1_ta_bights_v2.2.yaml"
    )
    experiment_id = EXPERIMENT_ID if args.region == "all" else f"p1_ta_{args.region}_viability_v2.2"
    preregistration_commit = (
        PREREGISTRATION_COMMIT if args.region == "all" else BIGHT_PREREGISTRATION_COMMIT
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = FrozenManifest.load(ROOT / "configs/frozen/data_manifest_v2.2.json")
    validation = manifest.validate(hash_mode="full")
    gateway = P1DataGateway(manifest)
    columns = ["obs_id", "salinity", "sst", "sss", "adt", "wspd", "pco2air", "temperature"]
    train = apply_bight_definition(
        prepare(gateway.load_labels("ta", Purpose.TRAIN, columns=columns)), args.region
    )
    development = apply_bight_definition(
        prepare(gateway.load_labels("ta", Purpose.SELECTION, columns=columns)), args.region
    )
    estimator = ESPER_LIR_TA.from_mat(args.esper_mat)
    train = add_carter(train, estimator)
    development = add_carter(development, estimator)
    train = train.loc[train.carter.notna()].reset_index(drop=True)
    development = development.loc[development.carter.notna()].reset_index(drop=True)
    train["nearest_carbon_km"] = 0.0
    development["nearest_carbon_km"] = nearest_support_km(train, development)
    seeds = [100, 101, 102]

    alpha, alpha_cv = select_alpha(train, [1.0, 10.0, 100.0, 1000.0])
    alpha_cv.to_csv(args.output / "alpha_selection_cv.csv", index=False)
    primary_rows, prediction_frames, strata_rows = [], [], []

    for name in [*DETERMINISTIC_MODELS, ORACLE_MODEL]:
        print(f"PRIMARY {name}", flush=True)
        prediction, _ = fit_predict(name, train, development, alpha=alpha)
        details, compact = metric_summary(development, prediction)
        compact.update(model=name, seed=100, phase="A0_A2")
        primary_rows.append(compact)
        strata_rows.extend(
            {"model": name, "seed": 100, **row}
            for row in add_strata(development, prediction, development.carter.to_numpy(float))
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "record_id": development.record_id,
                    "model": name,
                    "seed": 100,
                    "truth": development.truth,
                    "carter": development.carter,
                    "prediction": prediction,
                }
            )
        )
        details.to_parquet(args.output / f"metrics_detail_{name}.parquet", index=False)

    primary = pd.DataFrame(primary_rows)
    lookup = primary.set_index("model")
    a2 = lookup.loc["hierarchical_ta_sss"]
    a3_trigger = bool(
        a2.lme_macro_rmse < lookup.loc["carter_esper_prior", "lme_macro_rmse"]
        and a2.lme_macro_rmse < lookup.loc["lme_ta_sss", "lme_macro_rmse"]
        and a2.cruise_equal_rmse < lookup.loc["carter_esper_prior", "cruise_equal_rmse"]
        and a2.cruise_equal_rmse < lookup.loc["lme_ta_sss", "cruise_equal_rmse"]
    )
    print(f"A3 stopping-rule trigger={a3_trigger}", flush=True)
    if a3_trigger:
        for seed in seeds:
            print(f"PRIMARY {NEURAL_MODEL} seed={seed}", flush=True)
            result = train_neural(
                train,
                development,
                alpha=alpha,
                seed=seed,
                steps=args.steps,
                batch_size=args.batch_size,
                checkpoint_every=args.checkpoint_every,
                select_checkpoint=True,
            )
            torch.save(result.state, checkpoint_dir / f"primary_{NEURAL_MODEL}_seed{seed}.pt")
            _, compact = metric_summary(development, result.prediction)
            compact.update(
                model=NEURAL_MODEL,
                seed=seed,
                phase="A3",
                best_step=result.best_step,
            )
            primary_rows.append(compact)
            strata_rows.extend(
                {"model": NEURAL_MODEL, "seed": seed, **row}
                for row in add_strata(
                    development,
                    result.prediction,
                    development.carter.to_numpy(float),
                )
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "record_id": development.record_id,
                        "model": NEURAL_MODEL,
                        "seed": seed,
                        "truth": development.truth,
                        "carter": development.carter,
                        "prediction": result.prediction,
                    }
                )
            )

    primary = pd.DataFrame(primary_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    strata = pd.DataFrame(strata_rows)
    primary.to_csv(args.output / "metrics_by_seed.csv", index=False)
    predictions.to_parquet(args.output / "candidate_predictions.parquet", index=False)
    strata.to_csv(args.output / "stratified_metrics.csv", index=False)
    summary = (
        primary.groupby("model")
        .agg(
            seeds=("seed", "nunique"),
            pooled_rmse_mean=("pooled_rmse", "mean"),
            pooled_rmse_sd=("pooled_rmse", "std"),
            pooled_mae_mean=("pooled_mae", "mean"),
            pooled_bias_mean=("pooled_bias", "mean"),
            pooled_r2_mean=("pooled_r2", "mean"),
            cruise_equal_rmse_mean=("cruise_equal_rmse", "mean"),
            lme_macro_rmse_mean=("lme_macro_rmse", "mean"),
            regime_macro_rmse_mean=("regime_macro_rmse", "mean"),
            worst_lme_rmse_mean=("worst_lme_rmse", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(args.output / "summary.csv", index=False)
    eligible = summary.loc[~summary.model.isin(["carter_esper_prior", ORACLE_MODEL])].sort_values(
        "lme_macro_rmse_mean"
    )
    selected = str(eligible.iloc[0].model)

    cv_rows, cv_prediction_frames = [], []
    cv_models = [*DETERMINISTIC_MODELS]
    if a3_trigger:
        cv_models.append(NEURAL_MODEL)
    for fold in range(5):
        fit = train.loc[train.cv_fold.ne(fold)].reset_index(drop=True)
        held = train.loc[train.cv_fold.eq(fold)].reset_index(drop=True)
        held["nearest_carbon_km"] = nearest_support_km(fit, held)
        for name in cv_models:
            print(f"CV fold={fold} {name}", flush=True)
            if name == NEURAL_MODEL:
                result = train_neural(
                    fit,
                    held,
                    alpha=alpha,
                    seed=100,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    checkpoint_every=args.steps,
                    select_checkpoint=False,
                )
                prediction = result.prediction
            else:
                prediction, _ = fit_predict(name, fit, held, alpha=alpha)
            _, compact = metric_summary(held, prediction)
            compact.update(model=name, fold=fold, seed=100)
            cv_rows.append(compact)
            cv_prediction_frames.append(
                pd.DataFrame(
                    {
                        "record_id": held.record_id,
                        "model": name,
                        "fold": fold,
                        "truth": held.truth,
                        "prediction": prediction,
                    }
                )
            )
    cv = pd.DataFrame(cv_rows)
    cv_predictions = pd.concat(cv_prediction_frames, ignore_index=True)
    cv.to_csv(args.output / "cv_metrics.csv", index=False)
    cv_predictions.to_parquet(args.output / "cv_predictions.parquet", index=False)

    forward_fit, forward_held = safe_forward_frames(train, development)
    forward_rows = []
    forward_models = list(
        dict.fromkeys(["carter_esper_prior", "lme_ta_sss", "hierarchical_ta_sss", selected])
    )
    if len(forward_fit) and len(forward_held):
        forward_held["nearest_carbon_km"] = nearest_support_km(forward_fit, forward_held)
        for name in forward_models:
            if name == NEURAL_MODEL:
                predictions_seed = []
                for seed in seeds:
                    result = train_neural(
                        forward_fit,
                        forward_held,
                        alpha=alpha,
                        seed=seed,
                        steps=args.steps,
                        batch_size=args.batch_size,
                        checkpoint_every=args.steps,
                        select_checkpoint=False,
                    )
                    predictions_seed.append(result.prediction)
                prediction = np.mean(predictions_seed, axis=0)
            else:
                prediction, _ = fit_predict(name, forward_fit, forward_held, alpha=alpha)
            _, compact = metric_summary(forward_held, prediction)
            compact.update(model=name)
            forward_rows.append(compact)
    else:
        forward_rows.extend(
            {"model": name, "n": 0, "pooled_rmse": math.nan} for name in forward_models
        )
    forward = pd.DataFrame(forward_rows).drop_duplicates("model")
    forward.to_csv(args.output / "forward_metrics.csv", index=False)

    lme_rows = []
    lme_counts = train.groupby("lme_id").agg(
        rows=("truth", "size"), cruises=("group_key", "nunique")
    )
    eligible_lmes = lme_counts.loc[(lme_counts.rows >= 30) & (lme_counts.cruises >= 3)].index
    for lme in eligible_lmes:
        fit = train.loc[train.lme_id.ne(lme)].reset_index(drop=True)
        held = train.loc[train.lme_id.eq(lme)].reset_index(drop=True)
        held["nearest_carbon_km"] = nearest_support_km(fit, held)
        for name in ["carter_esper_prior", "hierarchical_ta_sss"]:
            prediction, _ = fit_predict(name, fit, held, alpha=alpha)
            _, compact = metric_summary(held, prediction)
            compact.update(model=name, held_lme=int(lme), cruises=int(held.group_key.nunique()))
            lme_rows.append(compact)
    leave_lme = pd.DataFrame(lme_rows)
    leave_lme.to_csv(args.output / "leave_lme_out_metrics.csv", index=False)

    selected_predictions = predictions.loc[predictions.model.eq(selected)]
    ensemble = (
        selected_predictions.groupby("record_id")
        .agg(
            truth=("truth", "first"),
            carter=("carter", "first"),
            prediction=("prediction", "mean"),
            ensemble_sd=("prediction", "std"),
            seed_count=("seed", "nunique"),
        )
        .reset_index()
    )
    meta = development.drop_duplicates("record_id").set_index("record_id")
    ensemble = ensemble.join(meta, on="record_id", rsuffix="_meta")
    oof = cv_predictions.loc[cv_predictions.model.eq(selected)].copy()
    q90 = float(np.quantile(np.abs(oof.truth - oof.prediction), 0.90))
    ensemble["ensemble_sd"] = ensemble.ensemble_sd.fillna(0.0)
    ensemble["lower"] = ensemble.prediction - q90
    ensemble["upper"] = ensemble.prediction + q90
    coverage = float(
        ((ensemble.truth >= ensemble.lower) & (ensemble.truth <= ensemble.upper)).mean()
    )
    ensemble["ood"] = (ensemble.nearest_carbon_km > 500) | ensemble.lme_id.isin(
        set(development.lme_id) - set(train.lme_id)
    )
    ensemble.to_parquet(args.output / "selected_development_predictions.parquet", index=False)

    selected_row = summary.set_index("model").loc[selected]
    carter_row = summary.set_index("model").loc["carter_esper_prior"]
    lme_row = summary.set_index("model").loc["lme_ta_sss"]
    selected_strata = (
        strata.loc[strata.model.eq(selected)]
        .groupby(["stratum", "group"], as_index=False)
        .agg(
            n=("n", "max"),
            cruises=("cruises", "max"),
            model_rmse=("model_rmse", "mean"),
            carter_rmse=("carter_rmse", "mean"),
            skill_vs_carter=("skill_vs_carter", "mean"),
        )
    )
    supported_lmes = selected_strata.loc[
        selected_strata.stratum.eq("lme")
        & (selected_strata.n >= 30)
        & (selected_strata.cruises >= 3)
    ]
    support_bins = selected_strata.loc[selected_strata.stratum.eq("support_distance")]
    forward_lookup = forward.set_index("model")
    forward_skill = skill(
        float(forward_lookup.loc[selected, "pooled_rmse"]),
        float(forward_lookup.loc["carter_esper_prior", "pooled_rmse"]),
    )
    if len(leave_lme):
        leave_pivot = leave_lme.pivot(index="held_lme", columns="model", values="pooled_rmse")
        leave_skills = 1 - (
            leave_pivot["hierarchical_ta_sss"] ** 2 / leave_pivot["carter_esper_prior"] ** 2
        )
    else:
        leave_skills = pd.Series(dtype=float)
    checks = {
        "lme_macro_improvement_vs_carter_ge_10pct": (
            1 - selected_row.lme_macro_rmse_mean**2 / carter_row.lme_macro_rmse_mean**2
        )
        >= 0.10,
        "lme_macro_improvement_vs_lme_linear_ge_5pct": (
            1 - selected_row.lme_macro_rmse_mean**2 / lme_row.lme_macro_rmse_mean**2
        )
        >= 0.05,
        "cruise_equal_improves_both": selected_row.cruise_equal_rmse_mean
        < min(carter_row.cruise_equal_rmse_mean, lme_row.cruise_equal_rmse_mean),
        "worst_lme_degradation_le_10pct": selected_row.worst_lme_rmse_mean
        <= 1.10 * min(carter_row.worst_lme_rmse_mean, lme_row.worst_lme_rmse_mean),
        "supported_lme_positive_skill_fraction_ge_80pct": float(
            (supported_lmes.skill_vs_carter > 0).mean()
        )
        >= 0.80,
        "safe_forward_skill_vs_carter_positive": forward_skill > 0,
        "leave_lme_out_mean_skill_vs_carter_positive": float(leave_skills.mean()) > 0,
        "coverage_90_between_85_and_95pct": 0.85 <= coverage <= 0.95,
        "support_bin_skill_vs_carter_positive": all_populated_support_bins_positive(
            selected_strata
        ),
    }
    if args.region != "all":
        checks.pop("leave_lme_out_mean_skill_vs_carter_positive")
        checks = {name.replace("lme", "subregion"): value for name, value in checks.items()}
    checks = {name: bool(value) for name, value in checks.items()}
    gate_passed = all(checks.values())
    if args.region == "all":
        decision = (
            "nominate_for_issue_11_locked_gate"
            if gate_passed
            else "diagnostic_only_do_not_open_locked_test"
        )
    elif gate_passed:
        decision = "pass_regional"
    elif selected_row.pooled_rmse_mean < carter_row.pooled_rmse_mean:
        decision = "diagnostic_only"
    else:
        decision = "fail"
    gate = {
        "selected_model": selected,
        "development_gate_passed": gate_passed,
        "decision": decision,
        "checks": checks,
        "metrics": {
            "lme_macro_improvement_vs_carter": float(
                1 - selected_row.lme_macro_rmse_mean**2 / carter_row.lme_macro_rmse_mean**2
            ),
            "lme_macro_improvement_vs_lme_linear": float(
                1 - selected_row.lme_macro_rmse_mean**2 / lme_row.lme_macro_rmse_mean**2
            ),
            "supported_lme_positive_skill_fraction": float(
                (supported_lmes.skill_vs_carter > 0).mean()
            ),
            "safe_forward_skill_vs_carter": forward_skill,
            "leave_lme_out_mean_skill_vs_carter": float(leave_skills.mean()),
            "development_coverage_90": coverage,
            "conformal_absolute_error_q90": q90,
            "worst_support_bin_skill_vs_carter": float(support_bins.skill_vs_carter.min()),
        },
        "locked_test_opened": False,
        "external_independent_opened": False,
        "region": args.region,
    }
    (args.output / "development_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    selection = {
        "selected_model": selected,
        "criterion": "development LME-macro RMSE",
        "hierarchical_alpha": alpha,
        "a3_stopping_rule_triggered": a3_trigger,
        "conformal_source": "five-fold cruise-grouped training OOF residuals",
        "conformal_absolute_error_q90": q90,
        "development_coverage_90": coverage,
        "locked_test_opened": False,
        "external_independent_opened": False,
        "region": args.region,
    }
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    protocol = {
        "experiment_id": experiment_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_commit": preregistration_commit,
        "run_git_commit": git_head(),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "manifest_validation": validation,
        "esper_mat": str(args.esper_mat),
        "esper_mat_sha256": sha256(args.esper_mat),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "checkpoint_every": args.checkpoint_every,
        "seeds": seeds,
        "train_rows": len(train),
        "development_rows": len(development),
        "train_cruises": int(train.group_key.nunique()),
        "development_cruises": int(development.group_key.nunique()),
        "region": args.region,
        "region_definition": BIGHT_DEFINITIONS.get(args.region),
        "safe_forward_train_rows": len(forward_fit),
        "safe_forward_development_rows": len(forward_held),
        "primary_salinity_input": "background sss",
        "oracle_salinity_input": "collocated in-situ salinity, diagnostic only",
        "locked_test_opened": False,
        "external_independent_opened": False,
        "elapsed_seconds": time.time() - start,
        "frozen_config": config,
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    hashes = {
        str(path.relative_to(args.output)).replace("\\", "/"): sha256(path)
        for path in args.output.rglob("*")
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (args.output / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
