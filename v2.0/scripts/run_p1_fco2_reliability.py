"""Run Issue #26 support-aware fCO2 development and selective-product gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from catboost import CatBoostRegressor
from torch import nn

from recad.evaluate.applicability import outer_splits
from recad.evaluate.fco2_reliability import (
    BinnedIntervalModel,
    apply_shrinkage,
    assign_reliability_grade,
    interval_coverage,
)
from recad.evaluate.p1_framework import FrozenManifest, P1DataGateway, Purpose

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_fco2_reliability_v2.2"
NUMERIC = [
    "sss_input",
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
MODEL_COLUMNS = ["grid_flat", "sst", "sss", "adt", "wspd", "pco2air", "xco2air"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[frame.truth.notna()].copy().reset_index(drop=True)
    result["longitude"] = np.mod(result.longitude.to_numpy(float), 360.0)
    radians = np.deg2rad(result.longitude.to_numpy(float))
    phase = 2.0 * np.pi * (result.month.to_numpy(float) - 1.0) / 12.0
    result["lon_sin"], result["lon_cos"] = np.sin(radians), np.cos(radians)
    result["month_sin"], result["month_cos"] = np.sin(phase), np.cos(phase)
    result["sss_input"] = result.sss
    result["record_id"] = (
        "fco2:"
        + result.group_key.astype(str)
        + ":"
        + result.year.astype(str)
        + ":"
        + result.month.astype(str)
        + ":"
        + result.grid_flat.astype(str)
    )
    return result


class SeasonalTrendClimatology:
    """Training-only LME-month climatology with a shared linear trend."""

    def fit(self, frame: pd.DataFrame) -> SeasonalTrendClimatology:
        self.center_year = float(frame.year.median())
        time = frame.year.to_numpy(float) - self.center_year
        truth = frame.truth.to_numpy(float)
        self.slope = float(np.cov(time, truth, ddof=0)[0, 1] / max(np.var(time), 1e-12))
        work = frame.assign(detrended=truth - self.slope * time)
        self.global_mean = float(work.detrended.mean())
        self.month = work.groupby("month").detrended.mean().to_dict()
        grouped = work.groupby(["lme_id", "month"]).detrended.agg(["mean", "count"])
        prior = np.array([self.month.get(i[1], self.global_mean) for i in grouped.index])
        weight = grouped["count"].to_numpy() / (grouped["count"].to_numpy() + 50.0)
        grouped["estimate"] = weight * grouped["mean"].to_numpy() + (1.0 - weight) * prior
        self.group = grouped.estimate.to_dict()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        seasonal = np.asarray(
            [
                self.group.get((r.lme_id, r.month), self.month.get(r.month, self.global_mean))
                for r in frame.itertuples()
            ]
        )
        return seasonal + self.slope * (frame.year.to_numpy(float) - self.center_year)


def balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    keys = ["lme_id", "group_key", "month"]
    n_regions = frame.lme_id.nunique()
    cruises = frame.groupby("lme_id").group_key.transform("nunique")
    months = frame.groupby(["lme_id", "group_key"])["month"].transform("nunique")
    records = frame.groupby(keys)["month"].transform("size")
    values = (1.0 / (n_regions * cruises * months * records)).to_numpy(float)
    return values / values.sum()


class TabularTransform:
    def fit(self, frame: pd.DataFrame) -> TabularTransform:
        values = frame[NUMERIC].to_numpy(float)
        self.median = np.nanmedian(values, axis=0)
        values = np.where(np.isfinite(values), values, self.median)
        self.mean, self.std = values.mean(axis=0), values.std(axis=0)
        self.std[self.std < 1e-7] = 1.0
        self.categories = []
        for column in CATEGORICAL:
            categories = sorted(frame[column].fillna(-999).astype(int).astype(str).unique())
            self.categories.append([*categories, "__unknown__"])
        return self

    def apply(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[NUMERIC].to_numpy(float)
        values = np.where(np.isfinite(values), values, self.median)
        parts = [(values - self.mean) / self.std]
        for column, categories in zip(CATEGORICAL, self.categories, strict=True):
            mapping = {value: index for index, value in enumerate(categories)}
            unknown = len(categories) - 1
            codes = np.fromiter(
                (
                    mapping.get(value, unknown)
                    for value in frame[column].fillna(-999).astype(int).astype(str)
                ),
                int,
            )
            one_hot = np.zeros((len(frame), len(categories)), dtype=np.float32)
            one_hot[np.arange(len(frame)), codes] = 1.0
            parts.append(one_hot)
        return np.concatenate(parts, axis=1).astype(np.float32)


class ResidualMLP(nn.Module):
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

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def train_predict_mlp(
    fit: pd.DataFrame,
    held: pd.DataFrame,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    loss_name: str,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = TabularTransform().fit(fit)
    x_fit = torch.as_tensor(transform.apply(fit), device=device)
    x_held = transform.apply(held)
    residual = fit.residual.to_numpy(np.float32)
    center, scale = float(np.mean(residual)), float(np.std(residual))
    scale = max(scale, 1e-6)
    y_fit = torch.as_tensor((residual - center) / scale, device=device)
    probabilities = torch.as_tensor(balanced_weights(fit).astype(np.float32), device=device)
    model = ResidualMLP(x_fit.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(steps):
        index = torch.multinomial(probabilities, batch_size, replacement=True)
        prediction = model(x_fit[index])
        if loss_name == "mse":
            loss = nn.functional.mse_loss(prediction, y_fit[index])
        else:
            loss = nn.functional.huber_loss(prediction, y_fit[index], delta=30.0 / scale)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(held), 131_072):
            batch = torch.as_tensor(x_held[start : start + 131_072], device=device)
            output.append(model(batch).cpu().numpy())
    standardized = np.concatenate(output)
    return held.background.to_numpy(float) + center + scale * standardized


def train_predict_catboost(
    fit: pd.DataFrame,
    held: pd.DataFrame,
    *,
    seed: int,
    iterations: int,
) -> np.ndarray:
    columns = NUMERIC + CATEGORICAL
    x_fit, x_held = fit[columns].copy(), held[columns].copy()
    for column in NUMERIC:
        median = float(x_fit[column].median())
        x_fit[column] = x_fit[column].fillna(median)
        x_held[column] = x_held[column].fillna(median)
    for column in CATEGORICAL:
        x_fit[column] = x_fit[column].fillna(-999).astype(int).astype(str)
        x_held[column] = x_held[column].fillna(-999).astype(int).astype(str)
    parameters = {
        "iterations": iterations,
        "depth": 8,
        "learning_rate": 0.05,
        "loss_function": "RMSE",
        "random_seed": seed,
        "random_strength": 0.5,
        "l2_leaf_reg": 5.0,
        "allow_writing_files": False,
        "verbose": False,
        "task_type": "GPU",
        "devices": "0",
    }
    model = CatBoostRegressor(**parameters)
    model.fit(
        x_fit,
        fit.residual,
        cat_features=[columns.index(column) for column in CATEGORICAL],
        sample_weight=balanced_weights(fit) * len(fit),
        verbose=False,
    )
    return held.background.to_numpy(float) + model.predict(x_held)


def join_support(frame: pd.DataFrame, source: Path, scheme: str, fold: int) -> pd.DataFrame:
    columns = [
        "record_id",
        "environment_k64",
        "geographic_km",
        "unique_cruises",
        "effective_groups",
        "lme_month_count",
    ]
    support = pd.read_parquet(
        source,
        filters=[
            ("target", "==", "fco2"),
            ("outer_scheme", "==", scheme),
            ("outer_fold", "==", fold),
        ],
        columns=columns,
    )
    result = frame.merge(support, on="record_id", how="left", validate="one_to_one")
    if result.environment_k64.isna().any():
        raise RuntimeError(f"Issue #24 support join failed for {scheme}/{fold}")
    result["risk_environment_k64"] = result.environment_k64
    return result


def add_crossfit_sss(frame: pd.DataFrame, path: Path, scheme: str) -> pd.DataFrame:
    sss = pd.read_parquet(
        path,
        filters=[("outer_scheme", "==", scheme)],
        columns=["record_id", "prediction"],
    ).drop_duplicates("record_id")
    sss["record_id"] = "fco2:" + sss.record_id.str.removeprefix("sss:")
    sss = sss.rename(columns={"prediction": "sss_oof"})
    result = frame.merge(sss, on="record_id", how="left", validate="one_to_one")
    result["sss_input"] = result.sss_oof.fillna(result.sss)
    result["sss_oof_available"] = result.sss_oof.notna()
    return result


def metric_values(frame: pd.DataFrame, prediction: str) -> dict[str, float | int]:
    error = frame[prediction].to_numpy(float) - frame.truth.to_numpy(float)
    background_error = frame.background.to_numpy(float) - frame.truth.to_numpy(float)
    denominator = np.mean(background_error**2)
    return {
        "n": len(frame),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "skill_vs_background": float(1.0 - np.mean(error**2) / denominator),
        "background_rmse": float(np.sqrt(denominator)),
    }


def run_outer(config: dict[str, object], gateway: P1DataGateway, output: Path) -> pd.DataFrame:
    cached = output / "model_outer_oof.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    train = prepare(gateway.load_labels("fco2", Purpose.TRAIN, columns=MODEL_COLUMNS))
    development = prepare(gateway.load_labels("fco2", Purpose.SELECTION, columns=MODEL_COLUMNS))
    support_path = Path(config["applicability_output"]) / "outer_predictions.parquet"
    sss_path = Path(config["sss_reliability_output"]) / "development_oof_predictions.parquet"
    parts = []
    for scheme in config["outer_schemes"]:
        source = (
            pd.concat([train, development], ignore_index=True) if scheme == "forward" else train
        )
        source_sss = add_crossfit_sss(source, sss_path, str(scheme))
        for split in outer_splits(source, str(scheme), n_folds=int(config["outer_folds"])):
            completed = output / "partitions" / f"{scheme}_{split.fold}.parquet"
            if completed.exists():
                parts.append(pd.read_parquet(completed))
                continue
            fit = source.iloc[split.fit_index].copy().reset_index(drop=True)
            held = source.iloc[split.held_index].copy().reset_index(drop=True)
            fit_sss = source_sss.iloc[split.fit_index].copy().reset_index(drop=True)
            held_sss = source_sss.iloc[split.held_index].copy().reset_index(drop=True)
            background = SeasonalTrendClimatology().fit(fit)
            for frame in (fit, held, fit_sss, held_sss):
                frame["background"] = background.predict(frame)
                frame["residual"] = frame.truth - frame.background
            print(f"OUTER {scheme}/{split.fold} fit={len(fit):,} held={len(held):,}", flush=True)
            candidate_predictions = {
                "catboost": train_predict_catboost(
                    fit,
                    held,
                    seed=int(config["models"]["oof_seed"]),
                    iterations=int(config["models"]["catboost_iterations"]),
                ),
                "mlp_mse_standardized": train_predict_mlp(
                    fit,
                    held,
                    seed=int(config["models"]["oof_seed"]),
                    steps=int(config["models"]["neural_steps"]),
                    batch_size=int(config["models"]["batch_size"]),
                    loss_name="mse",
                ),
                "mlp_huber_30uatm": train_predict_mlp(
                    fit,
                    held,
                    seed=int(config["models"]["oof_seed"]),
                    steps=int(config["models"]["neural_steps"]),
                    batch_size=int(config["models"]["batch_size"]),
                    loss_name="huber",
                ),
                "catboost_sss_oof": train_predict_catboost(
                    fit_sss,
                    held_sss,
                    seed=int(config["models"]["oof_seed"]),
                    iterations=int(config["models"]["catboost_iterations"]),
                ),
            }
            meta = held[
                [
                    "record_id",
                    "group_key",
                    "year",
                    "month",
                    "latitude",
                    "longitude",
                    "lme_id",
                    "basin_id",
                    "regime_id",
                    "truth",
                    "background",
                ]
            ].copy()
            meta = join_support(meta, support_path, str(scheme), split.fold)
            fold_parts = []
            for candidate, prediction in candidate_predictions.items():
                block = meta.copy()
                block.insert(0, "candidate", candidate)
                block.insert(1, "outer_scheme", scheme)
                block.insert(2, "outer_fold", split.fold)
                block["raw_prediction"] = prediction
                fold_parts.append(block)
            result = pd.concat(fold_parts, ignore_index=True)
            completed.parent.mkdir(parents=True, exist_ok=True)
            result.to_parquet(completed, index=False)
            parts.append(result)
    result = pd.concat(parts, ignore_index=True)
    result.to_parquet(cached, index=False)
    return result


def select_candidate(
    predictions: pd.DataFrame, config: dict[str, object]
) -> tuple[str, str, pd.DataFrame]:
    rows = []
    for candidate in config["models"]["candidates"]:
        part = predictions.loc[predictions.candidate.eq(candidate)].copy()
        for shrinkage in config["shrinkage_candidates"]:
            part["prediction"] = apply_shrinkage(
                part.background, part.raw_prediction, part.environment_k64, str(shrinkage)
            )
            for scheme, group in part.groupby("outer_scheme"):
                rows.append(
                    {
                        "candidate": candidate,
                        "shrinkage": shrinkage,
                        "outer_scheme": scheme,
                        **metric_values(group, "prediction"),
                    }
                )
    table = pd.DataFrame(rows)
    table["normalized_rmse"] = table.rmse / table.background_rmse
    ranking = table.groupby(["candidate", "shrinkage"]).normalized_rmse.mean().sort_values()
    selected_candidate, selected_shrinkage = ranking.index[0]
    return str(selected_candidate), str(selected_shrinkage), table


def calibrate(predictions: pd.DataFrame, *, minimum_cell: int) -> pd.DataFrame:
    result = predictions.copy()
    result["absolute_error"] = np.abs(result.prediction - result.truth)
    widths = []
    for (scheme, fold), evaluation in result.groupby(["outer_scheme", "outer_fold"]):
        if scheme == "forward":
            calibration = result.loc[result.outer_scheme.eq("cruise")]
        else:
            calibration = result.loc[result.outer_scheme.eq(scheme) & result.outer_fold.ne(fold)]
        model = BinnedIntervalModel.fit(calibration, minimum_cell=minimum_cell)
        estimate = model.predict(evaluation)
        estimate.index = evaluation.index
        widths.append(estimate)
    estimated = pd.concat(widths).sort_index()
    for column in estimated:
        result[column] = estimated[column]
    result["grade"] = assign_reliability_grade(result)
    return result


def evaluate_gate(
    frame: pd.DataFrame, config: dict[str, object]
) -> tuple[dict[str, object], pd.DataFrame]:
    gate = config["development_gate"]
    publishable = frame.loc[frame.grade.astype(str).isin(gate["publishable_grades"])].copy()
    rows = []
    for scheme, part in publishable.groupby("outer_scheme"):
        values = metric_values(part, "prediction")
        lme_rows = []
        for lme, region in part.groupby("lme_id"):
            if len(region) >= 100:
                score = metric_values(region, "prediction")
                lme_rows.append({"lme_id": lme, **score})
        lmes = pd.DataFrame(lme_rows)
        positive = float((lmes.skill_vs_background > 0).mean()) if len(lmes) else 0.0
        worst_ratio = float((lmes.rmse / lmes.background_rmse).max()) if len(lmes) else math.inf
        rows.append(
            {
                "outer_scheme": scheme,
                **values,
                "retained_fraction": len(part) / len(frame.loc[frame.outer_scheme.eq(scheme)]),
                "coverage50": interval_coverage(part, "width50") if len(part) else math.nan,
                "coverage90": interval_coverage(part, "width90") if len(part) else math.nan,
                "eligible_lmes": len(lmes),
                "positive_lme_fraction": positive,
                "worst_lme_relative_rmse": worst_ratio,
            }
        )
    table = pd.DataFrame(rows)
    required = set(gate["required_schemes"])
    checks = {
        "all_schemes_present": set(table.outer_scheme) == required,
        "skill_positive": bool((table.skill_vs_background > gate["skill_vs_background_gt"]).all()),
        "mae": bool((table.mae <= gate["grade_ab_mae_max_uatm"]).all()),
        "coverage50": bool(table.coverage50.between(*gate["coverage50_range"]).all()),
        "coverage90": bool(table.coverage90.between(*gate["coverage90_range"]).all()),
        "positive_lme_fraction": bool(
            (table.positive_lme_fraction >= gate["positive_lme_fraction_minimum"]).all()
        ),
        "worst_lme": bool(
            (table.worst_lme_relative_rmse <= gate["worst_lme_relative_rmse_maximum"]).all()
        ),
        "retained_fraction": bool(
            (table.retained_fraction >= gate["minimum_retained_fraction"]).all()
        ),
    }
    passed = all(checks.values())
    return {
        "status": "pass_regional" if passed else "diagnostic_only",
        "passed": passed,
        "checks": checks,
        "locked_test_opened": False,
        "external_independent_opened": False,
    }, table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/p1_fco2_reliability_v2.2.yaml"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / f"outputs/experiments/{EXPERIMENT_ID}"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["locked_test_access"] != "forbidden_for_issue_26":
        raise ValueError("Issue #26 must keep locked fCO2 labels sealed")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "configs/frozen/data_manifest_v2.2.json"
    manifest = FrozenManifest.load(manifest_path)
    validation = manifest.validate(hash_mode="full")
    gateway = P1DataGateway(manifest)
    predictions = run_outer(config, gateway, args.output)
    if "risk_environment_k64" not in predictions:
        predictions["risk_environment_k64"] = predictions.environment_k64
    candidate, shrinkage, comparison = select_candidate(predictions, config)
    selected = predictions.loc[predictions.candidate.eq(candidate)].copy()
    selected["prediction"] = apply_shrinkage(
        selected.background, selected.raw_prediction, selected.environment_k64, shrinkage
    )
    calibrated = calibrate(selected, minimum_cell=int(config["intervals"]["minimum_cell_records"]))
    decision, gate = evaluate_gate(calibrated, config)
    calibrated.to_parquet(args.output / "development_oof_predictions.parquet", index=False)
    comparison.to_csv(args.output / "candidate_shrinkage_comparison.csv", index=False)
    gate.to_csv(args.output / "development_gate_by_scheme.csv", index=False)
    selection = {"selected_candidate": candidate, "selected_shrinkage": shrinkage, **decision}
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    protocol = {
        "git_commit": git_head(),
        "config_sha256": sha256(args.config),
        "manifest_sha256": sha256(manifest_path),
        "manifest_validation": validation,
        "selected_candidate": candidate,
        "selected_shrinkage": shrinkage,
        "row_count": len(calibrated),
        "locked_test_opened": False,
        "external_independent_opened": False,
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
