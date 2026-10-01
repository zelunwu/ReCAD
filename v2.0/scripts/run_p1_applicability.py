"""Run Issue #24 nested OOF applicability analysis and build its reviewer archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.spatial import cKDTree

from recad.evaluate.applicability import (
    CoastalGraphIndex,
    ReasonBit,
    SupportIndex,
    _chord_to_km,
    _unit_sphere,
    add_label_free_risk_scores,
    nested_grouped_ridge,
    outer_splits,
    reliability_schema,
    risk_coverage_curve,
    support_error_summary,
    validate_reliability_frame,
)
from recad.evaluate.p1_framework import FrozenManifest, P1DataGateway, Purpose

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_applicability_v2.2"
ARCHIVE = ROOT / "docs" / "experiment_archive" / EXPERIMENT_ID
RISK_COLUMNS = [
    "risk_geographic",
    "risk_region",
    "risk_graph",
    "risk_environment_k8",
    "risk_environment_k16",
    "risk_environment_k32",
    "risk_environment_k64",
    "risk_hybrid",
]
SIMPLE_METHODS = {"geographic", "region"}
COMPLEX_METHODS = {
    "graph",
    "environment_k8",
    "environment_k16",
    "environment_k32",
    "environment_k64",
    "hybrid",
}


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    longitude = np.mod(result.longitude.to_numpy(float), 360.0)
    result["longitude"] = longitude
    result["latitude_sin"] = np.sin(np.deg2rad(result.latitude))
    result["longitude_sin"] = np.sin(np.deg2rad(longitude))
    result["longitude_cos"] = np.cos(np.deg2rad(longitude))
    result["month_sin"] = np.sin(2.0 * np.pi * (result.month - 1) / 12.0)
    result["month_cos"] = np.cos(2.0 * np.pi * (result.month - 1) / 12.0)
    if "grid_flat" in result:
        identity = result.grid_flat.astype(str)
    elif "obs_id" in result:
        identity = result.obs_id.astype(str)
    else:
        identity = result.index.astype(str)
    result["record_id"] = (
        target
        + ":"
        + result.group_key.astype(str)
        + ":"
        + result.year.astype(str)
        + ":"
        + result.month.astype(str)
        + ":"
        + identity
    )
    return result


def load_target_frames(gateway: P1DataGateway, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["grid_flat", "coastal_node", "sst", "sss", "adt", "wspd", "pco2air"]
    if target in {"ta", "dic"}:
        columns.append("obs_id")
    train = prepare(gateway.load_labels(target, Purpose.TRAIN, columns=columns), target)
    development = prepare(gateway.load_labels(target, Purpose.SELECTION, columns=columns), target)
    return train, development


def run_outer_predictions(
    gateway: P1DataGateway,
    graph: CoastalGraphIndex,
    config: dict[str, object],
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_parts: list[pd.DataFrame] = []
    inner_parts: list[pd.DataFrame] = []
    numeric = list(config["numeric_predictors"])
    categorical = list(config["categorical_predictors"])
    environmental = list(config["environmental_predictors"])
    for target in config["targets"]:
        train, development = load_target_frames(gateway, str(target))
        for scheme in config["outer_schemes"]:
            source = (
                pd.concat([train, development], ignore_index=True)
                if scheme == "forward"
                else train.copy()
            )
            for split in outer_splits(source, str(scheme), n_folds=int(config["outer_folds"])):
                fit = source.iloc[split.fit_index].reset_index(drop=True)
                held = source.iloc[split.held_index].reset_index(drop=True)
                baseline = "sss" if target == "sss" else None
                model_result = nested_grouped_ridge(
                    fit,
                    held.drop(columns=["truth"]),
                    numeric=numeric,
                    categorical=categorical,
                    alphas=config["ridge_alphas"],
                    baseline_column=baseline,
                    inner_splits=int(config["inner_group_folds"]),
                )
                selected = model_result.inner_scores.copy()
                selected.insert(0, "target", target)
                selected.insert(1, "outer_scheme", scheme)
                selected.insert(2, "outer_fold", split.fold)
                selected["selected_alpha"] = model_result.selected_alpha
                inner_parts.append(selected)

                support_index = SupportIndex(
                    environmental_columns=environmental,
                    k_values=config["k_values"],
                    radii_km=config["radii_km"],
                ).fit(fit)
                support = support_index.query(held)
                graph_distance = graph.distance_to_sources(fit.coastal_node.dropna().astype(int))
                held_nodes = held.coastal_node.fillna(-1).astype(int).to_numpy()
                connected = np.full(len(held), np.inf)
                valid_node = (held_nodes >= 0) & (held_nodes < len(graph_distance))
                connected[valid_node] = graph_distance[held_nodes[valid_node]]
                support["water_connected_km"] = connected
                support = add_label_free_risk_scores(support)

                frozen = held[
                    [
                        "record_id",
                        "group_key",
                        "year",
                        "month",
                        "latitude",
                        "longitude",
                        "coastal_node",
                        "lme_id",
                        "basin_id",
                        "regime_id",
                        "truth",
                    ]
                ].reset_index(drop=True)
                frozen.insert(0, "target", target)
                frozen.insert(1, "outer_scheme", scheme)
                frozen.insert(2, "outer_fold", split.fold)
                frozen["prediction"] = model_result.prediction
                frozen["selected_alpha"] = model_result.selected_alpha
                frozen = pd.concat([frozen, support.reset_index(drop=True)], axis=1)
                frozen["absolute_error"] = np.abs(frozen.prediction - frozen.truth)
                prediction_parts.append(frozen)
                print(
                    f"{target} {scheme} fold={split.fold}: fit={len(fit):,} held={len(held):,} "
                    f"alpha={model_result.selected_alpha:g}",
                    flush=True,
                )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    inner_scores = pd.concat(inner_parts, ignore_index=True)
    predictions.to_parquet(output / "outer_predictions.parquet", index=False)
    inner_scores.to_csv(output / "nested_selection.csv", index=False)
    return predictions, inner_scores


def evaluate_support(
    predictions: pd.DataFrame, config: dict[str, object], output: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_parts: list[pd.DataFrame] = []
    curve_parts: list[pd.DataFrame] = []
    for (target, scheme), part in predictions.groupby(["target", "outer_scheme"], sort=True):
        metrics = support_error_summary(part, RISK_COLUMNS)
        metrics.insert(0, "target", target)
        metrics.insert(1, "outer_scheme", scheme)
        metric_parts.append(metrics)
        for risk_column in RISK_COLUMNS:
            curve = risk_coverage_curve(part.truth, part.prediction, part[risk_column])
            curve.insert(0, "target", target)
            curve.insert(1, "outer_scheme", scheme)
            curve.insert(2, "method", risk_column.removeprefix("risk_"))
            curve_parts.append(curve)
    metrics = pd.concat(metric_parts, ignore_index=True)
    curves = pd.concat(curve_parts, ignore_index=True)
    decisions = []
    margin = float(config["complex_method_margin_relative_aurc"])
    monotonic_gate = float(config["monotonic_steps_gate"])
    aggregate = metrics.groupby(["target", "method"], as_index=False).agg(
        schemes=("outer_scheme", "nunique"),
        mean_aurc=("risk_curve_auc", "mean"),
        mean_spearman=("spearman_abs_error", "mean"),
        min_monotonic_steps=("monotonic_steps", "min"),
    )
    for target, part in aggregate.groupby("target"):
        simple = part.loc[part.method.isin(SIMPLE_METHODS)].sort_values("mean_aurc").iloc[0]
        complex_row = part.loc[part.method.isin(COMPLEX_METHODS)].sort_values("mean_aurc").iloc[0]
        improvement = 1.0 - float(complex_row.mean_aurc) / float(simple.mean_aurc)
        retain_complex = bool(
            improvement >= margin and complex_row.min_monotonic_steps >= monotonic_gate
        )
        retained = complex_row if retain_complex else simple
        target_monotonic = bool((part.min_monotonic_steps >= monotonic_gate).any())
        decisions.append(
            {
                "target": target,
                "best_simple": simple.method,
                "best_simple_mean_aurc": simple.mean_aurc,
                "best_complex": complex_row.method,
                "best_complex_mean_aurc": complex_row.mean_aurc,
                "complex_relative_improvement": improvement,
                "complex_margin_pass": retain_complex,
                "retained_method": retained.method,
                "retained_mean_aurc": retained.mean_aurc,
                "monotonic_method_exists": target_monotonic,
            }
        )
    decision_frame = pd.DataFrame(decisions)
    metrics.to_csv(output / "support_method_metrics.csv", index=False)
    curves.to_csv(output / "risk_coverage.csv", index=False)
    decision_frame.to_csv(output / "decisions.csv", index=False)
    return metrics, curves, decision_frame


def prepared_environment(
    dataset: xr.Dataset,
    support: xr.Dataset,
    year: int,
    month: int,
    nodes: np.ndarray,
) -> pd.DataFrame:
    lat_index = support.lat_index.to_numpy()[nodes]
    lon_index = support.lon_index.to_numpy()[nodes]
    data: dict[str, np.ndarray] = {}
    for name in ("sst", "sss", "adt", "wspd", "pco2air"):
        plane = dataset[name].sel(year=year, month=month).load().to_numpy()
        data[name] = plane[lat_index, lon_index]
    latitude = support.latitude.to_numpy()[nodes]
    longitude = np.mod(support.longitude.to_numpy()[nodes], 360.0)
    data.update(
        latitude=latitude,
        longitude=longitude,
        latitude_sin=np.sin(np.deg2rad(latitude)),
        longitude_sin=np.sin(np.deg2rad(longitude)),
        longitude_cos=np.cos(np.deg2rad(longitude)),
        month_sin=np.full(len(nodes), np.sin(2.0 * np.pi * (month - 1) / 12.0)),
        month_cos=np.full(len(nodes), np.cos(2.0 * np.pi * (month - 1) / 12.0)),
        year=np.full(len(nodes), year),
        month=np.full(len(nodes), month),
        lme_id=support.lme_id.to_numpy()[nodes],
        basin_id=support.basin_id.to_numpy()[nodes],
        regime_id=support.regime_id.to_numpy()[nodes],
    )
    return pd.DataFrame(data)


def nearest_geographic_km(tree: cKDTree, frame: pd.DataFrame) -> np.ndarray:
    chord, _ = tree.query(_unit_sphere(frame.latitude, frame.longitude), k=1, workers=-1)
    return _chord_to_km(chord)


def grid_support_indices(
    gateway: P1DataGateway,
    graph: CoastalGraphIndex,
    config: dict[str, object],
    predictions: pd.DataFrame,
    decisions: pd.DataFrame,
    prepared_path: Path,
    output: Path,
) -> pd.DataFrame:
    frozen_root = gateway.manifest.data_dir
    grid_root = output / "grid_month_support"
    grid_root.mkdir(parents=True, exist_ok=True)
    with (
        xr.open_dataset(frozen_root / "spatial_support_v2.2.nc") as support_ds,
        xr.open_dataset(frozen_root / "inference_availability_v2.2.nc") as availability,
        xr.open_dataset(prepared_path) as prepared,
    ):
        support = support_ds.load()
        available = availability.strict_all_inputs_ready.load()
        grade_rows = []
        all_nodes = np.arange(support.sizes["node"])
        for target in config["targets"]:
            target = str(target)
            train, calibration = load_target_frames(gateway, target)
            support_index = SupportIndex(
                environmental_columns=config["environmental_predictors"],
                k_values=config["k_values"],
                radii_km=config["radii_km"],
            ).fit(train)
            calibration_tree = cKDTree(_unit_sphere(calibration.latitude, calibration.longitude))
            graph_distance = graph.distance_to_sources(train.coastal_node.dropna().astype(int))

            # Geographic/diversity terms are spatial and are computed once per target.
            reference = prepared_environment(prepared, support, 2025, 1, all_nodes)
            spatial_terms = support_index.query(reference)
            spatial_terms["water_connected_km"] = graph_distance
            selected_method = str(
                decisions.loc[decisions.target.eq(target), "retained_method"].iloc[0]
            )
            oof_risk = predictions.loc[
                predictions.target.eq(target), f"risk_{selected_method}"
            ].dropna()
            q50, q75, q90 = oof_risk.quantile([0.5, 0.75, 0.9]).to_numpy(float)

            for year in config["grid_years"]:
                for month in range(1, 13):
                    mask = available.sel(year=int(year), month=month).to_numpy().astype(bool)
                    nodes = np.flatnonzero(mask)
                    if not len(nodes):
                        continue
                    query = prepared_environment(prepared, support, int(year), month, nodes)
                    terms = spatial_terms.iloc[nodes].reset_index(drop=True).copy()
                    environment = query[list(config["environmental_predictors"])].to_numpy(float)
                    missing = ~np.isfinite(environment).all(axis=1)
                    environment = np.where(
                        np.isfinite(environment), environment, support_index.environment_median
                    )
                    standardized = (
                        environment - support_index.environment_mean
                    ) / support_index.environment_scale
                    max_k = min(max(config["k_values"]), len(train))
                    environmental_distance, _ = support_index.environment_tree.query(
                        standardized, k=max_k, workers=-1
                    )
                    if max_k == 1:
                        environmental_distance = environmental_distance[:, None]
                    for k in config["k_values"]:
                        terms[f"environment_k{k}"] = environmental_distance[
                            :, : min(int(k), max_k)
                        ].mean(axis=1)
                    terms["missing_predictor"] = missing
                    for name, columns in {
                        "lme_month": ["lme_id", "month"],
                        "basin_month": ["basin_id", "month"],
                        "regime_month": ["regime_id", "month"],
                    }.items():
                        mapping = support_index.region_counts[name]
                        keys = zip(
                            *(query[column].fillna(-9999).astype(int) for column in columns),
                            strict=True,
                        )
                        terms[f"{name}_count"] = [mapping.get(tuple(key), 0) for key in keys]
                    terms = add_label_free_risk_scores(terms)
                    risk = terms[f"risk_{selected_method}"].to_numpy(float)
                    grade = np.select(
                        [risk <= q50, risk <= q75, risk <= q90], ["A", "B", "C"], default="D"
                    )
                    calibration_km = nearest_geographic_km(calibration_tree, query)
                    reason = np.full(
                        len(nodes), int(ReasonBit.INDEPENDENT_SUPPORT_SEALED), dtype=np.int32
                    )
                    reason |= np.where(
                        terms.geographic_km.to_numpy() > 200,
                        int(ReasonBit.NO_TRAINING_WITHIN_200KM),
                        0,
                    ).astype(np.int32)
                    reason |= np.where(
                        terms.lme_month_count.to_numpy() == 0,
                        int(ReasonBit.UNSEEN_LME),
                        0,
                    ).astype(np.int32)
                    reason |= np.where(
                        terms.regime_month_count.to_numpy() == 0,
                        int(ReasonBit.UNSEEN_REGIME),
                        0,
                    ).astype(np.int32)
                    reason |= np.where(
                        ~np.isfinite(terms.water_connected_km.to_numpy()),
                        int(ReasonBit.DISCONNECTED_COASTAL_GRAPH),
                        0,
                    ).astype(np.int32)
                    reason |= np.where(
                        terms.environment_k16.to_numpy()
                        > np.nanquantile(
                            predictions.loc[predictions.target.eq(target), "environment_k16"], 0.9
                        ),
                        int(ReasonBit.ENVIRONMENTAL_EXTRAPOLATION),
                        0,
                    ).astype(np.int32)
                    reason |= np.where(
                        terms.effective_groups.to_numpy() < 2,
                        int(ReasonBit.LOW_GROUP_DIVERSITY),
                        0,
                    ).astype(np.int32)
                    reason |= np.where(missing, int(ReasonBit.MISSING_PREDICTOR), 0).astype(
                        np.int32
                    )
                    reason |= np.where(
                        calibration_km > 200,
                        int(ReasonBit.CALIBRATION_SUPPORT_SPARSE),
                        0,
                    ).astype(np.int32)
                    frame = pd.DataFrame(
                        {
                            "target": target,
                            "year": int(year),
                            "month": month,
                            "coastal_node": nodes,
                            "latitude": query.latitude,
                            "longitude": query.longitude,
                            "training_support_km": terms.geographic_km,
                            "calibration_support_km": calibration_km,
                            "independent_support_km": np.nan,
                            "water_connected_support_km": terms.water_connected_km,
                            "environmental_dissimilarity": terms.environment_k16,
                            "unique_cruises": terms.unique_cruises,
                            "effective_groups": terms.effective_groups,
                            "sampled_years": terms.sampled_years,
                            "sampled_months": terms.sampled_months,
                            "support_entropy": terms.support_entropy,
                            "applicability_flag": grade,
                            "reason_bits": reason,
                            "model_version": "nested_ridge_probe_v1",
                            "calibration_version": "issue24_oof_v1",
                            "evidence_stage": "development",
                        }
                    )
                    validate_reliability_frame(frame)
                    partition = grid_root / f"target={target}" / f"year={year}"
                    partition.mkdir(parents=True, exist_ok=True)
                    frame.to_parquet(partition / f"month={month:02d}.parquet", index=False)
                    for flag, count in frame.applicability_flag.value_counts().items():
                        grade_rows.append(
                            {
                                "target": target,
                                "year": year,
                                "month": month,
                                "grade": flag,
                                "grid_months": int(count),
                                "retained_method": selected_method,
                            }
                        )
                    print(
                        f"grid {target} {year}-{month:02d}: {len(frame):,} available nodes",
                        flush=True,
                    )
    grade_frame = pd.DataFrame(grade_rows)
    grade_frame.to_csv(output / "grid_grade_counts.csv", index=False)
    (output / "reliability_schema.json").write_text(
        json.dumps(reliability_schema(), indent=2), encoding="utf-8"
    )
    return grade_frame


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_archive(
    output: Path,
    config: dict[str, object],
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    decisions: pd.DataFrame,
    inner: pd.DataFrame,
    grades: pd.DataFrame,
    training_commit: str,
) -> None:
    if ARCHIVE.exists():
        shutil.rmtree(ARCHIVE)
    figures = ARCHIVE / "figures"
    tables = ARCHIVE / "tables"
    figures.mkdir(parents=True)
    tables.mkdir(parents=True)
    scope = predictions.groupby(["target", "outer_scheme"], as_index=False).agg(
        records=("record_id", "size"),
        cruises=("group_key", "nunique"),
        folds=("outer_fold", "nunique"),
    )
    table_map = {
        "table01_outer_scope.csv": scope,
        "table02_support_method_metrics.csv": metrics,
        "table03_risk_coverage.csv": curves,
        "table04_decisions.csv": decisions,
        "table05_grid_grade_counts.csv": grades,
        "table06_nested_selection.csv": inner,
    }
    for name, frame in table_map.items():
        frame.to_csv(tables / name, index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = scope.pivot(index="target", columns="outer_scheme", values="records")
    pivot.plot.bar(ax=ax)
    ax.set_ylabel("Untouched outer predictions")
    ax.set_xlabel("Target")
    ax.set_yscale("log")
    save_figure(fig, figures / "fig01_outer_scope.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    aggregate = metrics.groupby(["target", "method"], as_index=False).risk_curve_auc.mean()
    for target, part in aggregate.groupby("target"):
        ax.plot(part.method, part.risk_curve_auc, marker="o", label=target)
    ax.set_ylabel("Mean normalized selective-risk AUC (lower is better)")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    save_figure(fig, figures / "fig02_method_aurc.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    association = metrics.groupby(["target", "method"], as_index=False).spearman_abs_error.mean()
    for target, part in association.groupby("target"):
        ax.plot(part.method, part.spearman_abs_error, marker="o", label=target)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Spearman association with absolute error")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    save_figure(fig, figures / "fig03_error_association.png")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for ax, target in zip(axes.ravel(), sorted(decisions.target), strict=True):
        method = decisions.loc[decisions.target.eq(target), "retained_method"].iloc[0]
        part = curves.loc[(curves.target.eq(target)) & curves.method.eq(method)]
        summary = part.groupby("coverage", as_index=False).rmse.mean()
        ax.plot(summary.coverage, summary.rmse, marker="o")
        ax.set_title(f"{target}: {method}")
        ax.set_ylabel("RMSE")
        ax.set_xlabel("Retained coverage")
    save_figure(fig, figures / "fig04_retained_risk_coverage.png")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for ax, target in zip(axes.ravel(), sorted(decisions.target), strict=True):
        method = decisions.loc[decisions.target.eq(target), "retained_method"].iloc[0]
        row = metrics.loc[(metrics.target.eq(target)) & metrics.method.eq(method)]
        values = [row[f"q{i}_rmse"].mean() for i in range(1, 6)]
        ax.plot(range(1, 6), values, marker="o")
        ax.set_title(f"{target}: {method}")
        ax.set_ylabel("Outer RMSE")
        ax.set_xlabel("Support-risk quintile")
    save_figure(fig, figures / "fig05_error_quintiles.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    grade_summary = grades.groupby(["target", "grade"]).grid_months.sum().unstack(fill_value=0)
    grade_fraction = grade_summary.div(grade_summary.sum(axis=1), axis=0)
    grade_fraction.reindex(columns=["A", "B", "C", "D"], fill_value=0).plot.bar(
        stacked=True, ax=ax, color=["#2ca25f", "#99d8c9", "#fdae6b", "#de2d26"]
    )
    ax.set_ylabel("Fraction of available 2025-2026 grid-months")
    ax.set_xlabel("Target")
    save_figure(fig, figures / "fig06_grid_reliability_grades.png")

    captions = {
        "fig01_outer_scope.png": "Figure 1. Untouched outer-prediction counts for cruise, fixed spatial-block, complete-LME, and forward-time evaluation. The logarithmic axis shows the much smaller TA and DIC anchor sets without treating record count as independent evidence.",
        "fig02_method_aurc.png": "Figure 2. Mean normalized selective-risk area under the risk-coverage curve for each label-free support method. Lower values mean that low-risk records retain lower error as mapped coverage expands.",
        "fig03_error_association.png": "Figure 3. Mean Spearman association between each support-risk score and untouched outer absolute error. Positive association is desirable but is interpreted together with monotonic strata and selective-risk area.",
        "fig04_retained_risk_coverage.png": "Figure 4. Outer RMSE versus retained coverage for the method retained by the preregistered simple-versus-complex rule. Each line averages evaluation schemes rather than mixing their records.",
        "fig05_error_quintiles.png": "Figure 5. Untouched outer RMSE from the lowest to highest support-risk quintile for each retained method. Monotonic increases provide the operational evidence needed for A/B/C/D grading.",
        "fig06_grid_reliability_grades.png": "Figure 6. Development-stage A/B/C/D support grades across strict-input-ready 2025-2026 coastal grid-months. These are applicability grades, not independent accuracy validation or permission to publish a target product.",
    }
    source_map = {
        "fig01_outer_scope.png": ["table01_outer_scope.csv"],
        "fig02_method_aurc.png": ["table02_support_method_metrics.csv"],
        "fig03_error_association.png": ["table02_support_method_metrics.csv"],
        "fig04_retained_risk_coverage.png": ["table03_risk_coverage.csv", "table04_decisions.csv"],
        "fig05_error_quintiles.png": [
            "table02_support_method_metrics.csv",
            "table04_decisions.csv",
        ],
        "fig06_grid_reliability_grades.png": ["table05_grid_grade_counts.csv"],
    }
    decision_lines = []
    for row in decisions.itertuples():
        outcome = (
            "retained complex method" if row.complex_margin_pass else "rejected complex method"
        )
        decision_lines.append(
            f"- **{row.target}:** retained `{row.retained_method}`; {outcome}. Best complex versus simple AURC change: {row.complex_relative_improvement:+.1%}. Monotonic method exists: {row.monotonic_method_exists}."
        )
    report = f"""# Quantitative applicability infrastructure for ReCAD P1.5

## Scientific question and permitted claim

Issue #24 asks whether label-free geographic, regional, water-connected, or environmental support can predict untouched outer-fold error for SSS, fCO2, TA, and DIC. The permitted claim is an applicability-method decision and a development-stage grid-month support index. It is not a final product validation result.

## Data, splits, and leakage controls

The run used only labels exposed by the frozen v2.2 train/development gateway. Locked-test and external-independent labels remained sealed. Cruise folds, fixed 5-degree spatial blocks, complete LME assignments, and the frozen forward split were evaluated separately. For every outer partition, ridge alpha was selected by three-fold cruise GroupKFold inside the outer-training rows. Outer labels were removed from the prediction call and were consulted only after predictions and support indices were frozen.

Training, calibration, and independent support have separate fields. The independent-support field is null and carries the `INDEPENDENT_SUPPORT_SEALED` reason bit until P3. Water-connected distance uses the frozen 148,795-node coastal graph and cannot cross disconnected land barriers. Environmental scaling and nearest-neighbour trees are fitted from the corresponding outer-training rows only.

## Candidate models and training

The prediction model is a deterministic nested Ridge residual probe, used to create calibration-safe residuals rather than to choose the final SSS/fCO2/TA architecture. SSS is fitted as a residual over background SSS; the other targets are direct probes. Candidate alpha values are {config["ridge_alphas"]}. Applicability candidates are geographic distance, LME/basin/regime-month support, coastal-graph distance, standardized environmental k-neighbour distance for k={config["k_values"]}, and a label-free rank hybrid. The complex-method gate requires at least {float(config["complex_method_margin_relative_aurc"]):.1%} lower normalized AURC than the best simple method and monotonic-step score of at least {float(config["monotonic_steps_gate"]):.2f}.

## Main development results

{chr(10).join(decision_lines)}

The output contains {len(predictions):,} untouched outer predictions and {int(grades.grid_months.sum()):,} target-specific strict-input-ready grid-month support records for 2025-2026. Exact values by target, scheme, method, and grade are in Tables 1-6.

## Decision and limitations

The Issue #24 infrastructure gate {"passes" if decisions.monotonic_method_exists.all() else "fails"}: every target {"has" if decisions.monotonic_method_exists.all() else "does not have"} at least one method with the preregistered monotonic outer-error behavior. Complex graph/environment/hybrid metrics are retained target by target only where their AURC margin passes; otherwise the simpler geographic/region baseline is the required operational choice.

The probe model is deliberately not a production model. Chl-a, bathymetry, and explicit coast distance are absent from the frozen v2.2 observation cache and therefore were not silently reconstructed after preregistration. TA/DIC support is North-American evidence even though the schema is globally mappable. A/B/C/D grades quantify demonstrated support, while final target accuracy, interval calibration, and publishability remain the responsibility of Issues #25-#28.

## Figure and table index

Every figure is backed by CSV source data under `tables/`; hashes and mappings are in `archive_manifest.json`.

"""
    for figure_name, caption in captions.items():
        report += f"![{figure_name}](figures/{figure_name})\n\n{caption}\n\n"
    (ARCHIVE / "REPORT.md").write_text(report, encoding="utf-8")
    caption_text = "# Figure captions\n\n" + "\n\n".join(
        f"## {name}\n\n{caption}" for name, caption in captions.items()
    )
    (ARCHIVE / "CAPTIONS.md").write_text(caption_text + "\n", encoding="utf-8")
    (ARCHIVE / "README.md").write_text(
        "# Issue #24 reviewer archive\n\nRun `python scripts/run_p1_applicability.py --prepared <prepared_global_p32.nc>`, then verify this directory with `python scripts/verify_experiment_archive.py docs/experiment_archive/p1_applicability_v2.2`. Large row-level predictions and grid-month indices remain under ignored `outputs/`.\n",
        encoding="utf-8",
    )
    files = [ARCHIVE / "README.md", ARCHIVE / "REPORT.md", ARCHIVE / "CAPTIONS.md"]
    files += sorted(figures.glob("*.png")) + sorted(tables.glob("*.csv"))
    manifest_path = ROOT / "configs" / "frozen" / "data_manifest_v2.2.json"
    frozen_root = FrozenManifest.load(manifest_path).data_dir
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_git_commit": training_commit,
        "data_manifest_sha256": file_sha256(manifest_path),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "v2.2 train/development nested outer OOF plus 2025-2026 support index",
        "decision": "applicability_infrastructure_pass"
        if decisions.monotonic_method_exists.all()
        else "applicability_infrastructure_fail",
        "archive_builder_sha256": file_sha256(Path(__file__)),
        "analysis_script_sha256": file_sha256(Path(__file__)),
        "source_artifacts_sha256": {
            "applicability_config": file_sha256(ROOT / "configs" / "p1_applicability_v2.2.yaml"),
            "coastal_graph": file_sha256(frozen_root / "coastal_graph_v2.2.npz"),
            "spatial_support": file_sha256(frozen_root / "spatial_support_v2.2.nc"),
        },
        "figure_source_data": source_map,
        "figure_captions": captions,
        "files_sha256": {
            str(path.relative_to(ARCHIVE)).replace("\\", "/"): file_sha256(path) for path in files
        },
    }
    (ARCHIVE / "archive_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/p1_applicability_v2.2.yaml")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / f"outputs/experiments/{EXPERIMENT_ID}"
    )
    parser.add_argument("--reuse-oof", action="store_true")
    parser.add_argument("--skip-grid-index", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    training_commit = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain")
    if status and not args.reuse_oof:
        raise RuntimeError("formal applicability run requires a clean preregistered worktree")
    manifest_path = args.config.parent / str(config["data_manifest"])
    manifest = FrozenManifest.load(manifest_path)
    validation = manifest.validate(hash_mode="full")
    gateway = P1DataGateway(manifest)
    graph = CoastalGraphIndex.from_frozen(
        manifest.data_dir / "coastal_graph_v2.2.npz",
        manifest.data_dir / "spatial_support_v2.2.nc",
    )
    if args.reuse_oof:
        predictions = pd.read_parquet(args.output / "outer_predictions.parquet")
        inner = pd.read_csv(args.output / "nested_selection.csv")
    else:
        predictions, inner = run_outer_predictions(gateway, graph, config, args.output)
    metrics, curves, decisions = evaluate_support(predictions, config, args.output)
    if args.skip_grid_index:
        grades = pd.DataFrame(
            columns=["target", "year", "month", "grade", "grid_months", "retained_method"]
        )
    else:
        grades = grid_support_indices(
            gateway, graph, config, predictions, decisions, args.prepared, args.output
        )
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "training_git_commit": training_commit,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_validation": validation,
        "locked_test_opened": False,
        "external_independent_opened": False,
        "acceptance_gate_passed": bool(decisions.monotonic_method_exists.all()),
        "targets": decisions.to_dict(orient="records"),
    }
    (args.output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (args.output / "protocol.json").write_text(
        json.dumps(
            {
                "config": config,
                "training_git_commit": training_commit,
                "config_sha256": file_sha256(args.config),
                "script_sha256": file_sha256(Path(__file__)),
                "prepared_path": str(args.prepared.resolve()),
                "prepared_size": args.prepared.stat().st_size,
                "locked_test_opened": False,
                "external_independent_opened": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    build_archive(
        args.output, config, predictions, metrics, curves, decisions, inner, grades, training_commit
    )
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
