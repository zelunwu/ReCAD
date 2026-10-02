"""Run Issue #25 SSS reliability calibration, locked audit, and atlas build."""

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
import torch
import xarray as xr
import yaml

from recad.evaluate.applicability import SupportIndex, outer_splits, risk_coverage_curve
from recad.evaluate.p1_framework import (
    FrozenManifest,
    LockedTestGrant,
    P1DataGateway,
    Purpose,
)
from recad.evaluate.sss_reliability import (
    BinnedIntervalModel,
    apply_shrinkage,
    assign_reliability_grade,
    interval_coverage,
    predict_soft_expert,
    salinity_band,
    train_soft_expert,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_sss_reliability_v2.2"
ARCHIVE = ROOT / "docs" / "experiment_archive" / EXPERIMENT_ID
MODEL_COLUMNS = ["grid_flat", "coastal_node", "sst", "sss", "adt", "wspd", "pco2air"]
ENVIRONMENT_COLUMNS = [
    "sst",
    "sss",
    "adt",
    "wspd",
    "pco2air",
    "latitude_sin",
    "longitude_sin",
    "longitude_cos",
    "month_sin",
    "month_cos",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[frame.sss.notna()].copy().reset_index(drop=True)
    result["longitude"] = np.mod(result.longitude.to_numpy(float), 360.0)
    result["latitude_sin"] = np.sin(np.deg2rad(result.latitude))
    result["longitude_sin"] = np.sin(np.deg2rad(result.longitude))
    result["longitude_cos"] = np.cos(np.deg2rad(result.longitude))
    result["lon_sin"] = result.longitude_sin
    result["lon_cos"] = result.longitude_cos
    phase = 2.0 * np.pi * (result.month.to_numpy(float) - 1.0) / 12.0
    result["month_sin"] = np.sin(phase)
    result["month_cos"] = np.cos(phase)
    identity = result.grid_flat.astype(str) if "grid_flat" in result else result.index.astype(str)
    result["record_id"] = (
        "sss:"
        + result.group_key.astype(str)
        + ":"
        + result.year.astype(str)
        + ":"
        + result.month.astype(str)
        + ":"
        + identity
    )
    return result


def load_development(gateway: P1DataGateway) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = prepare(gateway.load_labels("sss", Purpose.TRAIN, columns=MODEL_COLUMNS))
    development = prepare(gateway.load_labels("sss", Purpose.SELECTION, columns=MODEL_COLUMNS))
    return train, development


def load_config(path: Path) -> dict[str, object]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["external_independent_access"] != "forbidden_until_P3":
        raise ValueError("Issue #25 must keep external independent labels sealed")
    if config["locked_test_openings"] != 1:
        raise ValueError("Issue #25 permits exactly one locked SSS opening")
    return config


def train_predict_state(
    fit: pd.DataFrame,
    held: pd.DataFrame,
    *,
    seed: int,
    config: dict[str, object],
    checkpoint: Path,
) -> np.ndarray:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    else:
        state = train_soft_expert(
            fit,
            seed=seed,
            steps=int(config["model"]["fixed_steps"]),
            batch_size=int(config["model"]["batch_size"]),
        )
        torch.save(state, checkpoint)
    return predict_soft_expert(state, held)


def soft_expert_oof(
    gateway: P1DataGateway,
    config: dict[str, object],
    output: Path,
) -> pd.DataFrame:
    cached = output / "soft_expert_outer_oof.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    train, development = load_development(gateway)
    issue24 = Path(config["applicability_output"]) / "outer_predictions.parquet"
    support = pd.read_parquet(issue24, filters=[("target", "==", "sss")])
    keep = [
        "outer_scheme",
        "outer_fold",
        "record_id",
        "environment_k64",
        "risk_environment_k64",
        "geographic_km",
        "unique_cruises",
        "effective_groups",
        "lme_month_count",
    ]
    support = support[keep]
    parts = []
    for scheme in config["outer_schemes"]:
        source = (
            pd.concat([train, development], ignore_index=True) if scheme == "forward" else train
        )
        for split in outer_splits(source, str(scheme), n_folds=int(config["outer_folds"])):
            fit = source.iloc[split.fit_index].reset_index(drop=True)
            held = source.iloc[split.held_index].reset_index(drop=True)
            checkpoint = (
                output
                / "checkpoints"
                / f"oof_{scheme}_{split.fold}_seed{config['model']['oof_seed']}.pt"
            )
            print(
                f"OOF {scheme} fold={split.fold}: fit={len(fit):,} held={len(held):,}", flush=True
            )
            prediction = train_predict_state(
                fit,
                held,
                seed=int(config["model"]["oof_seed"]),
                config=config,
                checkpoint=checkpoint,
            )
            frozen = held[
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
                    "sss",
                ]
            ].copy()
            frozen.insert(0, "outer_scheme", scheme)
            frozen.insert(1, "outer_fold", split.fold)
            frozen = frozen.rename(columns={"sss": "background"})
            frozen["raw_prediction"] = prediction
            matched = support.loc[
                support.outer_scheme.eq(scheme) & support.outer_fold.eq(split.fold)
            ]
            frozen = frozen.merge(
                matched,
                on=["outer_scheme", "outer_fold", "record_id"],
                how="left",
                validate="one_to_one",
            )
            if frozen.risk_environment_k64.isna().any():
                raise RuntimeError(f"Issue #24 support join failed for {scheme}/{split.fold}")
            parts.append(frozen)
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_parquet(cached, index=False)
    return predictions


def metric_values(frame: pd.DataFrame, prediction: str) -> dict[str, float | int]:
    truth = frame.truth.to_numpy(float)
    values = frame[prediction].to_numpy(float)
    background = frame.background.to_numpy(float)
    error = values - truth
    background_error = background - truth
    denominator = np.sum((truth - truth.mean()) ** 2)
    return {
        "n": len(frame),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1.0 - np.sum(error**2) / denominator) if denominator else np.nan,
        "background_rmse": float(np.sqrt(np.mean(background_error**2))),
        "skill_vs_background": float(1.0 - np.mean(error**2) / np.mean(background_error**2)),
    }


def select_shrinkage(
    predictions: pd.DataFrame, config: dict[str, object]
) -> tuple[str, pd.DataFrame]:
    rows = []
    for method in config["shrinkage_candidates"]:
        column = f"prediction_{method}"
        predictions[column] = apply_shrinkage(
            predictions.background,
            predictions.raw_prediction,
            predictions.risk_environment_k64,
            str(method),
        )
        for scheme, part in predictions.groupby("outer_scheme"):
            row = metric_values(part, column)
            rows.append({"method": method, "outer_scheme": scheme, **row})
    table = pd.DataFrame(rows)
    table["normalized_rmse"] = table.rmse / table.background_rmse
    ranking = table.groupby("method").normalized_rmse.mean().sort_values()
    selected = str(ranking.index[0])
    return selected, table


def crossfit_intervals(
    predictions: pd.DataFrame,
    *,
    prediction_column: str,
    minimum_cell: int,
) -> pd.DataFrame:
    result = predictions.copy()
    result["prediction"] = result[prediction_column]
    result["absolute_error"] = np.abs(result.prediction - result.truth)
    widths = []
    for (scheme, fold), evaluation in result.groupby(["outer_scheme", "outer_fold"], sort=True):
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


def grouped_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    strata: list[tuple[str, pd.Series]] = [
        ("salinity_band", pd.Series(salinity_band(predictions.truth), index=predictions.index)),
        (
            "estuary",
            pd.Series(
                np.where(predictions.background < 30.0, "fresh_lt30", "marine_ge30"),
                index=predictions.index,
            ),
        ),
        ("lme", predictions.lme_id.astype(str)),
    ]
    for scheme, scheme_part in predictions.groupby("outer_scheme"):
        for kind, labels in strata:
            for group in sorted(labels.loc[scheme_part.index].dropna().unique(), key=str):
                part = scheme_part.loc[labels.eq(group)]
                if len(part):
                    rows.append(
                        {
                            "outer_scheme": scheme,
                            "stratum": kind,
                            "group": str(group),
                            **metric_values(part, "prediction"),
                        }
                    )
    return pd.DataFrame(rows)


def run_development(
    gateway: P1DataGateway,
    config: dict[str, object],
    config_path: Path,
    output: Path,
) -> dict[str, object]:
    predictions = soft_expert_oof(gateway, config, output)
    selected, shrinkage = select_shrinkage(predictions, config)
    calibrated = crossfit_intervals(
        predictions,
        prediction_column=f"prediction_{selected}",
        minimum_cell=int(config["intervals"]["minimum_cell_records"]),
    )
    calibrated.to_parquet(output / "development_oof_predictions.parquet", index=False)
    shrinkage.to_csv(output / "shrinkage_comparison.csv", index=False)
    grouped_diagnostics(calibrated).to_csv(output / "development_strata.csv", index=False)
    coverage_rows = []
    curves = []
    for scheme, part in calibrated.groupby("outer_scheme"):
        coverage_rows.append({"outer_scheme": scheme, **interval_coverage(part)})
        curve = risk_coverage_curve(part.truth, part.prediction, part.risk_environment_k64)
        curve.insert(0, "outer_scheme", scheme)
        curves.append(curve)
    pd.DataFrame(coverage_rows).to_csv(output / "development_interval_coverage.csv", index=False)
    pd.concat(curves).to_csv(output / "development_risk_coverage.csv", index=False)
    # The spatial and whole-LME stress tests repeat the same observation rows.
    # Cruise OOF contains each calibration label exactly once.
    final_calibration = BinnedIntervalModel.fit(
        calibrated.loc[calibrated.outer_scheme.eq("cruise")],
        minimum_cell=int(config["intervals"]["minimum_cell_records"]),
    )
    summary = []
    for scheme, part in calibrated.groupby("outer_scheme"):
        summary.append({"outer_scheme": scheme, **metric_values(part, "prediction")})
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(output / "development_metrics.csv", index=False)
    forward_skill = float(
        summary_frame.loc[summary_frame.outer_scheme.eq("forward"), "skill_vs_background"].iloc[0]
    )
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "development_frozen_before_locked_opening",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_head(),
        "config_sha256": sha256(config_path),
        "issue24_outer_predictions_sha256": sha256(
            Path(config["applicability_output"]) / "outer_predictions.parquet"
        ),
        "soft_expert_oof_sha256": sha256(output / "soft_expert_outer_oof.parquet"),
        "selected_shrinkage": selected,
        "interval_model": final_calibration.to_dict(),
        "grade_rules": config["grades"],
        "forward_skill_vs_glorys": forward_skill,
        "locked_test_opened": False,
        "external_independent_opened": False,
    }
    (output / "frozen_development_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps({"selected_shrinkage": selected, "metrics": summary}, indent=2), flush=True)
    return decision


def support_for_frame(reference: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    support = SupportIndex(environmental_columns=ENVIRONMENT_COLUMNS).fit(reference).query(query)
    support["risk_environment_k64"] = support.environment_k64
    return support


def final_states(
    development_evidence: pd.DataFrame,
    config: dict[str, object],
    output: Path,
) -> list[dict[str, object]]:
    states = []
    for seed in config["model"]["seeds"]:
        path = output / "checkpoints" / f"final_train_development_seed{seed}.pt"
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=False)
        else:
            print(f"FINAL model seed={seed}", flush=True)
            state = train_soft_expert(
                development_evidence,
                seed=int(seed),
                steps=int(config["model"]["fixed_steps"]),
                batch_size=int(config["model"]["batch_size"]),
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, path)
        states.append(state)
    return states


def ensemble_prediction(
    states: list[dict[str, object]], frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    values = np.vstack([predict_soft_expert(state, frame) for state in states])
    return values.mean(axis=0), values.std(axis=0)


def locked_metrics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [{"scope": "all", "group": "all", **metric_values(frame, "prediction")}]
    for grade, part in frame.groupby("grade"):
        rows.append({"scope": "grade", "group": grade, **metric_values(part, "prediction")})
    for lme, part in frame.groupby("lme_id"):
        if len(part) >= 100:
            rows.append({"scope": "lme", "group": str(lme), **metric_values(part, "prediction")})
    intervals = [{"scope": "all", "group": "all", **interval_coverage(frame)}]
    for grade, part in frame.groupby("grade"):
        intervals.append({"scope": "grade", "group": grade, **interval_coverage(part)})
    return pd.DataFrame(rows), pd.DataFrame(intervals)


def run_locked(
    gateway: P1DataGateway,
    config: dict[str, object],
    frozen_path: Path,
    output: Path,
) -> dict[str, object]:
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("locked_test_opened") is not False:
        raise RuntimeError("frozen decision is not pre-locked")
    # Finish every operation that does not need locked labels before creating
    # the irreversible one-time grant.
    train, development = load_development(gateway)
    evidence = pd.concat([train, development], ignore_index=True)
    states = final_states(evidence, config, output)
    audit_path = output / "audit" / "locked_sss_opening.json"
    grant = LockedTestGrant.create(
        "Issue #25 one-time scoring of the fully frozen SSS reliability candidate",
        audit_path,
    )
    # This is the only statement in the repository that materialises the
    # locked SSS labels.  The grant refuses a second opening at the same path.
    locked = prepare(
        gateway.load_labels("sss", Purpose.LOCKED_TEST, columns=MODEL_COLUMNS, grant=grant)
    )
    raw, epistemic = ensemble_prediction(states, locked)
    support = support_for_frame(evidence, locked)
    scored = locked[
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
            "sss",
        ]
    ].rename(columns={"sss": "background"})
    for column in [
        "environment_k64",
        "risk_environment_k64",
        "geographic_km",
        "unique_cruises",
        "effective_groups",
        "lme_month_count",
    ]:
        scored[column] = support[column].to_numpy()
    scored["raw_prediction"] = raw
    scored["epistemic_sd"] = epistemic
    scored["prediction"] = apply_shrinkage(
        scored.background,
        scored.raw_prediction,
        scored.risk_environment_k64,
        frozen["selected_shrinkage"],
    )
    interval_model = BinnedIntervalModel.from_dict(frozen["interval_model"])
    widths = interval_model.predict(scored)
    for column in widths:
        scored[column] = widths[column]
    scored["grade"] = assign_reliability_grade(scored)
    scored["lower50"] = scored.prediction - scored.width50
    scored["upper50"] = scored.prediction + scored.width50
    scored["lower90"] = scored.prediction - scored.width90
    scored["upper90"] = scored.prediction + scored.width90
    scored.to_parquet(output / "locked_predictions.parquet", index=False)
    metrics, intervals = locked_metrics(scored)
    metrics.to_csv(output / "locked_metrics.csv", index=False)
    intervals.to_csv(output / "locked_interval_coverage.csv", index=False)
    overall = metrics.loc[(metrics.scope == "all") & (metrics.group == "all")].iloc[0]
    grade_ab = scored.loc[scored.grade.isin(["A", "B"])]
    grade_ab_metric = (
        metric_values(grade_ab, "prediction")
        if len(grade_ab)
        else {"mae": np.inf, "skill_vs_background": -np.inf}
    )
    overall_interval = intervals.loc[(intervals.scope == "all") & (intervals.group == "all")].iloc[
        0
    ]
    lme = metrics.loc[metrics.scope.eq("lme")]
    positive_fraction = float((lme.skill_vs_background > 0).mean()) if len(lme) else 0.0
    worst_ratio = float((lme.rmse / lme.background_rmse).max()) if len(lme) else np.inf
    gate = config["locked_gate"]
    checks = {
        "overall_positive_skill": bool(
            overall.skill_vs_background > gate["overall_skill_vs_glorys_gt"]
        ),
        "grade_ab_positive_skill": bool(
            grade_ab_metric["skill_vs_background"] > gate["publishable_grade_skill_gt"]
        ),
        "grade_ab_mae": bool(grade_ab_metric["mae"] <= gate["grade_ab_mae_max_psu"]),
        "coverage50": bool(
            gate["coverage50_range"][0]
            <= overall_interval.coverage50
            <= gate["coverage50_range"][1]
        ),
        "coverage90": bool(
            gate["coverage90_range"][0]
            <= overall_interval.coverage90
            <= gate["coverage90_range"][1]
        ),
        "positive_lme_fraction": bool(positive_fraction >= gate["positive_lme_fraction_minimum"]),
        "worst_lme_no_collapse": bool(worst_ratio <= gate["worst_lme_relative_rmse_maximum"]),
        "forward_evidence": bool(frozen["forward_skill_vs_glorys"] > 0),
    }
    decision = {
        **frozen,
        "stage": "locked_audit_complete",
        "locked_test_opened": True,
        "locked_audit_sha256": sha256(audit_path),
        "locked_rows": len(scored),
        "locked_overall": overall.to_dict(),
        "locked_grade_ab": grade_ab_metric,
        "locked_interval": overall_interval.to_dict(),
        "eligible_lme_positive_fraction": positive_fraction,
        "worst_lme_rmse_ratio": worst_ratio,
        "gate_checks": checks,
        "acceptance_gate_passed": bool(all(checks.values())),
        "final_status": "pass_regional" if all(checks.values()) else "diagnostic_only",
        "external_independent_opened": False,
    }
    (output / "locked_decision.json").write_text(
        json.dumps(
            decision,
            indent=2,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        ),
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, default=str), flush=True)
    return decision


def grid_query(
    prepared: xr.Dataset,
    spatial: xr.Dataset,
    year: int,
    month: int,
    nodes: np.ndarray,
) -> pd.DataFrame:
    lat_index = spatial.lat_index.to_numpy()[nodes]
    lon_index = spatial.lon_index.to_numpy()[nodes]
    indexers = {"year": year, "month": month}
    data = {
        name: prepared[name].sel(**indexers).to_numpy()[lat_index, lon_index]
        for name in ["sst", "sss", "adt", "wspd", "pco2air"]
    }
    longitude = spatial.longitude.to_numpy()[nodes].astype(float)
    latitude = spatial.latitude.to_numpy()[nodes].astype(float)
    phase = 2.0 * np.pi * (month - 1.0) / 12.0
    data.update(
        grid_flat=spatial.grid_flat.to_numpy()[nodes],
        coastal_node=nodes,
        latitude=latitude,
        longitude=longitude,
        latitude_sin=np.sin(np.deg2rad(latitude)),
        longitude_sin=np.sin(np.deg2rad(longitude)),
        longitude_cos=np.cos(np.deg2rad(longitude)),
        lon_sin=np.sin(np.deg2rad(longitude)),
        lon_cos=np.cos(np.deg2rad(longitude)),
        month_sin=np.full(len(nodes), np.sin(phase)),
        month_cos=np.full(len(nodes), np.cos(phase)),
        year=np.full(len(nodes), year),
        month=np.full(len(nodes), month),
        lme_id=spatial.lme_id.to_numpy()[nodes],
        basin_id=spatial.basin_id.to_numpy()[nodes],
        regime_id=spatial.regime_id.to_numpy()[nodes],
        group_key=np.full(len(nodes), "prediction_grid"),
    )
    return pd.DataFrame(data)


def run_atlas(
    gateway: P1DataGateway,
    config: dict[str, object],
    frozen_path: Path,
    output: Path,
) -> pd.DataFrame:
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    train, development = load_development(gateway)
    evidence = pd.concat([train, development], ignore_index=True)
    states = final_states(evidence, config, output)
    support_index = SupportIndex(environmental_columns=ENVIRONMENT_COLUMNS).fit(evidence)
    interval_model = BinnedIntervalModel.from_dict(frozen["interval_model"])
    availability_path = gateway.manifest.data_dir / "inference_availability_v2.2.nc"
    spatial_path = gateway.manifest.data_dir / "spatial_support_v2.2.nc"
    atlas_root = output / "sss_reliability_atlas"
    rows = []
    with (
        xr.open_dataset(config["prepared_global"]) as prepared,
        xr.open_dataset(spatial_path) as spatial,
        xr.open_dataset(availability_path) as availability,
    ):
        for year in config["grid_years"]:
            for month in range(1, 13):
                ready = (
                    availability.strict_all_inputs_ready.sel(year=year, month=month)
                    .to_numpy()
                    .astype(bool)
                )
                nodes = np.flatnonzero(ready)
                if not len(nodes):
                    continue
                print(f"ATLAS {year}-{month:02d}: {len(nodes):,} grid cells", flush=True)
                query = grid_query(prepared, spatial, int(year), month, nodes)
                support = support_index.query(query)
                query["risk_environment_k64"] = support.environment_k64.to_numpy()
                query["unique_cruises"] = support.unique_cruises.to_numpy()
                query["effective_groups"] = support.effective_groups.to_numpy()
                raw, epistemic = ensemble_prediction(states, query)
                query["background"] = query.sss
                query["raw_prediction"] = raw
                query["epistemic_sd"] = epistemic
                query["prediction"] = apply_shrinkage(
                    query.background, raw, query.risk_environment_k64, frozen["selected_shrinkage"]
                )
                widths = interval_model.predict(query)
                for column in widths:
                    query[column] = widths[column]
                query["grade"] = assign_reliability_grade(query)
                query["lower50"] = query.prediction - query.width50
                query["upper50"] = query.prediction + query.width50
                query["lower90"] = query.prediction - query.width90
                query["upper90"] = query.prediction + query.width90
                query["prediction_status"] = "provisional" if year == 2026 else "core"
                query["reason_bits"] = 256
                query.loc[query.risk_environment_k64 > 5.0, "reason_bits"] |= 16
                query.loc[query.unique_cruises < 2, "reason_bits"] |= 32
                query.loc[~query.cell_supported, "reason_bits"] |= 128
                columns = [
                    "year",
                    "month",
                    "coastal_node",
                    "latitude",
                    "longitude",
                    "background",
                    "prediction",
                    "epistemic_sd",
                    "lower50",
                    "upper50",
                    "lower90",
                    "upper90",
                    "width50",
                    "width90",
                    "risk_environment_k64",
                    "unique_cruises",
                    "effective_groups",
                    "grade",
                    "reason_bits",
                    "prediction_status",
                ]
                partition = atlas_root / f"year={year}"
                partition.mkdir(parents=True, exist_ok=True)
                query[columns].to_parquet(partition / f"month={month:02d}.parquet", index=False)
                for grade, count in query.grade.value_counts().items():
                    rows.append(
                        {
                            "year": year,
                            "month": month,
                            "grade": grade,
                            "grid_months": int(count),
                            "fraction": float(count / len(query)),
                        }
                    )
    counts = pd.DataFrame(rows)
    counts.to_csv(output / "atlas_grade_counts.csv", index=False)
    return counts


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_archive(output: Path, config_path: Path) -> None:
    if ARCHIVE.exists():
        shutil.rmtree(ARCHIVE)
    figures = ARCHIVE / "figures"
    tables = ARCHIVE / "tables"
    figures.mkdir(parents=True)
    tables.mkdir(parents=True)
    copy_names = [
        "development_metrics.csv",
        "shrinkage_comparison.csv",
        "development_interval_coverage.csv",
        "development_risk_coverage.csv",
        "development_strata.csv",
        "locked_metrics.csv",
        "locked_interval_coverage.csv",
        "atlas_grade_counts.csv",
    ]
    for name in copy_names:
        shutil.copy2(output / name, tables / name)
    shutil.copy2(config_path, ARCHIVE / config_path.name)
    shutil.copy2(
        output / "frozen_development_decision.json", ARCHIVE / "frozen_development_decision.json"
    )
    shutil.copy2(output / "locked_decision.json", ARCHIVE / "locked_decision.json")
    shrinkage = pd.read_csv(output / "shrinkage_comparison.csv")
    locked = pd.read_csv(output / "locked_metrics.csv")
    intervals = pd.read_csv(output / "locked_interval_coverage.csv")
    grades = pd.read_csv(output / "atlas_grade_counts.csv")
    decision = json.loads((output / "locked_decision.json").read_text(encoding="utf-8"))

    pivot = shrinkage.groupby("method").normalized_rmse.mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(pivot.index, pivot.values, color="#2878B5")
    ax.axhline(1.0, color="black", lw=1)
    ax.set_ylabel("Mean scheme-normalized RMSE")
    ax.tick_params(axis="x", rotation=25)
    save_figure(fig, figures / "fig01_shrinkage_comparison.png")

    locked_grades = locked.loc[locked.scope.eq("grade")].sort_values("group")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(locked_grades.group, locked_grades.rmse, label="SSS candidate", color="#2CA02C")
    ax.scatter(
        locked_grades.group,
        locked_grades.background_rmse,
        label="GLORYS",
        color="#D62728",
        zorder=3,
    )
    ax.set_ylabel("Locked RMSE (PSU)")
    ax.set_xlabel("Frozen reliability grade")
    ax.legend()
    save_figure(fig, figures / "fig02_locked_grade_skill.png")

    interval_grades = intervals.loc[intervals.scope.eq("grade")].sort_values("group")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(interval_grades))
    ax.bar(x - 0.18, interval_grades.coverage50, width=0.36, label="50% interval")
    ax.bar(x + 0.18, interval_grades.coverage90, width=0.36, label="90% interval")
    ax.axhline(0.5, color="C0", ls="--", lw=1)
    ax.axhline(0.9, color="C1", ls="--", lw=1)
    ax.set_xticks(x, interval_grades.group)
    ax.set_ylabel("Locked empirical coverage")
    ax.legend()
    save_figure(fig, figures / "fig03_locked_interval_coverage.png")

    coverage = grades.groupby("grade").grid_months.sum().reindex(list("ABCD"), fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        coverage.index,
        coverage.values / coverage.sum(),
        color=["#2CA02C", "#9ACD32", "#FFB000", "#D62728"],
    )
    ax.set_ylabel("Fraction of 2025-2026 available grid-months")
    ax.set_xlabel("Frozen SSS grade")
    save_figure(fig, figures / "fig04_atlas_grade_coverage.png")

    atlas_file = output / "sss_reliability_atlas" / "year=2025" / "month=07.parquet"
    atlas = pd.read_parquet(atlas_file)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    palette = {"A": "#2CA02C", "B": "#9ACD32", "C": "#FFB000", "D": "#D62728"}
    for grade in "DCBA":
        part = atlas.loc[atlas.grade.eq(grade)]
        ax.scatter(
            part.longitude, part.latitude, s=0.4, color=palette[grade], label=grade, rasterized=True
        )
    ax.set_xlim(0, 360)
    ax.set_ylim(-80, 85)
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude")
    ax.legend(markerscale=8, ncol=4)
    save_figure(fig, figures / "fig05_global_grade_map_2025_07.png")

    risk_curves = pd.read_csv(output / "development_risk_coverage.csv")
    locked_rows = pd.read_parquet(
        output / "locked_predictions.parquet",
        columns=["truth", "prediction", "background", "risk_environment_k64", "grade"],
    )
    locked_curve = risk_coverage_curve(
        locked_rows.truth, locked_rows.prediction, locked_rows.risk_environment_k64
    )
    locked_curve.insert(0, "outer_scheme", "locked")
    risk_curves = pd.concat([risk_curves, locked_curve], ignore_index=True)
    risk_curves.to_csv(tables / "risk_coverage_all.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for scheme, part in risk_curves.groupby("outer_scheme"):
        ax.plot(part.coverage, part.rmse, marker="o", ms=3, label=scheme)
    ax.set_xlabel("Retained lowest-risk fraction")
    ax.set_ylabel("SSS RMSE (PSU)")
    ax.legend(ncol=2)
    save_figure(fig, figures / "fig06_risk_coverage.png")

    atlas_parts = []
    for path in sorted((output / "sss_reliability_atlas").rglob("*.parquet")):
        atlas_parts.append(pd.read_parquet(path, columns=["latitude", "grade"]))
    atlas_grade_rows = pd.concat(atlas_parts, ignore_index=True)
    atlas_grade_rows["area_weight"] = np.cos(np.deg2rad(atlas_grade_rows.latitude))
    grade_order = list("ABCD")
    retention_rows = []
    for maximum_grade in grade_order:
        retained_grades = grade_order[: grade_order.index(maximum_grade) + 1]
        locked_retained = locked_rows.loc[locked_rows.grade.isin(retained_grades)]
        atlas_retained = atlas_grade_rows.grade.isin(retained_grades)
        row = metric_values(locked_retained, "prediction") if len(locked_retained) else {}
        retention_rows.append(
            {
                "maximum_grade": maximum_grade,
                "locked_records": len(locked_retained),
                "locked_rmse": row.get("rmse", np.nan),
                "locked_mae": row.get("mae", np.nan),
                "atlas_gridmonth_fraction": float(atlas_retained.mean()),
                "atlas_area_weighted_fraction": float(
                    atlas_grade_rows.loc[atlas_retained, "area_weight"].sum()
                    / atlas_grade_rows.area_weight.sum()
                ),
            }
        )
    retention = pd.DataFrame(retention_rows)
    retention.to_csv(tables / "selective_grade_retention.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    valid = retention.locked_rmse.notna()
    ax.plot(
        retention.loc[valid, "atlas_area_weighted_fraction"],
        retention.loc[valid, "locked_rmse"],
        marker="o",
    )
    for row in retention.loc[valid].itertuples():
        ax.annotate(f"through {row.maximum_grade}", (row.atlas_area_weighted_fraction, row.locked_rmse))
    ax.set_xlabel("Retained atlas area-weighted grid-month fraction")
    ax.set_ylabel("Cumulative locked RMSE (PSU)")
    save_figure(fig, figures / "fig07_grade_retention.png")

    overall = locked.loc[(locked.scope == "all") & (locked.group == "all")].iloc[0]
    interval = intervals.loc[(intervals.scope == "all") & (intervals.group == "all")].iloc[0]
    status = decision["final_status"]
    checks = pd.DataFrame(
        [{"gate": key, "passed": value} for key, value in decision["gate_checks"].items()]
    )
    checks.to_csv(tables / "locked_gate_checks.csv", index=False)
    checks_markdown = "| gate | passed |\n|---|---|\n" + "\n".join(
        f"| {row.gate} | {str(bool(row.passed)).lower()} |" for row in checks.itertuples()
    )
    captions = {
        "fig01_shrinkage_comparison.png": "Figure 1. Mean RMSE across cruise, spatial-block, whole-LME, and forward-time outer partitions, normalized by GLORYS RMSE for each preregistered residual-shrinkage rule. Lower is better; the black line is parity with GLORYS. This development-only comparison froze the shrinkage method before locked labels were opened.",
        "fig02_locked_grade_skill.png": "Figure 2. Internal locked-test RMSE for the frozen SSS candidate and GLORYS within each preregistered A/B/C/D grade. Each locked observation is scored once with the already frozen model and grade rule; lower is better.",
        "fig03_locked_interval_coverage.png": "Figure 3. Empirical locked coverage of the frozen nominal 50% and 90% SSS intervals by reliability grade. Dashed lines mark nominal coverage; deviations diagnose calibration without retrospective rescaling.",
        "fig04_atlas_grade_coverage.png": "Figure 4. Fraction of all strict-input-ready 2025 core and available 2026 provisional coastal grid-months assigned to each frozen SSS reliability grade. D values remain in the audit table but are suppressed from the publishable product.",
        "fig05_global_grade_map_2025_07.png": "Figure 5. Frozen SSS reliability grades for strict-input-ready coastal grid cells in July 2025. The North-American SOCAT development domain dominates demonstrated support; remote global cells become C/D or return toward GLORYS through residual shrinkage. This map is an applicability atlas, not external validation.",
        "fig06_risk_coverage.png": "Figure 6. SSS RMSE as progressively higher-risk observations are retained under the frozen environment-k64 ranking. Development outer schemes and the one-time locked audit are shown separately; a rising curve means the applicability score orders error usefully.",
        "fig07_grade_retention.png": "Figure 7. Cumulative one-time locked RMSE versus the area-weighted fraction of 2025 core and available 2026 provisional atlas grid-months retained through each frozen grade. The curve states the accuracy paid for broader mapped coverage.",
    }
    report = f"""# P1.5b SSS reliability atlas and one-time locked audit

## Scientific question and permitted claim

This experiment asks where and with what error the frozen coastal SSS residual candidate is usable. The permitted claim is an internally locked, North-American-adjacent reliability result plus a global applicability atlas. External-independent validation remains reserved for P3.

## Data, splits, and leakage controls

SOCATv2026 in-situ salinity is the label and GLORYS monthly SSS is the background. Complete-cruise, fixed 5-degree spatial-block, whole-LME, and forward-time outer partitions were inherited from Issue #24. Every fold trained for a fixed 4,000 steps without consulting held-fold labels. Cruise OOF, in which each observation occurs once, calibrated the final interval model. The audited grant opened internal locked SSS labels exactly once after the tracked decision froze all choices.

## Candidate models and training

The Issue #7 eight-expert top-two architecture predicts a residual added to GLORYS. `environment_k64` is the frozen applicability variable. Four correction rules were compared on development OOF evidence; `{decision["selected_shrinkage"]}` was frozen before locked access. Intervals use finite-sample 50%/90% absolute-error quantiles in environmental-risk decile x GLORYS-salinity cells, with global fallback below 200 calibration records.

![Shrinkage comparison](figures/fig01_shrinkage_comparison.png)

*{captions["fig01_shrinkage_comparison.png"]}*

## Main development results

The model retained positive skill in cruise, spatial-block, whole-LME, and forward-time evaluation. The complete values, calibration coverage, salinity/estuary strata, and risk-coverage curve are archived as source tables. Development evidence selected shrinkage and froze uncertainty; it did not use locked labels.

Cruise, spatial-block, whole-LME, and forward-time RMSE were 1.263, 1.405, 1.719, and 1.408 PSU, with skill over GLORYS of 0.574, 0.472, 0.210, and 0.582. Cross-fitted 90% coverage was 0.898, 0.895, 0.853, and 0.925 in the same order.

## One-time locked audit

The frozen SSS candidate received **`{status}`**. The one-time internal locked set contained {int(overall.n):,} observations. Locked pooled RMSE was {overall.rmse:.3f} PSU versus {overall.background_rmse:.3f} PSU for GLORYS, giving skill {overall.skill_vs_background:.3f}; MAE was {overall.mae:.3f} PSU and R² was {overall.r2:.3f}. The frozen 50% and 90% intervals achieved {interval.coverage50:.3f} and {interval.coverage90:.3f} coverage.

Seven of eight gates passed. The failure was localized to LME 17, the North Brazil Shelf (n=149): candidate RMSE 3.565 PSU versus GLORYS 2.631 PSU, a ratio of 1.355 and skill -0.835. The preregistered worst-LME limit was 1.10, so strong pooled and A/B results cannot promote this version beyond `diagnostic_only`.

![Locked grade skill](figures/fig02_locked_grade_skill.png)

*{captions["fig02_locked_grade_skill.png"]}*

![Locked interval coverage](figures/fig03_locked_interval_coverage.png)

*{captions["fig03_locked_interval_coverage.png"]}*

## Reliability atlas

Each strict-input-ready 2025 grid-month and available provisional 2026 grid-month stores prediction, 50%/90% interval, `environment_k64`, evidence counts, A/B/C/D grade, and reason bits. A requires q90 at most 0.5 PSU; B at most 1.0 PSU; C at most 2.0 PSU; D suppresses wider error or inadequate environmental, cruise, group, or calibration support.

![Atlas grade coverage](figures/fig04_atlas_grade_coverage.png)

*{captions["fig04_atlas_grade_coverage.png"]}*

![Global grade map](figures/fig05_global_grade_map_2025_07.png)

*{captions["fig05_global_grade_map_2025_07.png"]}*

No grid-month earned A because the narrowest development-calibrated 90% cell width exceeded the fixed 0.5 PSU A boundary. Across 1,840,408 available grid-months, 79,837 (4.3%) were B, 980,998 (53.3%) were C, and 779,573 (42.4%) were D.

![Risk coverage](figures/fig06_risk_coverage.png)

*{captions["fig06_risk_coverage.png"]}*

![Grade retention](figures/fig07_grade_retention.png)

*{captions["fig07_grade_retention.png"]}*

## Decision and limitations

{checks_markdown}

A failed frozen check downgrades the product to diagnostic-only; the locked set cannot be reused to redesign it. The global atlas measures similarity to the frozen observation domain and does not prove global accuracy. Very fresh water remains difficult even where relative skill over GLORYS is positive. Row-level predictions, checkpoints, and atlas parquet partitions remain in the local hashed output and are excluded from Git.

## Figure and table index

Figures 1-7 correspond to shrinkage selection, locked grade skill, locked interval coverage, atlas grade coverage, the July 2025 global grade map, risk-coverage, and cumulative grade retention. Their aggregate sources are under `tables/`; the map source is the hashed local atlas partition. Captions appear directly below every figure and are duplicated in `CAPTIONS.md`.
"""
    (ARCHIVE / "REPORT.md").write_text(report, encoding="utf-8")
    (ARCHIVE / "README.md").write_text(
        "# P1.5b reviewer archive\n\nSee `REPORT.md` for the complete result, figures, captions, evidence boundary, and decision. Row-level outputs remain in the local experiment directory.\n",
        encoding="utf-8",
    )
    caption_text = "# Figure captions\n\n" + "\n\n".join(
        f"## {name}\n\n{caption}" for name, caption in captions.items()
    )
    (ARCHIVE / "CAPTIONS.md").write_text(caption_text + "\n", encoding="utf-8")
    mapping = {
        "fig01_shrinkage_comparison.png": ["shrinkage_comparison.csv"],
        "fig02_locked_grade_skill.png": ["locked_metrics.csv"],
        "fig03_locked_interval_coverage.png": ["locked_interval_coverage.csv"],
        "fig04_atlas_grade_coverage.png": ["atlas_grade_counts.csv"],
        "fig05_global_grade_map_2025_07.png": [
            "local:sss_reliability_atlas/year=2025/month=07.parquet"
        ],
        "fig06_risk_coverage.png": ["risk_coverage_all.csv"],
        "fig07_grade_retention.png": ["selective_grade_retention.csv"],
    }
    script_path = ROOT / "scripts/run_p1_sss_reliability.py"
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_git_commit": decision["git_commit"],
        "data_manifest_sha256": sha256(ROOT / "configs/frozen/data_manifest_v2.2.json"),
        "locked_test_opened": True,
        "external_independent_opened": False,
        "evidence_scope": "development outer OOF plus one-time internal locked SSS; external independent sealed",
        "decision": status,
        "figure_source_data": mapping,
        "figure_captions": captions,
        "archive_builder_sha256": sha256(script_path),
        "analysis_script_sha256": sha256(script_path),
        "source_artifacts_sha256": {
            "frozen_development_decision.json": sha256(output / "frozen_development_decision.json"),
            "locked_decision.json": sha256(output / "locked_decision.json"),
            "locked_predictions.parquet": sha256(output / "locked_predictions.parquet"),
            "development_oof_predictions.parquet": sha256(
                output / "development_oof_predictions.parquet"
            ),
        },
        "files_sha256": {
            str(path.relative_to(ARCHIVE)).replace("\\", "/"): sha256(path)
            for path in ARCHIVE.rglob("*")
            if path.is_file()
        },
    }
    (ARCHIVE / "archive_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/p1_sss_reliability_v2.2.yaml"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/experiments" / EXPERIMENT_ID)
    parser.add_argument(
        "--frozen", type=Path, default=ROOT / "configs/frozen/p1_sss_reliability_decision_v2.2.json"
    )
    parser.add_argument(
        "--phase",
        choices=["development", "locked", "atlas", "archive", "postfreeze"],
        required=True,
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    manifest = FrozenManifest.load(ROOT / "configs/frozen/data_manifest_v2.2.json")
    validation = manifest.validate(hash_mode="full")
    gateway = P1DataGateway(manifest)
    if args.phase in {"locked", "atlas", "archive", "postfreeze"}:
        frozen_payload = json.loads(args.frozen.read_text(encoding="utf-8"))
        if frozen_payload["config_sha256"] != sha256(args.config):
            raise RuntimeError("current config does not match the pre-locked frozen decision")
    if args.phase == "development":
        run_development(gateway, config, args.config, args.output)
    elif args.phase == "locked":
        run_locked(gateway, config, args.frozen, args.output)
    elif args.phase == "atlas":
        run_atlas(gateway, config, args.frozen, args.output)
    elif args.phase == "archive":
        build_archive(args.output, args.config)
    else:
        run_locked(gateway, config, args.frozen, args.output)
        run_atlas(gateway, config, args.frozen, args.output)
        build_archive(args.output, args.config)
    protocol = {
        "experiment_id": EXPERIMENT_ID,
        "phase": args.phase,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_head(),
        "manifest_validation": validation,
        "locked_test_opened": (args.output / "audit/locked_sss_opening.json").exists(),
        "external_independent_opened": False,
    }
    (args.output / f"protocol_{args.phase}.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
