"""Run the preregistered conditional regional-expert stage for Issue #26."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from recad.evaluate.applicability import outer_splits
from recad.evaluate.fco2_reliability import apply_shrinkage, finite_quantile
from recad.evaluate.p1_framework import FrozenManifest, P1DataGateway, Purpose
from run_p1_fco2_reliability import (
    MODEL_COLUMNS,
    ROOT,
    SeasonalTrendClimatology,
    add_crossfit_sss,
    join_support,
    metric_values,
    prepare,
    train_predict_catboost,
)

EXPERIMENT_ID = "p1_fco2_reliability_v2.2"


def eligible_lmes(train: pd.DataFrame, rules: dict[str, int]) -> list[int]:
    summary = train.groupby("lme_id").agg(
        records=("truth", "size"),
        cruises=("group_key", "nunique"),
        years=("year", "nunique"),
        months=("month", "nunique"),
    )
    keep = summary.loc[
        (summary.records >= rules["minimum_records"])
        & (summary.cruises >= rules["minimum_cruises"])
        & (summary.years >= rules["minimum_years"])
        & (summary.months >= rules["minimum_months"])
        & summary.index.to_series().ne(-1)
    ]
    return sorted(int(value) for value in keep.index)


def run_outer(config: dict[str, object], gateway: P1DataGateway, output: Path) -> pd.DataFrame:
    cached = output / "regional_expert_outer_oof.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    train = prepare(gateway.load_labels("fco2", Purpose.TRAIN, columns=MODEL_COLUMNS))
    development = prepare(gateway.load_labels("fco2", Purpose.SELECTION, columns=MODEL_COLUMNS))
    stage = config["regional_expert_stage"]
    regions = eligible_lmes(train, stage["eligibility_from_training_metadata"])
    support_path = Path(config["applicability_output"]) / "outer_predictions.parquet"
    sss_path = Path(config["sss_reliability_output"]) / "development_oof_predictions.parquet"
    parts = []
    for scheme in config["outer_schemes"]:
        source = (
            pd.concat([train, development], ignore_index=True) if scheme == "forward" else train
        )
        source = add_crossfit_sss(source, sss_path, str(scheme))
        for split in outer_splits(source, str(scheme), n_folds=int(config["outer_folds"])):
            path = output / "regional_partitions" / f"{scheme}_{split.fold}.parquet"
            if path.exists():
                parts.append(pd.read_parquet(path))
                continue
            fit = source.iloc[split.fit_index].copy().reset_index(drop=True)
            held = source.iloc[split.held_index].copy().reset_index(drop=True)
            background = SeasonalTrendClimatology().fit(fit)
            for frame in (fit, held):
                frame["background"] = background.predict(frame)
                frame["residual"] = frame.truth - frame.background
            held["raw_prediction"] = held.background
            held["expert_available"] = False
            if scheme != "whole_lme":
                for lme in regions:
                    fit_region = fit.loc[fit.lme_id.eq(lme)].reset_index(drop=True)
                    held_mask = held.lme_id.eq(lme)
                    held_region = held.loc[held_mask].reset_index(drop=True)
                    if len(fit_region) < 2000 or not len(held_region):
                        continue
                    print(
                        f"REGIONAL {scheme}/{split.fold} LME={lme} "
                        f"fit={len(fit_region):,} held={len(held_region):,}",
                        flush=True,
                    )
                    prediction = train_predict_catboost(
                        fit_region,
                        held_region,
                        seed=int(config["models"]["oof_seed"]),
                        iterations=int(stage["iterations"]),
                    )
                    held.loc[held_mask, "raw_prediction"] = prediction
                    held.loc[held_mask, "expert_available"] = True
            result = held[
                [
                    "record_id",
                    "group_key",
                    "year",
                    "month",
                    "latitude",
                    "longitude",
                    "lme_id",
                    "truth",
                    "background",
                    "raw_prediction",
                    "expert_available",
                ]
            ].copy()
            result.insert(0, "outer_scheme", scheme)
            result.insert(1, "outer_fold", split.fold)
            result = join_support(result, support_path, str(scheme), split.fold)
            result["prediction"] = apply_shrinkage(
                result.background,
                result.raw_prediction,
                result.risk_environment_k64,
                "linear_2_5",
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            result.to_parquet(path, index=False)
            parts.append(result)
    result = pd.concat(parts, ignore_index=True)
    result.to_parquet(cached, index=False)
    return result


def crossfit_region_intervals(predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    result["absolute_error"] = np.abs(result.prediction - result.truth)
    result["width50"] = np.nan
    result["width90"] = np.nan
    for (scheme, fold, lme), held in result.groupby(["outer_scheme", "outer_fold", "lme_id"]):
        if scheme == "whole_lme":
            continue
        cruise = result.loc[
            result.outer_scheme.eq("cruise") & result.lme_id.eq(lme) & result.expert_available
        ]
        calibration = cruise.loc[cruise.outer_fold.ne(fold)] if scheme == "cruise" else cruise
        if len(calibration) < 200:
            continue
        result.loc[held.index, "width50"] = finite_quantile(calibration.absolute_error, 0.50)
        result.loc[held.index, "width90"] = finite_quantile(calibration.absolute_error, 0.90)
    support = (
        result.expert_available
        & result.width90.notna()
        & result.risk_environment_k64.le(5.0)
        & result.unique_cruises.ge(2)
        & result.effective_groups.ge(1.5)
    )
    result["grade"] = "D"
    result.loc[support & result.width90.le(60.0), "grade"] = "C"
    result.loc[support & result.width90.le(35.0), "grade"] = "B"
    result.loc[support & result.width90.le(20.0), "grade"] = "A"
    return result


def region_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scheme, lme), part in frame.groupby(["outer_scheme", "lme_id"]):
        values = metric_values(part, "prediction")
        error = np.abs(part.prediction - part.truth)
        row = {
            "outer_scheme": scheme,
            "lme_id": int(lme),
            **values,
            "q90_absolute_error": float(np.quantile(error, 0.90)),
            "rmse_ratio": values["rmse"] / values["background_rmse"],
            "coverage50": float((error <= part.width50).mean())
            if part.width50.notna().any()
            else np.nan,
            "coverage90": float((error <= part.width90).mean())
            if part.width90.notna().any()
            else np.nan,
            "publishable_fraction": float(part.grade.isin(["A", "B"]).mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def decide(metrics: pd.DataFrame, config: dict[str, object]) -> dict[str, object]:
    stage = config["regional_expert_stage"]
    nomination = stage["nomination_rules"]
    cruise = metrics.loc[metrics.outer_scheme.eq("cruise")]
    nominated = cruise.loc[
        (cruise.n >= nomination["minimum_rows"])
        & (cruise.q90_absolute_error <= nomination["q90_absolute_error_max_uatm"])
        & (cruise.skill_vs_background > nomination["skill_vs_background_gt"])
    ].lme_id.astype(int)
    confirmation = stage["confirmation_rules"]
    confirmed = []
    checks_by_lme = {}
    for lme in nominated:
        rows = metrics.loc[
            metrics.lme_id.eq(lme) & metrics.outer_scheme.isin(stage["confirmation_sources"])
        ].set_index("outer_scheme")
        checks = {
            "sources_present": set(rows.index) == set(stage["confirmation_sources"]),
            "minimum_rows": bool((rows.n >= confirmation["minimum_rows"]).all()),
            "q90": bool(
                (rows.q90_absolute_error <= confirmation["q90_absolute_error_max_uatm"]).all()
            ),
            "skill": bool(
                (rows.skill_vs_background > confirmation["skill_vs_background_gt"]).all()
            ),
            "rmse_ratio": bool((rows.rmse_ratio <= confirmation["rmse_ratio_maximum"]).all()),
            "coverage50": bool(rows.coverage50.between(*confirmation["coverage50_range"]).all()),
            "coverage90": bool(rows.coverage90.between(*confirmation["coverage90_range"]).all()),
        }
        checks_by_lme[str(lme)] = checks
        if all(checks.values()):
            confirmed.append(int(lme))
    return {
        "status": "pass_regional" if confirmed else "diagnostic_only",
        "passed": bool(confirmed),
        "nominated_lmes": sorted(int(value) for value in nominated),
        "confirmed_lmes": confirmed,
        "checks_by_lme": checks_by_lme,
        "whole_lme_policy": stage["whole_lme_policy"],
        "locked_test_opened": False,
        "external_independent_opened": False,
    }


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
    manifest = FrozenManifest.load(ROOT / "configs/frozen/data_manifest_v2.2.json")
    gateway = P1DataGateway(manifest)
    predictions = run_outer(config, gateway, args.output)
    calibrated = crossfit_region_intervals(predictions)
    metrics = region_metrics(calibrated)
    decision = decide(metrics, config)
    calibrated.to_parquet(args.output / "regional_expert_predictions.parquet", index=False)
    metrics.to_csv(args.output / "regional_expert_metrics.csv", index=False)
    (args.output / "regional_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
