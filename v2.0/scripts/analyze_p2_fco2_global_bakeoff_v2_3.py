"""Analyze the frozen Issue #41 global coastal fCO2 bake-off."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from recad.evaluate.fco2_bakeoff import fco2_band, interval_scores

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path(r"C:\backup\phd\data\processed\recad_v2_3\p2_fco2_global_bakeoff_v2.3")
PARTITIONS = EXTERNAL / "partitions"
TABLES = EXTERNAL / "tables"
MODELS = [
    "seasonal_trend",
    "v11_random_forest",
    "catboost_direct",
    "catboost_residual",
    "catboost_residual_sss",
    "lme_adapters",
    "gmm_experts",
]
SCHEMES = ["cruise", "spatial", "whole_lme", "forward"]
MODERN_MODELS = [
    "catboost_direct",
    "catboost_residual",
    "catboost_residual_sss",
    "lme_adapters",
    "gmm_experts",
]
TIME_BLOCKS = [
    (1993, 1997, "1993_1997"),
    (1998, 2001, "1998_2001"),
    (2002, 2020, "2002_2020"),
    (2021, 2025, "2021_2025"),
]


def load_pair(scheme: str, model: str) -> pd.DataFrame:
    paths = sorted(PARTITIONS.glob(f"fCO2_{scheme}_fold*_{model}.parquet"))
    expected = 1 if scheme == "forward" else 5
    if len(paths) != expected:
        raise RuntimeError(
            f"expected {expected} partitions for {scheme}/{model}, found {len(paths)}"
        )
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def metrics(frame: pd.DataFrame, **keys: object) -> dict[str, object]:
    error = frame.prediction.to_numpy(float) - frame.truth.to_numpy(float)
    background_error = frame.background.to_numpy(float) - frame.truth.to_numpy(float)
    variance = np.sum((frame.truth - frame.truth.mean()) ** 2)
    return {
        **keys,
        "n": len(frame),
        "cruises": frame.group_key.nunique(),
        "lmes": frame.lme_id.nunique(),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1 - np.sum(error**2) / variance) if variance > 0 else np.nan,
        "skill_vs_seasonal_trend": float(1 - np.mean(error**2) / np.mean(background_error**2)),
    }


def common_grid_metrics(frame: pd.DataFrame, **keys: object) -> dict[str, object]:
    grouped = (
        frame.groupby(["yr", "mon", "latitude", "longitude"], observed=True)
        .agg(
            truth=("truth", "mean"),
            prediction=("prediction", "mean"),
            background=("background", "mean"),
        )
        .reset_index()
    )
    error = grouped.prediction - grouped.truth
    background_error = grouped.background - grouped.truth
    variance = np.sum((grouped.truth - grouped.truth.mean()) ** 2)
    return {
        **keys,
        "grid_months": len(grouped),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1 - np.sum(error**2) / variance) if variance > 0 else np.nan,
        "skill_vs_seasonal_trend": float(1 - np.mean(error**2) / np.mean(background_error**2)),
    }


def calibration_widths(scheme: str, model: str, held: pd.DataFrame) -> pd.DataFrame:
    """Attach leakage-safe symmetric 50/90 interval half-widths."""

    output = held.copy()
    output["width50"] = np.nan
    output["width90"] = np.nan
    if scheme == "forward":
        calibration_source = load_pair("cruise", model)
        calibration_source = calibration_source.loc[calibration_source.yr.le(2018)]
    for fold, fold_rows in output.groupby("fold"):
        if scheme == "forward":
            calibration = calibration_source
        else:
            calibration = output.loc[output.fold.ne(fold)]
        absolute = (calibration.prediction - calibration.truth).abs()
        global_widths = absolute.quantile([0.5, 0.9]).to_numpy(float)
        cells = (
            calibration.assign(
                absolute_error=absolute,
                calibration_band=fco2_band(calibration.background).astype(str),
            )
            .groupby("calibration_band", observed=True)
            .absolute_error.agg(
                size="size",
                q50=lambda values: values.quantile(0.5),
                q90=lambda values: values.quantile(0.9),
            )
        )
        held_band = fco2_band(fold_rows.background).astype(str)
        for band, indices in (
            pd.Series(held_band, index=fold_rows.index).groupby(held_band).groups.items()
        ):
            width = global_widths
            if band in cells.index and cells.loc[band, "size"] >= 200:
                width = cells.loc[band, ["q50", "q90"]].to_numpy(float)
            output.loc[indices, ["width50", "width90"]] = width
    return output


def paired_bootstrap(repeats: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(65023)
    rows = []
    for scheme in ("cruise", "spatial"):
        reference = load_pair(scheme, "v11_random_forest")
        reference = reference[["record_id", "group_key", "truth", "prediction"]].rename(
            columns={"prediction": "reference"}
        )
        for model in MODERN_MODELS:
            candidate = load_pair(scheme, model)[["record_id", "prediction"]].rename(
                columns={"prediction": "candidate"}
            )
            joined = reference.merge(candidate, on="record_id", validate="one_to_one")
            joined["delta"] = (joined.candidate - joined.truth) ** 2 - (
                joined.reference - joined.truth
            ) ** 2
            grouped = joined.groupby("group_key").delta.mean().to_numpy(float)
            draws = np.empty(repeats)
            for index in range(repeats):
                draws[index] = rng.choice(grouped, len(grouped), replace=True).mean()
            rows.append(
                {
                    "scheme": scheme,
                    "candidate": model,
                    "reference": "v11_random_forest",
                    "cruises": len(grouped),
                    "mean_delta_mse": grouped.mean(),
                    "ci_low_delta_mse": np.quantile(draws, 0.025),
                    "ci_high_delta_mse": np.quantile(draws, 0.975),
                    "probability_improved": np.mean(draws < 0),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    if len(list(PARTITIONS.glob("*.parquet"))) != 112:
        raise RuntimeError("expected 112 complete prediction partitions")
    pooled_rows = []
    equal_rows = []
    common_rows = []
    lme_rows = []
    strata_rows = []
    time_rows = []
    annual_rows = []
    extreme_rows = []
    interval_rows = []
    risk_rows = []
    historical_rows = []
    for scheme in SCHEMES:
        for model in MODELS:
            frame = load_pair(scheme, model)
            pooled_rows.append(metrics(frame, scheme=scheme, model=model))
            per_cruise = (
                frame.assign(sq=(frame.prediction - frame.truth) ** 2)
                .groupby("group_key")
                .agg(mse=("sq", "mean"), n=("truth", "size"))
            )
            equal_rows.append(
                {
                    "scheme": scheme,
                    "model": model,
                    "cruises": len(per_cruise),
                    "cruise_equal_rmse": np.sqrt(per_cruise.mse.mean()),
                    "median_cruise_rmse": np.median(np.sqrt(per_cruise.mse)),
                }
            )
            common_rows.append(common_grid_metrics(frame, scheme=scheme, model=model))
            for lme_id, group in frame.groupby("lme_id", observed=True):
                if len(group) >= 20:
                    lme_rows.append(metrics(group, scheme=scheme, model=model, lme_id=lme_id))
            for column in ("fco2_band", "coast_band", "shelf_core_mask"):
                for value, group in frame.groupby(column, observed=True):
                    if len(group) >= 20:
                        strata_rows.append(
                            metrics(
                                group,
                                scheme=scheme,
                                model=model,
                                stratum_type=column,
                                stratum=str(value),
                            )
                        )
            for start, end, label in TIME_BLOCKS:
                group = frame.loc[frame.yr.between(start, end)]
                if len(group):
                    time_rows.append(metrics(group, scheme=scheme, model=model, time_block=label))
            annual = (
                frame.assign(error=frame.prediction - frame.truth)
                .groupby("yr", observed=True)
                .agg(
                    rows=("truth", "size"),
                    cruises=("group_key", "nunique"),
                    truth_mean=("truth", "mean"),
                    prediction_mean=("prediction", "mean"),
                    bias=("error", "mean"),
                )
            )
            annual["model"] = model
            annual["scheme"] = scheme
            annual_rows.extend(annual.reset_index().to_dict("records"))
            lower, upper = frame.truth.quantile([0.1, 0.9])
            for label, group in (
                ("low_decile", frame.loc[frame.truth.le(lower)]),
                ("central", frame.loc[frame.truth.between(lower, upper)]),
                ("high_decile", frame.loc[frame.truth.ge(upper)]),
            ):
                extreme_rows.append(metrics(group, scheme=scheme, model=model, extreme_band=label))
            historical = frame.loc[frame.yr.isin([2004, 2005])]
            if len(historical):
                historical_rows.append(
                    metrics(historical, scheme=scheme, model=model, period="2004_2005_descriptive")
                )
            calibrated = calibration_widths(scheme, model, frame)
            interval_rows.append(
                {
                    "scheme": scheme,
                    "model": model,
                    **interval_scores(
                        calibrated.truth.to_numpy(),
                        calibrated.prediction.to_numpy(),
                        calibrated.width50.to_numpy(),
                        calibrated.width90.to_numpy(),
                    ),
                }
            )
            ordered = calibrated.assign(
                error_sq=(calibrated.prediction - calibrated.truth) ** 2
            ).sort_values("width90")
            for retained in (0.2, 0.4, 0.6, 0.8, 1.0):
                selected = ordered.iloc[: max(1, int(len(ordered) * retained))]
                risk_rows.append(
                    {
                        "scheme": scheme,
                        "model": model,
                        "retained_fraction": retained,
                        "n": len(selected),
                        "rmse": np.sqrt(selected.error_sq.mean()),
                        "mean_width90": selected.width90.mean(),
                    }
                )

    pooled = pd.DataFrame(pooled_rows)
    equal = pd.DataFrame(equal_rows)
    common = pd.DataFrame(common_rows)
    lme = pd.DataFrame(lme_rows)
    strata = pd.DataFrame(strata_rows)
    intervals = pd.DataFrame(interval_rows)
    pooled.to_csv(TABLES / "table01_native_metrics.csv", index=False)
    equal.to_csv(TABLES / "table02_cruise_equal_metrics.csv", index=False)
    common.to_csv(TABLES / "table03_common_grid_metrics.csv", index=False)
    lme.to_csv(TABLES / "table04_lme_metrics.csv", index=False)
    lme.groupby(["scheme", "model"]).agg(
        lme_count=("lme_id", "nunique"),
        macro_lme_rmse=("rmse", "mean"),
        median_lme_rmse=("rmse", "median"),
        worst_lme_rmse=("rmse", "max"),
    ).reset_index().to_csv(TABLES / "table05_lme_macro_worst.csv", index=False)
    strata.to_csv(TABLES / "table06_stratified_metrics.csv", index=False)
    pd.DataFrame(time_rows).to_csv(TABLES / "table07_time_block_metrics.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(TABLES / "table08_annual_diagnostics.csv", index=False)
    pd.DataFrame(extreme_rows).to_csv(TABLES / "table09_extreme_metrics.csv", index=False)
    intervals.to_csv(TABLES / "table10_interval_metrics.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(TABLES / "table11_risk_coverage.csv", index=False)
    bootstrap = paired_bootstrap()
    bootstrap.to_csv(TABLES / "table12_paired_bootstrap.csv", index=False)
    pd.DataFrame(historical_rows).to_csv(TABLES / "table13_historical_2004_2005.csv", index=False)

    qualification_rows = []
    for model in MODELS:
        pivot = lme.loc[
            lme.model.eq(model) & lme.scheme.isin(["cruise", "spatial", "forward"])
        ].pivot(
            index="lme_id", columns="scheme", values=["rmse", "skill_vs_seasonal_trend", "cruises"]
        )
        for lme_id, row in pivot.iterrows():
            complete = all(
                (metric, scheme) in row.index and np.isfinite(row[(metric, scheme)])
                for metric in ("rmse", "skill_vs_seasonal_trend")
                for scheme in ("cruise", "spatial", "forward")
            )
            qualified = (
                complete
                and row[("rmse", "cruise")] < 30
                and row[("rmse", "spatial")] < 40
                and row[("rmse", "forward")] < 40
                and min(
                    row[("skill_vs_seasonal_trend", scheme)]
                    for scheme in ("cruise", "spatial", "forward")
                )
                > 0
                and min(row[("cruises", scheme)] for scheme in ("cruise", "spatial", "forward"))
                >= 2
            )
            qualification_rows.append(
                {
                    "model": model,
                    "lme_id": lme_id,
                    "qualified": qualified,
                    "complete_three_scheme_evidence": complete,
                }
            )
    qualification = pd.DataFrame(qualification_rows)
    qualification.to_csv(TABLES / "table14_lme_qualification.csv", index=False)

    definitions = pd.DataFrame(
        [
            ("seasonal_trend", "training-fold trend/season plus shrunk LME-month climatology"),
            ("v11_random_forest", "honest v1.x random forest with outer-fold cross-fitted SSS"),
            ("catboost_direct", "frozen Issue #54 direct A2 CatBoost"),
            ("catboost_residual", "CatBoost residual around fold-fitted seasonal/trend background"),
            (
                "catboost_residual_sss",
                "same residual CatBoost plus strict outer-fold cross-fitted SSS",
            ),
            ("lme_adapters", "residual-SSS CatBoost plus shrunk fixed-LME/month correction"),
            (
                "gmm_experts",
                "eight probabilistic environmental residual experts shrunk toward global residual-SSS CatBoost",
            ),
        ],
        columns=["model", "definition"],
    )
    definitions.to_csv(TABLES / "table15_model_definitions.csv", index=False)
    pd.read_csv(EXTERNAL / "training_log.csv").to_csv(
        TABLES / "table16_training_log.csv", index=False
    )

    reference = equal.set_index(["scheme", "model"]).cruise_equal_rmse
    pooled_lookup = pooled.set_index(["scheme", "model"])
    lme_summary = pd.read_csv(TABLES / "table05_lme_macro_worst.csv").set_index(["scheme", "model"])
    issue26_caribbean = {"cruise": 9.538, "spatial": 10.033, "forward": 13.571}
    gate_rows = []
    for model in MODERN_MODELS:
        improvement = {
            scheme: 1
            - reference.loc[(scheme, model)] / reference.loc[(scheme, "v11_random_forest")]
            for scheme in SCHEMES
        }
        lme12 = lme.loc[lme.model.eq(model) & lme.lme_id.eq(12)].set_index("scheme")
        candidate_bootstrap = bootstrap.loc[bootstrap.candidate.eq(model)]
        candidate_interval = intervals.loc[intervals.model.eq(model)]
        checks = {
            "cruise-equal improvement over strict v1 >= 5%": improvement["cruise"] >= 0.05,
            "spatial cruise-equal improvement over strict v1 >= 5%": improvement["spatial"] >= 0.05,
            "global spatial skill positive": pooled_lookup.loc[
                ("spatial", model), "skill_vs_seasonal_trend"
            ]
            > 0,
            "global forward skill positive": pooled_lookup.loc[
                ("forward", model), "skill_vs_seasonal_trend"
            ]
            > 0,
            "whole-LME does not degrade from strict v1": improvement["whole_lme"] >= 0,
            "worst-LME ratio <= 1.10": lme_summary.loc[("whole_lme", model), "worst_lme_rmse"]
            / lme_summary.loc[("whole_lme", "v11_random_forest"), "worst_lme_rmse"]
            <= 1.10,
            "Caribbean Issue 26 skill maintained": all(
                lme12.loc[scheme, "rmse"] <= issue26_caribbean[scheme] * 1.05
                for scheme in issue26_caribbean
            ),
            "cruise/spatial bootstrap upper CI < 0": len(candidate_bootstrap) == 2
            and (candidate_bootstrap.ci_high_delta_mse < 0).all(),
            "50% interval coverage in range": candidate_interval.coverage50.between(
                0.45, 0.55
            ).all(),
            "90% interval coverage in range": candidate_interval.coverage90.between(
                0.85, 0.95
            ).all(),
        }
        gate_rows.extend(
            {"candidate": model, "gate": gate, "passed": passed} for gate, passed in checks.items()
        )
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(TABLES / "table17_selection_gate.csv", index=False)
    inventory = pd.DataFrame(
        [
            {
                "artifact": "prediction_partitions",
                "count": len(list(PARTITIONS.glob("*.parquet"))),
                "location": str(PARTITIONS),
            },
            {
                "artifact": "crossfit_sss_partitions",
                "count": len(list((EXTERNAL / "crossfit_sss").glob("*.parquet"))),
                "location": str(EXTERNAL / "crossfit_sss"),
            },
            {
                "artifact": "model_checkpoints",
                "count": len(list((EXTERNAL / "checkpoints").glob("*.*"))),
                "location": str(EXTERNAL / "checkpoints"),
            },
        ]
    )
    inventory.to_csv(TABLES / "table18_source_inventory.csv", index=False)
    centers = (
        load_pair("cruise", "v11_random_forest")
        .groupby("lme_id", observed=True)
        .agg(
            longitude=("longitude", "median"),
            latitude=("latitude", "median"),
            rows=("truth", "size"),
            cruises=("group_key", "nunique"),
        )
        .reset_index()
    )
    centers.to_csv(TABLES / "table19_lme_centers.csv", index=False)
    summary_rows = []
    for model in MODELS:
        row: dict[str, object] = {
            "model": model,
            "qualified_lmes": int(
                qualification.loc[qualification.model.eq(model), "qualified"].sum()
            ),
        }
        for scheme in SCHEMES:
            row[f"{scheme}_rmse"] = pooled_lookup.loc[(scheme, model), "rmse"]
            row[f"{scheme}_cruise_equal_rmse"] = reference.loc[(scheme, model)]
        if model in MODERN_MODELS:
            row["gates_passed"] = int(gate.loc[gate.candidate.eq(model), "passed"].sum())
            row["gates_total"] = int(gate.candidate.eq(model).sum())
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(TABLES / "table20_candidate_summary.csv", index=False)
    annual_frame = pd.DataFrame(annual_rows)
    trend_rows = []
    for (scheme, model), group in annual_frame.groupby(["scheme", "model"]):
        year = group.yr.to_numpy(float)
        trend_rows.append(
            {
                "scheme": scheme,
                "model": model,
                "observed_slope_uatm_per_year": np.polyfit(year, group.truth_mean, 1)[0],
                "predicted_slope_uatm_per_year": np.polyfit(year, group.prediction_mean, 1)[0],
                "bias_slope_uatm_per_year": np.polyfit(year, group.bias, 1)[0],
            }
        )
    pd.DataFrame(trend_rows).to_csv(TABLES / "table21_trend_slopes.csv", index=False)
    forward_rows = []
    for model in MODELS:
        frame = load_pair("forward", model)
        for start, end, label in ((2019, 2021, "2019_2021"), (2022, 2025, "2022_2025")):
            forward_rows.append(
                metrics(
                    frame.loc[frame.yr.between(start, end)],
                    scheme=f"forward_{label}",
                    model=model,
                )
            )
    pd.DataFrame(forward_rows).to_csv(TABLES / "table22_forward_period_metrics.csv", index=False)
    summary = {
        "promoted_model": None,
        "retained_baseline": "v11_random_forest",
        "best_forward_model": pooled.loc[pooled.scheme.eq("forward")]
        .sort_values("rmse")
        .iloc[0]
        .model,
        "all_candidate_gates_passed": False,
        "locked_test_opened": False,
        "external_independent_opened": False,
    }
    (EXTERNAL / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
