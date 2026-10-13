"""Analyze the frozen Issue #42 temporal-context and coastal-connectivity experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from recad.evaluate.fco2_bakeoff import fco2_band, interval_scores

EXTERNAL = Path(r"C:\backup\phd\data\processed\recad_v2_3\p2_context_connectivity_v2.3")
PARTITIONS = EXTERNAL / "partitions"
TABLES = EXTERNAL / "tables"
MODELS = ["v11_random_forest", "lag_catboost", "gru_context", "st96_local", "st96_graph"]
CANDIDATES = MODELS[1:]
SCHEMES = ["cruise", "spatial", "whole_lme", "forward"]
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
        raise RuntimeError(f"expected {expected} {scheme}/{model} partitions, found {len(paths)}")
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
        "r2": float(1 - np.sum(error**2) / variance) if variance else np.nan,
        "skill_vs_seasonal_trend": float(1 - np.mean(error**2) / np.mean(background_error**2)),
    }


def common_grid_metrics(frame: pd.DataFrame, **keys: object) -> dict[str, object]:
    grouped = (
        frame.groupby(["time_index", "coastal_node"], observed=True)
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
        "r2": float(1 - np.sum(error**2) / variance) if variance else np.nan,
        "skill_vs_seasonal_trend": float(1 - np.mean(error**2) / np.mean(background_error**2)),
    }


def calibration_widths(scheme: str, model: str, held: pd.DataFrame) -> pd.DataFrame:
    output = held.copy()
    output["width50"] = np.nan
    output["width90"] = np.nan
    if scheme == "forward":
        calibration_source = load_pair("cruise", model)
        calibration_source = calibration_source.loc[calibration_source.yr.le(2018)]
    for fold, fold_rows in output.groupby("fold"):
        calibration = (
            calibration_source if scheme == "forward" else output.loc[output.fold.ne(fold)]
        )
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
        held_band = pd.Series(fco2_band(fold_rows.background).astype(str), index=fold_rows.index)
        for band, indices in held_band.groupby(held_band).groups.items():
            width = global_widths
            if band in cells.index and cells.loc[band, "size"] >= 200:
                width = cells.loc[band, ["q50", "q90"]].to_numpy(float)
            output.loc[indices, ["width50", "width90"]] = width
    return output


def paired_bootstrap(repeats: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(76023)
    rows = []
    for scheme in ("cruise", "spatial"):
        reference = load_pair(scheme, "v11_random_forest")[
            ["record_id", "group_key", "truth", "prediction"]
        ].rename(columns={"prediction": "reference"})
        for model in CANDIDATES:
            candidate = load_pair(scheme, model)[["record_id", "prediction"]].rename(
                columns={"prediction": "candidate"}
            )
            joined = reference.merge(candidate, on="record_id", validate="one_to_one")
            joined["delta"] = (joined.candidate - joined.truth) ** 2 - (
                joined.reference - joined.truth
            ) ** 2
            grouped = joined.groupby("group_key", observed=True).delta.mean().to_numpy(float)
            draws = np.empty(repeats)
            for start in range(0, repeats, 100):
                count = min(100, repeats - start)
                index = rng.integers(0, len(grouped), size=(count, len(grouped)))
                draws[start : start + count] = grouped[index].mean(axis=1)
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


def graph_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    neighbours = np.load(EXTERNAL / "context_cache" / "neighbours.npy", mmap_mode="r")
    graph_rows = []
    front_rows = []
    rng = np.random.default_rng(42)
    for model in MODELS:
        frame = load_pair("cruise", model)
        nodes = (
            frame.groupby(["time_index", "coastal_node"], as_index=False)
            .agg(
                truth=("truth", "mean"),
                prediction=("prediction", "mean"),
                lme_id=("lme_id", "first"),
            )
            .reset_index(drop=True)
        )
        nodes["residual"] = nodes.truth - nodes.prediction
        pairs = []
        for slot in (1, 3):
            left = nodes.copy()
            left["neighbour_node"] = neighbours[left.coastal_node.to_numpy(int), slot]
            left = left.loc[left.neighbour_node.ge(0)]
            right = nodes.rename(
                columns={
                    "coastal_node": "neighbour_node",
                    "truth": "neighbour_truth",
                    "prediction": "neighbour_prediction",
                    "residual": "neighbour_residual",
                    "lme_id": "neighbour_lme",
                }
            )[
                [
                    "time_index",
                    "neighbour_node",
                    "neighbour_truth",
                    "neighbour_prediction",
                    "neighbour_residual",
                    "neighbour_lme",
                ]
            ]
            pairs.append(left.merge(right, on=["time_index", "neighbour_node"], how="inner"))
        pair = pd.concat(pairs, ignore_index=True)
        correlation = pair.residual.corr(pair.neighbour_residual)
        permuted = []
        for _, group in pair.groupby("time_index", observed=True):
            if len(group) < 3:
                continue
            values = group.neighbour_residual.to_numpy().copy()
            rng.shuffle(values)
            permuted.append(np.corrcoef(group.residual, values)[0, 1])
        graph_rows.append(
            {
                "model": model,
                "pairs": len(pair),
                "adjacent_residual_correlation": correlation,
                "within_time_permutation_correlation": np.nanmean(permuted),
                "same_sign_fraction": np.mean(
                    np.sign(pair.residual) == np.sign(pair.neighbour_residual)
                ),
            }
        )
        truth_gradient = np.abs(pair.truth - pair.neighbour_truth)
        prediction_gradient = np.abs(pair.prediction - pair.neighbour_prediction)
        truth_front = truth_gradient >= np.quantile(truth_gradient, 0.9)
        prediction_front = prediction_gradient >= np.quantile(prediction_gradient, 0.9)
        for boundary, selection in (
            ("within_lme", pair.lme_id.eq(pair.neighbour_lme)),
            ("cross_lme", pair.lme_id.ne(pair.neighbour_lme)),
            ("all", np.ones(len(pair), dtype=bool)),
        ):
            if selection.sum() < 20:
                continue
            front_rows.append(
                {
                    "model": model,
                    "boundary": boundary,
                    "pairs": int(selection.sum()),
                    "truth_mean_abs_gradient": truth_gradient[selection].mean(),
                    "prediction_mean_abs_gradient": prediction_gradient[selection].mean(),
                    "gradient_bias": (
                        prediction_gradient[selection] - truth_gradient[selection]
                    ).mean(),
                    "gradient_correlation": np.corrcoef(
                        truth_gradient[selection], prediction_gradient[selection]
                    )[0, 1],
                    "front_recall": np.mean(prediction_front[selection & truth_front]),
                }
            )
    return pd.DataFrame(graph_rows), pd.DataFrame(front_rows)


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    if len(list(PARTITIONS.glob("*.parquet"))) != 80:
        raise RuntimeError("expected 80 complete prediction partitions")
    pooled_rows, equal_rows, common_rows, lme_rows = [], [], [], []
    strata_rows, time_rows, annual_rows, extreme_rows = [], [], [], []
    interval_rows, risk_rows, historical_rows = [], [], []
    for scheme in SCHEMES:
        for model in MODELS:
            frame = load_pair(scheme, model)
            pooled_rows.append(metrics(frame, scheme=scheme, model=model))
            per_cruise = (
                frame.assign(sq=(frame.prediction - frame.truth) ** 2)
                .groupby("group_key", observed=True)
                .sq.mean()
            )
            equal_rows.append(
                {
                    "scheme": scheme,
                    "model": model,
                    "cruises": len(per_cruise),
                    "cruise_equal_rmse": np.sqrt(per_cruise.mean()),
                    "median_cruise_rmse": np.median(np.sqrt(per_cruise)),
                }
            )
            common_rows.append(common_grid_metrics(frame, scheme=scheme, model=model))
            for lme_id, group in frame.groupby("lme_id", observed=True):
                if len(group) >= 20:
                    lme_rows.append(metrics(group, scheme=scheme, model=model, lme_id=lme_id))
            for column in ("support_band", "coast_band", "shelf_core_mask"):
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
                .reset_index()
            )
            annual["model"] = model
            annual["scheme"] = scheme
            annual_rows.extend(annual.to_dict("records"))
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
                        calibrated.truth,
                        calibrated.prediction,
                        calibrated.width50,
                        calibrated.width90,
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
    pd.DataFrame(strata_rows).to_csv(TABLES / "table06_stratified_metrics.csv", index=False)
    pd.DataFrame(time_rows).to_csv(TABLES / "table07_time_block_metrics.csv", index=False)
    annual = pd.DataFrame(annual_rows)
    annual.to_csv(TABLES / "table08_annual_diagnostics.csv", index=False)
    pd.DataFrame(extreme_rows).to_csv(TABLES / "table09_extreme_metrics.csv", index=False)
    intervals.to_csv(TABLES / "table10_interval_metrics.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(TABLES / "table11_risk_coverage.csv", index=False)
    bootstrap = paired_bootstrap()
    bootstrap.to_csv(TABLES / "table12_paired_bootstrap.csv", index=False)
    pd.DataFrame(historical_rows).to_csv(TABLES / "table13_historical_2004_2005.csv", index=False)
    graph, fronts = graph_diagnostics()
    graph.to_csv(TABLES / "table14_graph_residual_structure.csv", index=False)
    fronts.to_csv(TABLES / "table15_front_boundary_metrics.csv", index=False)
    resources = pd.read_csv(EXTERNAL / "training_log.csv")
    resources.to_csv(TABLES / "table16_training_resources.csv", index=False)
    pd.read_csv(EXTERNAL / "learning_curve.csv").to_csv(
        TABLES / "table17_learning_curve.csv", index=False
    )
    definitions = pd.DataFrame(
        [
            ("v11_random_forest", "Issue #41 strict point random-forest baseline"),
            ("lag_catboost", "residual CatBoost with six months of label-free local predictors"),
            ("gru_context", "two-layer 64-unit GRU over six local predictor months"),
            ("st96_local", "two-layer d_model=96 temporal transformer over local history"),
            ("st96_graph", "same ST96 plus four-neighbour water-connected predictor history"),
        ],
        columns=["model", "definition"],
    )
    definitions.to_csv(TABLES / "table18_model_definitions.csv", index=False)
    effective = pd.DataFrame(
        [
            {
                "rows": 887811,
                "cruises": 5143,
                "coastal_nodes": 117324,
                "node_months": 788556,
                "lmes": 54,
                "median_rows_per_cruise": 123,
                "p90_rows_per_cruise": 389,
            }
        ]
    )
    effective.to_csv(TABLES / "table19_effective_sample_size.csv", index=False)

    equal_lookup = equal.set_index(["scheme", "model"]).cruise_equal_rmse
    lme_summary = pd.read_csv(TABLES / "table05_lme_macro_worst.csv").set_index(["scheme", "model"])
    gate_rows = []
    for model in CANDIDATES:
        improvement = {
            scheme: 1
            - equal_lookup.loc[(scheme, model)] / equal_lookup.loc[(scheme, "v11_random_forest")]
            for scheme in SCHEMES
        }
        candidate_intervals = intervals.loc[intervals.model.eq(model)]
        candidate_bootstrap = bootstrap.loc[bootstrap.candidate.eq(model)]
        checks = {
            "at least two outer schemes improve >=5%": sum(
                value >= 0.05 for value in improvement.values()
            )
            >= 2,
            "no outer scheme degrades over 1%": min(improvement.values()) >= -0.01,
            "cruise/spatial paired bootstrap upper CI <0": len(candidate_bootstrap) == 2
            and (candidate_bootstrap.ci_high_delta_mse < 0).all(),
            "worst whole-LME ratio <=1.10": lme_summary.loc[("whole_lme", model), "worst_lme_rmse"]
            / lme_summary.loc[("whole_lme", "v11_random_forest"), "worst_lme_rmse"]
            <= 1.10,
            "50% interval coverage calibrated": candidate_intervals.coverage50.between(
                0.45, 0.55
            ).all(),
            "90% interval coverage calibrated": candidate_intervals.coverage90.between(
                0.85, 0.95
            ).all(),
        }
        for gate, passed in checks.items():
            gate_rows.append(
                {
                    "candidate": model,
                    "gate": gate,
                    "passed": bool(passed),
                    "cruise_improvement": improvement["cruise"],
                    "spatial_improvement": improvement["spatial"],
                    "whole_lme_improvement": improvement["whole_lme"],
                    "forward_improvement": improvement["forward"],
                }
            )
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(TABLES / "table20_selection_gate.csv", index=False)
    trend_rows = []
    for (scheme, model), group in annual.groupby(["scheme", "model"]):
        trend_rows.append(
            {
                "scheme": scheme,
                "model": model,
                "bias_slope_per_year": np.polyfit(group.yr, group.bias, 1)[0],
                "truth_slope_per_year": np.polyfit(group.yr, group.truth_mean, 1)[0],
                "prediction_slope_per_year": np.polyfit(group.yr, group.prediction_mean, 1)[0],
            }
        )
    pd.DataFrame(trend_rows).to_csv(TABLES / "table21_trend_slopes.csv", index=False)
    forward_period_rows = []
    for model in MODELS:
        frame = load_pair("forward", model)
        for start, end, label in ((2019, 2021, "2019_2021"), (2022, 2025, "2022_2025")):
            forward_period_rows.append(
                metrics(
                    frame.loc[frame.yr.between(start, end)],
                    scheme="forward",
                    model=model,
                    period=label,
                )
            )
    pd.DataFrame(forward_period_rows).to_csv(
        TABLES / "table22_forward_period_metrics.csv", index=False
    )
    summary = {
        "decision": "stop_scaling_context_family",
        "promoted_model": None,
        "retained_baseline": "v11_random_forest",
        "graph_triggered": True,
        "graph_trigger_correlation": json.loads(
            (EXTERNAL / "graph_trigger.json").read_text(encoding="utf-8")
        )["correlation"],
        "best_forward_model": pooled.loc[pooled.scheme.eq("forward")]
        .sort_values("rmse")
        .iloc[0]
        .model,
        "best_forward_rmse": pooled.loc[pooled.scheme.eq("forward")].rmse.min(),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "provisional_2026_opened": False,
    }
    (EXTERNAL / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
