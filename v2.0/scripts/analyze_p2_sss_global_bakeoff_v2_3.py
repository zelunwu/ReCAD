"""Analyze the frozen Issue #40 SSS model-family bake-off."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from recad.evaluate.sss_bakeoff import interval_scores

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path(r"C:\backup\phd\data\processed\recad_v2_3\p2_sss_global_bakeoff_v2.3")
PARTITIONS = EXTERNAL / "partitions"
TABLES = EXTERNAL / "tables"
MATCHUP = Path(
    r"C:\backup\phd\data\processed\recad_v2_3\p2_predictor_ablation_v2.3"
    r"\socat_predictor_matchup_v2.3.parquet"
)
CUBE = Path(
    r"C:\backup\phd\data\processed\recad_v2_3"
    r"\global_coastal_predictor_cube_v2.3.nc"
)
FRONT_CACHE = EXTERNAL / "front_diagnostic.parquet"
MODELS = [
    "raw_glorys",
    "robust_regional",
    "catboost_global",
    "gmm_experts",
    "top2_experts",
    "st96",
]
SCHEMES = ["cruise", "spatial", "whole_lme", "forward"]
TIME_BLOCKS = [
    (1993, 1997, "1993_1997"),
    (1998, 2001, "1998_2001"),
    (2002, 2020, "2002_2020"),
    (2021, 2025, "2021_2025"),
]


def metric_row(frame: pd.DataFrame, **keys: object) -> dict[str, object]:
    error = frame.prediction.to_numpy() - frame.truth.to_numpy()
    baseline = frame.sss_bg.to_numpy() - frame.truth.to_numpy()
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
        "skill_vs_raw": float(1 - np.mean(error**2) / np.nanmean(baseline**2)),
    }


def load_predictions() -> pd.DataFrame:
    paths = sorted(PARTITIONS.glob("*.parquet"))
    if len(paths) != 96:
        raise RuntimeError(f"expected 96 prediction partitions, found {len(paths)}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def add_front_strength(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach local GLORYS SSS gradient at each observed node and month."""
    if FRONT_CACHE.is_file():
        front = pd.read_parquet(FRONT_CACHE)
        frame = frame.merge(front, on="record_id", how="left", validate="many_to_one")
        frame["front_band"] = (
            pd.cut(
                frame.front_strength_psu_km,
                [-np.inf, 0.01, 0.03, 0.1, np.inf],
                labels=["lt0.01", "0.01_0.03", "0.03_0.10", "ge0.10"],
            )
            .astype("string")
            .fillna("unavailable")
        )
        return frame
    source = pd.read_parquet(
        MATCHUP,
        columns=["record_id", "coastal_node", "time_index", "latitude"],
    )
    source = source.loc[source.record_id.isin(frame.record_id.unique())].drop_duplicates(
        "record_id"
    )
    with xr.open_dataset(CUBE, decode_times=False) as cube:
        ilat = cube.grid_lat_index.values.astype(np.int32)
        ilon = cube.grid_lon_index.values.astype(np.int32)
        nlon = int(ilon.max()) + 1
        flat = ilat.astype(np.int64) * nlon + ilon
        lookup = dict(
            zip(flat.tolist(), np.arange(len(flat), dtype=np.int32).tolist(), strict=True)
        )
        neighbor = np.full((len(flat), 4), -1, dtype=np.int32)
        for column, delta in enumerate((-nlon, nlon, -1, 1)):
            neighbor[:, column] = np.fromiter(
                (lookup.get(int(key + delta), -1) for key in flat),
                dtype=np.int32,
                count=len(flat),
            )
        source["front_strength_psu_km"] = np.nan
        for time_index, rows in source.groupby("time_index", sort=True):
            nodes = rows.coastal_node.to_numpy(np.int32)
            center = cube.sss_bg.isel(time=int(time_index), node=nodes).values
            slopes = np.full((len(nodes), 4), np.nan, dtype=np.float32)
            lat = rows.latitude.to_numpy(float)
            distances = np.column_stack(
                [
                    np.full(len(nodes), 13.9),
                    np.full(len(nodes), 13.9),
                    13.9 * np.cos(np.deg2rad(lat)),
                    13.9 * np.cos(np.deg2rad(lat)),
                ]
            )
            for column in range(4):
                adjacent = neighbor[nodes, column]
                valid = adjacent >= 0
                if valid.any():
                    values = cube.sss_bg.isel(time=int(time_index), node=adjacent[valid]).values
                    slopes[valid, column] = (
                        np.abs(values - center[valid]) / distances[valid, column]
                    )
            available = np.isfinite(slopes).any(axis=1)
            strength = np.full(len(rows), np.nan, dtype=np.float32)
            strength[available] = np.nanmax(slopes[available], axis=1)
            source.loc[rows.index, "front_strength_psu_km"] = strength
    front = source[["record_id", "front_strength_psu_km"]]
    front.to_parquet(FRONT_CACHE, index=False)
    frame = frame.merge(front, on="record_id", how="left", validate="many_to_one")
    frame["front_band"] = (
        pd.cut(
            frame.front_strength_psu_km,
            [-np.inf, 0.01, 0.03, 0.1, np.inf],
            labels=["lt0.01", "0.01_0.03", "0.03_0.10", "ge0.10"],
        )
        .astype("string")
        .fillna("unavailable")
    )
    return frame


def calibration_widths(frame: pd.DataFrame) -> pd.DataFrame:
    """Estimate symmetric 50/90 intervals without using each held fold's labels."""
    output = frame.copy()
    output["width50"] = np.nan
    output["width90"] = np.nan
    cruise = frame.loc[frame.scheme.eq("cruise") & frame.yr.le(2018)]
    for (scheme, model, fold), held in frame.groupby(["scheme", "model", "fold"]):
        if scheme == "forward":
            calibration = cruise.loc[cruise.model.eq(model)]
        else:
            calibration = frame.loc[
                frame.scheme.eq(scheme) & frame.model.eq(model) & frame.fold.ne(fold)
            ]
        absolute = (calibration.prediction - calibration.truth).abs()
        global_widths = absolute.quantile([0.5, 0.9]).to_numpy()
        by_band = (
            calibration.assign(absolute_error=absolute)
            .groupby("salinity_band", observed=True)
            .absolute_error.agg(
                ["size", lambda values: values.quantile(0.5), lambda values: values.quantile(0.9)]
            )
        )
        by_band.columns = ["size", "q50", "q90"]
        for band, indices in held.groupby("salinity_band", observed=True).groups.items():
            widths = global_widths
            if band in by_band.index and by_band.loc[band, "size"] >= 200:
                widths = by_band.loc[band, ["q50", "q90"]].to_numpy(float)
            output.loc[indices, ["width50", "width90"]] = widths
    return output


def paired_bootstrap(frame: pd.DataFrame, repeats: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(64023)
    rows = []
    for scheme in ("cruise", "spatial"):
        reference = frame.loc[
            frame.scheme.eq(scheme) & frame.model.eq("catboost_global"),
            ["record_id", "group_key", "truth", "prediction"],
        ].rename(columns={"prediction": "reference"})
        for model in ("gmm_experts", "top2_experts", "st96"):
            candidate = frame.loc[
                frame.scheme.eq(scheme) & frame.model.eq(model),
                ["record_id", "prediction"],
            ].rename(columns={"prediction": "candidate"})
            joined = reference.merge(candidate, on="record_id", validate="one_to_one")
            joined["delta"] = (joined.candidate - joined.truth) ** 2 - (
                joined.reference - joined.truth
            ) ** 2
            cruise_delta = joined.groupby("group_key").delta.mean().to_numpy()
            draws = np.empty(repeats)
            for index in range(repeats):
                draws[index] = rng.choice(cruise_delta, len(cruise_delta), replace=True).mean()
            rows.append(
                {
                    "scheme": scheme,
                    "candidate": model,
                    "reference": "catboost_global",
                    "cruises": len(cruise_delta),
                    "mean_delta_mse": cruise_delta.mean(),
                    "ci_low_delta_mse": np.quantile(draws, 0.025),
                    "ci_high_delta_mse": np.quantile(draws, 0.975),
                    "probability_improved": np.mean(draws < 0),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    frame = add_front_strength(load_predictions())
    frame = calibration_widths(frame)

    pooled = pd.DataFrame(
        [
            metric_row(group, scheme=scheme, model=model)
            for (scheme, model), group in frame.groupby(["scheme", "model"])
        ]
    )
    pooled.to_csv(TABLES / "table01_pooled_metrics.csv", index=False)

    cruise_equal_rows = []
    for (scheme, model), group in frame.groupby(["scheme", "model"]):
        per_cruise = (
            group.assign(sq=(group.prediction - group.truth) ** 2)
            .groupby("group_key")
            .agg(mse=("sq", "mean"), n=("truth", "size"))
        )
        cruise_equal_rows.append(
            {
                "scheme": scheme,
                "model": model,
                "cruises": len(per_cruise),
                "cruise_equal_rmse": np.sqrt(per_cruise.mse.mean()),
                "median_cruise_rmse": np.median(np.sqrt(per_cruise.mse)),
            }
        )
    cruise_equal = pd.DataFrame(cruise_equal_rows)
    cruise_equal.to_csv(TABLES / "table02_cruise_equal_metrics.csv", index=False)

    lme_rows = []
    for (scheme, model, lme_id), group in frame.groupby(["scheme", "model", "lme_id"]):
        if len(group) >= 20:
            lme_rows.append(metric_row(group, scheme=scheme, model=model, lme_id=lme_id))
    lme = pd.DataFrame(lme_rows)
    lme.to_csv(TABLES / "table03_lme_metrics.csv", index=False)
    lme.groupby(["scheme", "model"]).agg(
        lme_count=("lme_id", "nunique"),
        macro_lme_rmse=("rmse", "mean"),
        median_lme_rmse=("rmse", "median"),
        worst_lme_rmse=("rmse", "max"),
    ).reset_index().to_csv(TABLES / "table04_lme_macro_worst.csv", index=False)

    strata_rows = []
    for column in ("salinity_band", "coast_band", "shelf_core_mask", "front_band"):
        for (scheme, model, value), group in frame.groupby(
            ["scheme", "model", column], observed=True
        ):
            if len(group) >= 20:
                strata_rows.append(
                    metric_row(
                        group, scheme=scheme, model=model, stratum_type=column, stratum=str(value)
                    )
                )
    strata = pd.DataFrame(strata_rows)
    strata.to_csv(TABLES / "table05_stratified_metrics.csv", index=False)

    time_rows = []
    for start, end, label in TIME_BLOCKS:
        subset = frame.loc[frame.yr.between(start, end)]
        for (scheme, model), group in subset.groupby(["scheme", "model"]):
            time_rows.append(metric_row(group, scheme=scheme, model=model, time_block=label))
    pd.DataFrame(time_rows).to_csv(TABLES / "table06_time_block_metrics.csv", index=False)

    interval_rows = []
    risk_rows = []
    for (scheme, model), group in frame.groupby(["scheme", "model"]):
        diagnostics = interval_scores(
            group.truth.to_numpy(),
            group.prediction.to_numpy(),
            group.width50.to_numpy(),
            group.width90.to_numpy(),
        )
        interval_rows.append({"scheme": scheme, "model": model, **diagnostics})
        ordered = group.assign(error_sq=(group.prediction - group.truth) ** 2).sort_values(
            "width90"
        )
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
    intervals = pd.DataFrame(interval_rows)
    intervals.to_csv(TABLES / "table07_interval_metrics.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(TABLES / "table08_risk_coverage.csv", index=False)
    paired_bootstrap(frame).to_csv(TABLES / "table09_paired_bootstrap.csv", index=False)

    forward_rows = []
    for start, end, label in ((2019, 2021, "2019_2021"), (2022, 2025, "2022_2025")):
        subset = frame.loc[frame.scheme.eq("forward") & frame.yr.between(start, end)]
        for model, group in subset.groupby("model"):
            forward_rows.append(metric_row(group, scheme=f"forward_{label}", model=model))
    pd.DataFrame(forward_rows).to_csv(TABLES / "table10_forward_period_metrics.csv", index=False)

    definitions = pd.DataFrame(
        [
            ("raw_glorys", "Uncorrected GLORYS SSS background"),
            ("robust_regional", "Fold-fitted robust region/season affine residual correction"),
            ("catboost_global", "Frozen Issue #54 global CatBoost residual baseline"),
            (
                "gmm_experts",
                "Eight probabilistic environmental experts, shrunk 0.5 toward global CatBoost",
            ),
            ("top2_experts", "Eight environmental experts with top-two soft gating"),
            ("st96", "Single-target grouped-token ST96 transformer residual model"),
        ],
        columns=["model", "definition"],
    )
    definitions.to_csv(TABLES / "table11_model_definitions.csv", index=False)
    pd.read_csv(EXTERNAL / "training_log.csv").to_csv(
        TABLES / "table12_training_log.csv", index=False
    )

    baseline = pooled.set_index(["scheme", "model"]).rmse
    effects = []
    for model in MODELS:
        if model == "catboost_global":
            continue
        for scheme in SCHEMES:
            effects.append(
                {
                    "model": model,
                    "scheme": scheme,
                    "reference": "catboost_global",
                    "relative_rmse_improvement": 1
                    - baseline.loc[(scheme, model)] / baseline.loc[(scheme, "catboost_global")],
                }
            )
    pd.DataFrame(effects).to_csv(TABLES / "table13_effects_vs_catboost.csv", index=False)

    bootstrap = pd.read_csv(TABLES / "table09_paired_bootstrap.csv")
    equal = cruise_equal.set_index(["scheme", "model"]).cruise_equal_rmse
    equal_effect = pd.Series(
        {
            scheme: 1 - equal.loc[(scheme, "gmm_experts")] / equal.loc[(scheme, "catboost_global")]
            for scheme in SCHEMES
        }
    )
    low_salinity = strata.loc[
        strata.scheme.eq("cruise")
        & strata.stratum_type.eq("salinity_band")
        & strata.stratum.isin(["lt20", "20_30"])
        & strata.model.isin(["catboost_global", "gmm_experts"])
    ].pivot(index="stratum", columns="model", values="rmse")
    lme17 = lme.loc[lme.lme_id.eq(17) & lme.model.isin(["catboost_global", "gmm_experts"])].pivot(
        index="scheme", columns="model", values="rmse"
    )
    gmm_intervals = intervals.loc[intervals.model.eq("gmm_experts")]
    gate = pd.DataFrame(
        [
            ("median cruise-equal improvement >= 5%", equal_effect.median() >= 0.05),
            ("cruise-equal improvement >= 5%", equal_effect.cruise >= 0.05),
            ("spatial cruise-equal improvement >= 5%", equal_effect.spatial >= 0.05),
            ("whole-LME cruise-equal does not degrade", equal_effect.whole_lme >= 0),
            ("forward cruise-equal does not degrade", equal_effect.forward >= 0),
            (
                "low-salinity cruise strata do not degrade",
                bool((low_salinity.gmm_experts <= low_salinity.catboost_global).all()),
            ),
            (
                "LME 17 does not degrade in any scheme",
                bool((lme17.gmm_experts <= lme17.catboost_global).all()),
            ),
            (
                "cruise paired bootstrap upper CI < 0",
                bootstrap.loc[
                    (bootstrap.candidate == "gmm_experts") & (bootstrap.scheme == "cruise"),
                    "ci_high_delta_mse",
                ].iloc[0]
                < 0,
            ),
            (
                "spatial paired bootstrap upper CI < 0",
                bootstrap.loc[
                    (bootstrap.candidate == "gmm_experts") & (bootstrap.scheme == "spatial"),
                    "ci_high_delta_mse",
                ].iloc[0]
                < 0,
            ),
            (
                "50% interval coverage in frozen range",
                bool(gmm_intervals.coverage50.between(0.45, 0.55).all()),
            ),
            (
                "90% interval coverage in frozen range",
                bool(gmm_intervals.coverage90.between(0.85, 0.95).all()),
            ),
        ],
        columns=["gate", "passed"],
    )
    gate["candidate"] = "gmm_experts"
    gate.to_csv(TABLES / "table14_selection_gate.csv", index=False)

    inventory = pd.DataFrame(
        [
            {
                "artifact": "prediction_partitions",
                "count": len(list(PARTITIONS.glob("*.parquet"))),
                "location": str(PARTITIONS),
            },
            {
                "artifact": "model_checkpoints",
                "count": len(list((EXTERNAL / "checkpoints").rglob("*.*"))),
                "location": str(EXTERNAL / "checkpoints"),
            },
            {
                "artifact": "training_log",
                "count": 1,
                "location": str(EXTERNAL / "training_log.csv"),
            },
        ]
    )
    inventory.to_csv(TABLES / "table15_source_inventory.csv", index=False)
    summary = {
        "candidate": "gmm_experts",
        "all_release_gates_passed": bool(gate.passed.all()),
        "decision": "retain_catboost_global_production_baseline_gmm_as_supported_domain_challenger",
        "locked_test_opened": False,
        "external_independent_opened": False,
    }
    (EXTERNAL / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
