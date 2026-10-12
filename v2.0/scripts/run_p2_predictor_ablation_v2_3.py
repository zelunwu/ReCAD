"""Run the frozen Issue #54 SSS/fCO2 predictor-ablation experiment.

Large matchups and predictions are written outside Git.  The script is deliberately
resumable: every completed fold/variant prediction is an independent Parquet file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from netCDF4 import Dataset

from recad.evaluate.predictor_ablation import (
    AblationVariant,
    balanced_sample_weight,
    frozen_splits,
    paired_group_bootstrap,
    qualifies_addition,
    regression_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:\backup\phd\data\processed\recad_v2_3")
CUBE = DATA / "global_coastal_predictor_cube_v2.3.nc"
OBSERVATIONS = DATA / "global_socat_cache_v2.3.parquet"
SPLITS = DATA / "global_group_split_manifest_v2.3.parquet"
OUTPUT = DATA / "p2_predictor_ablation_v2.3"
MATCHUP = OUTPUT / "socat_predictor_matchup_v2.3.parquet"
PREDICTIONS = OUTPUT / "fold_predictions"

GEO = (
    "year_centered",
    "month_sin",
    "month_cos",
    "latitude_sin",
    "latitude_cos",
    "longitude_sin",
    "longitude_cos",
)
GEOMETRY = (
    "distance_to_significant_land_km",
    "bathymetry_m",
    "shelf_core_mask",
    "slope_deep_mask",
)
MANDATORY = ("sst", "wspd", "xco2air")
SSS_BACKGROUND = ("sss_bg", "sss_bg_available")
ADT = ("adt", "adt_available")
VARIANTS = (
    AblationVariant("A0_geo_time", GEO),
    AblationVariant("A1_geometry", GEO + GEOMETRY),
    AblationVariant("A2_environment", GEO + GEOMETRY + MANDATORY),
    AblationVariant("A3_background_sss", GEO + GEOMETRY + MANDATORY + SSS_BACKGROUND),
    AblationVariant("A4_adt", GEO + GEOMETRY + MANDATORY + ADT),
    AblationVariant("A5_sss_adt", GEO + GEOMETRY + MANDATORY + SSS_BACKGROUND + ADT),
    AblationVariant(
        "A6_missing_robust",
        GEO + GEOMETRY + MANDATORY + SSS_BACKGROUND + ADT,
        dropout_sss=0.05,
        dropout_adt=0.15,
    ),
)
SCHEMES = ("cruise", "spatial", "whole_lme", "forward")
TARGETS = {"SSS": "salinity", "fCO2": "fco2"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_matchup(*, force: bool = False) -> Path:
    """Join frozen target rows to the exact monthly coastal-node predictors."""

    if MATCHUP.exists() and not force:
        print(f"reuse matchup {MATCHUP}", flush=True)
        return MATCHUP
    OUTPUT.mkdir(parents=True, exist_ok=True)
    observations = pd.read_parquet(OBSERVATIONS).reset_index(names="record_id")
    observations = observations.loc[
        observations.yr.between(1993, 2025)
        & observations.training_eligible
        & observations.on_v2_2_product_mask
    ].reset_index(drop=True)
    observations["time_index"] = (observations.yr - 1993) * 12 + observations.mon - 1
    observations["year_centered"] = (observations.yr - 2009).astype("float32")

    static_names = (
        "grid_flat",
        "latitude_sin",
        "latitude_cos",
        "longitude_sin",
        "longitude_cos",
        "distance_to_significant_land_km",
        "bathymetry_m",
        "shelf_core_mask",
        "slope_deep_mask",
    )
    dynamic_names = (
        "sst",
        "wspd",
        "xco2air",
        "sss_bg",
        "sss_bg_available",
        "adt",
        "adt_available",
    )
    with Dataset(CUBE) as cube:
        cube_flat = np.asarray(cube.variables["grid_flat"][:], dtype=np.int64)
        lookup = np.full(1297 * 2880, -1, dtype=np.int32)
        lookup[cube_flat] = np.arange(len(cube_flat), dtype=np.int32)
        mapped_node = lookup[observations.grid_flat.to_numpy(np.int64)]
        inside = mapped_node >= 0
        if not inside.all():
            print(
                f"exclude {(~inside).sum():,} legacy-mask rows outside frozen cube geometry",
                flush=True,
            )
            observations = observations.loc[inside].reset_index(drop=True)
            mapped_node = mapped_node[inside]
        node = mapped_node.astype(np.int64)
        observations["coastal_node"] = node.astype(np.int32)
        for name in static_names[1:]:
            observations[name] = np.asarray(cube.variables[name][node])
        observations["month_sin"] = np.asarray(
            cube.variables["month_sin"][observations.time_index.to_numpy()]
        )
        observations["month_cos"] = np.asarray(
            cube.variables["month_cos"][observations.time_index.to_numpy()]
        )
        for name in dynamic_names:
            observations[name] = (
                np.nan if name in {"sst", "wspd", "xco2air", "sss_bg", "adt"} else 0
            )
        for time_index, row_index in observations.groupby("time_index", sort=True).groups.items():
            selected = np.asarray(row_index, dtype=np.int64)
            selected_nodes = node[selected]
            for name in dynamic_names:
                observations.loc[selected, name] = np.asarray(
                    cube.variables[name][int(time_index), selected_nodes]
                )
            if int(time_index) % 24 == 0:
                print(f"matched month {int(time_index) + 1}/396", flush=True)

    float_columns = set(GEO + GEOMETRY + MANDATORY + SSS_BACKGROUND + ADT) - {
        "shelf_core_mask",
        "slope_deep_mask",
        "sss_bg_available",
        "adt_available",
    }
    for name in float_columns:
        observations[name] = observations[name].astype("float32")
    observations.to_parquet(MATCHUP, index=False, compression="zstd")
    metadata = {
        "rows": len(observations),
        "columns": list(observations.columns),
        "cube_sha256": sha256(CUBE),
        "observations_sha256": sha256(OBSERVATIONS),
        "splits_sha256": sha256(SPLITS),
        "matchup_sha256": sha256(MATCHUP),
        "targets_in_cube": False,
    }
    (OUTPUT / "matchup_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {MATCHUP} ({len(observations):,} rows)", flush=True)
    return MATCHUP


def apply_dropout(fit: pd.DataFrame, variant: AblationVariant, *, seed: int) -> pd.DataFrame:
    """Apply deterministic training-only field dropout for A6."""

    if not variant.dropout_sss and not variant.dropout_adt:
        return fit.loc[:, variant.features]
    values = fit.loc[:, variant.features].copy()
    rng = np.random.default_rng(seed)
    if variant.dropout_sss:
        selected = rng.random(len(values)) < variant.dropout_sss
        values.loc[selected, "sss_bg"] = np.nan
        values.loc[selected, "sss_bg_available"] = 0
    if variant.dropout_adt:
        selected = rng.random(len(values)) < variant.dropout_adt
        values.loc[selected, "adt"] = np.nan
        values.loc[selected, "adt_available"] = 0
    return values


def run_models(*, iterations: int = 300, task_type: str = "GPU") -> None:
    """Fit every frozen target/scheme/fold/variant, resuming completed files."""

    if not MATCHUP.exists():
        build_matchup()
    frame = pd.read_parquet(MATCHUP)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT / "training_log_current.csv"
    run_log: list[dict[str, object]] = (
        pd.read_csv(log_path).to_dict("records") if log_path.exists() else []
    )
    total_start = time.perf_counter()
    for target_index, (target, truth_name) in enumerate(TARGETS.items()):
        eligible = frame[truth_name].notna()
        target_frame = frame.loc[eligible].reset_index(drop=True)
        print(f"target {target}: {len(target_frame):,} rows", flush=True)
        for scheme_index, scheme in enumerate(SCHEMES):
            for split in frozen_splits(target_frame, scheme):
                fit = target_frame.iloc[split.fit_index]
                held = target_frame.iloc[split.held_index]
                for variant_index, variant in enumerate(VARIANTS):
                    destination = PREDICTIONS / (
                        f"{target}_{scheme}_fold{split.fold}_{variant.name}.parquet"
                    )
                    if destination.exists():
                        print(f"reuse {destination.name}", flush=True)
                        continue
                    seed = (
                        54023
                        + target_index * 1000
                        + scheme_index * 100
                        + split.fold * 10
                        + variant_index
                    )
                    model = CatBoostRegressor(
                        iterations=iterations,
                        depth=8,
                        learning_rate=0.05,
                        loss_function="RMSE",
                        random_seed=seed,
                        task_type=task_type,
                        devices="0" if task_type == "GPU" else None,
                        random_strength=0.5,
                        l2_leaf_reg=5.0,
                        allow_writing_files=False,
                        verbose=False,
                    )
                    start = time.perf_counter()
                    residual_sss = target == "SSS" and "sss_bg" in variant.features
                    model_fit = fit.loc[fit.sss_bg.notna()] if residual_sss else fit
                    model_truth = model_fit[truth_name].to_numpy(float)
                    if residual_sss:
                        model_truth = model_truth - model_fit.sss_bg.to_numpy(float)
                    model.fit(
                        apply_dropout(model_fit, variant, seed=seed),
                        model_truth,
                        sample_weight=balanced_sample_weight(model_fit),
                    )
                    if residual_sss:
                        baseline_path = PREDICTIONS / (
                            f"{target}_{scheme}_fold{split.fold}_A2_environment.parquet"
                        )
                        baseline = pd.read_parquet(
                            baseline_path, columns=["record_id", "prediction"]
                        ).set_index("record_id")
                        prediction = held.record_id.map(baseline.prediction).to_numpy(
                            "float32", copy=True
                        )
                        available = held.sss_bg.notna().to_numpy()
                        correction = model.predict(held.loc[available, variant.features]).astype(
                            "float32"
                        )
                        prediction[available] = (
                            held.loc[available, "sss_bg"].to_numpy("float32") + correction
                        )
                    else:
                        prediction = model.predict(held.loc[:, variant.features]).astype("float32")
                    result = held.loc[
                        :,
                        [
                            "record_id",
                            "group_key",
                            "yr",
                            "mon",
                            "lme_id",
                            "sss_bg_available",
                            "adt_available",
                        ],
                    ].copy()
                    result["truth"] = held[truth_name].to_numpy("float32")
                    result["prediction"] = prediction
                    result["target"] = target
                    result["scheme"] = scheme
                    result["fold"] = split.fold
                    result["variant"] = variant.name
                    result.to_parquet(destination, index=False, compression="zstd")
                    elapsed = time.perf_counter() - start
                    entry = {
                        "target": target,
                        "scheme": scheme,
                        "fold": split.fold,
                        "variant": variant.name,
                        "fit_rows": len(model_fit),
                        "held_rows": len(held),
                        "features": len(variant.features),
                        "iterations": iterations,
                        "task_type": task_type,
                        "target_mode": "background_residual" if residual_sss else "direct",
                        "seconds": elapsed,
                        "seed": seed,
                    }
                    run_log = [
                        row
                        for row in run_log
                        if not (
                            row["target"] == target
                            and row["scheme"] == scheme
                            and int(row["fold"]) == split.fold
                            and row["variant"] == variant.name
                        )
                    ]
                    run_log.append(entry)
                    pd.DataFrame(run_log).to_csv(log_path, index=False)
                    print(
                        f"done {destination.name}: {elapsed:.1f}s; total "
                        f"{(time.perf_counter() - total_start) / 60:.1f} min",
                        flush=True,
                    )


def prediction_matrix(target: str, scheme: str) -> pd.DataFrame:
    """Load one OOF prediction matrix with one column per variant."""

    merged: pd.DataFrame | None = None
    keys = [
        "record_id",
        "group_key",
        "yr",
        "mon",
        "lme_id",
        "sss_bg_available",
        "adt_available",
        "truth",
    ]
    for variant in VARIANTS:
        paths = sorted(PREDICTIONS.glob(f"{target}_{scheme}_fold*_{variant.name}.parquet"))
        expected = 1 if scheme == "forward" else 5
        if len(paths) != expected:
            raise FileNotFoundError(
                f"{target}/{scheme}/{variant.name}: expected {expected} folds, found {len(paths)}"
            )
        current = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        current = current[[*keys, "prediction"]].rename(columns={"prediction": variant.name})
        merged = (
            current if merged is None else merged.merge(current, on=keys, validate="one_to_one")
        )
    assert merged is not None
    return merged


def metric_row(
    frame: pd.DataFrame, prediction: str, target: str, scheme: str, support: str
) -> dict[str, object]:
    values = regression_metrics(
        frame.truth.to_numpy(), frame[prediction].to_numpy(), frame.group_key.to_numpy()
    )
    return {
        "target": target,
        "scheme": scheme,
        "support": support,
        "variant": prediction,
        **values,
    }


def summarize() -> None:
    """Create all model-selection and diagnostic source tables."""

    tables = OUTPUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pooled: list[dict[str, object]] = []
    lme_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    matrices: dict[tuple[str, str], pd.DataFrame] = {}
    time_blocks = pd.cut(
        np.arange(1993, 2026),
        [1992, 2000, 2008, 2016, 2025],
        labels=["1993-2000", "2001-2008", "2009-2016", "2017-2025"],
    )
    year_to_block = dict(zip(np.arange(1993, 2026), time_blocks, strict=True))
    for target in TARGETS:
        for scheme in SCHEMES:
            matrix = prediction_matrix(target, scheme)
            matrices[(target, scheme)] = matrix
            periods = {scheme: matrix}
            if scheme == "forward":
                periods = {
                    "forward_2019_2021": matrix.loc[matrix.yr.between(2019, 2021)],
                    "forward_2022_2025": matrix.loc[matrix.yr.between(2022, 2025)],
                }
            for period_name, period_frame in periods.items():
                common = period_frame.sss_bg_available.eq(1) & period_frame.adt_available.eq(1)
                for variant in VARIANTS:
                    pooled.append(
                        metric_row(period_frame, variant.name, target, period_name, "all")
                    )
                    pooled.append(
                        metric_row(
                            period_frame.loc[common],
                            variant.name,
                            target,
                            period_name,
                            "strict_common",
                        )
                    )
            for variant in VARIANTS:
                for lme_id, group in matrix.groupby("lme_id", observed=True):
                    if len(group) >= 20:
                        lme_rows.append(
                            {
                                "lme_id": lme_id,
                                **metric_row(group, variant.name, target, scheme, "all"),
                            }
                        )
                matrix["time_block"] = matrix.yr.map(year_to_block).astype(str)
                for block, group in matrix.groupby("time_block", observed=True):
                    time_rows.append(
                        {
                            "time_block": block,
                            **metric_row(group, variant.name, target, scheme, "all"),
                        }
                    )
                for status, group in matrix.groupby(
                    ["sss_bg_available", "adt_available"], observed=True
                ):
                    missing_rows.append(
                        {
                            "sss_bg_available": status[0],
                            "adt_available": status[1],
                            **metric_row(group, variant.name, target, scheme, "all"),
                        }
                    )
            if scheme in {"cruise", "spatial"}:
                for candidate, reference, group_name in (
                    ("A3_background_sss", "A2_environment", "background_sss"),
                    ("A4_adt", "A2_environment", "adt"),
                    ("A6_missing_robust", "A5_sss_adt", "missing_robustness"),
                ):
                    result = paired_group_bootstrap(matrix, candidate, reference)
                    bootstrap_rows.append(
                        {
                            "target": target,
                            "scheme": scheme,
                            "group": group_name,
                            "candidate": candidate,
                            "reference": reference,
                            **result,
                        }
                    )

    pooled_frame = pd.DataFrame(pooled)
    pooled_frame.to_csv(tables / "table01_pooled_metrics.csv", index=False)
    pd.DataFrame(lme_rows).to_csv(tables / "table02_lme_metrics.csv", index=False)
    pd.DataFrame(time_rows).to_csv(tables / "table03_time_block_metrics.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(tables / "table04_missing_strata_metrics.csv", index=False)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    bootstrap_frame.to_csv(tables / "table05_paired_bootstrap.csv", index=False)

    effects = []
    boot_lookup = bootstrap_frame.set_index(["target", "scheme", "group"])
    for target in TARGETS:
        for group_name, candidate, reference in (
            ("background_sss", "A3_background_sss", "A2_environment"),
            ("adt", "A4_adt", "A2_environment"),
            ("missing_robustness", "A6_missing_robust", "A5_sss_adt"),
        ):
            for scheme in (
                "cruise",
                "spatial",
                "whole_lme",
                "forward_2019_2021",
                "forward_2022_2025",
            ):
                subset = pooled_frame.loc[
                    (pooled_frame.target == target) & (pooled_frame.scheme == scheme)
                ]
                all_rows = subset.loc[subset.support == "all"].set_index("variant")
                common_rows = subset.loc[subset.support == "strict_common"].set_index("variant")
                boot_high = np.nan
                if (target, scheme, group_name) in boot_lookup.index:
                    boot_high = boot_lookup.loc[(target, scheme, group_name)].ci_high_delta_mse
                effects.append(
                    {
                        "target": target,
                        "group": group_name,
                        "candidate": candidate,
                        "reference": reference,
                        "scheme": scheme,
                        "relative_rmse_improvement": 1
                        - all_rows.loc[candidate, "rmse"] / all_rows.loc[reference, "rmse"],
                        "common_relative_rmse_improvement": 1
                        - common_rows.loc[candidate, "rmse"] / common_rows.loc[reference, "rmse"],
                        "bootstrap_ci_high_delta_mse": boot_high,
                    }
                )
    effect_frame = pd.DataFrame(effects)
    effect_frame.to_csv(tables / "table06_effects.csv", index=False)
    decisions = []
    for (target, group), rows in effect_frame.groupby(["target", "group"], observed=True):
        passed, reasons = qualifies_addition(rows)
        decisions.append(
            {
                "target": target,
                "group": group,
                "retained": passed,
                "reasons": ";".join(reasons) if reasons else "passed_all_gates",
            }
        )
    pd.DataFrame(decisions).to_csv(tables / "table07_retained_decision.csv", index=False)
    variant_rows = [
        {
            "variant": variant.name,
            "features": ";".join(variant.features),
            "feature_count": len(variant.features),
            "training_dropout_sss": variant.dropout_sss,
            "training_dropout_adt": variant.dropout_adt,
        }
        for variant in VARIANTS
    ]
    pd.DataFrame(variant_rows).to_csv(tables / "table08_variant_definitions.csv", index=False)
    inventory = [
        {"artifact": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in (CUBE, OBSERVATIONS, SPLITS, MATCHUP)
    ]
    pd.DataFrame(inventory).to_csv(tables / "table09_source_inventory.csv", index=False)
    print(f"wrote summary tables to {tables}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("match", "train", "summarize", "all"))
    parser.add_argument("--force-match", action="store_true")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--task-type", choices=("GPU", "CPU"), default="GPU")
    args = parser.parse_args()
    if args.phase in {"match", "all"}:
        build_matchup(force=args.force_match)
    if args.phase in {"train", "all"}:
        run_models(iterations=args.iterations, task_type=args.task_type)
    if args.phase in {"summarize", "all"}:
        summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
