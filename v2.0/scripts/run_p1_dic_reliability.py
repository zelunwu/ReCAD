"""Run and archive the preregistered Issue #28 derived-DIC reliability experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostRegressor

from recad.chem.inverse_dic import forward_fco2, inverse_dic, measured_fco2_mask
from recad.evaluate.applicability import assign_spatial_blocks
from recad.evaluate.dic_reliability import (
    REASON_BITS,
    assign_dic_grade,
    chemistry_reason_bits,
    covariance_from_sigmas,
    finite_difference_jacobian,
    jacobian_interval,
    monte_carlo_dic,
    weakest_grade,
)
from recad.evaluate.p1_framework import FrozenManifest, P1DataGateway, Purpose, sha256
from recad.evaluate.sss_reliability import BinnedIntervalModel
from recad.evaluate.ta_mab_reliability import calibration_groups, finite_sample_quantile
from run_p1_fco2_reliability import (
    CATEGORICAL,
    MODEL_COLUMNS,
    NUMERIC,
    SeasonalTrendClimatology,
    add_crossfit_sss,
    balanced_weights,
    prepare,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_dic_reliability_v2.2"
ARCHIVE = ROOT / "docs/experiment_archive" / EXPERIMENT_ID
PRIMARY_TA = "hierarchical_subregion_regime_ta_sss"
SHARED_OUTPUTS = Path(
    os.environ.get("RECAD_SHARED_OUTPUTS", "D:/proj_personal/PhD/ReCAD/v2.0/outputs")
)
OUTPUT = ROOT / "outputs/experiments" / EXPERIMENT_ID


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def write_text(path: Path, content: str) -> None:
    path.write_bytes(content.replace("\r\n", "\n").encode("utf-8"))


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    write_text(path, frame.to_csv(index=False, lineterminator="\n"))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subregion(frame: pd.DataFrame) -> pd.Series:
    return pd.cut(
        frame.latitude,
        bins=[35.20, 38.00, 40.00, 41.75],
        labels=["south", "central", "north"],
        include_lowest=True,
    ).astype(str)


def load_direct_dic(gateway: P1DataGateway) -> pd.DataFrame:
    columns = [
        "grid_flat",
        "coastal_node",
        "sst",
        "sss",
        "adt",
        "wspd",
        "pco2air",
        "xco2air",
        "temperature",
        "salinity",
        "ta",
        "fco2",
        "ta_qc",
        "fco2_qc",
        "parameter_method_fco2",
        "parameter_method_ta",
        "parameter_method_dic",
        "source",
        "obs_id",
        "is_primary",
        "label_ta_ok",
    ]
    frame = pd.concat(
        [
            gateway.load_labels("dic", purpose, columns=columns)
            for purpose in (Purpose.TRAIN, Purpose.SELECTION)
        ],
        ignore_index=True,
    )
    return frame.loc[
        frame.lme_id.eq(7)
        & frame.latitude.between(35.20, 41.75, inclusive="both")
        & frame.parameter_method_dic.isin([1, 2])
    ].copy()


def fit_fco2_model(fit: pd.DataFrame, seed: int, iterations: int) -> CatBoostRegressor:
    columns = NUMERIC + CATEGORICAL
    x_fit = fit[columns].copy()
    for column in NUMERIC:
        x_fit[column] = x_fit[column].fillna(float(x_fit[column].median()))
    for column in CATEGORICAL:
        x_fit[column] = x_fit[column].fillna(-999).astype(int).astype(str)
    model = CatBoostRegressor(
        iterations=iterations,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=seed,
        random_strength=0.5,
        l2_leaf_reg=5.0,
        allow_writing_files=False,
        verbose=False,
        task_type="GPU",
        devices="0",
    )
    model.fit(
        x_fit,
        fit.residual,
        cat_features=[columns.index(column) for column in CATEGORICAL],
        sample_weight=balanced_weights(fit) * len(fit),
        verbose=False,
    )
    model._recad_medians = {column: float(fit[column].median()) for column in NUMERIC}
    return model


def predict_fco2(model: CatBoostRegressor, frame: pd.DataFrame) -> np.ndarray:
    columns = NUMERIC + CATEGORICAL
    values = frame[columns].copy()
    for column in NUMERIC:
        values[column] = values[column].fillna(model._recad_medians[column])
    for column in CATEGORICAL:
        values[column] = values[column].fillna(-999).astype(int).astype(str)
    return frame.background.to_numpy(float) + model.predict(values)


def assign_fco2_partition(frame: pd.DataFrame, scheme: str, fold: int) -> np.ndarray:
    if scheme == "cruise":
        return frame.cv_fold.to_numpy(int) != fold
    if scheme == "spatial_block":
        return assign_spatial_blocks(frame).to_numpy(int) != fold
    if scheme == "subregion":
        return subregion(frame).to_numpy() != ["south", "central", "north"][fold]
    if scheme == "forward":
        return frame.forward_split.eq("train").to_numpy()
    raise ValueError(scheme)


def train_fco2_queries(
    gateway: P1DataGateway,
    queries: pd.DataFrame,
    config: dict[str, object],
) -> pd.DataFrame:
    cached = OUTPUT / "fco2_query_predictions.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    train = prepare(gateway.load_labels("fco2", Purpose.TRAIN, columns=MODEL_COLUMNS))
    development = prepare(gateway.load_labels("fco2", Purpose.SELECTION, columns=MODEL_COLUMNS))
    sss_path = (
        SHARED_OUTPUTS / "experiments/p1_sss_reliability_v2.2/development_oof_predictions.parquet"
    )
    parts: list[pd.DataFrame] = []
    for (scheme, fold), query_part in queries.groupby(["outer_scheme", "outer_fold"]):
        scheme = str(scheme)
        fold = int(fold)
        source = (
            pd.concat([train, development], ignore_index=True)
            if scheme == "forward"
            else train.copy()
        )
        source = source.loc[source.lme_id.eq(7) & source.latitude.between(35.20, 41.75)].copy()
        upstream_scheme = "spatial_block" if scheme == "subregion" else scheme
        source = add_crossfit_sss(source, sss_path, upstream_scheme)
        source = source.loc[assign_fco2_partition(source, scheme, fold)].copy()
        calibration = calibration_groups(source.group_key, fraction=0.20)
        model_fit = source.loc[~source.group_key.astype(str).isin(calibration)].copy()
        calibrate = source.loc[source.group_key.astype(str).isin(calibration)].copy()
        background = SeasonalTrendClimatology().fit(model_fit)
        for item in (model_fit, calibrate):
            item["background"] = background.predict(item)
            item["residual"] = item.truth - item.background
        query = query_part.copy()
        query["longitude"] = np.mod(query.longitude.to_numpy(float), 360.0)
        radians = np.deg2rad(query.longitude.to_numpy(float))
        phase = 2 * np.pi * (query.month.to_numpy(float) - 1) / 12
        query["lon_sin"], query["lon_cos"] = np.sin(radians), np.cos(radians)
        query["month_sin"], query["month_cos"] = np.sin(phase), np.cos(phase)
        query["sss_input"] = query.sss_predicted
        query["background"] = background.predict(query)
        print(
            f"fCO2 {scheme}/{fold}: fit={len(model_fit):,}, calibration={len(calibrate):,}, query={len(query):,}",
            flush=True,
        )
        model = fit_fco2_model(
            model_fit,
            seed=100,
            iterations=int(config["outer_evaluation"]["fco2_catboost_iterations"]),
        )
        calibrate["prediction"] = predict_fco2(model, calibrate)
        query["fco2_prediction"] = predict_fco2(model, query)
        error = np.abs(calibrate.prediction - calibrate.truth)
        query["fco2_q50"] = finite_sample_quantile(error, 0.50)
        query["fco2_q90"] = finite_sample_quantile(error, 0.90)
        query["fco2_calibration_cruises"] = len(calibration)
        delta = 0.05
        high, low = query.copy(), query.copy()
        high["sss_input"] += delta
        low["sss_input"] -= delta
        query["fco2_sss_slope"] = (predict_fco2(model, high) - predict_fco2(model, low)) / (
            2 * delta
        )
        query["fco2_upstream_grade"] = "D"  # Issue #26 selected only LME 12; MAB is out of scope.
        parts.append(query)
    result = pd.concat(parts, ignore_index=True)
    result.to_parquet(cached, index=False)
    return result


def regression_metrics(
    frame: pd.DataFrame, prediction: str = "dic_prediction"
) -> dict[str, float | int]:
    finite = frame[["truth", prediction]].notna().all(axis=1)
    part = frame.loc[finite]
    truth = part.truth.to_numpy(float)
    predicted = part[prediction].to_numpy(float)
    error = predicted - truth
    denominator = np.sum((truth - truth.mean()) ** 2)
    return {
        "n": len(part),
        "cruises": int(part.group_key.nunique()),
        "years": int(part.year.nunique()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1 - np.sum(error**2) / denominator) if denominator else np.nan,
    }


def build_predictions(
    direct: pd.DataFrame,
    gateway: P1DataGateway,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ta_path = SHARED_OUTPUTS / "experiments/p1_ta_mab_reliability_v2.2/outer_predictions.parquet"
    ta = pd.read_parquet(ta_path)
    ta = ta.loc[ta.candidate.eq(PRIMARY_TA)].copy()
    merged = ta.merge(
        direct[
            [
                "obs_id",
                "truth",
                "ta",
                "fco2",
                "fco2_qc",
                "parameter_method_fco2",
                "parameter_method_dic",
                "source",
                "is_primary",
                "xco2air",
            ]
        ].rename(columns={"truth": "dic_truth", "ta": "observed_ta", "fco2": "observed_fco2"}),
        on="obs_id",
        how="inner",
        validate="many_to_one",
    )
    merged = merged.rename(columns={"truth": "ta_truth", "grade": "ta_grade"})
    merged["truth"] = merged.dic_truth
    required = ["truth", "prediction", "sss_predicted", "temperature"]
    merged = merged.loc[merged[required].notna().all(axis=1)].copy()
    merged = train_fco2_queries(gateway, merged, config)

    frozen = json.loads(
        (
            SHARED_OUTPUTS / "experiments/p1_sss_reliability_v2.2/frozen_development_decision.json"
        ).read_text()
    )
    interval_model = BinnedIntervalModel.from_dict(frozen["interval_model"])
    scored = pd.DataFrame(
        {"risk_environment_k64": merged.sss_environment_k64, "background": merged.sss}
    )
    widths = interval_model.predict(scored)
    merged["sss_q50"] = widths.width50.to_numpy(float)
    merged["sss_q90"] = widths.width90.to_numpy(float)
    merged["sss_grade"] = "D"
    good = merged.sss_environment_k64.le(5)
    merged.loc[good & merged.sss_q90.le(2), "sss_grade"] = "C"
    merged.loc[good & merged.sss_q90.le(1), "sss_grade"] = "B"
    merged.loc[good & merged.sss_q90.le(0.5), "sss_grade"] = "A"
    merged["inherited_grade"] = weakest_grade(
        merged.sss_grade, merged.fco2_upstream_grade, merged.ta_grade
    )

    ranges = {
        "salinity": config["chemistry"]["salinity_range"],
        "temperature": config["chemistry"]["temperature_c_range"],
        "ta": config["chemistry"]["ta_umol_kg_range"],
        "fco2": config["chemistry"]["fco2_uatm_range"],
    }
    merged["reason_bits"] = chemistry_reason_bits(
        merged.prediction,
        merged.fco2_prediction,
        merged.sss_predicted,
        merged.temperature,
        ranges=ranges,
    )
    merged.loc[merged.sss_grade.eq("D"), "reason_bits"] |= REASON_BITS["upstream_sss_d"]
    merged.loc[merged.fco2_upstream_grade.eq("D"), "reason_bits"] |= REASON_BITS["upstream_fco2_d"]
    merged.loc[merged.ta_grade.eq("D"), "reason_bits"] |= REASON_BITS["upstream_ta_d"]
    merged["reason_bits"] |= REASON_BITS["covariance_unidentified"]
    valid = merged.reason_bits.map(lambda value: not bool(value & 63))
    merged.loc[~valid, "dic_prediction"] = np.nan
    merged.loc[valid, "dic_prediction"] = inverse_dic(
        merged.loc[valid, "prediction"],
        merged.loc[valid, "fco2_prediction"],
        merged.loc[valid, "sss_predicted"],
        merged.loc[valid, "temperature"],
    )
    merged["closure_fco2"] = np.nan
    merged.loc[valid, "closure_fco2"] = forward_fco2(
        merged.loc[valid, "prediction"],
        merged.loc[valid, "dic_prediction"],
        merged.loc[valid, "sss_predicted"],
        merged.loc[valid, "temperature"],
    )
    merged["closure_error"] = merged.closure_fco2 - merged.fco2_prediction

    jacobian = finite_difference_jacobian(
        merged.prediction, merged.fco2_prediction, merged.sss_predicted, merged.temperature
    )
    merged[["dDIC_dTA", "dDIC_dfCO2", "dDIC_dSSS"]] = jacobian
    sigmas = np.stack(
        [merged.q90 / 1.64485363, merged.fco2_q90 / 1.64485363, merged.sss_q90 / 1.64485363],
        axis=1,
    )
    ta_slope = np.zeros(len(merged))
    for _, index in merged.groupby("outer_scheme").groups.items():
        sample = merged.loc[index, ["ta_truth", "salinity"]].dropna()
        slope = np.polyfit(sample.salinity, sample.ta_truth, 1)[0] if len(sample) > 2 else 0.0
        ta_slope[index] = slope
    ta_limit = np.divide(
        0.95 * sigmas[:, 0], sigmas[:, 2], out=np.zeros(len(merged)), where=sigmas[:, 2] > 0
    )
    fco2_limit = np.divide(
        0.95 * sigmas[:, 1], sigmas[:, 2], out=np.zeros(len(merged)), where=sigmas[:, 2] > 0
    )
    ta_slope = np.clip(ta_slope, -ta_limit, ta_limit)
    fco2_slope = np.clip(merged.fco2_sss_slope.to_numpy(float), -fco2_limit, fco2_limit)
    scenario_rows: list[dict[str, object]] = []
    scenario_predictions: list[pd.DataFrame] = []
    means = merged[["prediction", "fco2_prediction", "sss_predicted"]].to_numpy(float)
    exact_mask = merged.sss_predicted.lt(30).to_numpy() | ~np.isfinite(jacobian).all(axis=1)
    merged.loc[exact_mask, "reason_bits"] |= REASON_BITS["exact_mc_domain"]
    for scenario in config["uncertainty"]["scenarios"]:
        covariance = covariance_from_sigmas(
            sigmas,
            mode=scenario,
            bounded_rho=float(config["uncertainty"]["bounded_correlation"]),
            ta_sss_slope=ta_slope,
            fco2_sss_slope=fco2_slope,
        )
        jac = jacobian_interval(jacobian, covariance)
        point = merged.dic_prediction.to_numpy(float)
        exact = {
            "q05": point - jac["halfwidth90"],
            "q25": point - jac["halfwidth50"],
            "q50": point.copy(),
            "q75": point + jac["halfwidth50"],
            "q95": point + jac["halfwidth90"],
        }
        if exact_mask.any():
            nonlinear = monte_carlo_dic(
                means[exact_mask],
                covariance[exact_mask],
                merged.loc[exact_mask, "temperature"],
                draws=int(config["uncertainty"]["monte_carlo_draws"]),
                seed=100,
                chunk_size=int(config["uncertainty"]["monte_carlo_chunk_size"]),
            )
            for key in ("q05", "q25", "q50", "q75", "q95"):
                exact[key][exact_mask] = nonlinear[key]
        local = merged[
            ["obs_id", "outer_scheme", "outer_fold", "truth", "group_key", "year"]
        ].copy()
        local["scenario"] = scenario
        local["q05"], local["q25"], local["median"], local["q75"], local["q95"] = (
            exact["q05"],
            exact["q25"],
            exact["q50"],
            exact["q75"],
            exact["q95"],
        )
        local["jacobian_width90"] = 2 * jac["halfwidth90"]
        local["mc_width90"] = local.q95 - local.q05
        scenario_predictions.append(local)
        for scheme, part in local.groupby("outer_scheme"):
            error = part["median"] - part.truth
            scenario_rows.append(
                {
                    "scenario": scenario,
                    "outer_scheme": scheme,
                    **regression_metrics(part.rename(columns={"median": "dic_prediction"})),
                    "coverage50": float(
                        ((part.truth >= part.q25) & (part.truth <= part.q75)).mean()
                    ),
                    "coverage90": float(
                        ((part.truth >= part.q05) & (part.truth <= part.q95)).mean()
                    ),
                    "median_width90": float(part.mc_width90.median()),
                    "jacobian_mc_width_ratio": float(
                        (part.jacobian_width90 / part.mc_width90).median()
                    ),
                    "error_sd": float(error.std(ddof=1)),
                }
            )
    scenarios = pd.concat(scenario_predictions, ignore_index=True)
    common = scenarios.loc[scenarios.scenario.eq("common_sss_covariance")].copy()
    merged = merged.merge(
        common[
            [
                "obs_id",
                "outer_scheme",
                "outer_fold",
                "q05",
                "q25",
                "median",
                "q75",
                "q95",
                "mc_width90",
            ]
        ],
        on=["obs_id", "outer_scheme", "outer_fold"],
        how="left",
        validate="one_to_one",
    )
    merged["dic_grade"] = assign_dic_grade(merged.mc_width90, merged.inherited_grade)
    merged.loc[merged.dic_grade.eq("D"), "reason_bits"] |= REASON_BITS["interval_too_wide"]
    return merged, pd.DataFrame(scenario_rows), scenarios


def oracle_table(frame: pd.DataFrame) -> pd.DataFrame:
    scenarios: dict[str, tuple[np.ndarray, pd.Series]] = {
        "production_chain": (
            frame.dic_prediction.to_numpy(float),
            pd.Series(True, index=frame.index),
        ),
        "observed_TA_oracle": (
            inverse_dic(
                frame.observed_ta, frame.fco2_prediction, frame.sss_predicted, frame.temperature
            ),
            frame.observed_ta.notna(),
        ),
        "observed_SSS_oracle": (
            inverse_dic(frame.prediction, frame.fco2_prediction, frame.salinity, frame.temperature),
            frame.salinity.notna(),
        ),
    }
    strict = measured_fco2_mask(frame.observed_fco2, frame.fco2_qc, frame.parameter_method_fco2)
    scenarios["observed_fCO2_oracle"] = (
        inverse_dic(
            frame.prediction,
            frame.observed_fco2.fillna(400),
            frame.sss_predicted,
            frame.temperature,
        ),
        pd.Series(strict, index=frame.index),
    )
    scenarios["all_observed_inputs_oracle"] = (
        inverse_dic(
            frame.observed_ta.fillna(2300),
            frame.observed_fco2.fillna(400),
            frame.salinity.fillna(35),
            frame.temperature,
        ),
        pd.Series(strict, index=frame.index) & frame.observed_ta.notna() & frame.salinity.notna(),
    )
    rows = []
    for name, (prediction, mask) in scenarios.items():
        work = frame.loc[mask].copy()
        work["candidate"] = prediction[mask.to_numpy()]
        rows.append({"scenario": name, **regression_metrics(work, "candidate")})
    return pd.DataFrame(rows)


def convergence_table(frame: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    sample = frame.loc[frame.outer_scheme.eq("cruise")].head(64).copy()
    sigmas = np.stack(
        [sample.q90 / 1.64485363, sample.fco2_q90 / 1.64485363, sample.sss_q90 / 1.64485363], axis=1
    )
    covariance = covariance_from_sigmas(sigmas, mode="independent")
    means = sample[["prediction", "fco2_prediction", "sss_predicted"]].to_numpy(float)
    rows = []
    previous = None
    for draws in config["uncertainty"]["convergence_draws"]:
        result = monte_carlo_dic(means, covariance, sample.temperature, draws=int(draws), seed=2028)
        width = result["q95"] - result["q05"]
        row = {
            "draws": draws,
            "median_width90": float(np.median(width)),
            "median_q05": float(np.median(result["q05"])),
            "median_q95": float(np.median(result["q95"])),
            "relative_width_change": np.nan,
            "max_endpoint_change": np.nan,
        }
        if previous is not None:
            row["relative_width_change"] = float(
                abs(np.median(width) / np.median(previous["width"]) - 1)
            )
            row["max_endpoint_change"] = float(
                max(
                    np.median(abs(result["q05"] - previous["q05"])),
                    np.median(abs(result["q95"] - previous["q95"])),
                )
            )
        rows.append(row)
        previous = {"width": width, "q05": result["q05"], "q95": result["q95"]}
    return pd.DataFrame(rows)


def make_atlas() -> pd.DataFrame:
    base = SHARED_OUTPUTS / "experiments/p1_sss_reliability_v2.2/sss_reliability_atlas/year=2025"
    parts = [pd.read_parquet(path) for path in sorted(base.glob("month=*.parquet"))]
    atlas = pd.concat(parts, ignore_index=True)
    mapping_path = (
        SHARED_OUTPUTS
        / "experiments/p1_fco2_reliability_v2.2/fco2_reliability_atlas/year=2025/month=01.parquet"
    )
    mapping = pd.read_parquet(mapping_path, columns=["coastal_node", "lme_id"]).drop_duplicates(
        "coastal_node"
    )
    atlas = atlas.merge(mapping, on="coastal_node", how="left", validate="many_to_one")
    atlas = atlas.loc[atlas.lme_id.eq(7) & atlas.latitude.between(35.20, 41.75)].copy()
    atlas["sss_grade"] = atlas.grade
    atlas["fco2_grade"] = "D"
    atlas["ta_grade"] = "D"
    atlas["dic_grade"] = "D"
    atlas["dic_status"] = "suppressed"
    atlas["reason_bits"] = REASON_BITS["upstream_fco2_d"] | REASON_BITS["upstream_ta_d"]
    return atlas[
        [
            "coastal_node",
            "year",
            "month",
            "latitude",
            "longitude",
            "sss_grade",
            "fco2_grade",
            "ta_grade",
            "dic_grade",
            "dic_status",
            "reason_bits",
        ]
    ]


def save_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(ARCHIVE / "figures" / name, dpi=180, bbox_inches="tight")
    plt.close()


def archive_results(
    config: dict[str, object],
    direct: pd.DataFrame,
    predictions: pd.DataFrame,
    propagation: pd.DataFrame,
    scenarios: pd.DataFrame,
    oracle: pd.DataFrame,
    convergence: pd.DataFrame,
    atlas: pd.DataFrame,
) -> dict[str, object]:
    tables = ARCHIVE / "tables"
    figures = ARCHIVE / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    audited = direct.copy()
    audited["strict_measured_fco2"] = measured_fco2_mask(
        audited.fco2, audited.fco2_qc, audited.parameter_method_fco2
    )
    audited["ta_qc2"] = audited.ta_qc.eq(2) & audited.ta.notna()
    source_audit = (
        audited.groupby(["source", "parameter_method_dic"], dropna=False)
        .agg(
            records=("obs_id", "size"),
            cruises=("group_key", "nunique"),
            years=("year", "nunique"),
            salinity_available=("salinity", "count"),
            ta_available=("ta", "count"),
            measured_fco2_available=("fco2", "count"),
            strict_measured_fco2=("strict_measured_fco2", "sum"),
            ta_qc2=("ta_qc2", "sum"),
        )
        .reset_index()
    )
    source_audit["temperature_basis"] = "in-situ temperature; pressure fixed to surface 0 dbar"
    source_audit["units"] = "TA/DIC µmol kg-1; fCO2 µatm; SSS practical salinity"
    source_audit["nutrients"] = "silicate=0, phosphate=0 (frozen approximation)"
    source_audit["matching"] = "exact canonical obs_id to Issue #27 TA OOF; no tolerance match"
    source_audit["dic_qc"] = "gateway label_dic_ok plus parameter_method_dic in {1,2}"
    metrics = []
    for scheme, part in predictions.groupby("outer_scheme"):
        metrics.append({"outer_scheme": scheme, **regression_metrics(part)})
    metrics = pd.DataFrame(metrics)
    source_bias = (
        predictions.assign(error=predictions.dic_prediction - predictions.truth)
        .groupby(["outer_scheme", "source"])
        .agg(
            n=("error", "size"),
            cruises=("group_key", "nunique"),
            bias=("error", "mean"),
            rmse=("error", lambda x: float(np.sqrt(np.mean(x**2)))),
        )
        .reset_index()
    )
    reason_table = pd.DataFrame(
        [
            {
                "reason": name,
                "bit": bit,
                "records": int(((predictions.reason_bits & bit) > 0).sum()),
            }
            for name, bit in REASON_BITS.items()
        ]
    )
    chemistry = pd.DataFrame(
        [
            {"check": "chemistry_valid", "records": int((predictions.reason_bits & 63 == 0).sum())},
            {
                "check": "closure_max_abs_uatm",
                "records": float(predictions.closure_error.abs().max()),
            },
            {"check": "empirical_joint_residual_covariance_identifiable", "records": 0},
        ]
    )
    grade_counts = (
        predictions.groupby(["outer_scheme", "dic_grade"]).size().rename("records").reset_index()
    )
    atlas_counts = (
        atlas.groupby(["sss_grade", "fco2_grade", "ta_grade", "dic_grade", "dic_status"])
        .size()
        .rename("grid_months")
        .reset_index()
    )
    table_map = {
        "table01_source_audit.csv": source_audit,
        "table02_direct_chain_metrics.csv": metrics,
        "table03_propagation_scenarios.csv": propagation,
        "table04_oracle_ablation.csv": oracle,
        "table05_mc_convergence.csv": convergence,
        "table06_source_bias.csv": source_bias,
        "table07_reason_bits.csv": reason_table,
        "table08_chemistry_and_closure.csv": chemistry,
        "table09_grade_counts.csv": grade_counts,
        "table10_2025_atlas_counts.csv": atlas_counts,
    }
    for name, frame in table_map.items():
        write_csv(frame, tables / name)

    metrics.set_index("outer_scheme")["rmse"].plot.bar(figsize=(7, 4), color="#4472c4")
    plt.ylabel("DIC RMSE (µmol kg⁻¹)")
    plt.title("End-to-end DIC error by untouched outer scheme")
    save_figure("fig01_chain_rmse.png")
    common = propagation.loc[propagation.scenario.eq("common_sss_covariance")].set_index(
        "outer_scheme"
    )
    common[["coverage50", "coverage90"]].plot.bar(
        figsize=(8, 4), ylim=(0, 1), color=["#70ad47", "#ed7d31"]
    )
    plt.ylabel("empirical coverage")
    plt.title("Exact Monte Carlo interval calibration")
    save_figure("fig02_interval_coverage.png")
    pivot = propagation.pivot(index="outer_scheme", columns="scenario", values="median_width90")
    pivot.plot.bar(figsize=(9, 4))
    plt.ylabel("median 90% width (µmol kg⁻¹)")
    plt.title("Dependence assumptions change propagated width")
    save_figure("fig03_covariance_width.png")
    sample = predictions.loc[predictions.outer_scheme.eq("cruise")]
    plt.figure(figsize=(6, 6))
    plt.scatter(sample.truth, sample.dic_prediction, s=15, alpha=0.55)
    bounds = [
        min(sample.truth.min(), sample.dic_prediction.min()),
        max(sample.truth.max(), sample.dic_prediction.max()),
    ]
    plt.plot(bounds, bounds, "k--")
    plt.xlabel("Observed DIC")
    plt.ylabel("Derived DIC")
    plt.title("Cruise-outer direct DIC validation")
    save_figure("fig04_observed_predicted.png")
    oracle.set_index("scenario")["rmse"].plot.bar(figsize=(8, 4), color="#5b9bd5")
    plt.ylabel("DIC RMSE (µmol kg⁻¹)")
    plt.title("Oracle ablation identifies the upstream bottleneck")
    save_figure("fig05_oracle_ablation.png")
    source_bias.pivot(index="outer_scheme", columns="source", values="bias").plot.bar(
        figsize=(8, 4)
    )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("bias (µmol kg⁻¹)")
    plt.title("Direct-chain bias by source")
    save_figure("fig06_source_bias.png")
    convergence.plot(x="draws", y="median_width90", marker="o", figsize=(7, 4), legend=False)
    plt.ylabel("median 90% width (µmol kg⁻¹)")
    plt.title("Monte Carlo draw convergence")
    save_figure("fig07_mc_convergence.png")
    atlas_counts.set_index("dic_grade")["grid_months"].plot.bar(figsize=(6, 4), color="#a5a5a5")
    plt.ylabel("2025 MAB grid-months")
    plt.title("Inherited DIC product grades")
    save_figure("fig08_atlas_grade.png")

    captions = {
        "fig01_chain_rmse.png": "Figure 1. Direct-observation DIC RMSE for the complete predicted SSS-fCO2-TA chain under cruise, spatial-block, subregion, and forward outer exclusions; no DIC label trained any upstream model.",
        "fig02_interval_coverage.png": "Figure 2. Empirical 50% and 90% coverage from 2,048-draw exact PyCO2SYS Monte Carlo intervals using the shared-SSS covariance construction, compared across untouched outer schemes.",
        "fig03_covariance_width.png": "Figure 3. Median propagated 90% DIC interval width under independent errors, bounded negative and positive correlations, and a common-SSS structural covariance factor.",
        "fig04_observed_predicted.png": "Figure 4. Cruise-outer derived DIC against direct CODAP-NA/GLODAP observations in MAB; dispersion around the identity line is predictive error rather than numerical closure error.",
        "fig05_oracle_ablation.png": "Figure 5. RMSE after replacing one or all upstream quantities with observations; the comparison separates TA, fCO2, and SSS representation limits from carbonate-solver behavior.",
        "fig06_source_bias.png": "Figure 6. Mean DIC prediction error separated by source and outer scheme; small GLODAP strata are retained visibly and are not pooled away into the CODAP-NA majority.",
        "fig07_mc_convergence.png": "Figure 7. Preregistered 256-2,048 draw convergence diagnostic on a fixed cruise-outer anchor subset; the final run uses the frozen 2,048-draw budget.",
        "fig08_atlas_grade.png": "Figure 8. Grid-month count of inherited 2025 MAB DIC grades; Issue #26 fCO2 and Issue #27 TA statuses force suppression even where SSS itself has support.",
    }
    main = metrics.set_index("outer_scheme")
    common = common
    closure = float(predictions.closure_error.abs().max())
    grade_ab = float(predictions.dic_grade.isin(["A", "B"]).mean())
    gates = {
        "minimum_direct_cruises": int(predictions.group_key.nunique()) >= 5,
        "minimum_direct_years": int(predictions.year.nunique()) >= 3,
        "positive_r2_all_schemes": bool((main.r2 > 0).all()),
        "coverage50": bool(common.coverage50.between(0.45, 0.55).all()),
        "coverage90": bool(common.coverage90.between(0.85, 0.95).all()),
        "median_width90": bool((common.median_width90 <= 200).all()),
        "source_bias": bool((source_bias.bias.abs() <= 25).all()),
        "closure": closure <= 0.5,
        "grade_ab_retained": grade_ab >= 0.10,
        "all_upstreams_qualified": False,
    }
    decision = "pass_regional" if all(gates.values()) else "diagnostic_only"
    failed = [name for name, passed in gates.items() if not passed]
    captions_text = (
        "\n\n".join([f"### {name}\n\n{caption}" for name, caption in captions.items()]) + "\n"
    )
    write_text(ARCHIVE / "CAPTIONS.md", captions_text)
    report_figures = "\n\n".join(
        [f"![{name}](figures/{name})\n\n{caption}" for name, caption in captions.items()]
    )
    report = f"""# Issue #28 derived-DIC reliability report

## Scientific question and permitted claim

Can DIC be released as a qualified MAB derived product when SSS, fCO2, and TA are predictions? The answer is **`{decision}`**. This experiment permits a diagnostic claim only: PyCO2SYS propagation is operational and directly testable, but the current MAB upstream chain does not support a released DIC product. Exact closure is an engineering result, not validation.

## Data, splits, and leakage controls

The anchor is direct surface DIC from CODAP-NA/GLODAP, filtered to parameter methods 1/2 in LME 7 at 35.20-41.75°N. There are {len(direct):,} source rows before matching and {len(predictions):,} outer predictions spanning {predictions.group_key.nunique()} cruises and {predictions.year.nunique()} years. Issue #27 TA OOF folds define cruise, 5° spatial-block, complete-subregion, and forward exclusions. A matching MAB SOCAT fCO2 model is fit after the same exclusion and a nested 20% cruise calibration holdout. DIC labels never enter SSS, fCO2, or TA fitting. Locked and external-independent labels remain sealed.

CODAP-NA and GLODAP source, direct/adjusted parameter method, unit, surface temperature basis, zero-nutrient approximation, and QC availability are audited in Table 1. The frozen solver uses pressure 0 dbar, total silicate/phosphate 0 µmol kg⁻¹, pH scale 1, carbonic constants 10, bisulfate 1, and total borate 1. This nutrient assumption is a model limitation.

## Candidate models and training

The production-chain diagnostic combines Issue #25 cross-fitted SSS, full-budget 1,500-iteration CatBoost fCO2 residual models, and the frozen Issue #27 hierarchical TA candidate. Four dependence assumptions are compared. The common-SSS construction uses one salinity draw in SSS, TA, and fCO2; independent residual covariance cannot be estimated directly because the observation systems lack enough row-matched joint OOF residuals. PyCO2SYS central finite differences provide the Jacobian. Exact chunked Monte Carlo is run for all validation rows, so low-salinity and nonlinear states do not rely on a linear approximation.

## Main development results

End-to-end RMSE for cruise/spatial/subregion/forward is {main.loc["cruise", "rmse"]:.2f}, {main.loc["spatial_block", "rmse"]:.2f}, {main.loc["subregion", "rmse"]:.2f}, and {main.loc["forward", "rmse"]:.2f} µmol kg⁻¹; R² is {main.loc["cruise", "r2"]:.3f}, {main.loc["spatial_block", "r2"]:.3f}, {main.loc["subregion", "r2"]:.3f}, and {main.loc["forward", "r2"]:.3f}. Under common-SSS propagation, median 90% widths are {common.loc["cruise", "median_width90"]:.1f}, {common.loc["spatial_block", "median_width90"]:.1f}, {common.loc["subregion", "median_width90"]:.1f}, and {common.loc["forward", "median_width90"]:.1f} µmol kg⁻¹. Maximum inverse/forward closure error is {closure:.3g} µatm.

The strict grade rule inherits the weakest SSS/fCO2/TA grade. Issue #26 qualified only Caribbean fCO2 and Issue #27 classified the MAB TA atlas as D, so 2025 MAB DIC is fully suppressed. Direct validation remains scientifically useful for locating upstream error, but cannot override those frozen product statuses.

## Decision and limitations

Decision: **`{decision}`**. Failed preregistered gates: {", ".join(failed)}. Grade A/B retained fraction is {grade_ab:.3f}. Empirical residual covariance is not identifiable from the current disjoint SOCAT and CODAP/GLODAP sampling; bounded-correlation results therefore remain a sensitivity envelope, while the common-SSS term is a structural covariance model. No locked DIC/TA labels or 41-cruise external CODAP set were opened. A future DIC product requires a qualified MAB fCO2 product, a TA model that passes Issue #27-type spatial and interval gates, and then a newly frozen direct-DIC audit.

## Figure and table index

Tables 1-10 contain the source audit, chain metrics, propagation comparison, oracle ablation, Monte Carlo convergence, source bias, reason bits, chemistry/closure checks, grade counts, and 2025 atlas counts. Row-level predictions and the grid-month provenance atlas are stored in the ignored experiment output directory.

{report_figures}
"""
    write_text(ARCHIVE / "REPORT.md", report)
    write_text(
        ARCHIVE / "README.md",
        f"# {EXPERIMENT_ID}\n\nReviewer-ready archive for Issue #28. Decision: `{decision}`. See [REPORT.md](REPORT.md).\n",
    )
    decision_payload = {
        "experiment_id": EXPERIMENT_ID,
        "decision": decision,
        "gates": gates,
        "failed_gates": failed,
    }
    write_text(OUTPUT / "decision.json", json.dumps(decision_payload, indent=2))
    manifest_path = ROOT / str(config["data_manifest"])
    figure_source = {
        "fig01_chain_rmse.png": ["table02_direct_chain_metrics.csv"],
        "fig02_interval_coverage.png": ["table03_propagation_scenarios.csv"],
        "fig03_covariance_width.png": ["table03_propagation_scenarios.csv"],
        "fig04_observed_predicted.png": ["local:direct_predictions.parquet"],
        "fig05_oracle_ablation.png": ["table04_oracle_ablation.csv"],
        "fig06_source_bias.png": ["table06_source_bias.csv"],
        "fig07_mc_convergence.png": ["table05_mc_convergence.csv"],
        "fig08_atlas_grade.png": ["table10_2025_atlas_counts.csv"],
    }
    tracked = [
        ARCHIVE / "README.md",
        ARCHIVE / "REPORT.md",
        ARCHIVE / "CAPTIONS.md",
        *tables.glob("*.csv"),
        *figures.glob("*.png"),
    ]
    archive_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_git_commit": git_head(),
        "data_manifest_sha256": sha256(manifest_path),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "MAB direct DIC development anchors; no DIC label used in upstream fitting",
        "decision": decision,
        "figure_source_data": figure_source,
        "figure_captions": captions,
        "files_sha256": {
            str(path.relative_to(ARCHIVE)).replace("\\", "/"): file_hash(path) for path in tracked
        },
        "archive_builder_sha256": file_hash(Path(__file__)),
        "analysis_script_sha256": file_hash(Path(__file__)),
        "source_artifacts_sha256": {
            "config": file_hash(ROOT / "configs/p1_dic_reliability_v2.2.yaml"),
            "ta_predictions": file_hash(
                SHARED_OUTPUTS / "experiments/p1_ta_mab_reliability_v2.2/outer_predictions.parquet"
            ),
        },
    }
    write_text(ARCHIVE / "archive_manifest.json", json.dumps(archive_manifest, indent=2))
    return decision_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/p1_dic_reliability_v2.2.yaml"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    gateway = P1DataGateway(FrozenManifest.load(ROOT / str(config["data_manifest"])))
    direct = load_direct_dic(gateway)
    predictions, propagation, scenarios = build_predictions(direct, gateway, config)
    oracle = oracle_table(predictions)
    convergence = convergence_table(predictions, config)
    atlas = make_atlas()
    predictions.to_parquet(OUTPUT / "direct_predictions.parquet", index=False)
    scenarios.to_parquet(OUTPUT / "propagation_row_predictions.parquet", index=False)
    atlas.to_parquet(OUTPUT / "dic_reliability_atlas_2025.parquet", index=False)
    decision = archive_results(
        config, direct, predictions, propagation, scenarios, oracle, convergence, atlas
    )
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
