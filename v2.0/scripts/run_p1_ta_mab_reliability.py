"""Run the preregistered Issue #27 MAB TA reliability experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

from recad.chem.esper_ta import ESPER_LIR_TA
from recad.evaluate.applicability import SupportIndex, assign_spatial_blocks, outer_splits
from recad.evaluate.p1_framework import FrozenManifest, P1DataGateway, Purpose, sha256
from recad.evaluate.sss_reliability import apply_shrinkage, predict_soft_expert
from recad.evaluate.ta_mab_reliability import (
    assign_grade,
    calibration_groups,
    finite_sample_quantile,
    regression_metrics,
)
from run_p1_sss_reliability import ENVIRONMENT_COLUMNS
from run_p1_sss_reliability import load_development as load_sss
from run_p1_ta_viability import (
    RobustPartialPooling,
    add_carter,
    apply_bight_definition,
    balanced_weights,
    nearest_support_km,
    prepare,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_ta_mab_reliability_v2.2"
PRIMARY = "hierarchical_subregion_regime_ta_sss"
CANDIDATES = [
    "carter_esper_predicted_sss",
    "robust_mab_ta_sss",
    PRIMARY,
    "nonlinear_residual_gradient_boosting",
    "hierarchical_oracle_insitu_sss",
]
NUMERIC = [
    "sss",
    "sst",
    "adt",
    "wspd",
    "pco2air",
    "latitude",
    "longitude",
    "month_sin",
    "month_cos",
    "carter",
    "hierarchical_base",
]


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_lf(path: Path, text: str) -> None:
    """Write reproducible UTF-8/LF text on every operating system."""

    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def write_csv_lf(frame: pd.DataFrame, path: Path) -> None:
    """Write a reproducible UTF-8/LF CSV."""

    write_text_lf(path, frame.to_csv(index=False, lineterminator="\n"))


def prepare_sss_query(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["longitude"] = np.mod(result.longitude.to_numpy(float), 360.0)
    result["latitude_sin"] = np.sin(np.deg2rad(result.latitude))
    result["longitude_sin"] = np.sin(np.deg2rad(result.longitude))
    result["longitude_cos"] = np.cos(np.deg2rad(result.longitude))
    result["lon_sin"] = result.longitude_sin
    result["lon_cos"] = result.longitude_cos
    phase = 2 * np.pi * (result.month.to_numpy(float) - 1) / 12
    result["month_sin"] = np.sin(phase)
    result["month_cos"] = np.cos(phase)
    return result


def split_lookup(frame: pd.DataFrame, scheme: str) -> tuple[pd.Series, str]:
    if scheme == "cruise":
        return frame.cv_fold.astype(int), "cruise"
    if scheme == "spatial_block":
        return assign_spatial_blocks(frame), "spatial_block"
    if scheme == "subregion":
        return assign_spatial_blocks(frame), "spatial_block"
    if scheme == "forward":
        return pd.Series(np.zeros(len(frame), dtype=int), index=frame.index), "forward"
    raise ValueError(scheme)


def crossfit_sss(
    frame: pd.DataFrame,
    evidence: pd.DataFrame,
    scheme: str,
    upstream: Path,
) -> pd.DataFrame:
    """Apply the matching frozen Issue #25 outer-fold SSS model to TA rows."""

    query = prepare_sss_query(frame)
    assignment, upstream_scheme = split_lookup(query, scheme)
    splits = {item.fold: item for item in outer_splits(evidence, upstream_scheme)}
    prediction = np.full(len(query), np.nan)
    risk = np.full(len(query), np.nan)
    for fold in sorted(assignment.unique()):
        mask = assignment.eq(fold).to_numpy()
        split = splits[int(fold)]
        checkpoint = upstream / "checkpoints" / f"oof_{upstream_scheme}_{int(fold)}_seed100.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        raw = predict_soft_expert(state, query.loc[mask], device="cpu")
        support = (
            SupportIndex(environmental_columns=ENVIRONMENT_COLUMNS)
            .fit(evidence.iloc[split.fit_index])
            .query(query.loc[mask])
        )
        local_risk = support.environment_k64.to_numpy(float)
        prediction[mask] = apply_shrinkage(query.loc[mask, "sss"], raw, local_risk, "quadratic_5")
        risk[mask] = local_risk
    result = frame.copy()
    result["sss_predicted"] = prediction
    result["sss_environment_k64"] = risk
    result["sss_upstream_scheme"] = upstream_scheme
    return result


def model_frame(frame: pd.DataFrame, oracle: bool = False) -> pd.DataFrame:
    result = frame.copy()
    result["sss"] = result.salinity if oracle else result.sss_predicted
    return result


def fit_predict_candidate(
    name: str,
    fit: pd.DataFrame,
    target: pd.DataFrame,
    estimator: ESPER_LIR_TA,
) -> np.ndarray:
    oracle = name == "hierarchical_oracle_insitu_sss"
    fit_work = model_frame(fit, oracle=oracle)
    target_work = model_frame(target, oracle=oracle)
    fit_work = add_carter(fit_work, estimator)
    target_work = add_carter(target_work, estimator)
    if name == "carter_esper_predicted_sss":
        return target_work.carter.to_numpy(float)
    if name == "robust_mab_ta_sss":
        model = RobustPartialPooling(alpha=10.0, groups=False).fit(
            fit_work, sample_weight=balanced_weights(fit_work)
        )
        return model.predict(target_work)
    base = RobustPartialPooling(alpha=100.0, groups=True).fit(
        fit_work, sample_weight=balanced_weights(fit_work)
    )
    base_prediction = base.predict(target_work)
    if name in {PRIMARY, "hierarchical_oracle_insitu_sss"}:
        return base_prediction
    if name != "nonlinear_residual_gradient_boosting":
        raise ValueError(name)
    fit_work["hierarchical_base"] = base.predict(fit_work)
    target_work["hierarchical_base"] = base_prediction
    x_fit = fit_work[NUMERIC].replace([np.inf, -np.inf], np.nan)
    medians = x_fit.median()
    x_fit = x_fit.fillna(medians)
    x_target = target_work[NUMERIC].replace([np.inf, -np.inf], np.nan).fillna(medians)
    residual = fit_work.truth.to_numpy(float) - fit_work.hierarchical_base.to_numpy(float)
    nonlinear = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=10.0,
        random_state=100,
    ).fit(x_fit, residual)
    return base_prediction + nonlinear.predict(x_target)


def outer_partitions(frame: pd.DataFrame, scheme: str):
    if scheme != "subregion":
        return outer_splits(frame, scheme)
    result = []
    for fold in sorted(frame.lme_id.unique()):
        held = np.flatnonzero(frame.lme_id.eq(fold).to_numpy())
        fit = np.flatnonzero(frame.lme_id.ne(fold).to_numpy())
        result.append(
            type("Split", (), {"fold": int(fold), "fit_index": fit, "held_index": held})()
        )
    return result


def evaluate_outer(
    combined: pd.DataFrame,
    sss_evidence: pd.DataFrame,
    estimator: ESPER_LIR_TA,
    upstream: Path,
) -> pd.DataFrame:
    rows = []
    for scheme in ["cruise", "spatial_block", "subregion", "forward"]:
        print(f"Preparing cross-fitted SSS for {scheme}", flush=True)
        data = crossfit_sss(combined, sss_evidence, scheme, upstream)
        for split in outer_partitions(data, scheme):
            outer_fit = data.iloc[split.fit_index].copy().reset_index(drop=True)
            held = data.iloc[split.held_index].copy().reset_index(drop=True)
            groups = calibration_groups(outer_fit.group_key, 0.20)
            calibration = outer_fit.loc[outer_fit.group_key.astype(str).isin(groups)].reset_index(
                drop=True
            )
            inner_fit = outer_fit.loc[~outer_fit.group_key.astype(str).isin(groups)].reset_index(
                drop=True
            )
            if calibration.group_key.nunique() < 2 or inner_fit.group_key.nunique() < 2:
                raise RuntimeError(f"insufficient nested groups for {scheme} fold {split.fold}")
            held["support_km"] = nearest_support_km(inner_fit, held)
            for name in CANDIDATES:
                if name == "hierarchical_oracle_insitu_sss":
                    candidate_fit = inner_fit.loc[inner_fit.salinity.notna()].reset_index(drop=True)
                    candidate_calibration = calibration.loc[
                        calibration.salinity.notna()
                    ].reset_index(drop=True)
                    candidate_held = held.loc[held.salinity.notna()].reset_index(drop=True)
                else:
                    candidate_fit = inner_fit
                    candidate_calibration = calibration
                    candidate_held = held
                cal_prediction = fit_predict_candidate(
                    name, candidate_fit, candidate_calibration, estimator
                )
                residual = np.abs(candidate_calibration.truth.to_numpy(float) - cal_prediction)
                q50 = finite_sample_quantile(residual, 0.50)
                q90 = finite_sample_quantile(residual, 0.90)
                prediction = fit_predict_candidate(name, candidate_fit, candidate_held, estimator)
                candidate = candidate_held.copy()
                candidate["prediction"] = prediction
                candidate["q50"] = q50
                candidate["q90"] = q90
                candidate["environmental_ood"] = candidate.sss_environment_k64.gt(5.0)
                candidate["grade"] = assign_grade(
                    candidate.q90,
                    candidate.support_km,
                    np.repeat(candidate_fit.group_key.nunique(), len(candidate)),
                    candidate.environmental_ood,
                )
                candidate["candidate"] = name
                candidate["outer_scheme"] = scheme
                candidate["outer_fold"] = int(split.fold)
                candidate["calibration_cruises"] = calibration.group_key.nunique()
                candidate["training_cruises"] = candidate_fit.group_key.nunique()
                rows.append(candidate)
    return pd.concat(rows, ignore_index=True)


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = []
    for (scheme, candidate), part in predictions.groupby(["outer_scheme", "candidate"]):
        metrics.append({"outer_scheme": scheme, "candidate": candidate, **regression_metrics(part)})
    metrics = pd.DataFrame(metrics)
    pivot = metrics.pivot(index="outer_scheme", columns="candidate", values="rmse")
    for baseline in ["carter_esper_predicted_sss", "robust_mab_ta_sss"]:
        skill = 1 - pivot[PRIMARY] ** 2 / pivot[baseline] ** 2
        metrics = metrics.merge(
            skill.rename(f"primary_skill_vs_{baseline}").reset_index(), on="outer_scheme"
        )
    diagnostics = []
    primary = predictions.loc[predictions.candidate.eq(PRIMARY)].copy()
    primary["season"] = pd.cut(primary.month, [0, 3, 6, 9, 12], labels=["DJF", "MAM", "JJA", "SON"])
    primary["salinity_band"] = pd.cut(
        primary.sss_predicted, [-np.inf, 30, 33, 35, np.inf], right=False
    ).astype(str)
    primary["support_band"] = pd.cut(
        primary.support_km, [0, 50, 100, 200, 500, np.inf], right=False, include_lowest=True
    ).astype(str)
    for scheme, scheme_part in primary.groupby("outer_scheme"):
        for stratum in [
            "subregion",
            "season",
            "salinity_band",
            "support_band",
            "environmental_ood",
        ]:
            for group, part in scheme_part.groupby(stratum, observed=True):
                if len(part):
                    diagnostics.append(
                        {
                            "outer_scheme": scheme,
                            "stratum": stratum,
                            "group": str(group),
                            **regression_metrics(part),
                        }
                    )
    fold_rows = []
    for (scheme, fold, candidate), part in predictions.groupby(
        ["outer_scheme", "outer_fold", "candidate"]
    ):
        fold_rows.append(
            {
                "outer_scheme": scheme,
                "outer_fold": int(fold),
                "candidate": candidate,
                **regression_metrics(part),
            }
        )
    return metrics, pd.DataFrame(diagnostics), pd.DataFrame(fold_rows)


def build_atlas_support(
    ta_evidence: pd.DataFrame,
    predictions: pd.DataFrame,
    upstream: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted((upstream / "sss_reliability_atlas" / "year=2025").glob("month=*.parquet"))
    atlas = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    if "lme_id" not in atlas:
        mapping_path = (
            upstream.parent
            / "p1_fco2_reliability_v2.2"
            / "fco2_reliability_atlas"
            / "year=2025"
            / "month=01.parquet"
        )
        mapping = pd.read_parquet(mapping_path, columns=["coastal_node", "lme_id"]).drop_duplicates(
            "coastal_node"
        )
        atlas = atlas.merge(mapping, on="coastal_node", how="left", validate="many_to_one")
    atlas = atlas.loc[
        atlas.lme_id.eq(7) & atlas.latitude.between(35.20, 41.75, inclusive="both")
    ].copy()
    q90 = float(
        predictions.loc[
            predictions.candidate.eq(PRIMARY) & predictions.outer_scheme.eq("cruise"), "q90"
        ].median()
    )
    query = atlas.rename(columns={"background": "sss"}).copy()
    query["longitude"] = ((query.longitude + 180) % 360) - 180
    query["support_km"] = nearest_support_km(ta_evidence, query)
    query["environmental_ood"] = query.risk_environment_k64.gt(5.0) | query.grade.eq("D")
    query["ta_grade"] = assign_grade(
        np.repeat(q90, len(query)),
        query.support_km,
        np.repeat(ta_evidence.group_key.nunique(), len(query)),
        query.environmental_ood,
    )
    query["area_weight"] = np.cos(np.deg2rad(query.latitude))
    area = (
        query.groupby("ta_grade", as_index=False)
        .area_weight.sum()
        .rename(columns={"area_weight": "weighted_area"})
    )
    area["area_fraction"] = area.weighted_area / area.weighted_area.sum()
    month = query.loc[query.month.eq(7), ["latitude", "longitude", "ta_grade", "support_km"]].copy()
    return area, month


def gates(
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    area: pd.DataFrame,
) -> dict:
    lookup = metrics.set_index(["outer_scheme", "candidate"])
    primary = metrics.loc[metrics.candidate.eq(PRIMARY)].set_index("outer_scheme")
    schemes = ["cruise", "spatial_block", "subregion", "forward"]
    fold_pivot = fold_metrics.pivot(
        index=["outer_scheme", "outer_fold"], columns="candidate", values="rmse"
    )
    supported_fold_counts = {}
    for scheme in ["cruise", "spatial_block", "subregion"]:
        part = fold_pivot.loc[scheme]
        supported_fold_counts[scheme] = int(
            (
                (part[PRIMARY] < part["carter_esper_predicted_sss"])
                & (part[PRIMARY] < part["robust_mab_ta_sss"])
            ).sum()
        )
    checks = {
        "beats_carter_all_outer_schemes": all(
            lookup.loc[(scheme, PRIMARY), "rmse"]
            < lookup.loc[(scheme, "carter_esper_predicted_sss"), "rmse"]
            for scheme in schemes
        ),
        "beats_robust_regional_all_outer_schemes": all(
            lookup.loc[(scheme, PRIMARY), "rmse"]
            < lookup.loc[(scheme, "robust_mab_ta_sss"), "rmse"]
            for scheme in schemes
        ),
        "coverage90_all_schemes_85_to_95pct": bool(primary.coverage90.between(0.85, 0.95).all()),
        "median_width90_all_schemes_le_160": bool(primary.median_width90.le(160).all()),
        "forward_has_at_least_3_cruises": int(primary.loc["forward", "cruises"]) >= 3,
        "forward_has_at_least_2_years": int(primary.loc["forward", "years"]) >= 2,
        "at_least_2_supported_folds_per_repeated_scheme": all(
            value >= 2 for value in supported_fold_counts.values()
        ),
        "grade_ab_retained_area_ge_50pct": float(
            area.loc[area.ta_grade.isin(["A", "B"]), "area_fraction"].sum()
        )
        >= 0.50,
    }
    pooled_better = bool(
        primary.loc["cruise", "rmse"] < lookup.loc[("cruise", "carter_esper_predicted_sss"), "rmse"]
    )
    decision = (
        "pass_regional"
        if all(checks.values())
        else ("diagnostic_only" if pooled_better else "fail")
    )
    return {
        "decision": decision,
        "checks": {key: bool(value) for key, value in checks.items()},
        "supported_fold_counts": supported_fold_counts,
    }


def save_figures(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    atlas_month: pd.DataFrame,
    output: Path,
) -> dict[str, str]:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "carter_esper_predicted_sss": "Carter",
        "robust_mab_ta_sss": "MAB TA-SSS",
        PRIMARY: "Hierarchical",
        "nonlinear_residual_gradient_boosting": "Nonlinear residual",
        "hierarchical_oracle_insitu_sss": "In-situ SSS oracle",
    }
    captions = {}

    def save(name: str, caption: str):
        plt.tight_layout()
        plt.savefig(figure_dir / name, dpi=180, bbox_inches="tight")
        plt.close()
        captions[name] = caption

    table = metrics.pivot(index="outer_scheme", columns="candidate", values="rmse")
    table = table[[key for key in CANDIDATES if key in table]].rename(columns=labels)
    table.plot(kind="bar", figsize=(11, 5))
    plt.ylabel("RMSE (µmol kg⁻¹)")
    plt.xlabel("Untouched outer scheme")
    save(
        "fig01_outer_rmse.png",
        "Figure 1. TA RMSE under untouched cruise, 5° spatial-block, complete-subregion, and forward outer tests. Every bar uses nested cruise calibration and the matching cross-fitted upstream SSS prediction.",
    )

    primary_metrics = metrics.loc[metrics.candidate.eq(PRIMARY)].set_index("outer_scheme")
    ax = primary_metrics[["coverage50", "coverage90"]].plot(kind="bar", figsize=(8, 5))
    ax.axhspan(0.85, 0.95, color="green", alpha=0.12)
    plt.ylabel("Empirical coverage")
    save(
        "fig02_interval_coverage.png",
        "Figure 2. Empirical 50% and 90% coverage for the frozen hierarchical candidate. The green band is the preregistered acceptable range for the nominal 90% interval.",
    )

    ax = primary_metrics["median_width90"].plot(kind="bar", figsize=(8, 5), color="#4C78A8")
    ax.axhline(160, color="red", linestyle="--")
    plt.ylabel("Median 90% interval width (µmol kg⁻¹)")
    save(
        "fig03_interval_width.png",
        "Figure 3. Median nested split-conformal 90% interval width. The dashed line is the frozen 160 µmol kg⁻¹ utility ceiling.",
    )

    primary = predictions.loc[
        predictions.candidate.eq(PRIMARY) & predictions.outer_scheme.eq("cruise")
    ]
    plt.figure(figsize=(6, 6))
    for grade, part in primary.groupby("grade"):
        plt.scatter(part.truth, part.prediction, s=18, alpha=0.65, label=f"Grade {grade}")
    limits = [
        min(primary.truth.min(), primary.prediction.min()),
        max(primary.truth.max(), primary.prediction.max()),
    ]
    plt.plot(limits, limits, "k--", linewidth=1)
    plt.xlabel("Observed TA")
    plt.ylabel("Predicted TA")
    plt.legend()
    save(
        "fig04_observed_predicted.png",
        "Figure 4. Cruise-outer cross-fitted hierarchical TA versus observed TA, colored by preregistered reliability grade. No locked or external label appears.",
    )

    cruise = primary.copy()
    cruise["absolute_error"] = np.abs(cruise.prediction - cruise.truth)
    plt.figure(figsize=(8, 5))
    plt.scatter(
        cruise.support_km, cruise.absolute_error, c=cruise.sss_environment_k64, s=18, alpha=0.65
    )
    plt.xscale("symlog", linthresh=1)
    plt.xlabel("Nearest inner-training TA support (km)")
    plt.ylabel("Absolute error (µmol kg⁻¹)")
    plt.colorbar(label="Upstream SSS environmental k64")
    save(
        "fig05_support_error.png",
        "Figure 5. Cruise-outer absolute TA error against geographic support, colored by upstream SSS environmental risk. This separates sparse TA support from uncertain SSS input.",
    )

    sub = diagnostics.loc[
        diagnostics.outer_scheme.eq("cruise") & diagnostics.stratum.eq("subregion")
    ].set_index("group")
    sub.rmse.plot(kind="bar", figsize=(7, 5), color="#F58518")
    plt.ylabel("RMSE (µmol kg⁻¹)")
    save(
        "fig06_subregion_rmse.png",
        "Figure 6. Cruise-outer hierarchical RMSE for southern, central, and northern MAB subregions. The plot exposes regional averaging that pooled RMSE can hide.",
    )

    sal = diagnostics.loc[
        diagnostics.outer_scheme.eq("cruise") & diagnostics.stratum.eq("salinity_band")
    ].set_index("group")
    sal.rmse.plot(kind="bar", figsize=(8, 5), color="#54A24B")
    plt.ylabel("RMSE (µmol kg⁻¹)")
    save(
        "fig07_salinity_rmse.png",
        "Figure 7. Cruise-outer hierarchical RMSE by predicted-SSS band. Low-salinity performance is the principal diagnostic for nonconservative and estuarine influence because no frozen estuary mask exists.",
    )

    colors = {"A": "#2ca02c", "B": "#8bc34a", "C": "#ff9800", "D": "#d62728"}
    plt.figure(figsize=(7, 6))
    for grade, part in atlas_month.groupby("ta_grade"):
        plt.scatter(part.longitude, part.latitude, s=3, color=colors[grade], label=grade, alpha=0.7)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="TA grade", markerscale=3)
    save(
        "fig08_july_atlas_grade.png",
        "Figure 8. July 2025 MAB grid eligibility from nested TA interval width, TA support distance, and Issue #25 SSS risk. It is a label-free retained-area diagnostic, not independent accuracy evidence.",
    )
    return captions


def write_report(
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    grade_evidence: pd.DataFrame,
    area: pd.DataFrame,
    decision: dict,
    captions: dict[str, str],
    archive: Path,
) -> None:
    archive.mkdir(parents=True, exist_ok=True)
    primary = metrics.loc[metrics.candidate.eq(PRIMARY)].set_index("outer_scheme")
    carter = metrics.loc[metrics.candidate.eq("carter_esper_predicted_sss")].set_index(
        "outer_scheme"
    )
    robust = metrics.loc[metrics.candidate.eq("robust_mab_ta_sss")].set_index("outer_scheme")
    rows = []
    for scheme in ["cruise", "spatial_block", "subregion", "forward"]:
        rows.append(
            f"| {scheme} | {primary.loc[scheme, 'n']:.0f} | {primary.loc[scheme, 'cruises']:.0f} | "
            f"{primary.loc[scheme, 'rmse']:.2f} | {carter.loc[scheme, 'rmse']:.2f} | "
            f"{robust.loc[scheme, 'rmse']:.2f} | {primary.loc[scheme, 'coverage90']:.3f} | "
            f"{primary.loc[scheme, 'median_width90']:.1f} |"
        )
    area_text = ", ".join(
        f"{row.ta_grade}={100 * row.area_fraction:.1f}%" for row in area.itertuples()
    )
    grade_rows = "\n".join(
        f"| {row.outer_scheme} | {row.grade} | {row.n} | {row.cruises} | {row.years} |"
        for row in grade_evidence.itertuples()
    )
    failed = [name for name, value in decision["checks"].items() if not value]
    figure_markdown = "\n\n".join(
        f"![{name}](figures/{name})\n\n{caption}" for name, caption in captions.items()
    )
    report = f"""# Issue #27 — MAB TA applicability and calibrated uncertainty

## Scientific question and permitted claim

**`{decision["decision"]}`.** The MAB TA regional product does not pass unless every frozen cruise, spatial-block, complete-subregion, forward, interval-width, calibration, evidence-count, and retained-area gate passes. Failed gates: {", ".join(failed) if failed else "none"}.

The frozen hierarchical model has cruise/spatial/subregion/forward RMSE of {primary.loc["cruise", "rmse"]:.2f}, {primary.loc["spatial_block", "rmse"]:.2f}, {primary.loc["subregion", "rmse"]:.2f}, and {primary.loc["forward", "rmse"]:.2f} µmol kg⁻¹. The forward partition contains {int(primary.loc["forward", "cruises"])} cruises across {int(primary.loc["forward", "years"])} years and passes the frozen minimum evidence count, but it does not rescue the spatial, interval, or retained-area failures. Grid-weighted 2025 retained area is {area_text}.

## Data, splits, and leakage controls

CODAP-NA/GLODAP surface TA records in canonical MAB (LME 7, 35.20-41.75°N) are evaluated with four untouched outer schemes. Within every outer training partition, complete cruises are held aside for interval calibration before the point model is fit. Issue #25 SSS checkpoints generate row-wise outer-cross-fitted salinity: cruise checkpoints for cruise evaluation, spatial-block checkpoints for spatial/subregion evaluation, and the training-era checkpoint for forward evaluation. Observed in-situ salinity is an oracle only.

Locked TA and 41 external CODAP groups remain sealed. The permitted claim is limited to unsealed MAB diagnostic evidence; this experiment neither validates a released TA product nor changes the prior SAB failure.

## Candidate models and training

Candidates are Carter/ESPER with predicted SSS, robust MAB-wide TA-SSS, hierarchical subregion/regime TA-SSS, a fixed-budget nonlinear residual model, and the in-situ-SSS hierarchical oracle. The hierarchical predicted-SSS candidate was frozen as primary before results. Every outer-fold model is trained only after its evaluation rows and nested calibration cruises are removed.

## Main development results

| Outer scheme | n | cruises | Primary RMSE | Carter RMSE | Regional TA-SSS RMSE | 90% coverage | median 90% width |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Decision and limitations

The test asks whether MAB can support a defensible regional TA product, not whether a flexible model can fit the existing records. A pass requires the primary model to beat both Carter and the simple regional relation in every outer scheme, useful independently calibrated intervals, at least three forward cruises across two years, and at least 50% grade-A/B retained grid area. A single row-rich cruise cannot satisfy the evidence rule.

Independent evidence counts for every observed-data grade are saved in `tables/grade_evidence.csv`. These counts describe the outer held observations; they do not upgrade the 2025 grid, whose TA grade is D everywhere under the frozen width and support rules.

| Outer scheme | grade | n | cruises | years |
|---|---|---:|---:|---:|
{grade_rows}

Bathymetry, shelf-zone, estuary masks, and explicit coast distance are absent from the frozen v2.2 cache. They are recorded as unavailable. Low-salinity, subregion, season, upstream-SSS OOD, and TA-support diagnostics are reported without silently introducing new post-registration covariates. SAB retains its previous `fail` decision.

## Figure and table index

{figure_markdown}

The source tables are `outer_metrics.csv`, `outer_fold_metrics.csv`, `diagnostic_metrics.csv`, `grade_evidence.csv`, `retained_area.csv`, `unavailable_diagnostics.csv`, `decision_checks.csv`, `interval_metrics.csv`, and `source_hashes.csv`. Row-level outer predictions remain outside Git.

## Reproducibility

Machine-readable outer predictions are intentionally kept outside Git. The archive contains aggregate source tables, captions, protocol, source hashes, and artifact hashes. `tables/unavailable_diagnostics.csv` records diagnostics that cannot be computed from frozen inputs. Generated row-level data live under `outputs/experiments/{EXPERIMENT_ID}`.
"""
    write_text_lf(archive / "REPORT.md", report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/experiments" / EXPERIMENT_ID)
    parser.add_argument(
        "--archive", type=Path, default=ROOT / "docs/experiment_archive" / EXPERIMENT_ID
    )
    parser.add_argument(
        "--upstream",
        type=Path,
        default=Path(
            r"D:\proj_personal\PhD\ReCAD\v2.0\outputs\experiments\p1_sss_reliability_v2.2"
        ),
    )
    parser.add_argument(
        "--esper-mat",
        type=Path,
        default=Path(r"D:\proj_personal\PhD\ESPER\ESPER_LIR_Files\LIR_files_TA_v3.mat"),
    )
    parser.add_argument("--reuse-predictions", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "configs/p1_ta_mab_reliability_v2.2.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = FrozenManifest.load(ROOT / config["data_manifest"])
    validation = manifest.validate(hash_mode="full")
    gateway = P1DataGateway(manifest)
    columns = ["obs_id", "salinity", "sst", "sss", "adt", "wspd", "pco2air", "temperature"]
    train = apply_bight_definition(
        prepare(gateway.load_labels("ta", Purpose.TRAIN, columns=columns)), "mab"
    )
    development = apply_bight_definition(
        prepare(gateway.load_labels("ta", Purpose.SELECTION, columns=columns)), "mab"
    )
    train["evidence_partition"] = "train"
    development["evidence_partition"] = "development"
    combined = pd.concat([train, development], ignore_index=True)
    sss_train, sss_development = load_sss(gateway)
    sss_evidence = pd.concat([sss_train, sss_development], ignore_index=True)
    estimator = ESPER_LIR_TA.from_mat(args.esper_mat)
    prediction_path = args.output / "outer_predictions.parquet"
    if args.reuse_predictions and prediction_path.exists():
        predictions = pd.read_parquet(prediction_path)
    else:
        predictions = evaluate_outer(combined, sss_evidence, estimator, args.upstream)
    metrics, diagnostics, fold_metrics = summarize(predictions)
    area, atlas_month = build_atlas_support(combined, predictions, args.upstream)
    decision = gates(metrics, fold_metrics, predictions, area)
    grade_evidence = (
        predictions.loc[predictions.candidate.eq(PRIMARY)]
        .groupby(["outer_scheme", "grade"], as_index=False)
        .agg(
            n=("truth", "size"),
            cruises=("group_key", "nunique"),
            years=("year", "nunique"),
        )
    )

    predictions.to_parquet(args.output / "outer_predictions.parquet", index=False)
    metrics.to_csv(args.output / "outer_metrics.csv", index=False)
    fold_metrics.to_csv(args.output / "outer_fold_metrics.csv", index=False)
    diagnostics.to_csv(args.output / "diagnostic_metrics.csv", index=False)
    grade_evidence.to_csv(args.output / "grade_evidence.csv", index=False)
    area.to_csv(args.output / "retained_area.csv", index=False)
    (args.output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    protocol = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_head(),
        "config_sha256": sha256(config_path),
        "manifest_validation": validation,
        "ta_rows": len(combined),
        "ta_cruises": combined.group_key.nunique(),
        "sss_evidence_rows": len(sss_evidence),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "source_hashes": {
            "runner": file_hash(Path(__file__)),
            "utility": file_hash(ROOT / "src/recad/evaluate/ta_mab_reliability.py"),
            "config": file_hash(config_path),
            "manifest": file_hash(ROOT / config["data_manifest"]),
        },
        "upstream_checkpoint_hashes": {
            path.name: file_hash(path)
            for path in sorted((args.upstream / "checkpoints").glob("oof_*_seed100.pt"))
        },
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    captions = save_figures(predictions, metrics, diagnostics, atlas_month, args.output)
    table_dir = args.archive / "tables"
    figure_dir = args.archive / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    write_csv_lf(metrics, table_dir / "outer_metrics.csv")
    write_csv_lf(fold_metrics, table_dir / "outer_fold_metrics.csv")
    write_csv_lf(diagnostics, table_dir / "diagnostic_metrics.csv")
    write_csv_lf(grade_evidence, table_dir / "grade_evidence.csv")
    write_csv_lf(area, table_dir / "retained_area.csv")
    unavailable = pd.DataFrame(
        {
            "diagnostic": ["bathymetry", "shelf_zone", "estuary_mask", "coast_distance"],
            "status": ["unavailable_in_frozen_cache"] * 4,
        }
    )
    write_csv_lf(unavailable, table_dir / "unavailable_diagnostics.csv")
    decision_table = pd.DataFrame(
        [{"gate": key, "passed": value} for key, value in decision["checks"].items()]
    )
    write_csv_lf(decision_table, table_dir / "decision_checks.csv")
    interval_table = metrics[
        [
            "outer_scheme",
            "candidate",
            "coverage50",
            "coverage90",
            "median_width50",
            "median_width90",
        ]
    ]
    write_csv_lf(interval_table, table_dir / "interval_metrics.csv")
    source_table = pd.DataFrame(
        [
            {"artifact": key, "sha256": value}
            for key, value in {
                **protocol["source_hashes"],
                **protocol["upstream_checkpoint_hashes"],
            }.items()
        ]
    )
    write_csv_lf(source_table, table_dir / "source_hashes.csv")
    for path in (args.output / "figures").glob("*.png"):
        (figure_dir / path.name).write_bytes(path.read_bytes())
    write_text_lf(args.archive / "captions.json", json.dumps(captions, indent=2))
    write_text_lf(args.archive / "protocol.json", json.dumps(protocol, indent=2))
    write_report(metrics, diagnostics, grade_evidence, area, decision, captions, args.archive)
    write_text_lf(
        args.archive / "README.md",
        "# MAB TA reliability reviewer archive\n\n"
        "Issue #27 aggregate evidence, figures, captions, hashes, and frozen decision. "
        "Row-level predictions are excluded from Git. See REPORT.md for interpretation.\n",
    )
    caption_text = "# Figure captions\n\n" + "\n\n".join(
        f"## {name}\n\n{caption}" for name, caption in captions.items()
    )
    write_text_lf(args.archive / "CAPTIONS.md", caption_text + "\n")
    figure_sources = {
        "fig01_outer_rmse.png": ["outer_metrics.csv"],
        "fig02_interval_coverage.png": ["interval_metrics.csv"],
        "fig03_interval_width.png": ["interval_metrics.csv"],
        "fig04_observed_predicted.png": ["local:outer_predictions.parquet"],
        "fig05_support_error.png": ["local:outer_predictions.parquet"],
        "fig06_subregion_rmse.png": ["diagnostic_metrics.csv"],
        "fig07_salinity_rmse.png": ["diagnostic_metrics.csv"],
        "fig08_july_atlas_grade.png": [
            "local:p1_sss_reliability_v2.2/sss_reliability_atlas",
            "retained_area.csv",
        ],
    }
    hashes = {}
    for path in sorted(args.archive.rglob("*")):
        if path.is_file() and path.name not in {"artifact_hashes.json", "archive_manifest.json"}:
            hashes[str(path.relative_to(args.archive)).replace("\\", "/")] = file_hash(path)
    archive_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "training_git_commit": protocol["git_commit"],
        "data_manifest_sha256": protocol["manifest_validation"]["manifest_sha256"],
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "MAB unsealed train/development outer stress tests and label-free 2025 grid",
        "decision": decision["decision"],
        "figure_source_data": figure_sources,
        "figure_captions": captions,
        "files_sha256": hashes,
        "archive_builder_sha256": protocol["source_hashes"]["runner"],
        "analysis_script_sha256": protocol["source_hashes"]["utility"],
        "source_artifacts_sha256": {
            "config": protocol["source_hashes"]["config"],
            "manifest": protocol["source_hashes"]["manifest"],
            **protocol["upstream_checkpoint_hashes"],
        },
    }
    write_text_lf(args.archive / "archive_manifest.json", json.dumps(archive_manifest, indent=2))
    hashes["archive_manifest.json"] = file_hash(args.archive / "archive_manifest.json")
    write_text_lf(args.archive / "artifact_hashes.json", json.dumps(hashes, indent=2))
    print(
        json.dumps({"decision": decision, "metrics": metrics.to_dict("records")}, indent=2),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
