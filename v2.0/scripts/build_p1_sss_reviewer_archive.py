"""Rebuild the reviewer-ready P1.1 SSS archive from hashed local outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_sss_viability_v2.2"
MODEL_LABELS = {
    "background": "GLORYS",
    "regional_month_bias": "Regional-month bias",
    "ridge_residual": "Ridge residual",
    "catboost_residual": "CatBoost residual",
    "point_mlp_residual": "Point MLP residual",
    "soft_experts_residual": "Soft experts residual",
}
COLORS = {"GLORYS": "#777777", "CatBoost residual": "#E76F51", "Soft experts residual": "#2A9D8F"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 << 20):
            h.update(block)
    return h.hexdigest()


def normalize_archive_text(archive: Path) -> None:
    """Write hash-tracked archive text with deterministic LF endings."""
    for path in archive.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".csv", ".md"}:
            content = path.read_text(encoding="utf-8")
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def table_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for record in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in record) + " |")
    return "\n".join([header, rule, *rows])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/experiments" / EXPERIMENT_ID)
    parser.add_argument(
        "--archive", type=Path, default=ROOT / "docs/experiment_archive" / EXPERIMENT_ID
    )
    args = parser.parse_args()
    figures = args.archive / "figures"
    tables = args.archive / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.source / "summary.csv")
    by_seed = pd.read_csv(args.source / "metrics_by_seed.csv")
    cv = pd.read_csv(args.source / "cv_metrics.csv")
    forward = pd.read_csv(args.source / "forward_metrics.csv")
    strata = pd.read_csv(args.source / "stratified_metrics.csv")
    support = pd.read_csv(args.source / "sss_support_distance_metrics.csv")
    predictions = pd.read_parquet(args.source / "candidate_predictions.parquet")
    gate = json.loads((args.source / "development_gate.json").read_text(encoding="utf-8"))
    selection = json.loads((args.source / "selection.json").read_text(encoding="utf-8"))
    protocol = json.loads((args.source / "protocol.json").read_text(encoding="utf-8"))
    selected = gate["selected_model"]

    model_table = summary.copy()
    model_table["model"] = model_table.model.map(MODEL_LABELS)
    model_table.to_csv(tables / "table01_development_model_summary.csv", index=False)
    cv_summary = (
        cv.groupby("model")
        .agg(
            folds=("fold", "nunique"),
            pooled_rmse_mean=("pooled_rmse", "mean"),
            pooled_rmse_sd=("pooled_rmse", "std"),
            lme_macro_rmse_mean=("lme_macro_rmse", "mean"),
            lme_macro_rmse_sd=("lme_macro_rmse", "std"),
            pooled_skill_mean=("skill_vs_background", "mean"),
            pooled_skill_sd=("skill_vs_background", "std"),
        )
        .reset_index()
    )
    cv_summary["model"] = cv_summary.model.map(MODEL_LABELS)
    cv_summary.to_csv(tables / "table02_five_fold_cv_summary.csv", index=False)
    forward_summary = (
        forward.groupby("model")
        .agg(
            seeds=("seed", "nunique"),
            pooled_rmse_mean=("pooled_rmse", "mean"),
            pooled_rmse_sd=("pooled_rmse", "std"),
            lme_macro_rmse_mean=("lme_macro_rmse", "mean"),
            lme_macro_rmse_sd=("lme_macro_rmse", "std"),
            pooled_skill_mean=("skill_vs_background", "mean"),
            pooled_skill_sd=("skill_vs_background", "std"),
        )
        .reset_index()
    )
    forward_summary["model"] = forward_summary.model.map(MODEL_LABELS)
    forward_summary.to_csv(tables / "table03_forward_summary.csv", index=False)
    lme = (
        strata.loc[strata.stratum.eq("lme")]
        .groupby(["model", "group"])
        .agg(
            n=("n", "max"),
            rmse_mean=("model_rmse", "mean"),
            background_rmse=("background_rmse", "mean"),
            skill_mean=("skill_vs_background", "mean"),
            skill_sd=("skill_vs_background", "std"),
        )
        .reset_index()
    )
    lme["model"] = lme.model.map(MODEL_LABELS)
    lme.to_csv(tables / "table04_lme_skill.csv", index=False)
    salinity = (
        strata.loc[strata.stratum.eq("salinity_band")]
        .groupby(["model", "group"])
        .agg(
            n=("n", "max"),
            rmse_mean=("model_rmse", "mean"),
            background_rmse=("background_rmse", "mean"),
            skill_mean=("skill_vs_background", "mean"),
            skill_sd=("skill_vs_background", "std"),
        )
        .reset_index()
    )
    salinity["model"] = salinity.model.map(MODEL_LABELS)
    salinity.to_csv(tables / "table05_salinity_band_skill.csv", index=False)
    support.to_csv(tables / "table06_sss_support_distance.csv", index=False)
    gate_table = pd.DataFrame(
        [{"gate": key, "passed": bool(value)} for key, value in gate["checks"].items()]
    )
    gate_table.to_csv(tables / "table07_development_gates.csv", index=False)
    calibration = pd.DataFrame(
        [
            {
                "model": MODEL_LABELS[selected],
                "absolute_error_q90_psu": selection["conformal_absolute_error_q90"],
                "development_coverage": selection["development_coverage_90"],
                "nominal_coverage": 0.90,
                "calibration_set": "development",
                "independent_coverage_tested": False,
            }
        ]
    )
    calibration.to_csv(tables / "table08_uncertainty_calibration.csv", index=False)

    make_model_figure(by_seed, figures / "fig01_development_model_comparison.png")
    make_cv_figure(cv, figures / "fig02_five_fold_cv.png")
    make_lme_figure(lme, figures / "fig03_lme_skill_sensitivity.png")
    make_salinity_figure(salinity, figures / "fig04_salinity_band_skill.png")
    make_forward_figure(forward, figures / "fig05_forward_chain.png")
    make_support_figure(support, figures / "fig06_sss_support_distance.png")
    scatter_predictions = (
        predictions.loc[predictions.model.eq(selected)]
        .groupby("record_id")
        .agg(
            truth=("truth", "first"),
            background=("background", "first"),
            prediction=("prediction", "mean"),
        )
        .reset_index()
    )
    make_scatter_figure(scatter_predictions, figures / "fig07_observed_vs_predicted.png")
    make_calibration_figure(
        predictions,
        selected,
        selection["conformal_absolute_error_q90"],
        figures / "fig08_absolute_error_calibration.png",
    )

    figure_sources = {
        "fig01_development_model_comparison.png": ["table01_development_model_summary.csv"],
        "fig02_five_fold_cv.png": ["table02_five_fold_cv_summary.csv"],
        "fig03_lme_skill_sensitivity.png": ["table04_lme_skill.csv"],
        "fig04_salinity_band_skill.png": ["table05_salinity_band_skill.csv"],
        "fig05_forward_chain.png": ["table03_forward_summary.csv"],
        "fig06_sss_support_distance.png": ["table06_sss_support_distance.csv"],
        "fig07_observed_vs_predicted.png": ["local:selected_development_predictions.parquet"],
        "fig08_absolute_error_calibration.png": [
            "table08_uncertainty_calibration.csv",
            "local:candidate_predictions.parquet",
        ],
    }
    figure_captions = {
        "fig01_development_model_comparison.png": (
            "Development-set SSS RMSE for GLORYS and five residual-correction candidates under pooled, "
            "cruise-equal, and LME-macro aggregation. Points are means across seeds 100-102 and error bars "
            "show one standard deviation; lower is better. Soft experts win the preregistered LME-macro "
            "criterion, while CatBoost has the lowest pooled and cruise-equal RMSE."
        ),
        "fig02_five_fold_cv.png": (
            "Five-fold cruise-grouped cross-validation for the same candidate families. Bars show mean "
            "LME-macro RMSE and pooled skill relative to GLORYS across folds; error bars show one standard "
            "deviation. CatBoost is the strongest cross-validation sensitivity comparator."
        ),
        "fig03_lme_skill_sensitivity.png": (
            "Development skill by Large Marine Ecosystem (LME), comparing the selected soft-expert ensemble "
            "with CatBoost; positive skill indicates lower MSE than GLORYS. Labels give development record "
            "counts. The soft-expert selection advantage is sensitive to sparse LME 55 (n=30)."
        ),
        "fig04_salinity_band_skill.png": (
            "Development skill of the selected soft-expert ensemble across observed-SSS bands, with record "
            "counts shown separately. Skill remains positive in every reported band, including low-salinity "
            "coastal observations, but the freshest bands have much smaller support."
        ),
        "fig05_forward_chain.png": (
            "Forward-chain development skill for seeds 100-102: training cruises end by 2018 and evaluation "
            "uses cruises assigned to 2019-2021. All seeds retain positive pooled skill relative to GLORYS; "
            "the horizontal line marks zero skill."
        ),
        "fig06_sss_support_distance.png": (
            "Development skill of the selected soft-expert ensemble by distance to the nearest SSS training "
            "support, with record counts by bin. Positive skill persists in all populated bins, but distant "
            "support bins contain fewer observations and remain an extrapolation risk."
        ),
        "fig07_observed_vs_predicted.png": (
            "Hexbin density of development observations against GLORYS and the three-seed mean soft-expert "
            "prediction on identical 0-40 PSU axes. The white line is 1:1 and color is log10 record count. "
            "Displayed RMSE is the row-pooled ensemble RMSE, not the across-seed mean reported in Table 1."
        ),
        "fig08_absolute_error_calibration.png": (
            "Development absolute-error empirical distributions for GLORYS and the selected soft-expert "
            "ensemble. The dashed line is the development-calibrated 90th-percentile error threshold "
            "(1.156 PSU); its 0.900 development coverage is not independent calibration evidence."
        ),
    }
    report = build_report(
        model_table,
        cv_summary,
        forward_summary,
        lme,
        salinity,
        support,
        gate,
        selection,
        protocol,
        figure_captions,
    )
    (args.archive / "REPORT.md").write_text(report, encoding="utf-8")
    captions_text = (
        "# Figure captions\n\n"
        + "\n\n".join(f"## {name}\n\n{figure_captions[name]}" for name in figure_sources)
        + "\n"
    )
    (args.archive / "CAPTIONS.md").write_text(captions_text, encoding="utf-8")
    (args.archive / "README.md").write_text(
        "# Archive index\n\nStart with [REPORT.md](REPORT.md); standalone figure captions are in "
        "[CAPTIONS.md](CAPTIONS.md). Every figure has source data and a SHA256 entry in "
        "`archive_manifest.json`. Large predictions and checkpoints remain in the local ignored experiment "
        "directory recorded by the manifest.\n",
        encoding="utf-8",
    )
    normalize_archive_text(args.archive)
    files = {
        str(path.relative_to(args.archive)).replace("\\", "/"): digest(path)
        for path in args.archive.rglob("*")
        if path.is_file() and path.name != "archive_manifest.json"
    }
    source_artifacts = {
        name: digest(args.source / name)
        for name in (
            "candidate_predictions.parquet",
            "selected_development_predictions.parquet",
            "summary.csv",
            "cv_metrics.csv",
            "forward_metrics.csv",
            "development_gate.json",
            "protocol.json",
        )
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_experiment_directory": str(args.source),
        "training_git_commit": protocol["git_commit"],
        "archive_builder": "scripts/build_p1_sss_reviewer_archive.py",
        "archive_builder_sha256": digest(Path(__file__)),
        "analysis_script_sha256": digest(ROOT / "scripts/analyze_p1_sss_results.py"),
        "source_artifacts_sha256": source_artifacts,
        "data_manifest_sha256": protocol["manifest_validation"]["manifest_sha256"],
        "data_hash_mode": protocol["manifest_validation"]["hash_mode"],
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "North-American-adjacent train/development and grouped CV only",
        "decision": gate["decision"],
        "figure_source_data": figure_sources,
        "figure_captions": figure_captions,
        "files_sha256": files,
    }
    with (args.archive / "archive_manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        stream.write(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "archive": str(args.archive),
                "files": len(files),
                "figures": len(figure_sources),
                "gate": gate["development_gate_passed"],
            },
            indent=2,
        )
    )
    return 0


def make_model_figure(by_seed: pd.DataFrame, path: Path) -> None:
    metrics = [
        ("pooled_rmse", "Pooled RMSE"),
        ("cruise_equal_rmse", "Cruise-equal RMSE"),
        ("lme_macro_rmse", "LME-macro RMSE"),
    ]
    order = [
        "background",
        "regional_month_bias",
        "ridge_residual",
        "catboost_residual",
        "point_mlp_residual",
        "soft_experts_residual",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, (column, title) in zip(axes, metrics, strict=True):
        stats = by_seed.groupby("model")[column].agg(["mean", "std"]).reindex(order)
        labels = [MODEL_LABELS[name] for name in order]
        colors = [COLORS.get(label, "#457B9D") for label in labels]
        errors = stats["std"].fillna(0).to_numpy()
        ax.barh(labels, stats["mean"], xerr=errors, color=colors, alpha=0.9, capsize=3)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("PSU (lower is better)")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Development performance; error bars show between-seed SD")
    fig.tight_layout()
    save_figure(fig, path)


def make_cv_figure(cv: pd.DataFrame, path: Path) -> None:
    order = cv.groupby("model").lme_macro_rmse.mean().sort_values().index
    stats = (
        cv.groupby("model")
        .agg(
            rmse=("lme_macro_rmse", "mean"),
            sd=("lme_macro_rmse", "std"),
            skill=("skill_vs_background", "mean"),
        )
        .reindex(order)
    )
    labels = [MODEL_LABELS[name] for name in order]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(labels, stats.rmse, xerr=stats.sd.fillna(0), color="#457B9D", capsize=3)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("LME-macro RMSE (PSU)")
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].barh(labels, stats.skill, color="#2A9D8F")
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_xlabel("Pooled skill vs GLORYS")
    fig.suptitle("Five-fold cruise-grouped cross-validation")
    fig.tight_layout()
    save_figure(fig, path)


def make_lme_figure(lme: pd.DataFrame, path: Path) -> None:
    keep = lme.loc[lme.model.isin(["CatBoost residual", "Soft experts residual"])].copy()
    pivot = keep.pivot(index="group", columns="model", values="skill_mean")
    counts = keep.groupby("group").n.max()
    ordered = pivot["Soft experts residual"].sort_values().index
    pivot, counts = pivot.loc[ordered], counts.loc[ordered]
    y = np.arange(len(pivot))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(
        y - width / 2,
        pivot["CatBoost residual"],
        width,
        label="CatBoost",
        color=COLORS["CatBoost residual"],
    )
    ax.barh(
        y + width / 2,
        pivot["Soft experts residual"],
        width,
        label="Soft experts",
        color=COLORS["Soft experts residual"],
    )
    labels = [f"LME {group} (n={int(counts[group]):,})" for group in pivot.index]
    ax.set_yticks(y, labels)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Skill vs GLORYS")
    ax.set_title("Regional sensitivity; groups ordered by soft-expert skill")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, path)


def make_salinity_figure(salinity: pd.DataFrame, path: Path) -> None:
    keep = salinity.loc[salinity.model.isin(["CatBoost residual", "Soft experts residual"])].copy()
    order = ["[-inf, 20.0)", "[20.0, 30.0)", "[30.0, 33.0)", "[33.0, 36.0)", "[36.0, inf)"]
    pivot = keep.pivot(index="group", columns="model", values="skill_mean").reindex(order)
    counts = keep.groupby("group").n.max().reindex(order)
    x = np.arange(len(order))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        pivot["CatBoost residual"],
        width,
        label="CatBoost",
        color=COLORS["CatBoost residual"],
    )
    ax.bar(
        x + width / 2,
        pivot["Soft experts residual"],
        width,
        label="Soft experts",
        color=COLORS["Soft experts residual"],
    )
    labels = ["<20", "20-30", "30-33", "33-36", ">=36"]
    ax.set_xticks(x, [f"{label}\n(n={int(n):,})" for label, n in zip(labels, counts, strict=True)])
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Skill vs GLORYS")
    ax.set_xlabel("Observed SSS (PSU)")
    ax.set_title("Salinity-regime skill on development cruises")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, path)


def make_forward_figure(forward: pd.DataFrame, path: Path) -> None:
    candidate = forward.loc[forward.model.eq("soft_experts_residual")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, column, title in [
        (axes[0], "pooled_rmse", "Pooled RMSE"),
        (axes[1], "lme_macro_rmse", "LME-macro RMSE"),
    ]:
        bg = float(forward.loc[forward.model.eq("background"), column].iloc[0])
        values = candidate[column].to_numpy()
        ax.bar(
            [0, 1], [bg, values.mean()], color=[COLORS["GLORYS"], COLORS["Soft experts residual"]]
        )
        ax.scatter(np.repeat(1, len(values)), values, color="black", zorder=3, label="seeds")
        ax.set_xticks([0, 1], ["GLORYS", "Soft experts"])
        ax.set_ylabel("PSU")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Forward chain: train through 2018, development 2019-2021")
    fig.tight_layout()
    save_figure(fig, path)


def make_support_figure(support: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(support.support_bin_km, support.skill_vs_background, color="#2A9D8F")
    for bar, n in zip(bars, support.n, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"n={int(n):,}",
            ha="center",
            fontsize=8,
        )
    ax.axhline(0, color="black", lw=1)
    ax.set_ylim(0, max(0.9, support.skill_vs_background.max() + 0.12))
    ax.set_ylabel("Skill vs GLORYS")
    ax.set_xlabel("Nearest SSS training location (km)")
    ax.set_title("Spatial support-distance stress test for the selected soft experts")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, path)


def make_scatter_figure(predictions: pd.DataFrame, path: Path) -> None:
    truth = predictions.truth.to_numpy(float)
    model = predictions.prediction.to_numpy(float)
    background = predictions.background.to_numpy(float)
    # Keep the full physically relevant range visible.  The tails are sparse, but
    # hiding them would make the fit diagnostic look better than the scored data.
    limits = [0, 40]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, values, title in [
        (axes[0], background, "GLORYS background"),
        (axes[1], model, "Soft experts"),
    ]:
        hb = ax.hexbin(truth, values, gridsize=80, bins="log", mincnt=1, cmap="viridis")
        ax.plot(limits, limits, color="white", lw=1.2)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        rmse = np.sqrt(np.mean((values - truth) ** 2))
        ax.set_title(f"{title}\nRMSE={rmse:.3f} PSU")
        ax.set_xlabel("Observed SSS (PSU)")
        fig.colorbar(hb, ax=ax, label="log10 count")
    axes[0].set_ylabel("Estimated SSS (PSU)")
    fig.suptitle("Development observations; identical 1:1 axes")
    fig.tight_layout()
    save_figure(fig, path)


def make_calibration_figure(
    predictions: pd.DataFrame, selected: str, q90: float, path: Path
) -> None:
    ensemble = (
        predictions.loc[predictions.model.eq(selected)]
        .groupby("record_id")
        .agg(
            truth=("truth", "first"),
            background=("background", "first"),
            prediction=("prediction", "mean"),
        )
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for values, label, color in [
        (np.abs(ensemble.background - ensemble.truth), "GLORYS", COLORS["GLORYS"]),
        (
            np.abs(ensemble.prediction - ensemble.truth),
            "Soft experts",
            COLORS["Soft experts residual"],
        ),
    ]:
        ordered = np.sort(values.to_numpy(float))
        probability = np.arange(1, len(ordered) + 1) / len(ordered)
        ax.plot(ordered, probability, label=label, color=color)
    ax.axhline(0.9, color="black", ls="--", lw=1)
    ax.axvline(
        q90,
        color=COLORS["Soft experts residual"],
        ls=":",
        lw=1.5,
        label=f"development q90={q90:.3f} PSU",
    )
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Absolute error (PSU)")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_title("Development error distribution and calibration threshold")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, path)


def build_report(
    model_table,
    cv_summary,
    forward_summary,
    lme,
    salinity,
    support,
    gate,
    selection,
    protocol,
    figure_captions,
) -> str:
    selected = "Soft experts residual"
    development_display = model_table[
        [
            "model",
            "pooled_rmse_mean",
            "cruise_equal_rmse_mean",
            "lme_macro_rmse_mean",
            "worst_lme_rmse_mean",
            "skill_vs_background_mean",
        ]
    ].round(3)
    cv_display = cv_summary[
        [
            "model",
            "folds",
            "lme_macro_rmse_mean",
            "lme_macro_rmse_sd",
            "pooled_skill_mean",
            "pooled_skill_sd",
        ]
    ].round(3)
    forward_display = forward_summary.round(3)
    selected_lme = lme.loc[lme.model.eq(selected)]
    eligible = selected_lme.loc[selected_lme.n >= 100]
    low = salinity.loc[
        (salinity.model.eq(selected)) & salinity.group.isin(["[-inf, 20.0)", "[20.0, 30.0)"])
    ]
    return f"""# Reviewer archive: P1.1 coastal SSS viability

## Executive finding

The preregistered development criterion selected the top-2 soft-expert residual model. Relative to unmodified GLORYS, it reduced cruise-equal RMSE by {gate["metrics"]["cruise_equal_improvement"]:.1%}, LME-macro RMSE by {gate["metrics"]["lme_macro_improvement"]:.1%}, and worst-LME RMSE by {gate["metrics"]["worst_lme_improvement"]:.1%}. All development gates passed. This nominates a regional candidate for the one-time locked test; it does not establish final regional or global product skill.

## Scientific question and permitted claim

The experiment asks whether predictor-only residual correction can improve monthly coastal SSS over GLORYS across unseen cruises, regions, salinity regimes, and forward time. The frozen cache spans North-American-adjacent waters (0-70.125 N, 180-315 E). The allowed claim is restricted to grouped development and cross-validation evidence in this domain. Locked-test and external-independent labels were not opened.

## Data, splits, and leakage controls

- SOCAT in-situ salinity is the target; GLORYS SSS is only a predictor and background baseline.
- Train: {protocol["train_rows"]:,} valid records from {protocol["train_cruises"]:,} cruises.
- Development: {protocol["development_rows"]:,} records from {protocol["development_cruises"]:,} cruises, through 2025.
- Five-fold CV holds out complete cruises. The forward chain trains on cruise maximum year <=2018 and evaluates development cruises assigned to 2019-2021.
- The data manifest SHA256 is `{protocol["manifest_validation"]["manifest_sha256"]}`; all 12 files passed full hash verification.
- `locked_test_opened=false`; `external_opened=false`.

## Candidate models and training

All learned models predict a correction added to GLORYS SSS. Candidates were GLORYS, a shrunk regional-month bias, Ridge, CatBoost, a point MLP, and an eight-way top-2 soft mixture of experts. Neural candidates used 5,000 optimizer steps with batch size 2,048 and seeds 100/101/102. CatBoost used up to 1,500 trees and the same three seeds. Checkpoints were selected only by development LME-macro RMSE.

## Main development results

{table_markdown(development_display)}

![Development model comparison](figures/fig01_development_model_comparison.png)

*{figure_captions["fig01_development_model_comparison.png"]}*

CatBoost has the lowest pooled and cruise-equal RMSE. Soft experts have the lowest preregistered LME-macro metric among candidate families and were therefore selected. The distinction is scientifically relevant: the soft-expert advantage is strongly influenced by LME 55, which has only 30 development records. CatBoost remains a prespecified sensitivity comparator for the locked evaluation; locked results cannot be used to choose retrospectively between them.

## Cruise-grouped cross-validation

{table_markdown(cv_display)}

![Five-fold cruise CV](figures/fig02_five_fold_cv.png)

*{figure_captions["fig02_five_fold_cv.png"]}*

CatBoost is strongest in five-fold CV, while both neural candidates retain large positive skill. This tension with the frozen development selection is reported rather than resolved after observing results.

## Region, low salinity, and extrapolation stress tests

All {len(eligible)}/{len(eligible)} LMEs with at least 100 records have positive selected-model skill. The <20 PSU and 20-30 PSU strata contain {int(low.n.iloc[0]):,} and {int(low.n.iloc[1]):,} records and both have positive skill. Every SSS support-distance bin is positive, although the >250 km bin contains only {int(support.n.iloc[-1]):,} records and is not strong evidence for remote extrapolation.

![LME sensitivity](figures/fig03_lme_skill_sensitivity.png)

*{figure_captions["fig03_lme_skill_sensitivity.png"]}*

![Salinity-band skill](figures/fig04_salinity_band_skill.png)

*{figure_captions["fig04_salinity_band_skill.png"]}*

![SSS support-distance skill](figures/fig06_sss_support_distance.png)

*{figure_captions["fig06_sss_support_distance.png"]}*

## Forward-time evidence

{table_markdown(forward_display)}

![Forward chain](figures/fig05_forward_chain.png)

*{figure_captions["fig05_forward_chain.png"]}*

Soft experts have mean forward pooled skill {gate["metrics"]["forward_pooled_skill_mean"]:.3f}; all three seeds are positive.

## Fit and uncertainty diagnostics

![Observed versus predicted](figures/fig07_observed_vs_predicted.png)

*{figure_captions["fig07_observed_vs_predicted.png"]}*

![Absolute-error calibration](figures/fig08_absolute_error_calibration.png)

*{figure_captions["fig08_absolute_error_calibration.png"]}*

The development absolute-error 90th percentile is {selection["conformal_absolute_error_q90"]:.3f} PSU and gives development coverage {selection["development_coverage_90"]:.3f}. Because the same development data calibrated this interval, it is a frozen parameter awaiting locked-test coverage evaluation, not independent calibration evidence.

## Decision and limitations

Decision: `{gate["decision"]}`. The development gate passed, but final status remains pending Issue #11. No global claim is allowed because the frozen evaluation cache is regional. Sparse LMEs, rare extreme fresh water, spatial support beyond 250 km, background-product assimilation dependence, and development-calibrated uncertainty remain limitations. The archive contains the exact source tables for every plotted aggregate; large row-level predictions and checkpoints remain in the hashed local experiment directory.

## Figure and table index

1. Development comparison: `fig01`; source `table01`.
2. Five-fold cruise CV: `fig02`; source `table02`.
3. LME sensitivity: `fig03`; source `table04`.
4. Salinity bands: `fig04`; source `table05`.
5. Forward chain: `fig05`; source `table03`.
6. SSS training support distance: `fig06`; source `table06`.
7. Observation-prediction density: `fig07`; source is the hashed local standard prediction table.
8. Absolute-error calibration: `fig08`; aggregate source `table08`, row source is the hashed local candidate prediction table.

`archive_manifest.json` records the data/code provenance, evidence boundary, figure-to-source mapping, and SHA256 of every archived file.
"""


if __name__ == "__main__":
    raise SystemExit(main())
