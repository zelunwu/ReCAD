"""Build the reviewer-ready SAB/MAB TA comparison archive."""

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
EXPERIMENT_ID = "p1_ta_bights_v2.2"
REGIONS = ["sab", "mab"]
REGION_LABELS = {"sab": "SAB", "mab": "MAB"}
MODEL_LABELS = {
    "carter_esper_prior": "Carter/ESPER",
    "carter_region_season_corrected": "Carter + subregion correction",
    "global_ta_sss": "Bight TA-SSS",
    "lme_ta_sss": "Subregion TA-SSS",
    "hierarchical_ta_sss": "Hierarchical TA-SSS",
    "hierarchical_residual": "Hierarchical residual MLP",
    "carter_esper_oracle_insitu_sss": "Carter oracle (in-situ SSS)",
}
SUBREGION_LABELS = {"0": "south", "1": "central", "2": "north"}
COLORS = {"SAB": "#D55E00", "MAB": "#0072B2"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        frame.to_csv(stream, index=False, lineterminator="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "outputs/experiments/p1_ta_bights_v2.2",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "docs/experiment_archive/p1_ta_bights_v2.2",
    )
    args = parser.parse_args()
    figures = args.archive / "figures"
    tables = args.archive / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    summaries = []
    cvs = []
    strata = []
    forwards = []
    leaves = []
    selected_frames: dict[str, pd.DataFrame] = {}
    gates: dict[str, dict] = {}
    protocols: dict[str, dict] = {}
    selections: dict[str, dict] = {}
    for region in REGIONS:
        source = args.source / region
        summary = pd.read_csv(source / "summary.csv")
        summary.insert(0, "region", REGION_LABELS[region])
        summary["model_label"] = summary.model.map(MODEL_LABELS)
        summaries.append(summary)

        cv = (
            pd.read_csv(source / "cv_metrics.csv")
            .groupby("model", as_index=False)
            .agg(
                folds=("fold", "nunique"),
                pooled_rmse_mean=("pooled_rmse", "mean"),
                pooled_rmse_sd=("pooled_rmse", "std"),
                cruise_equal_rmse_mean=("cruise_equal_rmse", "mean"),
                subregion_macro_rmse_mean=("lme_macro_rmse", "mean"),
                subregion_macro_rmse_sd=("lme_macro_rmse", "std"),
            )
        )
        cv.insert(0, "region", REGION_LABELS[region])
        cv["model_label"] = cv.model.map(MODEL_LABELS)
        cvs.append(cv)

        gate = read_json(source / "development_gate.json")
        gates[region] = gate
        protocols[region] = read_json(source / "protocol.json")
        selections[region] = read_json(source / "selection.json")
        selected = gate["selected_model"]

        stratified = pd.read_csv(source / "stratified_metrics.csv")
        selected_strata = (
            stratified.loc[stratified.model.eq(selected)]
            .groupby(["stratum", "group"], as_index=False)
            .agg(
                n=("n", "max"),
                cruises=("cruises", "max"),
                model_rmse=("model_rmse", "mean"),
                carter_rmse=("carter_rmse", "mean"),
                skill_vs_carter=("skill_vs_carter", "mean"),
            )
        )
        selected_strata.insert(0, "region", REGION_LABELS[region])
        strata.append(selected_strata)

        forward = pd.read_csv(source / "forward_metrics.csv")
        forward.insert(0, "region", REGION_LABELS[region])
        forward["model_label"] = forward.model.map(MODEL_LABELS)
        forwards.append(forward)

        leave = pd.read_csv(source / "leave_lme_out_metrics.csv")
        if len(leave):
            leave.insert(0, "region", REGION_LABELS[region])
            pivot = leave.pivot(index="held_lme", columns="model", values="pooled_rmse")
            skill = 1 - (pivot.hierarchical_ta_sss**2 / pivot.carter_esper_prior**2)
            leave_skill = skill.rename("skill_vs_carter").reset_index()
            leave_skill.insert(0, "region", REGION_LABELS[region])
            leave_skill["subregion"] = leave_skill.held_lme.astype(str).map(SUBREGION_LABELS)
            leaves.append(leave_skill)

        selected_frame = pd.read_parquet(source / "selected_development_predictions.parquet")
        selected_frame["region"] = REGION_LABELS[region]
        selected_frames[region] = selected_frame

    summary = pd.concat(summaries, ignore_index=True)
    cv_summary = pd.concat(cvs, ignore_index=True)
    strata_table = pd.concat(strata, ignore_index=True)
    forward_table = pd.concat(forwards, ignore_index=True)
    leave_table = pd.concat(leaves, ignore_index=True)
    write_csv(summary, tables / "table01_development_model_summary.csv")
    write_csv(cv_summary, tables / "table02_grouped_cv_summary.csv")
    write_csv(
        strata_table.loc[strata_table.stratum.eq("lme")],
        tables / "table03_subregion_skill.csv",
    )
    write_csv(
        strata_table.loc[strata_table.stratum.eq("support_distance")],
        tables / "table04_support_distance_skill.csv",
    )
    write_csv(
        strata_table.loc[strata_table.stratum.eq("salinity_band")],
        tables / "table05_salinity_band_skill.csv",
    )
    write_csv(forward_table, tables / "table06_safe_forward_metrics.csv")
    write_csv(leave_table, tables / "table07_leave_subregion_out.csv")

    ensemble_rows = []
    uncertainty_rows = []
    for region in REGIONS:
        frame = selected_frames[region]
        error = frame.prediction - frame.truth
        ensemble_rows.append(
            {
                "region": REGION_LABELS[region],
                "selected_model": gates[region]["selected_model"],
                "n": len(frame),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "bias": float(np.mean(error)),
                "r2": float(1 - np.sum(error**2) / np.sum((frame.truth - frame.truth.mean()) ** 2)),
            }
        )
        uncertainty_rows.append(
            {
                "region": REGION_LABELS[region],
                "nominal_coverage": 0.90,
                "development_coverage": selections[region]["development_coverage_90"],
                "oof_absolute_error_q90": selections[region]["conformal_absolute_error_q90"],
            }
        )
    ensemble_table = pd.DataFrame(ensemble_rows)
    uncertainty_table = pd.DataFrame(uncertainty_rows)
    write_csv(ensemble_table, tables / "table08_selected_ensemble_metrics.csv")
    write_csv(uncertainty_table, tables / "table09_uncertainty_calibration.csv")

    gate_rows = []
    for region, gate in gates.items():
        gate_rows.extend(
            {
                "region": REGION_LABELS[region],
                "check": name,
                "passed": passed,
            }
            for name, passed in gate["checks"].items()
        )
    gate_table = pd.DataFrame(gate_rows)
    write_csv(gate_table, tables / "table10_development_gates.csv")
    scope_table = pd.DataFrame(
        [
            {
                "region": REGION_LABELS[region],
                "train_rows": protocols[region]["train_rows"],
                "train_cruises": protocols[region]["train_cruises"],
                "development_rows": protocols[region]["development_rows"],
                "development_cruises": protocols[region]["development_cruises"],
                "safe_forward_train_rows": protocols[region]["safe_forward_train_rows"],
                "safe_forward_development_rows": protocols[region]["safe_forward_development_rows"],
                "decision": gates[region]["decision"],
            }
            for region in REGIONS
        ]
    )
    write_csv(scope_table, tables / "table11_data_scope.csv")

    make_model_figure(summary, figures / "fig01_development_model_comparison.png")
    make_cv_figure(cv_summary, figures / "fig02_grouped_cv.png")
    make_scatter_figure(selected_frames, figures / "fig03_observed_vs_predicted.png")
    make_skill_figure(
        strata_table.loc[strata_table.stratum.eq("lme")],
        "Subregion",
        figures / "fig04_subregion_skill.png",
        map_subregions=True,
    )
    make_skill_figure(
        strata_table.loc[strata_table.stratum.eq("support_distance")],
        "Distance to nearest training TA observation",
        figures / "fig05_support_distance_skill.png",
    )
    make_skill_figure(
        strata_table.loc[strata_table.stratum.eq("salinity_band")],
        "Observed salinity band (evaluation only)",
        figures / "fig06_salinity_band_skill.png",
    )
    make_forward_figure(forward_table, figures / "fig07_safe_forward.png")
    make_leave_figure(leave_table, figures / "fig08_leave_subregion_out.png")
    make_uncertainty_figure(uncertainty_table, figures / "fig09_uncertainty_calibration.png")
    make_gate_figure(gate_table, figures / "fig10_development_gates.png")

    captions = figure_captions(ensemble_table, protocols)
    report = build_report(
        summary,
        cv_summary,
        ensemble_table,
        uncertainty_table,
        scope_table,
        gates,
        captions,
    )
    with (args.archive / "REPORT.md").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(report)
    with (args.archive / "CAPTIONS.md").open("w", encoding="utf-8", newline="\n") as stream:
        caption_text = "# Figure captions\n\n" + "".join(
            f"## {name}\n\n{caption}\n\n" for name, caption in captions.items()
        )
        stream.write(caption_text.rstrip() + "\n")
    with (args.archive / "README.md").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            "# P1.3b SAB/MAB TA reviewer archive\n\n"
            "Complete development-only evidence for Issue #15. Read `REPORT.md`; "
            "locked-test and external-independent labels remain sealed.\n"
        )

    source_names = [
        "summary.csv",
        "cv_metrics.csv",
        "stratified_metrics.csv",
        "forward_metrics.csv",
        "leave_lme_out_metrics.csv",
        "selected_development_predictions.parquet",
        "development_gate.json",
        "selection.json",
        "protocol.json",
    ]
    source_hashes = {
        f"{region}/{name}": digest(args.source / region / name)
        for region in REGIONS
        for name in source_names
    }
    tracked = [
        args.archive / "README.md",
        args.archive / "REPORT.md",
        args.archive / "CAPTIONS.md",
        *sorted(figures.glob("*.png")),
        *sorted(tables.glob("*.csv")),
    ]
    hashes = {
        str(path.relative_to(args.archive)).replace("\\", "/"): digest(path) for path in tracked
    }
    figure_sources = {
        "fig01_development_model_comparison.png": ["table01_development_model_summary.csv"],
        "fig02_grouped_cv.png": ["table02_grouped_cv_summary.csv"],
        "fig03_observed_vs_predicted.png": [
            "local:sab/selected_development_predictions.parquet",
            "local:mab/selected_development_predictions.parquet",
        ],
        "fig04_subregion_skill.png": ["table03_subregion_skill.csv"],
        "fig05_support_distance_skill.png": ["table04_support_distance_skill.csv"],
        "fig06_salinity_band_skill.png": ["table05_salinity_band_skill.csv"],
        "fig07_safe_forward.png": ["table06_safe_forward_metrics.csv"],
        "fig08_leave_subregion_out.png": ["table07_leave_subregion_out.csv"],
        "fig09_uncertainty_calibration.png": ["table09_uncertainty_calibration.csv"],
        "fig10_development_gates.png": ["table10_development_gates.csv"],
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_experiment_directory": str(args.source),
        "training_git_commit": protocols["sab"]["run_git_commit"],
        "preregistration_git_commit": protocols["sab"]["preregistration_commit"],
        "archive_builder": "scripts/build_p1_ta_bights_reviewer_archive.py",
        "archive_builder_sha256": digest(Path(__file__)),
        "analysis_script_sha256": digest(ROOT / "scripts/run_p1_ta_viability.py"),
        "source_artifacts_sha256": source_hashes,
        "data_manifest_sha256": protocols["sab"]["manifest_validation"]["manifest_sha256"],
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "Separate SAB and broad-geographic-MAB train/development, grouped CV and safe-forward evidence",
        "decision": {region: gates[region]["decision"] for region in REGIONS},
        "figure_source_data": figure_sources,
        "figure_captions": captions,
        "files_sha256": hashes,
    }
    with (args.archive / "archive_manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        stream.write(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "archive": str(args.archive),
                "figures": len(captions),
                "tables": len(list(tables.glob("*.csv"))),
                "decision": manifest["decision"],
            },
            indent=2,
        )
    )
    return 0


def make_model_figure(table: pd.DataFrame, path: Path) -> None:
    metrics = [
        ("pooled_rmse_mean", "Pooled RMSE"),
        ("cruise_equal_rmse_mean", "Cruise-equal RMSE"),
        ("lme_macro_rmse_mean", "Subregion-macro RMSE"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for row, region in enumerate(["SAB", "MAB"]):
        part = table.loc[table.region.eq(region)].sort_values("lme_macro_rmse_mean")
        for ax, (column, title) in zip(axes[row], metrics, strict=True):
            ax.barh(part.model_label, part[column], color=COLORS[region], alpha=0.82)
            ax.invert_yaxis()
            ax.set_title(f"{region}: {title}")
            ax.set_xlabel("TA RMSE (µmol kg⁻¹)")
    fig.tight_layout()
    save_figure(fig, path)


def make_cv_figure(table: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, region in zip(axes, ["SAB", "MAB"], strict=True):
        part = table.loc[table.region.eq(region)].sort_values("subregion_macro_rmse_mean")
        ax.barh(
            part.model_label,
            part.subregion_macro_rmse_mean,
            xerr=part.subregion_macro_rmse_sd,
            color=COLORS[region],
            alpha=0.82,
        )
        ax.invert_yaxis()
        ax.set_title(f"{region}: five-fold cruise-grouped CV")
        ax.set_xlabel("Subregion-macro RMSE (µmol kg⁻¹)")
    fig.tight_layout()
    save_figure(fig, path)


def make_scatter_figure(frames: dict[str, pd.DataFrame], path: Path) -> None:
    values = np.concatenate(
        [frame[["truth", "carter", "prediction"]].to_numpy().ravel() for frame in frames.values()]
    )
    values = values[np.isfinite(values)]
    lower, upper = np.quantile(values, [0.002, 0.998])
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), sharex=True, sharey=True)
    for row, region in enumerate(REGIONS):
        frame = frames[region]
        for col, (field, title) in enumerate(
            [("carter", "Carter/ESPER"), ("prediction", "Selected model")]
        ):
            ax = axes[row, col]
            valid = np.isfinite(frame.truth) & np.isfinite(frame[field])
            ax.hexbin(
                frame.loc[valid, "truth"],
                frame.loc[valid, field],
                gridsize=45,
                bins="log",
                mincnt=1,
                cmap="viridis",
            )
            rmse = np.sqrt(np.mean((frame.loc[valid, field] - frame.loc[valid, "truth"]) ** 2))
            ax.plot([lower, upper], [lower, upper], color="black", lw=1)
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
            ax.set_title(f"{REGION_LABELS[region]} {title}\nRMSE={rmse:.1f}")
            ax.set_xlabel("Observed TA (µmol kg⁻¹)")
            ax.set_ylabel("Predicted TA (µmol kg⁻¹)")
    fig.tight_layout()
    save_figure(fig, path)


def make_skill_figure(
    table: pd.DataFrame,
    xlabel: str,
    path: Path,
    *,
    map_subregions: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, region in zip(axes, ["SAB", "MAB"], strict=True):
        part = table.loc[table.region.eq(region)].copy()
        if xlabel.startswith("Distance"):
            support_order = {
                "[0.0, 25.0)": 0,
                "[25.0, 50.0)": 1,
                "[50.0, 100.0)": 2,
                "[100.0, 200.0)": 3,
                "[200.0, 500.0)": 4,
                "[500.0, inf)": 5,
            }
            part["plot_order"] = part.group.astype(str).map(support_order)
            part = part.sort_values("plot_order")
        labels = part.group.astype(str)
        if map_subregions:
            labels = labels.map(SUBREGION_LABELS)
        ax.bar(labels, part.skill_vs_carter, color=COLORS[region], alpha=0.85)
        ax.axhline(0, color="black", lw=1)
        ax.set_title(region)
        ax.set_xlabel(xlabel)
        ax.tick_params(axis="x", rotation=30)
        for x, (_, item) in enumerate(part.iterrows()):
            ax.text(
                x, item.skill_vs_carter, f"n={int(item.n)}", ha="center", va="bottom", fontsize=8
            )
    axes[0].set_ylabel("MSE skill versus Carter/ESPER")
    fig.tight_layout()
    save_figure(fig, path)


def make_forward_figure(table: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, region in zip(axes, ["SAB", "MAB"], strict=True):
        part = table.loc[table.region.eq(region) & table.pooled_rmse.notna()]
        if part.empty:
            ax.text(0.5, 0.5, "No safe-forward development cruises", ha="center", va="center")
            ax.set_axis_off()
            ax.set_title(region)
            continue
        ax.barh(part.model_label, part.pooled_rmse, color=COLORS[region], alpha=0.85)
        ax.invert_yaxis()
        ax.set_xlabel("Forward RMSE (µmol kg⁻¹)")
        ax.set_title(f"{region}: n={int(part.n.max())}")
    fig.tight_layout()
    save_figure(fig, path)


def make_leave_figure(table: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, region in zip(axes, ["SAB", "MAB"], strict=True):
        part = table.loc[table.region.eq(region)]
        ax.bar(part.subregion, part.skill_vs_carter, color=COLORS[region], alpha=0.85)
        ax.axhline(0, color="black", lw=1)
        ax.set_title(region)
        ax.set_xlabel("Entire held-out subregion")
    axes[0].set_ylabel("Hierarchical TA-SSS skill versus Carter")
    fig.tight_layout()
    save_figure(fig, path)


def make_uncertainty_figure(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(table.region, table.development_coverage, color=[COLORS[x] for x in table.region])
    ax.axhline(0.90, color="black", lw=1, label="nominal 90%")
    ax.axhspan(0.85, 0.95, color="gray", alpha=0.15, label="frozen acceptance band")
    ax.set_ylim(0.8, 1.01)
    ax.set_ylabel("Development interval coverage")
    for bar, q90 in zip(bars, table.oof_absolute_error_q90, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"q90={q90:.1f}",
            ha="center",
            va="bottom",
        )
    ax.legend()
    fig.tight_layout()
    save_figure(fig, path)


def make_gate_figure(table: pd.DataFrame, path: Path) -> None:
    checks = list(dict.fromkeys(table.check))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, region in zip(axes, ["SAB", "MAB"], strict=True):
        lookup = table.loc[table.region.eq(region)].set_index("check").passed.to_dict()
        values = [bool(lookup.get(check, False)) for check in checks]
        y = np.arange(len(checks))
        ax.scatter(
            [0] * len(checks), y, s=160, color=["#009E73" if v else "#D55E00" for v in values]
        )
        for yi, value in zip(y, values, strict=True):
            ax.text(0.08, yi, "PASS" if value else "FAIL", va="center", fontweight="bold")
        ax.set_xlim(-0.2, 0.8)
        ax.set_xticks([])
        ax.set_yticks(y, [check.replace("_", " ") for check in checks])
        ax.invert_yaxis()
        ax.set_title(region)
    fig.tight_layout()
    save_figure(fig, path)


def figure_captions(ensemble: pd.DataFrame, protocols: dict[str, dict]) -> dict[str, str]:
    sab = ensemble.set_index("region").loc["SAB"]
    mab = ensemble.set_index("region").loc["MAB"]
    return {
        "fig01_development_model_comparison.png": "Development TA RMSE for independently trained SAB and MAB candidates under pooled, cruise-equal, and frozen subregion-macro aggregation. The in-situ-SSS Carter oracle is diagnostic only and excluded from selection; lower is better.",
        "fig02_grouped_cv.png": "Five-fold cruise-grouped training cross-validation subregion-macro RMSE. Error bars show one standard deviation across folds. SAB estimates are unstable because one frozen fold contains only one training record, while MAB favors the simpler hierarchical linear relation over the nonlinear residual model in CV.",
        "fig03_observed_vs_predicted.png": f"Development observation-prediction density for Carter/ESPER and each region's selected model. The selected SAB model has RMSE {sab.rmse:.1f} µmol kg⁻¹ and fails to improve pooled Carter error; the selected MAB ensemble reaches {mab.rmse:.1f} µmol kg⁻¹.",
        "fig04_subregion_skill.png": "Development MSE skill relative to Carter/ESPER in the preregistered south, central, and north latitude subregions. Labels give record counts. SAB central skill is negative and dominates the regional sample, whereas all three populated MAB subregions show positive skill.",
        "fig05_support_distance_skill.png": "Development MSE skill relative to Carter/ESPER by haversine distance to the nearest training TA observation. Every populated bin is retained regardless of sample size; SAB fails in its dominant 0-25 km bin, while all populated MAB bins are positive.",
        "fig06_salinity_band_skill.png": "Development skill relative to Carter/ESPER stratified by collocated observed salinity, which is used only for evaluation. Sparse low-salinity SAB records have very large errors, and the main 33-36 band also degrades; MAB remains positive but has weak improvement in the 33-36 band.",
        "fig07_safe_forward.png": f"Safe forward-time evaluation uses only primary-train groups assigned to forward train and primary-development groups assigned to forward development. SAB has {protocols['sab']['safe_forward_development_rows']} eligible development records, so no forward claim is possible; MAB has {protocols['mab']['safe_forward_development_rows']} records from one cruise.",
        "fig08_leave_subregion_out.png": "Leave-one-subregion-out transfer skill for the hierarchical linear TA-SSS relation relative to Carter/ESPER. Only subregions meeting the frozen minimum of 30 rows and three cruises are evaluated; sparse SAB edge subregions cannot support a complete transfer test.",
        "fig09_uncertainty_calibration.png": "Development coverage of symmetric conformal intervals calibrated from five-fold training OOF absolute residuals. SAB coverage falls inside the frozen 85-95% band, while MAB coverage is 98.5%, showing that its 90% interval is materially overconservative.",
        "fig10_development_gates.png": "Frozen development gates evaluated separately by bight. SAB fails multiple prediction and evidence gates and is classified fail. MAB passes every skill gate but fails uncertainty calibration, so it remains diagnostic-only and the locked test stays sealed.",
    }


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame[columns].copy()
    for column in view.select_dtypes(include=["number"]).columns:
        view[column] = view[column].map(lambda value: f"{value:.3f}" if pd.notna(value) else "NA")
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()]
    return "\n".join([header, divider, *rows])


def build_report(
    summary: pd.DataFrame,
    cv: pd.DataFrame,
    ensemble: pd.DataFrame,
    uncertainty: pd.DataFrame,
    scope: pd.DataFrame,
    gates: dict[str, dict],
    captions: dict[str, str],
) -> str:
    selected_names = {region: gates[region]["selected_model"] for region in REGIONS}
    selected_rows = pd.concat(
        [
            summary.loc[
                summary.region.eq(REGION_LABELS[region]) & summary.model.eq(selected_names[region])
            ]
            for region in REGIONS
        ]
    )
    report = f"""# Reviewer archive: P1.3b SAB/MAB TA viability

## Scientific question and permitted claim

This experiment asks whether surface TA reconstruction passes the frozen P1.3 development gate independently in the South Atlantic Bight (SAB) and broad geographic Mid-Atlantic Bight (MAB). SAB is the LME-6 shelf segment from Cape Canaveral to Cape Hatteras; MAB is the LME-7 segment from Cape Hatteras to Cape Cod and includes Southern New England. The permitted evidence is train/development, cruise-grouped CV, safe-forward, support-distance, and leave-subregion-out analysis only. No locked-test, external-independent, continuous-product, or global claim is allowed.

The decisions are **SAB: `{gates["sab"]["decision"]}`** and **MAB: `{gates["mab"]["decision"]}`**. SAB does not show deployable skill. MAB has strong development skill but its nominal 90% interval covers {uncertainty.set_index("region").loc["MAB", "development_coverage"]:.3f}, outside the frozen 0.85-0.95 acceptance band, so it cannot advance to the locked gate in this experiment.

## Data, splits, and leakage controls

{markdown_table(scope, ["region", "train_rows", "train_cruises", "development_rows", "development_cruises", "safe_forward_train_rows", "safe_forward_development_rows", "decision"])}

The v2.2 manifest, TA QC, primary-record rule, cruise groups, deterministic primary split, five frozen CV folds, background SSS input, and Carter/ESPER implementation are inherited unchanged from Issue #9. Each bight is filtered before fitting and trained independently. The offshore boundary comes from the frozen LME shelf polygons because the current cache has no bathymetry column. Estuary and shelf-depth stratification therefore remains a documented limitation. Locked and external labels were never materialized.

## Candidate models and training

Candidates are raw Carter/ESPER, train-only seasonal/subregion Carter correction, bight-wide robust TA-SSS, fixed subregion TA-SSS, hierarchical subregion/regime TA-SSS, and a three-seed residual MLP. The nonlinear candidate ran only where the frozen stopping rule triggered. SAB stopped after the hierarchical linear stage; MAB ran 4,000 steps for seeds 100-102. Partial-pooling alpha was selected from {{1, 10, 100, 1000}} using only five-fold cruise-grouped training CV.

## Main development results

{markdown_table(selected_rows, ["region", "model_label", "pooled_rmse_mean", "pooled_mae_mean", "pooled_bias_mean", "pooled_r2_mean", "cruise_equal_rmse_mean", "lme_macro_rmse_mean"])}

The selected SAB hierarchical linear model has pooled RMSE 99.255 µmol kg⁻¹ versus 98.468 for Carter/ESPER and R² -0.063. Its central subregion, containing 141 of 165 development records, has negative skill. It also has no safe-forward development cruise under the frozen intersection. SAB is therefore classified `fail` rather than diagnostic-only.

The selected MAB three-seed ensemble has pooled RMSE {ensemble.set_index("region").loc["MAB", "rmse"]:.3f} µmol kg⁻¹, MAE {ensemble.set_index("region").loc["MAB", "mae"]:.3f}, bias {ensemble.set_index("region").loc["MAB", "bias"]:.3f}, and R² {ensemble.set_index("region").loc["MAB", "r2"]:.3f}. The three-seed mean cruise-equal RMSE is 38.062 versus 110.474 for Carter and 71.939 for fixed subregion TA-SSS. Five-fold CV is less decisive: hierarchical linear reaches {cv.loc[(cv.region.eq("MAB")) & (cv.model.eq("hierarchical_ta_sss")), "subregion_macro_rmse_mean"].iloc[0]:.3f}, while the residual MLP reaches {cv.loc[(cv.region.eq("MAB")) & (cv.model.eq("hierarchical_residual")), "subregion_macro_rmse_mean"].iloc[0]:.3f} µmol kg⁻¹. This model-ranking disagreement and the single-cruise forward set limit the claim.

![Development model comparison](figures/fig01_development_model_comparison.png)

*{captions["fig01_development_model_comparison.png"]}*

![Grouped cross-validation](figures/fig02_grouped_cv.png)

*{captions["fig02_grouped_cv.png"]}*

![Observed versus predicted TA](figures/fig03_observed_vs_predicted.png)

*{captions["fig03_observed_vs_predicted.png"]}*

![Subregion skill](figures/fig04_subregion_skill.png)

*{captions["fig04_subregion_skill.png"]}*

![Support-distance skill](figures/fig05_support_distance_skill.png)

*{captions["fig05_support_distance_skill.png"]}*

![Salinity-band skill](figures/fig06_salinity_band_skill.png)

*{captions["fig06_salinity_band_skill.png"]}*

![Safe forward evidence](figures/fig07_safe_forward.png)

*{captions["fig07_safe_forward.png"]}*

![Leave-subregion-out transfer](figures/fig08_leave_subregion_out.png)

*{captions["fig08_leave_subregion_out.png"]}*

![Uncertainty calibration](figures/fig09_uncertainty_calibration.png)

*{captions["fig09_uncertainty_calibration.png"]}*

![Frozen gates](figures/fig10_development_gates.png)

*{captions["fig10_development_gates.png"]}*

## Decision and limitations

SAB is `fail`: the selected model is worse than Carter in pooled and cruise-equal error, its data-rich central subregion degrades, the dominant support-distance bin is negative, and no safe-forward development group exists. This experiment provides no basis for a SAB TA product.

MAB is `diagnostic_only`: its selected ensemble strongly beats both frozen baselines across all three subregions and every populated support-distance bin, but uncertainty is overconservative and the nonlinear model is not the CV winner. The correct next experiment is a preregistered MAB-only calibration study with more development cruises and explicit estuary/depth masks; the present result must not be promoted by simply retuning the interval on development.

The LME/end-point masks are reproducible but do not yet separate estuaries, inner shelf, middle shelf, and outer shelf. SAB has only four development cruises and severely imbalanced subregions. MAB has seven development cruises; its safe-forward evidence contains 31 records from one cruise. Sparse bins are retained and cannot be discarded after observing their errors.

## Figure and table index

Figures 1-10 are embedded above with complete captions. Source CSVs `table01`-`table11` are stored in `tables/`; per-observation predictions and checkpoints remain in the ignored source experiment directory and are referenced by SHA256 in `archive_manifest.json`.
"""
    return report


if __name__ == "__main__":
    raise SystemExit(main())
