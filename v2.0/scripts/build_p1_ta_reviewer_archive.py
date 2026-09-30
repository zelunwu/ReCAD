"""Build the reviewer-ready P1.3 TA archive from the formal run outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recad.evaluate.p1_framework import evaluate_predictions

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_ta_viability_v2.2"
MODEL_LABELS = {
    "carter_esper_prior": "Carter/ESPER",
    "carter_region_season_corrected": "Carter + regional correction",
    "global_ta_sss": "Global TA-SSS",
    "lme_ta_sss": "LME TA-SSS",
    "hierarchical_ta_sss": "Hierarchical TA-SSS",
    "hierarchical_residual": "Hierarchical residual MLP",
    "carter_esper_oracle_insitu_sss": "Carter oracle (in-situ SSS)",
}
COLORS = {
    "Carter/ESPER": "#777777",
    "LME TA-SSS": "#2A9D8F",
    "Hierarchical TA-SSS": "#457B9D",
    "Hierarchical residual MLP": "#E76F51",
    "Carter oracle (in-situ SSS)": "#7B2CBF",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 << 20):
            value.update(block)
    return value.hexdigest()


def normalize_archive_text(archive: Path) -> None:
    for path in archive.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".csv", ".md"}:
            text = path.read_text(encoding="utf-8")
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(text)


def table_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    rows = [
        "| " + " | ".join(str(value) for value in record) + " |"
        for record in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *rows])


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compact_metrics(frame: pd.DataFrame, prediction: str) -> dict[str, float | int | str]:
    scored = frame[
        ["truth", prediction, "group_key", "lme_id", "regime_id", "nearest_carbon_km"]
    ].rename(columns={prediction: "prediction"})
    details = evaluate_predictions(scored)
    lookup = {row.aggregation: row for row in details.itertuples() if row.group == "all"}
    worst = details.loc[details.aggregation.eq("worst_lme")].iloc[0]
    return {
        "n": int(lookup["pooled"].n),
        "pooled_rmse": float(lookup["pooled"].rmse),
        "pooled_mae": float(lookup["pooled"].mae),
        "pooled_bias": float(lookup["pooled"].bias),
        "pooled_r2": float(lookup["pooled"].r2),
        "cruise_equal_rmse": float(lookup["cruise_equal"].rmse),
        "lme_macro_rmse": float(lookup["lme_macro"].rmse),
        "regime_macro_rmse": float(lookup["regime_macro"].rmse),
        "worst_lme_rmse": float(worst.rmse),
        "worst_lme": str(worst.group),
    }


def stratified_ensemble(frame: pd.DataFrame, grouping: pd.Series, kind: str) -> pd.DataFrame:
    rows = []
    for group in sorted(pd.Series(grouping).dropna().unique(), key=str):
        mask = np.asarray(grouping == group)
        part = frame.loc[mask]
        model_rmse = float(np.sqrt(np.mean((part.prediction - part.truth) ** 2)))
        carter_rmse = float(np.sqrt(np.mean((part.carter - part.truth) ** 2)))
        rows.append(
            {
                "stratum": kind,
                "group": str(group),
                "n": len(part),
                "cruises": int(part.group_key.nunique()),
                "model_rmse": model_rmse,
                "carter_rmse": carter_rmse,
                "skill_vs_carter": 1 - model_rmse**2 / carter_rmse**2,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / f"outputs/experiments/{EXPERIMENT_ID}"
    )
    parser.add_argument(
        "--archive", type=Path, default=ROOT / f"docs/experiment_archive/{EXPERIMENT_ID}"
    )
    args = parser.parse_args()
    figures = args.archive / "figures"
    tables = args.archive / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.source / "summary.csv")
    cv = pd.read_csv(args.source / "cv_metrics.csv")
    forward = pd.read_csv(args.source / "forward_metrics.csv")
    leave_lme = pd.read_csv(args.source / "leave_lme_out_metrics.csv")
    predictions = pd.read_parquet(args.source / "candidate_predictions.parquet")
    selected_predictions = pd.read_parquet(args.source / "selected_development_predictions.parquet")
    gate = json.loads((args.source / "development_gate.json").read_text(encoding="utf-8"))
    selection = json.loads((args.source / "selection.json").read_text(encoding="utf-8"))
    protocol = json.loads((args.source / "protocol.json").read_text(encoding="utf-8"))
    selected = selection["selected_model"]

    selected_predictions = selected_predictions.copy()
    selected_predictions["prediction"] = selected_predictions.prediction.astype(float)
    ensemble_metric = compact_metrics(selected_predictions, "prediction")
    candidate_metrics = summary.copy()
    candidate_metrics["model"] = candidate_metrics.model.map(MODEL_LABELS)
    candidate_metrics.to_csv(tables / "table01_development_model_summary.csv", index=False)

    cv_summary = (
        cv.groupby("model")
        .agg(
            folds=("fold", "nunique"),
            pooled_rmse_mean=("pooled_rmse", "mean"),
            pooled_rmse_sd=("pooled_rmse", "std"),
            cruise_equal_rmse_mean=("cruise_equal_rmse", "mean"),
            lme_macro_rmse_mean=("lme_macro_rmse", "mean"),
            lme_macro_rmse_sd=("lme_macro_rmse", "std"),
        )
        .reset_index()
    )
    cv_summary["model"] = cv_summary.model.map(MODEL_LABELS)
    cv_summary.to_csv(tables / "table02_five_fold_cv_summary.csv", index=False)

    ensemble_table = pd.DataFrame([{"model": MODEL_LABELS[selected], **ensemble_metric}])
    ensemble_table.to_csv(tables / "table03_selected_ensemble_metrics.csv", index=False)
    lme = stratified_ensemble(selected_predictions, selected_predictions.lme_id, "lme")
    lme.to_csv(tables / "table04_lme_skill.csv", index=False)
    salinity_groups = pd.cut(
        selected_predictions.salinity,
        [-np.inf, 20, 30, 33, 36, np.inf],
        right=False,
    )
    salinity = stratified_ensemble(selected_predictions, salinity_groups, "salinity_band")
    salinity.to_csv(tables / "table05_salinity_band_skill.csv", index=False)
    support_groups = pd.cut(
        selected_predictions.nearest_carbon_km,
        [0, 25, 50, 100, 200, 500, np.inf],
        right=False,
        include_lowest=True,
    )
    support = stratified_ensemble(selected_predictions, support_groups, "support_distance")
    support_order = {
        label: index
        for index, label in enumerate(
            [
                "[0.0, 25.0)",
                "[25.0, 50.0)",
                "[50.0, 100.0)",
                "[100.0, 200.0)",
                "[200.0, 500.0)",
                "[500.0, inf)",
            ]
        )
    }
    support = support.sort_values(
        "group", key=lambda values: values.map(support_order)
    ).reset_index(drop=True)
    support.to_csv(tables / "table06_support_distance_skill.csv", index=False)
    forward_table = forward.copy()
    forward_table["model"] = forward_table.model.map(MODEL_LABELS)
    forward_table.to_csv(tables / "table07_safe_forward_metrics.csv", index=False)
    leave_table = leave_lme.copy()
    leave_table["model"] = leave_table.model.map(MODEL_LABELS)
    leave_table.to_csv(tables / "table08_leave_lme_out_metrics.csv", index=False)
    gate_table = pd.DataFrame(
        [{"gate": name, "passed": bool(passed)} for name, passed in gate["checks"].items()]
    )
    gate_table.to_csv(tables / "table09_development_gates.csv", index=False)
    uncertainty = pd.DataFrame(
        [
            {
                "model": MODEL_LABELS[selected],
                "calibration_source": selection["conformal_source"],
                "absolute_error_q90_umol_kg": selection["conformal_absolute_error_q90"],
                "development_coverage": selection["development_coverage_90"],
                "nominal_coverage": 0.90,
                "locked_coverage_tested": False,
            }
        ]
    )
    uncertainty.to_csv(tables / "table10_uncertainty_calibration.csv", index=False)
    data_table = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": protocol["train_rows"],
                "cruises": protocol["train_cruises"],
            },
            {
                "split": "development",
                "rows": protocol["development_rows"],
                "cruises": protocol["development_cruises"],
            },
            {
                "split": "safe_forward_train",
                "rows": protocol["safe_forward_train_rows"],
                "cruises": "not independently counted",
            },
            {
                "split": "safe_forward_development",
                "rows": protocol["safe_forward_development_rows"],
                "cruises": "not independently counted",
            },
        ]
    )
    data_table.to_csv(tables / "table11_data_scope.csv", index=False)

    make_model_figure(candidate_metrics, figures / "fig01_development_model_comparison.png")
    make_cv_figure(cv_summary, figures / "fig02_five_fold_cv.png")
    make_lme_figure(lme, figures / "fig03_lme_skill.png")
    make_salinity_figure(salinity, figures / "fig04_salinity_band_skill.png")
    make_support_figure(support, figures / "fig05_support_distance_skill.png")
    make_forward_figure(forward_table, figures / "fig06_safe_forward.png")
    make_leave_lme_figure(leave_lme, figures / "fig07_leave_lme_out.png")
    make_scatter_figure(
        predictions,
        selected_predictions,
        selected,
        figures / "fig08_observed_vs_predicted.png",
    )
    make_calibration_figure(
        selected_predictions,
        selection["conformal_absolute_error_q90"],
        figures / "fig09_uncertainty_calibration.png",
    )
    make_gate_figure(gate_table, figures / "fig10_development_gates.png")

    figure_sources = {
        "fig01_development_model_comparison.png": ["table01_development_model_summary.csv"],
        "fig02_five_fold_cv.png": ["table02_five_fold_cv_summary.csv"],
        "fig03_lme_skill.png": ["table04_lme_skill.csv"],
        "fig04_salinity_band_skill.png": ["table05_salinity_band_skill.csv"],
        "fig05_support_distance_skill.png": ["table06_support_distance_skill.csv"],
        "fig06_safe_forward.png": ["table07_safe_forward_metrics.csv"],
        "fig07_leave_lme_out.png": ["table08_leave_lme_out_metrics.csv"],
        "fig08_observed_vs_predicted.png": ["local:candidate_predictions.parquet"],
        "fig09_uncertainty_calibration.png": [
            "table10_uncertainty_calibration.csv",
            "local:selected_development_predictions.parquet",
        ],
        "fig10_development_gates.png": ["table09_development_gates.csv"],
    }
    figure_captions = captions(selection, ensemble_metric, support)
    report = build_report(
        candidate_metrics,
        cv_summary,
        ensemble_table,
        lme,
        salinity,
        support,
        forward_table,
        leave_lme,
        gate_table,
        gate,
        selection,
        protocol,
        figure_captions,
    )
    with (args.archive / "REPORT.md").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(report)
    captions_text = (
        "# Figure captions\n\n"
        + "\n\n".join(f"## {name}\n\n{figure_captions[name]}" for name in figure_sources)
        + "\n"
    )
    with (args.archive / "CAPTIONS.md").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(captions_text)
    with (args.archive / "README.md").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            "# Archive index\n\nStart with [REPORT.md](REPORT.md). Every figure is embedded there "
            "with its caption immediately below it; [CAPTIONS.md](CAPTIONS.md) is the standalone "
            "caption index. Source tables and SHA256 provenance are recorded in "
            "`archive_manifest.json`.\n"
        )

    normalize_archive_text(args.archive)
    files = {
        str(path.relative_to(args.archive)).replace("\\", "/"): digest(path)
        for path in args.archive.rglob("*")
        if path.is_file() and path.name != "archive_manifest.json"
    }
    source_names = [
        "candidate_predictions.parquet",
        "selected_development_predictions.parquet",
        "summary.csv",
        "cv_metrics.csv",
        "forward_metrics.csv",
        "leave_lme_out_metrics.csv",
        "development_gate.json",
        "selection.json",
        "protocol.json",
    ]
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_experiment_directory": str(args.source),
        "training_git_commit": protocol["run_git_commit"],
        "preregistration_git_commit": protocol["preregistration_commit"],
        "archive_builder": "scripts/build_p1_ta_reviewer_archive.py",
        "archive_builder_sha256": digest(Path(__file__)),
        "analysis_script_sha256": digest(ROOT / "scripts/run_p1_ta_viability.py"),
        "source_artifacts_sha256": {name: digest(args.source / name) for name in source_names},
        "data_manifest_sha256": protocol["manifest_validation"]["manifest_sha256"],
        "data_hash_mode": protocol["manifest_validation"]["hash_mode"],
        "esper_mat_sha256": protocol["esper_mat_sha256"],
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "North-American coastal train/development, grouped CV, safe-forward and leave-LME-out only",
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
                "decision": gate["decision"],
            },
            indent=2,
        )
    )
    return 0


def make_model_figure(summary: pd.DataFrame, path: Path) -> None:
    order = summary.sort_values("lme_macro_rmse_mean").model.tolist()
    metrics = [
        ("pooled_rmse_mean", "Pooled RMSE"),
        ("cruise_equal_rmse_mean", "Cruise-equal RMSE"),
        ("lme_macro_rmse_mean", "LME-macro RMSE"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, (column, title) in zip(axes, metrics, strict=True):
        part = summary.set_index("model").loc[order]
        colors = [COLORS.get(name, "#6C8EBF") for name in order]
        ax.barh(order, part[column], color=colors)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("µmol kg⁻¹ (lower is better)")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "TA development performance using deployable background SSS; oracle shown separately"
    )
    fig.tight_layout()
    save_figure(fig, path)


def make_cv_figure(cv: pd.DataFrame, path: Path) -> None:
    part = cv.sort_values("lme_macro_rmse_mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        part.model,
        part.lme_macro_rmse_mean,
        xerr=part.lme_macro_rmse_sd.fillna(0),
        color=[COLORS.get(name, "#6C8EBF") for name in part.model],
        capsize=3,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Five-fold LME-macro RMSE (µmol kg⁻¹)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, path)


def make_lme_figure(lme: pd.DataFrame, path: Path) -> None:
    part = lme.sort_values("skill_vs_carter")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(part.group, part.skill_vs_carter, color="#E76F51")
    for bar, row in zip(bars, part.itertuples(), strict=True):
        ax.text(
            bar.get_width() + (0.02 if bar.get_width() >= 0 else -0.02),
            bar.get_y() + bar.get_height() / 2,
            f"n={row.n}, c={row.cruises}",
            va="center",
            ha="left" if bar.get_width() >= 0 else "right",
            fontsize=8,
        )
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("MSE skill versus Carter/ESPER")
    ax.set_ylabel("LME id")
    fig.tight_layout()
    save_figure(fig, path)


def make_salinity_figure(salinity: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(salinity.group, salinity.skill_vs_carter, color="#2A9D8F")
    for bar, n in zip(bars, salinity.n, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={n}", ha="center")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("MSE skill versus Carter/ESPER")
    ax.set_xlabel("Observed salinity band (oracle stratification only)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save_figure(fig, path)


def make_support_figure(support: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = np.where(support.skill_vs_carter >= 0, "#2A9D8F", "#D1495B")
    bars = ax.bar(support.group, support.skill_vs_carter, color=colors)
    for bar, n in zip(bars, support.n, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={n}", ha="center")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("MSE skill versus Carter/ESPER")
    ax.set_xlabel("Distance to nearest training TA observation (km)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save_figure(fig, path)


def make_forward_figure(forward: pd.DataFrame, path: Path) -> None:
    part = forward.sort_values("lme_macro_rmse")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(part.model, part.pooled_rmse, color=[COLORS.get(x, "#6C8EBF") for x in part.model])
    axes[1].barh(
        part.model,
        part.lme_macro_rmse,
        color=[COLORS.get(x, "#6C8EBF") for x in part.model],
    )
    for ax, title in zip(axes, ["Pooled RMSE", "LME-macro RMSE"], strict=True):
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("µmol kg⁻¹")
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, path)


def make_leave_lme_figure(leave: pd.DataFrame, path: Path) -> None:
    pivot = leave.pivot(index="held_lme", columns="model", values="pooled_rmse")
    skill_value = 1 - pivot.hierarchical_ta_sss**2 / pivot.carter_esper_prior**2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        skill_value.index.astype(str),
        skill_value,
        color=np.where(skill_value >= 0, "#2A9D8F", "#D1495B"),
    )
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Entire held-out LME")
    ax.set_ylabel("Hierarchical TA-SSS skill versus Carter")
    fig.tight_layout()
    save_figure(fig, path)


def make_scatter_figure(
    predictions: pd.DataFrame,
    selected: pd.DataFrame,
    selected_name: str,
    path: Path,
) -> None:
    oracle = predictions.loc[predictions.model.eq("carter_esper_oracle_insitu_sss")]
    panels = [
        (selected.truth, selected.carter, "Carter/ESPER (background SSS)"),
        (selected.truth, selected.prediction, MODEL_LABELS[selected_name]),
        (oracle.truth, oracle.prediction, "Carter oracle (in-situ SSS)"),
    ]
    combined = np.concatenate(
        [np.asarray(values, dtype=float) for panel in panels for values in panel[:2]]
    )
    combined = combined[np.isfinite(combined)]
    lower, upper = np.quantile(combined, [0.001, 0.999])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for ax, (truth, prediction, title) in zip(axes, panels, strict=True):
        truth_array = np.asarray(truth, dtype=float)
        prediction_array = np.asarray(prediction, dtype=float)
        valid = np.isfinite(truth_array) & np.isfinite(prediction_array)
        hb = ax.hexbin(
            truth_array[valid],
            prediction_array[valid],
            gridsize=65,
            bins="log",
            mincnt=1,
            cmap="viridis",
        )
        ax.plot([lower, upper], [lower, upper], color="white", lw=1)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        rmse = np.sqrt(np.mean((prediction_array[valid] - truth_array[valid]) ** 2))
        ax.set_title(f"{title}\nRMSE={rmse:.1f}")
        ax.set_xlabel("Observed TA (µmol kg⁻¹)")
        fig.colorbar(hb, ax=ax, label="log10 count")
    axes[0].set_ylabel("Predicted TA (µmol kg⁻¹)")
    fig.tight_layout()
    save_figure(fig, path)


def make_calibration_figure(frame: pd.DataFrame, q90: float, path: Path) -> None:
    errors = {
        "Carter/ESPER": np.abs(frame.carter - frame.truth),
        "Hierarchical residual MLP": np.abs(frame.prediction - frame.truth),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values in errors.items():
        ordered = np.sort(np.asarray(values))
        ax.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), label=label)
    ax.axvline(q90, color="#E76F51", ls=":", label=f"training OOF q90={q90:.1f}")
    ax.axhline(0.9, color="black", ls="--", lw=1)
    ax.set_xlabel("Absolute TA error (µmol kg⁻¹)")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_xlim(0, np.quantile(errors["Carter/ESPER"], 0.99))
    ax.legend()
    fig.tight_layout()
    save_figure(fig, path)


def make_gate_figure(gates: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    y = np.arange(len(gates))
    ax.scatter(
        np.zeros(len(gates)),
        y,
        s=180,
        color=np.where(gates.passed, "#2A9D8F", "#D1495B"),
    )
    for row, y_value in zip(gates.itertuples(), y, strict=True):
        ax.text(0.08, y_value, "PASS" if row.passed else "FAIL", va="center", weight="bold")
    ax.set_yticks(y, gates.gate)
    ax.set_xlim(-0.1, 0.6)
    ax.set_xticks([])
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    save_figure(fig, path)


def captions(selection: dict, ensemble: dict, support: pd.DataFrame) -> dict[str, str]:
    return {
        "fig01_development_model_comparison.png": (
            "Development TA RMSE under pooled, cruise-equal, and LME-macro aggregation. All deployable "
            "models use background SSS; the purple Carter oracle uses collocated in-situ salinity only to "
            "show the information ceiling and is excluded from selection. Lower is better."
        ),
        "fig02_five_fold_cv.png": (
            "Five-fold cruise-grouped training cross-validation LME-macro RMSE. Error bars show one standard "
            "deviation across folds. Hierarchical TA-SSS is more stable than the nonlinear residual model, "
            "which contrasts with the development ranking and limits the nonlinear claim."
        ),
        "fig03_lme_skill.png": (
            "Development MSE skill of the three-seed hierarchical residual ensemble relative to Carter/ESPER "
            "for each LME. Labels show records and cruises. Negative skill in LME 3 and LME 54 is retained; "
            "LME 54 has only one development cruise."
        ),
        "fig04_salinity_band_skill.png": (
            "Development skill relative to Carter/ESPER stratified by collocated observed salinity. Observed "
            "salinity is used only for evaluation strata, never as a deployable model input; labels give record "
            "counts. Skill is negative in the 33-36 and >=36 bands, showing a high-salinity regime failure."
        ),
        "fig05_support_distance_skill.png": (
            "Development skill relative to Carter/ESPER by haversine distance to the nearest training TA "
            f"observation. Skill is negative in the two most distant populated bins; the worst bin skill is "
            f"{support.skill_vs_carter.min():.2f}. Sparse counts do not permit removing these preregistered failures."
        ),
        "fig06_safe_forward.png": (
            "Safe forward-time evidence using only primary-train groups assigned to the frozen <=2018 training "
            "period and primary-development groups assigned to 2019-2021. No primary locked groups are materialized."
        ),
        "fig07_leave_lme_out.png": (
            "Leave-one-LME-out transfer skill for hierarchical TA-SSS relative to Carter/ESPER. Positive mean "
            "skill is marginal and several complete held-out LMEs degrade, showing that regional transfer remains weak."
        ),
        "fig08_observed_vs_predicted.png": (
            f"Development observation-prediction density for Carter with background SSS, the selected three-seed "
            f"ensemble, and the in-situ-SSS Carter oracle. The selected ensemble pooled RMSE is "
            f"{ensemble['pooled_rmse']:.1f} µmol kg⁻¹. The oracle gap isolates the cost of deployable SSS inputs."
        ),
        "fig09_uncertainty_calibration.png": (
            f"Development absolute-error distributions. The selected interval half-width "
            f"({selection['conformal_absolute_error_q90']:.1f} µmol kg⁻¹) was calibrated only from five-fold "
            f"training OOF residuals and attains {selection['development_coverage_90']:.3f} development coverage."
        ),
        "fig10_development_gates.png": (
            "Frozen P1.3 development gates. Cruise-equal improvement over both comparators and positive skill in "
            "every populated support-distance bin fail, so the locked test remains sealed and TA is diagnostic-only."
        ),
    }


def build_report(
    summary,
    cv,
    ensemble,
    lme,
    salinity,
    support,
    forward,
    leave_lme,
    gates,
    gate,
    selection,
    protocol,
    captions_map,
) -> str:
    display = summary[
        [
            "model",
            "pooled_rmse_mean",
            "pooled_bias_mean",
            "pooled_r2_mean",
            "cruise_equal_rmse_mean",
            "lme_macro_rmse_mean",
            "worst_lme_rmse_mean",
        ]
    ].round(3)
    cv_display = cv[
        [
            "model",
            "folds",
            "pooled_rmse_mean",
            "cruise_equal_rmse_mean",
            "lme_macro_rmse_mean",
            "lme_macro_rmse_sd",
        ]
    ].round(3)
    forward_display = forward[
        ["model", "n", "pooled_rmse", "pooled_bias", "cruise_equal_rmse", "lme_macro_rmse"]
    ].round(3)
    selected = MODEL_LABELS[selection["selected_model"]]
    failed = gates.loc[~gates.passed, "gate"].tolist()
    leave_pivot = leave_lme.pivot(index="held_lme", columns="model", values="pooled_rmse")
    leave_skill = 1 - leave_pivot.hierarchical_ta_sss**2 / leave_pivot.carter_esper_prior**2
    return f"""# Reviewer archive: P1.3 North-American coastal TA viability

## Executive finding

The preregistered experiment selected **{selected}** by development LME-macro RMSE. Its three-seed mean LME-macro RMSE is 67.277 µmol kg⁻¹ versus 96.212 for Carter/ESPER and 81.115 for the fixed LME TA-SSS baseline. The ensemble pooled RMSE is {float(ensemble.pooled_rmse.iloc[0]):.3f} µmol kg⁻¹. However, the frozen development gate failed because `{", ".join(failed)}` did not pass. The final status is `{gate["decision"]}`; locked-test and external-independent labels remain sealed.

## Scientific question and permitted claim

This experiment asks whether TA can be reconstructed at North-American coastal grid cells using inputs available at product inference time. Background SSS is the primary salinity input. Collocated in-situ salinity appears only in an oracle Carter diagnostic and in evaluation strata. The allowed claim is development, grouped-CV, safe-forward, and leave-LME-out evidence within the frozen North-American domain. No global or independent-test claim is allowed.

## Data, splits, and leakage controls

- Product-ready train: {protocol["train_rows"]:,} records from {protocol["train_cruises"]:,} cruises.
- Product-ready development: {protocol["development_rows"]:,} records from {protocol["development_cruises"]:,} cruises.
- Rows require observed TA, background SSS, valid coordinates, and finite Carter coefficients; this explains the reduction from all TA-QC records.
- Safe forward evaluation uses {protocol["safe_forward_train_rows"]:,} train rows and {protocol["safe_forward_development_rows"]:,} development rows after intersecting primary and forward assignments. It never loads primary locked groups.
- Support distance is recomputed from training TA coordinates; the zero-valued observation-anchor cache field is not used.
- The frozen data manifest `{protocol["manifest_validation"]["manifest_sha256"]}` passed full verification of 12 files.
- `locked_test_opened=false`; `external_independent_opened=false`.

## Candidate models and training

Candidates are raw Carter/ESPER, train-only Carter regional correction, global robust TA-SSS, fixed LME TA-SSS, hierarchical varying-intercept/varying-slope TA-SSS, and a three-seed nonlinear residual MLP. Partial-pooling alpha 100 was selected from the frozen grid {{1, 10, 100, 1000}} using five cruise-grouped training folds. The A3 residual MLP ran only because hierarchical A2 beat Carter and LME-linear under both development LME-macro and cruise-equal RMSE. A3 used 4,000 steps and seeds 100-102.

## Main development results

{table_markdown(display)}

![Development model comparison](figures/fig01_development_model_comparison.png)

*{captions_map["fig01_development_model_comparison.png"]}*

The selected ensemble improves pooled and LME-macro error substantially over raw Carter. It does not beat the fixed LME-linear model on cruise-equal RMSE: the three-seed mean is 102.853 µmol kg⁻¹, compared with 90.369 for LME TA-SSS. This is the first failed gate.

## Cruise-grouped cross-validation

{table_markdown(cv_display)}

![Five-fold cruise CV](figures/fig02_five_fold_cv.png)

*{captions_map["fig02_five_fold_cv.png"]}*

Hierarchical TA-SSS has the best five-fold LME-macro RMSE (89.091 µmol kg⁻¹), while the nonlinear residual model is worse (95.99 µmol kg⁻¹). Development selects A3, but grouped CV favors the simpler A2 relation; this disagreement is reported rather than resolved after observing results.

## Region, salinity, and observation-support diagnostics

{int((lme.loc[(lme.n >= 30) & (lme.cruises >= 3), "skill_vs_carter"] > 0).sum())}/{len(lme.loc[(lme.n >= 30) & (lme.cruises >= 3)])} supported LMEs have positive ensemble skill relative to Carter.

![LME skill](figures/fig03_lme_skill.png)

*{captions_map["fig03_lme_skill.png"]}*

![Salinity-band skill](figures/fig04_salinity_band_skill.png)

*{captions_map["fig04_salinity_band_skill.png"]}*

The selected model degrades relative to Carter in both high-salinity bands (33-36 and >=36). This is not a separate frozen gate, but it is a material limitation for offshore and subtropical parts of the regional product.

![Training-support distance](figures/fig05_support_distance_skill.png)

*{captions_map["fig05_support_distance_skill.png"]}*

The 200-500 km and >500 km bins contain only {int(support.iloc[-2].n)} and {int(support.iloc[-1].n)} records, but their negative skill must remain in the frozen gate. This is the second failed gate and prevents spatial extrapolation claims.

## Safe forward-time and whole-region transfer

{table_markdown(forward_display)}

![Safe forward evidence](figures/fig06_safe_forward.png)

*{captions_map["fig06_safe_forward.png"]}*

![Leave-LME-out transfer](figures/fig07_leave_lme_out.png)

*{captions_map["fig07_leave_lme_out.png"]}*

Safe-forward pooled MSE skill versus Carter is {gate["metrics"]["safe_forward_skill_vs_carter"]:.3f}. Mean leave-LME-out skill for hierarchical TA-SSS is only {leave_skill.mean():.3f}; {int((leave_skill < 0).sum())}/{len(leave_skill)} held LMEs degrade. The mean gate technically passes, but the regional pattern does not support a broad transfer claim.

## Deployable-input gap and uncertainty

![Observed versus predicted](figures/fig08_observed_vs_predicted.png)

*{captions_map["fig08_observed_vs_predicted.png"]}*

The in-situ-SSS Carter oracle reaches pooled RMSE 85.412 and LME-macro RMSE 47.144 µmol kg⁻¹. This shows that salinity information is a major limiting factor: background SSS suppresses the full coastal/estuarine TA-SSS signal.

![Uncertainty calibration](figures/fig09_uncertainty_calibration.png)

*{captions_map["fig09_uncertainty_calibration.png"]}*

The training-OOF 90% conformal half-width is {selection["conformal_absolute_error_q90"]:.3f} µmol kg⁻¹ and development coverage is {selection["development_coverage_90"]:.3f}. This is leakage-safe development calibration evidence, not locked or external-independent coverage.

## Decision and limitations

![Frozen development gates](figures/fig10_development_gates.png)

*{captions_map["fig10_development_gates.png"]}*

Decision: `{gate["decision"]}`. TA is predictable within well-supported North-American regions, and hierarchical/nonlinear corrections materially improve Carter in aggregate. It is not yet a publishable regional TA product because cruise-level robustness against the simple LME-linear baseline and remote-support extrapolation failed. Issue #10 may use this TA result only as a diagnostic sensitivity; it must not treat TA as a passed product or open the locked test.

## Figure and table index

All figures above have captions immediately below them. Source CSV files are in `tables/`; `archive_manifest.json` maps every figure to its source and records SHA256 provenance for all archived files and large local run artifacts.
"""


if __name__ == "__main__":
    raise SystemExit(main())
