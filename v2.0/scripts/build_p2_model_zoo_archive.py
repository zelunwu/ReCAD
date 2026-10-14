"""Build the reviewer-ready archive for the Issue 62 model-family screen."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_4\p2_model_zoo_v2.4")
REPAIR = Path(r"C:\backup\phd\data\processed\recad_v2_4\p2_validation_repair_v2.4")
ARCHIVE = ROOT / "docs/experiment_archive/p2_model_zoo_v2.4"
FIGURES = ARCHIVE / "figures"
TABLES = ARCHIVE / "tables"
CONFIG = ROOT / "configs/p2_model_zoo_v2.4.json"
RUNNER = ROOT / "scripts/run_p2_model_zoo_v2_4.py"
MODEL_SOURCE = ROOT / "src/recad/evaluate/model_zoo.py"
SCHEMES = ["cruise", "geographic", "whole_region", "forward"]
FAMILIES = ["ridge", "extra_trees", "catboost", "xgboost", "residual_mlp", "ft_transformer"]
LABELS = {
    "ridge": "Ridge",
    "extra_trees": "ExtraTrees",
    "catboost": "CatBoost",
    "xgboost": "XGBoost",
    "residual_mlp": "Residual MLP",
    "ft_transformer": "FT-Transformer",
    "raw_glorys": "Raw GLORYS",
    "catboost_background_residual": "Issue 61 CatBoost",
    "strict_v1_rf": "Strict-v1 RF",
    "modernized_rf": "Modern RF",
    "direct_catboost": "Issue 61 direct CatBoost",
}
COLORS = {
    "ridge": "#999999",
    "extra_trees": "#56B4E9",
    "catboost": "#009E73",
    "xgboost": "#0072B2",
    "residual_mlp": "#D55E00",
    "ft_transformer": "#CC79A7",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_lf(path: Path) -> None:
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))


def save_figure(figure: plt.Figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(FIGURES / name, dpi=220, bbox_inches="tight")
    plt.close(figure)


def prepare_tables() -> dict[str, pd.DataFrame]:
    """Copy exact analysis products and materialize compact decision tables."""

    source_names = [
        "screen_candidate_metrics.csv",
        "issue61_control_metrics.csv",
        "all_seed_metrics.csv",
        "risk_coverage.csv",
        "stability_metrics.csv",
        "compute_metrics.csv",
        "checkpoint_manifest.csv",
        "inner_selection_metrics.csv",
        "graph_residual_diagnostics.csv",
    ]
    frames = {}
    for name in source_names:
        shutil.copy2(OUTPUT / "tables" / name, TABLES / name)
        frames[name] = pd.read_csv(TABLES / name)
    split_frames = []
    for target in ("SSS", "fCO2"):
        frame = pd.read_csv(REPAIR / "splits" / f"{target}_split_audit.csv")
        if "target" in frame:
            observed_targets = set(frame.target.dropna().astype(str))
            if observed_targets != {target}:
                raise ValueError(
                    f"unexpected targets in {target} split audit: {sorted(observed_targets)}"
                )
        else:
            frame.insert(0, "target", target)
        split_frames.append(frame)
    frames["split_audit.csv"] = pd.concat(split_frames, ignore_index=True)
    frames["split_audit.csv"].to_csv(TABLES / "split_audit.csv", index=False)

    finalists = json.loads((OUTPUT / "finalists.json").read_text(encoding="utf-8"))
    finalist_rows = []
    for target, decision in finalists["decisions"].items():
        for family in FAMILIES:
            finalist_rows.append(
                {
                    "target": target,
                    "family": family,
                    "selected": family in decision["selected"],
                    "qualified": family in decision["qualified"],
                    "nondominated": family in decision["nondominated"],
                    "status": decision["status"],
                    "mean_rank": decision["mean_rank"][family],
                }
            )
    frames["finalist_decisions.csv"] = pd.DataFrame(finalist_rows)
    frames["finalist_decisions.csv"].to_csv(TABLES / "finalist_decisions.csv", index=False)

    decision = json.loads((OUTPUT / "decision.json").read_text(encoding="utf-8"))
    trigger_rows = [
        {"target": target, **payload} for target, payload in decision["context_triggers"].items()
    ]
    frames["context_triggers.csv"] = pd.DataFrame(trigger_rows)
    frames["context_triggers.csv"].to_csv(TABLES / "context_triggers.csv", index=False)

    specification = json.loads(CONFIG.read_text(encoding="utf-8"))
    definitions = []
    for family in FAMILIES:
        definitions.append(
            {
                "family": family,
                "label": LABELS[family],
                "configurations": len(
                    specification["families"][family].get(
                        "configurations", specification["families"][family].get("alpha", [])
                    )
                ),
                "inner_metric": specification["inner_selection"]["metric"],
                "screen_seed": specification["screen_seed"],
                "finalist_seeds": ",".join(map(str, specification["finalist_seeds"])),
            }
        )
    frames["model_definitions.csv"] = pd.DataFrame(definitions)
    frames["model_definitions.csv"].to_csv(TABLES / "model_definitions.csv", index=False)
    return frames


def grouped_metric(
    frame: pd.DataFrame,
    target: str,
    metric: str,
    ylabel: str,
    name: str,
) -> None:
    selected = frame.loc[frame.target == target].copy()
    figure, axis = plt.subplots(figsize=(12.5, 5.8))
    width = 0.82 / len(FAMILIES)
    x = np.arange(len(SCHEMES))
    for offset, family in enumerate(FAMILIES):
        values = selected.loc[selected.model == family].set_index("scheme")[metric].reindex(SCHEMES)
        axis.bar(
            x + (offset - (len(FAMILIES) - 1) / 2) * width,
            values,
            width,
            label=LABELS[family],
            color=COLORS[family],
        )
    axis.set_xticks(x, [value.replace("_", "\n") for value in SCHEMES])
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3)
    save_figure(figure, name)


def heatmap(
    table: pd.DataFrame,
    title: str,
    color_label: str,
    name: str,
    *,
    center: float | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    values = table.to_numpy(float)
    maximum = np.nanmax(np.abs(values - center)) if center is not None else None
    image = axis.imshow(
        values,
        cmap="RdYlGn" if center is not None else "viridis_r",
        aspect="auto",
        vmin=(center - maximum) if center is not None else None,
        vmax=(center + maximum) if center is not None else None,
    )
    axis.set_xticks(
        range(len(table.columns)), [value.replace("_", "\n") for value in table.columns]
    )
    axis.set_yticks(range(len(table.index)), [LABELS.get(value, value) for value in table.index])
    for row in range(len(table)):
        for column in range(len(table.columns)):
            value = values[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}" if np.isfinite(value) else "NA",
                ha="center",
                va="center",
                fontsize=8,
            )
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=color_label)
    save_figure(figure, name)


def build_figures(frames: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    screen = frames["screen_candidate_metrics.csv"]
    grouped_metric(screen, "SSS", "rmse", "SSS RMSE (psu)", "fig01_sss_rmse.png")
    grouped_metric(screen, "fCO2", "rmse", "fCO2 RMSE (µatm)", "fig02_fco2_rmse.png")

    controls = frames["issue61_control_metrics.csv"]
    relative_parts = []
    for target, baseline_model in (
        ("SSS", "catboost_background_residual"),
        ("fCO2", "modernized_rf"),
    ):
        baseline = (
            controls.loc[(controls.target == target) & (controls.model == baseline_model)]
            .set_index("scheme")
            .rmse
        )
        part = screen.loc[screen.target == target].pivot(
            index="model", columns="scheme", values="rmse"
        )
        relative = pd.DataFrame(
            {
                scheme: 100 * (baseline[scheme] - part[scheme]) / baseline[scheme]
                for scheme in part.columns
            }
        )
        relative["target"] = target
        relative_parts.append(relative.reset_index())
    relative_frame = pd.concat(relative_parts, ignore_index=True)
    relative_frame.to_csv(TABLES / "relative_improvement.csv", index=False)
    for target, number in (("SSS", 3), ("fCO2", 4)):
        part = relative_frame.loc[relative_frame.target == target].set_index("model")[SCHEMES]
        heatmap(
            part,
            f"{target} improvement over Issue 61 baseline",
            "RMSE improvement (%)",
            f"fig{number:02d}_{target.lower()}_relative_improvement.png",
            center=0.0,
        )

    group_table = screen.pivot_table(
        index=["target", "model"], columns="scheme", values="group_equal_rmse"
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        part = group_table.loc[target].reindex(FAMILIES)[SCHEMES]
        image = axis.imshow(part, cmap="viridis_r", aspect="auto")
        axis.set_xticks(range(4), [value.replace("_", "\n") for value in SCHEMES])
        axis.set_yticks(range(6), [LABELS[value] for value in FAMILIES])
        axis.set_title(target)
        figure.colorbar(image, ax=axis, fraction=0.045)
    save_figure(figure, "fig05_group_equal_rmse.png")

    stability = frames["stability_metrics.csv"]
    finalist_decisions = frames["finalist_decisions.csv"]
    selected = finalist_decisions.loc[finalist_decisions.selected, ["target", "family"]]
    finalist_stability = stability.merge(
        selected, left_on=["target", "model"], right_on=["target", "family"]
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        part = finalist_stability.loc[finalist_stability.target == target]
        for family, family_part in part.groupby("model", observed=True):
            values = family_part.set_index("scheme").reindex(SCHEMES)
            axis.errorbar(
                range(4),
                values.rmse_mean,
                yerr=values.rmse_std.fillna(0),
                marker="o",
                capsize=3,
                label=LABELS[family],
                color=COLORS[family],
            )
        axis.set_xticks(range(4), [value.replace("_", "\n") for value in SCHEMES])
        axis.set_title(target)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Held-out RMSE, mean ± seed SD")
    save_figure(figure, "fig06_finalist_seed_stability.png")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        part = finalist_stability.loc[finalist_stability.target == target]
        for family, family_part in part.groupby("model", observed=True):
            values = family_part.set_index("scheme").reindex(SCHEMES)
            axis.plot(range(4), values.coverage90_mean, marker="o", label=LABELS[family])
        axis.axhspan(0.85, 0.95, color="#009E73", alpha=0.15)
        axis.axhline(0.9, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(range(4), [value.replace("_", "\n") for value in SCHEMES])
        axis.set_ylim(0.5, 1.02)
        axis.set_title(target)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Empirical nominal-90% coverage")
    save_figure(figure, "fig07_finalist_coverage90.png")

    risk = frames["risk_coverage.csv"].merge(
        selected, left_on=["target", "model"], right_on=["target", "family"]
    )
    risk = risk.loc[risk.scheme == "geographic"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        part = risk.loc[risk.target == target]
        summary = part.groupby(["model", "retained_fraction"], as_index=False).rmse.mean()
        for family, family_part in summary.groupby("model", observed=True):
            axis.plot(
                family_part.retained_fraction,
                family_part.rmse,
                marker="o",
                label=LABELS[family],
            )
        axis.set_xlabel("Fraction retained by narrowest interval")
        axis.set_title(target)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Geographic held-out RMSE")
    save_figure(figure, "fig08_geographic_risk_coverage.png")

    compute = frames["compute_metrics.csv"]
    tuning = frames["inner_selection_metrics.csv"]
    screen_compute = (
        compute.loc[compute.seed.eq(62024)]
        .groupby(["target", "family"], as_index=False)
        .agg(
            final_fit_seconds=("elapsed_seconds", "sum"),
            peak_gpu_bytes=("peak_gpu_bytes", "max"),
            parameter_count=("parameter_count", "median"),
        )
    )
    tune_compute = (
        tuning.groupby(["target", "family"], as_index=False)
        .elapsed_seconds.sum()
        .rename(columns={"elapsed_seconds": "tuning_seconds"})
    )
    compute_summary = screen_compute.merge(tune_compute, on=["target", "family"])
    compute_summary["total_screen_hours"] = (
        compute_summary.final_fit_seconds + compute_summary.tuning_seconds
    ) / 3600
    mean_rank = finalist_decisions[["target", "family", "mean_rank"]]
    compute_summary = compute_summary.merge(mean_rank, on=["target", "family"])
    compute_summary.to_csv(TABLES / "compute_summary.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        part = compute_summary.loc[compute_summary.target == target]
        for _, row in part.iterrows():
            axis.scatter(row.total_screen_hours, row.mean_rank, color=COLORS[row.family], s=65)
            axis.annotate(LABELS[row.family], (row.total_screen_hours, row.mean_rank), fontsize=8)
        axis.set_xlabel("Complete screen compute (hours)")
        axis.set_title(target)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean held-out rank (lower is better)")
    save_figure(figure, "fig09_compute_performance.png")

    selection = frames["inner_selection_metrics.csv"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        part = selection.loc[selection.target == target]
        values = [
            part.loc[part.family == family, "group_equal_rmse"].to_numpy() for family in FAMILIES
        ]
        axis.boxplot(values, tick_labels=[LABELS[value] for value in FAMILIES], showfliers=False)
        axis.tick_params(axis="x", rotation=25)
        axis.set_title(target)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Inner group-equal RMSE across configs and folds")
    save_figure(figure, "fig10_inner_search_distribution.png")

    all_seed = frames["all_seed_metrics.csv"].merge(
        selected, left_on=["target", "model"], right_on=["target", "family"]
    )
    transfer = all_seed.groupby(["target", "model", "scheme"], as_index=False).agg(
        late_period_bias=("late_period_bias", "mean"),
        worst_supported_region_rmse=("worst_supported_region_rmse", "mean"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for column, target in enumerate(("SSS", "fCO2")):
        part = transfer.loc[transfer.target == target]
        for row, metric in enumerate(("late_period_bias", "worst_supported_region_rmse")):
            table = part.pivot(index="model", columns="scheme", values=metric).reindex(
                columns=SCHEMES
            )
            image = axes[row, column].imshow(table, cmap="coolwarm", aspect="auto")
            axes[row, column].set_xticks(range(4), [value.replace("_", "\n") for value in SCHEMES])
            axes[row, column].set_yticks(
                range(len(table)), [LABELS[value] for value in table.index]
            )
            axes[row, column].set_title(f"{target}: {metric.replace('_', ' ')}")
            figure.colorbar(image, ax=axes[row, column], fraction=0.045)
    save_figure(figure, "fig11_late_bias_worst_region.png")

    graph = frames["graph_residual_diagnostics.csv"]
    graph_summary = graph.groupby(["target", "model"], as_index=False).agg(
        signal_mean=("correlation_above_control", "mean"),
        signal_std=("correlation_above_control", "std"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        part = graph_summary.loc[graph_summary.target == target]
        axis.bar(
            [LABELS[value] for value in part.model],
            part.signal_mean,
            yerr=part.signal_std.fillna(0),
            color=[COLORS[value] for value in part.model],
            capsize=3,
        )
        axis.axhline(0.2, color="black", linestyle="--", label="frozen trigger")
        axis.tick_params(axis="x", rotation=20)
        axis.set_title(target)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Adjacent residual correlation above permutation control")
    axes[1].legend(frameon=False)
    save_figure(figure, "fig12_graph_residual_signal.png")

    split = frames["split_audit.csv"]
    leakage_columns = [
        "group_overlap",
        "record_overlap",
        "fit_calibration_group_overlap",
        "forbidden_time_rows",
    ]
    leakage = split.groupby(["target", "scheme"])[leakage_columns].sum()
    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    image = axis.imshow(leakage, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    axis.set_yticks(range(len(leakage)), [" / ".join(value) for value in leakage.index])
    axis.set_xticks(range(4), [value.replace("_", "\n") for value in leakage_columns])
    for row in range(len(leakage)):
        for column in range(4):
            axis.text(column, row, int(leakage.iloc[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Overlap count")
    save_figure(figure, "fig13_leakage_audit.png")

    return {
        "fig01_sss_rmse.png": ["screen_candidate_metrics.csv"],
        "fig02_fco2_rmse.png": ["screen_candidate_metrics.csv"],
        "fig03_sss_relative_improvement.png": [
            "relative_improvement.csv",
            "issue61_control_metrics.csv",
        ],
        "fig04_fco2_relative_improvement.png": [
            "relative_improvement.csv",
            "issue61_control_metrics.csv",
        ],
        "fig05_group_equal_rmse.png": ["screen_candidate_metrics.csv"],
        "fig06_finalist_seed_stability.png": ["stability_metrics.csv", "finalist_decisions.csv"],
        "fig07_finalist_coverage90.png": ["stability_metrics.csv", "finalist_decisions.csv"],
        "fig08_geographic_risk_coverage.png": ["risk_coverage.csv", "finalist_decisions.csv"],
        "fig09_compute_performance.png": ["compute_summary.csv", "finalist_decisions.csv"],
        "fig10_inner_search_distribution.png": ["inner_selection_metrics.csv"],
        "fig11_late_bias_worst_region.png": ["all_seed_metrics.csv", "finalist_decisions.csv"],
        "fig12_graph_residual_signal.png": [
            "graph_residual_diagnostics.csv",
            "context_triggers.csv",
        ],
        "fig13_leakage_audit.png": ["split_audit.csv"],
    }


def captions(finalists: dict[str, object], triggers: pd.DataFrame) -> dict[str, str]:
    sss = ", ".join(LABELS[value] for value in finalists["decisions"]["SSS"]["selected"])
    fco2 = ", ".join(LABELS[value] for value in finalists["decisions"]["fCO2"]["selected"])
    graph_status = ", ".join(
        f"{row.target}={'triggered' if row.graph_triggered else 'not triggered'}"
        for row in triggers.itertuples()
    )
    return {
        "fig01_sss_rmse.png": "Figure 1. Held-out SSS RMSE for all six preregistered point-model families across cruise, true geographic-block plus 100 km buffer, whole-region, and rolling-forward validation; every bar pools the complete corrected outer predictions rather than an inner tuning subset.",
        "fig02_fco2_rmse.png": "Figure 2. Held-out fCO2 RMSE for the same six families and repaired validation designs, using outer-nested predicted SSS rather than observed salinity as an input; the strict transfer columns determine whether ordinary cruise skill generalizes.",
        "fig03_sss_relative_improvement.png": "Figure 3. Percent SSS RMSE improvement relative to the Issue 61 CatBoost background-residual baseline; positive values indicate improvement, while the four columns expose any gain obtained by sacrificing geographic, regional, or future transfer.",
        "fig04_fco2_relative_improvement.png": "Figure 4. Percent fCO2 RMSE improvement relative to the Issue 61 modernized random forest; the comparison uses identical outer folds and nested upstream SSS, so differences isolate model family and its fixed small search allowance.",
        "fig05_group_equal_rmse.png": "Figure 5. Group-equal RMSE assigns every held cruise or platform group equal influence and therefore checks whether dense observations dominate pooled RMSE; lower values and consistent color across columns indicate more stable transfer.",
        "fig06_finalist_seed_stability.png": f"Figure 6. Three-seed RMSE mean and standard deviation for the preregistered Pareto or fallback finalists (SSS: {sss}; fCO2: {fco2}); error bars quantify stochastic training sensitivity after configuration selection.",
        "fig07_finalist_coverage90.png": "Figure 7. Empirical coverage of nominal 90% prediction intervals for the finalists; the green band marks the frozen 0.85-0.95 acceptance range and reveals where calibration fitted inside outer training fails under distribution shift.",
        "fig08_geographic_risk_coverage.png": "Figure 8. Geographic-holdout RMSE after retaining observations with the narrowest prediction intervals; a monotone curve supports interval width as a risk-ranking device but does not validate unobserved coastal grid cells.",
        "fig09_compute_performance.png": "Figure 9. Complete screening compute time versus mean held-out rank, including all six inner configurations and final outer fits; the lower-left region identifies families that combine transfer skill with practical computational cost.",
        "fig10_inner_search_distribution.png": "Figure 10. Distribution of inner group-equal RMSE across every family configuration and outer fold; broad or shifted boxes show whether the equal six-configuration allowance yielded a robust family or a configuration-sensitive result.",
        "fig11_late_bias_worst_region.png": "Figure 11. Finalist late-period bias for 2020-2025 and worst RMSE among LMEs with at least 200 held observations; these diagnostics prevent good global averages from hiding temporal drift or severe supported-region failures.",
        "fig12_graph_residual_signal.png": f"Figure 12. Same-month four-neighbour residual correlation after subtracting a within-month permutation control, averaged over finalist seeds; the frozen graph trigger is 0.20 and the resulting statuses are {graph_status}.",
        "fig13_leakage_audit.png": "Figure 13. Summed group, record, calibration-group, and forbidden-future overlap counts inherited from the repaired Issue 61 folds; every displayed zero is required before model-family comparisons can be interpreted.",
    }


def report_text(
    frames: dict[str, pd.DataFrame],
    finalist_payload: dict[str, object],
    caption_map: dict[str, str],
) -> str:
    stability = frames["stability_metrics.csv"].set_index(["target", "scheme", "model"])
    decision = json.loads((OUTPUT / "decision.json").read_text(encoding="utf-8"))
    result_lines = []
    for target in ("SSS", "fCO2"):
        selection = finalist_payload["decisions"][target]
        for family in selection["selected"]:
            values = (
                stability.loc[(target, slice(None), family), :].reset_index().set_index("scheme")
            )
            result_lines.append(
                f"- **{target} / {LABELS[family]}** ({selection['status']}): "
                + ", ".join(
                    f"{scheme.replace('_', ' ')} RMSE {values.loc[scheme, 'rmse_mean']:.3f}"
                    for scheme in SCHEMES
                )
                + "."
            )
    figures = []
    for number, (name, caption) in enumerate(caption_map.items(), start=1):
        figures.append(f"### Figure {number}\n\n![{caption}](figures/{name})\n\n{caption}")
    trigger_text = "; ".join(
        f"{target}: temporal={payload['temporal_triggered']}, graph={payload['graph_triggered']}"
        for target, payload in decision["context_triggers"].items()
    )
    return f"""# P2 fair finite model zoo v2.4

## Scientific question and permitted claim

Issue #62 asks which point-model inductive biases deserve further work for global-coastal SSS and fCO2 after the Issue 61 validation repair. The permitted claim is a development-stage family comparison on corrected outer holdouts. No new locked, external-independent, or 2026 labels were opened, and this experiment does not itself publish a global gridded product.

## Data, splits, and leakage controls

Every family uses the frozen 1993-2025 SOCAT matchup and the exact five cruise, five 5-degree geographic plus 100 km buffer, five whole-region, and three past-only forward splits from Issue 61. Complete groups define outer fitting, inner 80/20 configuration selection, and the separate outer 80/20 conformal calibration partition. fCO2 receives only SSS predictions generated within the permitted lineage: inner-selection SSS never uses an inner-validation salinity label, and outer held/calibration SSS is predicted from outer model-fit labels. Figure 13 records zero group, record, calibration-group, and forbidden-future overlap.

## Candidate models and training

The screen compares Ridge, ExtraTrees, CatBoost, XGBoost, a residual MLP, and a compact numerical FT-Transformer. Each family receives six frozen configurations, the same capped inner training and validation budgets, group-equal RMSE selection, and a final refit on all outer model-fit rows. SSS uses a same-family GLORYS background-residual path with a direct fallback; fCO2 is direct and uses outer-nested predicted SSS. The screen seed is 62024. At most two Pareto or fallback stability finalists per target are repeated with seeds 62025 and 62026. Checkpoints, transforms, configuration scores, runtime, updates, parameter count, inference time, peak GPU memory, and SHA256 hashes are retained outside Git and indexed by archived tables.

## Main development results

{chr(10).join(result_lines)}

The finalist state is explicitly distinguished from promotion: `promotion_finalist` means the frozen relative-improvement rule passed, whereas `stability_finalist_only` preserves the best non-dominated families for a three-seed stability estimate without claiming practical superiority. Late-period bias, worst supported-region RMSE, group-equal error, interval coverage, risk-coverage, and compute cost remain part of the interpretation rather than post-hoc selection metrics.

Conditional context diagnostics were evaluated only after the point screen. Their recorded outcomes are: {trigger_text}. A triggered diagnostic identifies residual structure worth a separately controlled context experiment; it is not evidence that a particular graph or temporal network will improve predictions.

## Decision and limitations

The machine-readable decision is `{decision["status"]}`. The selected families are development candidates under observed-location transfer, not independently validated product models. Geographic and whole-region holdouts are the best current proxies for unsampled coastal space but do not reproduce every no-observation coastline. Prediction intervals calibrated on outer training can under-cover after spatial or temporal shift. The graph statistic uses observed same-month adjacent cells and a permutation control; it establishes residual dependence, not causality or guaranteed graph-model skill.

Issue 62 closes the finite fair family screen. Any context branch triggered here must be treated as a separately named follow-up with the same corrected folds and an explicit point-model comparison. Product promotion still requires later tuning, grid-month applicability flags, and independent evidence defined by the roadmap.

## Figure and table index

{chr(10).join(figures)}

Exact source CSVs are under `tables/`. `archive_manifest.json` maps every figure to its source data and hashes all archived files. Row-level predictions, configuration records, nested SSS predictions, and model checkpoints remain outside Git under `{OUTPUT}`.
"""


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    frames = prepare_tables()
    mappings = build_figures(frames)
    finalist_payload = json.loads((OUTPUT / "finalists.json").read_text(encoding="utf-8"))
    caption_map = captions(finalist_payload, frames["context_triggers.csv"])
    report = report_text(frames, finalist_payload, caption_map)
    (ARCHIVE / "REPORT.md").write_text(report, encoding="utf-8")
    (ARCHIVE / "CAPTIONS.md").write_text(
        "# Figure captions\n\n"
        + "\n\n".join(f"## {name}\n\n{caption}" for name, caption in caption_map.items())
        + "\n",
        encoding="utf-8",
    )
    (ARCHIVE / "README.md").write_text(
        "# Issue 62 reviewer archive\n\nStart with [REPORT.md](REPORT.md). Rebuild with "
        "`python scripts/build_p2_model_zoo_archive.py` and verify with "
        "`python scripts/verify_experiment_archive.py "
        "docs/experiment_archive/p2_model_zoo_v2.4`.\n",
        encoding="utf-8",
    )
    tracked = [ARCHIVE / "README.md", ARCHIVE / "REPORT.md", ARCHIVE / "CAPTIONS.md"]
    tracked += sorted(FIGURES.glob("*.png")) + sorted(TABLES.glob("*.csv"))
    for path in tracked:
        if path.suffix in {".md", ".csv"}:
            normalize_lf(path)
    files_sha256 = {
        str(path.relative_to(ARCHIVE)).replace("\\", "/"): sha256(path) for path in tracked
    }
    source_artifacts = [
        OUTPUT / "data_manifest.json",
        OUTPUT / "finalists.json",
        OUTPUT / "decision.json",
        CONFIG,
        RUNNER,
        MODEL_SOURCE,
        REPAIR / "splits/SSS_split_indices.npz",
        REPAIR / "splits/fCO2_split_indices.npz",
    ]
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "experiment_id": "p2_model_zoo_v2.4",
        "training_git_commit": git_commit,
        "data_manifest_sha256": sha256(OUTPUT / "data_manifest.json"),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "1993-2025 frozen development labels on corrected outer holdouts",
        "decision": json.loads((OUTPUT / "decision.json").read_text(encoding="utf-8")),
        "figure_source_data": mappings,
        "figure_captions": caption_map,
        "files_sha256": files_sha256,
        "archive_builder_sha256": sha256(Path(__file__)),
        "analysis_script_sha256": sha256(RUNNER),
        "source_artifacts_sha256": {str(path): sha256(path) for path in source_artifacts},
    }
    (ARCHIVE / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
