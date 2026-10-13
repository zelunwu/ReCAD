"""Build the reviewer-ready archive for the Issue #42 context experiment."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path(r"C:\backup\phd\data\processed\recad_v2_3\p2_context_connectivity_v2.3")
ARCHIVE = ROOT / "docs/experiment_archive/p2_context_connectivity_v2.3"
TABLES = ARCHIVE / "tables"
FIGURES = ARCHIVE / "figures"
RUNNER = ROOT / "scripts/run_p2_context_connectivity_v2_3.py"
ANALYSIS = ROOT / "scripts/analyze_p2_context_connectivity_v2_3.py"
CONFIG = ROOT / "configs/frozen/context_connectivity_contract_v2.3.json"
MODEL_SOURCE = ROOT / "src/recad/evaluate/context_connectivity.py"
MODELS = ["v11_random_forest", "lag_catboost", "gru_context", "st96_local", "st96_graph"]
COLORS = dict(zip(MODELS, ["#3f78a8", "#d05b62", "#4d9c73", "#8c6bb1", "#207f7c"], strict=True))

CAPTIONS = {
    "fig01_native_rmse_matrix.png": "Figure 1. Native held-out fCO2 RMSE across point, lag, recurrent, local-transformer, and water-connected transformer candidates; the random forest leads three transfer schemes and lag CatBoost gives only a small forward gain.",
    "fig02_cruise_equal_rmse.png": "Figure 2. Cruise-equal RMSE gives every independent cruise equal influence and confirms that no context candidate reaches the frozen five-percent improvement threshold.",
    "fig03_relative_improvement.png": "Figure 3. Relative cruise-equal improvement against the honest random forest; positive values indicate improvement, and only lag CatBoost is slightly positive in forward validation.",
    "fig04_forward_periods.png": "Figure 4. Strict forward RMSE in 2019-2021 and 2022-2025; lag CatBoost reduces temporal bias but remains above the 40 microatmosphere ceiling in both periods.",
    "fig05_learning_curve_rmse.png": "Figure 5. Cruise-fold-0 pooled learning curves under nested LME-stratified fCO2 cruise-label budgets; random-forest error continues to fall as coverage increases, lag CatBoost plateaus, and ST96 is unstable.",
    "fig06_learning_curve_cruise_equal.png": "Figure 6. Cruise-equal learning curves show that the apparent row count overstates independent information and that ST96 does not unlock additional capacity with the available cruise coverage.",
    "fig07_graph_residual_structure.png": "Figure 7. Water-adjacent residual correlation remains high for every model and is larger after the context networks, so the graph input does not remove the spatially connected error structure that triggered it.",
    "fig08_front_preservation.png": "Figure 8. Correlation between observed and predicted adjacent-grid fCO2 gradients; all candidates smooth fronts strongly and context networks preserve less gradient structure than the point random forest.",
    "fig09_boundary_discontinuity.png": "Figure 9. Mean predicted gradient within and across LME boundaries; residual models inherit artificial cross-LME jumps from the fold-fitted LME-month background while the direct random forest remains smoother.",
    "fig10_lme_distributions.png": "Figure 10. Per-LME RMSE distributions for cruise, spatial, and whole-LME validation expose a long geographic error tail that additional context does not reduce.",
    "fig11_low_support.png": "Figure 11. Cruise-held-out RMSE by the number of training cruises in the target LME; sparse support remains difficult and no complex model supplies a reliable low-support advantage.",
    "fig12_annual_bias.png": "Figure 12. Annual cruise-OOF bias; lag CatBoost follows the long-term signal better than the random forest in some years but pays for it with higher ordinary held-cruise error.",
    "fig13_interval_coverage.png": "Figure 13. Leakage-safe 50 and 90 percent interval coverage; forward coverage remains below the frozen ranges for every candidate despite wider context-model intervals.",
    "fig14_risk_coverage.png": "Figure 14. Forward RMSE after retaining cases with the narrowest empirical intervals; uncertainty ranking cannot reduce any candidate to the release ceiling at full coverage.",
    "fig15_bootstrap.png": "Figure 15. Cruise-level paired MSE bootstrap contrasts against the random forest; every context candidate has confidence intervals above zero in cruise and spatial validation.",
    "fig16_selection_gate.png": "Figure 16. Frozen promotion-gate totals; no context candidate passes the complete contract, so the current STTransformer, GRU, and graph families stop scaling.",
    "fig17_compute_cost.png": "Figure 17. Median outer-fold training time and peak GPU memory; added neural complexity has measurable cost without a generalization return.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style(ax: Any, title: str, ylabel: str = "RMSE (microatmospheres)") -> None:
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)


def copy_tables() -> dict[str, pd.DataFrame]:
    if ARCHIVE.exists():
        shutil.rmtree(ARCHIVE)
    TABLES.mkdir(parents=True)
    FIGURES.mkdir(parents=True)
    result = {}
    for source in sorted((EXTERNAL / "tables").glob("*.csv")):
        destination = TABLES / source.name
        shutil.copyfile(source, destination)
        result[source.name] = pd.read_csv(destination)
    if len(result) < 22:
        raise RuntimeError(f"expected at least 22 source tables, found {len(result)}")
    return result


def grouped_bars(
    pivot: pd.DataFrame,
    models: list[str],
    title: str,
    filename: str,
    *,
    target: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    pivot.loc[:, models].plot(kind="bar", ax=ax, color=[COLORS[model] for model in models])
    if target is not None:
        ax.axhline(target, color="black", linestyle="--", linewidth=1, label="minimum target")
    ax.tick_params(axis="x", rotation=20)
    style(ax, title)
    ax.legend(fontsize=8, ncol=3)
    save(fig, filename)


def build_figures(tables: dict[str, pd.DataFrame]) -> None:
    native = tables["table01_native_metrics.csv"]
    pivot = native.pivot(index="model", columns="scheme", values="rmse").loc[
        MODELS, ["cruise", "spatial", "whole_lme", "forward"]
    ]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    image = ax.imshow(pivot, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(4), ["cruise", "spatial", "whole LME", "forward"])
    ax.set_yticks(range(len(MODELS)), MODELS)
    for row in range(len(MODELS)):
        for column in range(4):
            ax.text(column, row, f"{pivot.iloc[row, column]:.2f}", ha="center", va="center")
    ax.set_title("Temporal context and coastal connectivity", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, label="RMSE (microatmospheres)")
    save(fig, "fig01_native_rmse_matrix.png")

    equal = tables["table02_cruise_equal_metrics.csv"]
    grouped_bars(
        equal.pivot(index="scheme", columns="model", values="cruise_equal_rmse"),
        MODELS,
        "Cruise-equal transfer performance",
        "fig02_cruise_equal_rmse.png",
    )
    equal_pivot = equal.pivot(index="scheme", columns="model", values="cruise_equal_rmse")
    relative = pd.DataFrame(
        {
            model: 100 * (1 - equal_pivot[model] / equal_pivot.v11_random_forest)
            for model in MODELS[1:]
        }
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    relative.plot(kind="bar", ax=ax, color=[COLORS[model] for model in MODELS[1:]])
    ax.axhline(5, color="black", linestyle="--", label="promotion threshold")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.tick_params(axis="x", rotation=20)
    style(ax, "Relative improvement over random forest", "improvement (%)")
    ax.legend(fontsize=8)
    save(fig, "fig03_relative_improvement.png")

    forward = tables["table22_forward_period_metrics.csv"]
    grouped_bars(
        forward.pivot(index="period", columns="model", values="rmse"),
        MODELS,
        "Strict forward-time periods",
        "fig04_forward_periods.png",
        target=40,
    )

    learning = tables["table17_learning_curve.csv"]
    for metric, filename, title in (
        ("rmse", "fig05_learning_curve_rmse.png", "Nested cruise-label learning curve"),
        (
            "cruise_equal_rmse",
            "fig06_learning_curve_cruise_equal.png",
            "Cruise-equal nested learning curve",
        ),
    ):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for model in ("v11_random_forest", "lag_catboost", "st96_local"):
            subset = learning.loc[learning.model.eq(model)]
            ax.plot(
                100 * subset.cruise_fraction,
                subset[metric],
                marker="o",
                label=model,
                color=COLORS[model],
            )
        style(ax, title)
        ax.set_xlabel("nested training cruises (%)")
        ax.legend(fontsize=8)
        save(fig, filename)

    graph = tables["table14_graph_residual_structure.csv"].set_index("model").loc[MODELS]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(graph.index, graph.adjacent_residual_correlation, color=[COLORS[m] for m in MODELS])
    ax.axhline(0.15, color="black", linestyle="--", label="graph trigger")
    ax.tick_params(axis="x", rotation=20)
    style(ax, "Residual structure on water-connected pairs", "residual correlation")
    ax.legend()
    save(fig, "fig07_graph_residual_structure.png")

    fronts = tables["table15_front_boundary_metrics.csv"]
    all_front = fronts.loc[fronts.boundary.eq("all")].set_index("model").loc[MODELS]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(all_front.index, all_front.gradient_correlation, color=[COLORS[m] for m in MODELS])
    ax.tick_params(axis="x", rotation=20)
    style(ax, "Adjacent-grid front preservation", "gradient correlation")
    save(fig, "fig08_front_preservation.png")
    boundary = (
        fronts.loc[fronts.boundary.isin(["within_lme", "cross_lme"])]
        .pivot(index="model", columns="boundary", values="prediction_mean_abs_gradient")
        .loc[MODELS]
    )
    fig, ax = plt.subplots(figsize=(9, 4.6))
    boundary.plot(kind="bar", ax=ax, color=["#7aa6c2", "#c65d57"])
    ax.axhline(7.31, color="black", linestyle="--", label="observed mean")
    ax.tick_params(axis="x", rotation=20)
    style(ax, "Within- and cross-LME discontinuity", "mean absolute gradient")
    ax.legend(fontsize=8)
    save(fig, "fig09_boundary_discontinuity.png")

    lme = tables["table04_lme_metrics.csv"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    for ax, scheme in zip(axes, ["cruise", "spatial", "whole_lme"], strict=True):
        values = [
            lme.loc[lme.scheme.eq(scheme) & lme.model.eq(model), "rmse"].to_numpy()
            for model in MODELS
        ]
        ax.boxplot(values, tick_labels=["RF", "lag", "GRU", "ST96", "graph"], showfliers=False)
        ax.tick_params(axis="x", rotation=25)
        style(ax, scheme.replace("_", " "))
    save(fig, "fig10_lme_distributions.png")

    strata = tables["table06_stratified_metrics.csv"]
    low = strata.loc[strata.scheme.eq("cruise") & strata.stratum_type.eq("support_band")].pivot(
        index="stratum", columns="model", values="rmse"
    )
    grouped_bars(low, MODELS, "Held-cruise error by LME support", "fig11_low_support.png")

    annual = tables["table08_annual_diagnostics.csv"]
    fig, ax = plt.subplots(figsize=(10, 4.7))
    for model in ("v11_random_forest", "lag_catboost", "st96_graph"):
        subset = annual.loc[annual.scheme.eq("cruise") & annual.model.eq(model)]
        ax.plot(subset.yr, subset.bias, marker="o", markersize=3, label=model, color=COLORS[model])
    ax.axhline(0, color="black", linewidth=0.8)
    style(ax, "Annual cruise-OOF bias", "bias (microatmospheres)")
    ax.legend(fontsize=8)
    save(fig, "fig12_annual_bias.png")

    intervals = tables["table10_interval_metrics.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, column, nominal in zip(axes, ["coverage50", "coverage90"], [0.5, 0.9], strict=True):
        intervals.pivot(index="scheme", columns="model", values=column)[MODELS].plot(
            kind="bar", ax=ax, color=[COLORS[m] for m in MODELS]
        )
        ax.axhline(nominal, color="black", linestyle="--")
        ax.tick_params(axis="x", rotation=20)
        style(ax, column, "empirical coverage")
    save(fig, "fig13_interval_coverage.png")

    risk = tables["table11_risk_coverage.csv"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for model in MODELS:
        subset = risk.loc[risk.scheme.eq("forward") & risk.model.eq(model)]
        ax.plot(
            100 * subset.retained_fraction,
            subset.rmse,
            marker="o",
            label=model,
            color=COLORS[model],
        )
    ax.axhline(40, color="black", linestyle="--")
    style(ax, "Forward risk-coverage")
    ax.set_xlabel("retained fraction (%)")
    ax.legend(fontsize=8)
    save(fig, "fig14_risk_coverage.png")

    bootstrap = tables["table12_paired_bootstrap.csv"]
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{row.candidate}\n{row.scheme}" for row in bootstrap.itertuples()]
    mean = bootstrap.mean_delta_mse.to_numpy()
    ax.errorbar(
        range(len(mean)),
        mean,
        yerr=[mean - bootstrap.ci_low_delta_mse, bootstrap.ci_high_delta_mse - mean],
        fmt="o",
        capsize=4,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    style(ax, "Paired cruise bootstrap", "candidate - random forest MSE")
    save(fig, "fig15_bootstrap.png")

    gate = tables["table20_selection_gate.csv"]
    gate_count = gate.groupby("candidate").passed.sum().reindex(MODELS[1:])
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.barh(gate_count.index, gate_count, color=[COLORS[m] for m in gate_count.index])
    ax.axvline(6, color="black", linestyle="--", label="all gates")
    style(ax, "Frozen promotion gate", "passed checks of 6")
    ax.legend()
    save(fig, "fig16_selection_gate.png")

    resources = tables["table16_training_resources.csv"]
    resource_summary = (
        resources.groupby("model")
        .agg(training_seconds=("training_seconds", "median"), peak_gpu_mb=("peak_gpu_mb", "max"))
        .reindex(MODELS)
    )
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(
        resource_summary.index, resource_summary.training_seconds, color=[COLORS[m] for m in MODELS]
    )
    ax.tick_params(axis="x", rotation=20)
    style(ax, "Median outer-fold training cost", "training seconds")
    memory = ax.twinx()
    memory.plot(resource_summary.index, resource_summary.peak_gpu_mb, color="black", marker="o")
    memory.set_ylabel("peak GPU memory (MB)")
    save(fig, "fig17_compute_cost.png")


def write_documents(tables: dict[str, pd.DataFrame]) -> None:
    native = tables["table01_native_metrics.csv"].set_index(["scheme", "model"])
    rows = [
        f"| {model} | {native.loc[('cruise', model), 'rmse']:.2f} | {native.loc[('spatial', model), 'rmse']:.2f} | {native.loc[('whole_lme', model), 'rmse']:.2f} | {native.loc[('forward', model), 'rmse']:.2f} |"
        for model in MODELS
    ]
    core_table = (
        "| model | cruise | spatial | whole-LME | forward |\n"
        "| --- | ---: | ---: | ---: | ---: |\n" + "\n".join(rows)
    )
    learning = tables["table17_learning_curve.csv"]
    curve_rows = []
    for fraction in sorted(learning.cruise_fraction.unique()):
        subset = learning.loc[learning.cruise_fraction.eq(fraction)].set_index("model")
        curve_rows.append(
            f"| {fraction:.0%} | {int(subset.fit_cruises.max()):,} | {subset.loc['v11_random_forest', 'rmse']:.2f} | {subset.loc['lag_catboost', 'rmse']:.2f} | {subset.loc['st96_local', 'rmse']:.2f} |"
        )
    curve_table = (
        "| cruise budget | cruises | RF | lag CatBoost | ST96 |\n"
        "| --- | ---: | ---: | ---: | ---: |\n" + "\n".join(curve_rows)
    )
    captions = "\n\n".join(
        f"### Figure {index}\n\n![{caption}](figures/{name})\n\n{caption}"
        for index, (name, caption) in enumerate(CAPTIONS.items(), 1)
    )
    report = (
        f"""# P2.6 temporal context and coastal connectivity

Issue: #42

Decision: **stop scaling the current GRU/STTransformer/graph family; promote no context candidate and retain the honest v1.x random forest as the point baseline**

## Scientific question and permitted claim

This experiment tests whether six months of local environmental history or four-neighbour water-connected context adds reliable information beyond the strongest Issue #41 point model. It supports a development-stage architecture and data-sufficiency decision. It does not authorize a global product, independent-validation claim, or use 2004-2005 as selection evidence.

## Data, splits, and leakage controls

All candidates use the same 887,811 SOCAT fCO2 rows, 5,143 cruises, 117,324 observed coastal nodes, 788,556 node-months, 54 LMEs, strict outer-fold cross-fitted SSS, and frozen cruise, spatial-block, whole-LME, and forward splits. Six-month histories contain only SST, wind speed, atmospheric xCO2, and background SSS from the observation-independent predictor cube. No lagged fCO2 or observed SSS is used. Locked, external-independent, and provisional 2026 labels remain sealed.

The nominal row count substantially exceeds the independent information: the median cruise contributes 123 rows and the 90th percentile contributes 389. Model selection therefore uses grouped outer validation and cruise-equal metrics.

## Candidate models and training

The matrix compares the exact Issue #41 random-forest predictions, residual lag-feature CatBoost, a two-layer 64-unit GRU, a two-layer d_model=96 local temporal transformer, and the same transformer with water-connected neighbour histories. The graph branch was legitimately triggered: adjacent RF residual correlation is 0.869, while the within-time permutation control is approximately zero.

## Main development results

{core_table}

Lag CatBoost is the only candidate with any forward gain: pooled forward RMSE changes from 47.55 to 47.16 microatmospheres, while bias changes from -15.06 to -5.14. This 0.8 percent gain is far below the frozen five-percent threshold and is accompanied by cruise, spatial, and whole-LME cruise-equal degradations of 7.5, 7.2, and 4.7 percent. Paired cruise bootstraps place every cruise and spatial candidate above the random-forest MSE.

The graph ST96 improves on local ST96 in cruise and forward tests, but remains much worse than the point baseline. It also leaves adjacent residual correlation at 0.919 rather than reducing the RF value of 0.869. Observed adjacent fCO2 gradients average about 7.31 microatmospheres; the direct RF predicts about 2.13, while all candidates preserve fronts poorly. Residual models additionally develop roughly 20-microatmosphere cross-LME jumps because their LME-month background is discontinuous. That background must not be used unchanged in a continuous product.

## Data-volume and model-capacity diagnostic

The nested cruise-label experiment holds outer fold 0 fixed, keeps SSS support fixed, and increases fCO2 training cruises within every LME:

{curve_table}

At ten percent of cruises, lag CatBoost is better than RF in pooled RMSE; as independent cruise coverage grows, RF continues improving and overtakes it. Lag CatBoost plateaus near 35.9 microatmospheres after 40 percent. ST96 is unstable and remains above 41 microatmospheres even at high coverage. More rows alone do not unlock the transformer: current histories lack enough independent process information, and their monthly/local representation does not identify the missing coastal dynamics.

## Decision and limitations

Ordinary cruise and spatial intervals are calibrated by construction, but forward coverage remains inadequate. Lag CatBoost reaches 0.409/0.815 at nominal 50/90 percent, compared with RF 0.323/0.776; neither passes the frozen ranges. Neural models use about 174 thousand parameters and up to roughly 426 MB of allocated GPU memory, yet provide no outer-generalization return.

No candidate passes the complete promotion contract. The current STTransformer, GRU, and graph aggregation families should stop scaling. The allowed conclusion is conditional: under the available SOCAT cruise coverage, frozen monthly predictors, and strict validation protocol, RF is the most suitable interpolation baseline. Future gains require new independent coverage or more process-resolving predictors and a continuous background formulation, rather than a larger network on the same information.

## Figure and table index

{captions}

### Tables

"""
        + "\n".join(
            f"- `{name}`: machine-readable source data used by this report."
            for name in sorted(tables)
        )
        + "\n"
    )
    (ARCHIVE / "REPORT.md").write_text(report, encoding="utf-8")
    (ARCHIVE / "README.md").write_text(
        "# p2_context_connectivity_v2.3\n\nReviewer-ready archive for Issue #42. See `REPORT.md`; every figure has a caption and archived source tables. Large contexts, predictions, and checkpoints remain outside Git.\n",
        encoding="utf-8",
    )
    (ARCHIVE / "CAPTIONS.md").write_text(
        "# Figure captions\n\n"
        + "\n\n".join(f"## {name}\n\n{caption}" for name, caption in CAPTIONS.items())
        + "\n",
        encoding="utf-8",
    )


def normalize_text_files() -> None:
    for pattern in ("*.md", "tables/*.csv"):
        for path in ARCHIVE.glob(pattern):
            content = path.read_text(encoding="utf-8")
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)


def write_manifest() -> None:
    tracked = [
        ARCHIVE / "README.md",
        ARCHIVE / "REPORT.md",
        ARCHIVE / "CAPTIONS.md",
        *sorted(TABLES.glob("*.csv")),
        *sorted(FIGURES.glob("*.png")),
    ]
    source_tables = [path.name for path in sorted(TABLES.glob("*.csv"))]
    context_manifest = json.loads(
        (EXTERNAL / "context_cache" / "manifest.json").read_text(encoding="utf-8")
    )
    manifest = {
        "experiment_id": "p2_context_connectivity_v2.3",
        "training_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "predictor_cube_sha256": context_manifest["cube_sha256"],
        "matchup_sha256": context_manifest["matchup_sha256"],
        "data_manifest_sha256": context_manifest["cube_sha256"],
        "locked_test_opened": False,
        "external_independent_opened": False,
        "provisional_2026_opened": False,
        "evidence_scope": "1993-2025 frozen development observations with cruise, spatial, whole-LME, forward, graph, front, boundary, resource, and nested cruise-label diagnostics",
        "decision": "stop scaling current context family; no model promoted; retain honest v1.x RF as point baseline",
        "figure_source_data": dict.fromkeys(CAPTIONS, source_tables),
        "figure_captions": CAPTIONS,
        "files_sha256": {
            str(path.relative_to(ARCHIVE)).replace("\\", "/"): sha256(path) for path in tracked
        },
        "archive_builder_sha256": sha256(Path(__file__)),
        "analysis_script_sha256": sha256(ANALYSIS),
        "source_artifacts_sha256": {
            "runner": sha256(RUNNER),
            "contract": sha256(CONFIG),
            "model_source": sha256(MODEL_SOURCE),
        },
    }
    (ARCHIVE / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    tables = copy_tables()
    build_figures(tables)
    write_documents(tables)
    normalize_text_files()
    write_manifest()
    print(ARCHIVE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
