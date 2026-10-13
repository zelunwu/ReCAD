"""Build the reviewer-ready archive for the Issue #41 fCO2 bake-off."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path(r"C:\backup\phd\data\processed\recad_v2_3\p2_fco2_global_bakeoff_v2.3")
ARCHIVE = ROOT / "docs/experiment_archive/p2_fco2_global_bakeoff_v2.3"
TABLES = ARCHIVE / "tables"
FIGURES = ARCHIVE / "figures"
RUNNER = ROOT / "scripts/run_p2_fco2_global_bakeoff_v2_3.py"
ANALYSIS = ROOT / "scripts/analyze_p2_fco2_global_bakeoff_v2_3.py"
CONFIG = ROOT / "configs/frozen/fco2_global_bakeoff_contract_v2.3.json"
MODEL_SOURCE = ROOT / "src/recad/evaluate/fco2_bakeoff.py"
MODELS = [
    "seasonal_trend",
    "v11_random_forest",
    "catboost_direct",
    "catboost_residual",
    "catboost_residual_sss",
    "lme_adapters",
    "gmm_experts",
]
COLORS = dict(
    zip(
        MODELS,
        ["#a9afb6", "#3f78a8", "#8c6bb1", "#bd8655", "#d05b62", "#4d9c73", "#207f7c"],
        strict=True,
    )
)

CAPTIONS = {
    "fig01_native_rmse_matrix.png": "Figure 1. Native held-out fCO2 RMSE for seven model families and four validation designs; the honest v1.x random forest leads cruise and spatial tests, while no model leads every transfer regime.",
    "fig02_cruise_equal_rmse.png": "Figure 2. Cruise-equal RMSE gives each cruise equal influence and confirms that none of the modern candidates beats the honest random-forest baseline by the frozen five-percent margin.",
    "fig03_common_grid_rmse.png": "Figure 3. Monthly common-grid RMSE preserves the same ranking as native records, showing that repeated sampling within grid months does not explain the random forest advantage.",
    "fig04_controlled_factors.png": "Figure 4. Controlled CatBoost effects isolate target residualization from adding cross-fitted SSS; SSS gives small consistent gains, while residualization alone does not improve ordinary held-cruise skill.",
    "fig05_forward_periods.png": "Figure 5. Strict forward-time RMSE in 2019-2021 and 2022-2025; residual sharing reduces the late-period error and bias but every candidate remains above the 40 microatmosphere minimum target.",
    "fig06_lme_distributions.png": "Figure 6. Per-LME RMSE distributions reveal a long geographic error tail and the tradeoff between random-forest interpolation and residual-model temporal transfer.",
    "fig07_failure_map.png": "Figure 7. Whole-LME GMM-minus-random-forest RMSE difference at observation-weighted LME centers; positive red points mark regions where soft experts transfer less reliably.",
    "fig08_caribbean.png": "Figure 8. Caribbean Sea LME 12 RMSE against the prior Issue 26 regional candidate; no global candidate preserves its cruise, spatial, and forward skill simultaneously.",
    "fig09_qualified_lmes.png": "Figure 9. Development-only LME qualification counts; GMM meets simple absolute/positive-skill thresholds in more LMEs, but this does not override failed global, calibration, and paired-improvement gates.",
    "fig10_extremes.png": "Figure 10. Cruise-held-out RMSE in low, central, and high observed-fCO2 bands; extreme values remain substantially harder than the central distribution.",
    "fig11_annual_bias.png": "Figure 11. Annual cruise-OOF bias for the principal models; the random forest underestimates the observed temporal rise more strongly than residual-sharing candidates.",
    "fig12_interval_coverage.png": "Figure 12. Leakage-safe 50 and 90 percent interval coverage; forward-time coverage collapses below the frozen calibration ranges for every leading model.",
    "fig13_risk_coverage.png": "Figure 13. Forward RMSE after retaining cases with the narrowest empirical 90 percent intervals; ranking risk helps triage cases but does not establish unsampled-grid reliability.",
    "fig14_bootstrap.png": "Figure 14. Cruise-level paired MSE bootstrap contrasts against the honest v1.x random forest; every modern candidate has a positive confidence interval in cruise and spatial validation.",
    "fig15_selection_gate.png": "Figure 15. Frozen selection-gate totals for each modern candidate; none passes more than three of ten checks, so no architecture is promoted.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
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


def build_figures(t: dict[str, pd.DataFrame]) -> None:
    native = t["table01_native_metrics.csv"]
    pivot = native.pivot(index="model", columns="scheme", values="rmse").loc[
        MODELS, ["cruise", "spatial", "whole_lme", "forward"]
    ]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    image = ax.imshow(pivot, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(4), ["cruise", "spatial", "whole LME", "forward"])
    ax.set_yticks(range(len(MODELS)), MODELS)
    for row in range(len(MODELS)):
        for column in range(4):
            ax.text(
                column, row, f"{pivot.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8
            )
    ax.set_title("Global coastal fCO2 model-family bake-off", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, label="RMSE (microatmospheres)")
    save(fig, "fig01_native_rmse_matrix.png")

    equal = t["table02_cruise_equal_metrics.csv"]
    grouped_bars(
        equal.pivot(index="scheme", columns="model", values="cruise_equal_rmse"),
        [
            "v11_random_forest",
            "catboost_direct",
            "catboost_residual_sss",
            "lme_adapters",
            "gmm_experts",
        ],
        "Cruise-equal transfer performance",
        "fig02_cruise_equal_rmse.png",
    )
    common = t["table03_common_grid_metrics.csv"]
    grouped_bars(
        common.pivot(index="scheme", columns="model", values="rmse"),
        ["v11_random_forest", "catboost_direct", "lme_adapters", "gmm_experts"],
        "Monthly common-grid performance",
        "fig03_common_grid_rmse.png",
    )

    factor_models = ["catboost_direct", "catboost_residual", "catboost_residual_sss"]
    factor = native.loc[native.model.isin(factor_models)].pivot(
        index="scheme", columns="model", values="rmse"
    )
    grouped_bars(
        factor, factor_models, "Controlled target and SSS factors", "fig04_controlled_factors.png"
    )

    forward = t["table22_forward_period_metrics.csv"]
    grouped_bars(
        forward.pivot(index="scheme", columns="model", values="rmse"),
        ["v11_random_forest", "catboost_residual_sss", "lme_adapters", "gmm_experts"],
        "Strict forward-time periods",
        "fig05_forward_periods.png",
        target=40,
    )

    lme = t["table04_lme_metrics.csv"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    for ax, scheme in zip(axes, ["cruise", "spatial", "whole_lme"], strict=True):
        models = ["v11_random_forest", "catboost_direct", "gmm_experts"]
        values = [
            lme.loc[lme.scheme.eq(scheme) & lme.model.eq(model), "rmse"].to_numpy()
            for model in models
        ]
        ax.boxplot(values, tick_labels=["RF", "CatBoost", "GMM"], showfliers=False)
        style(ax, scheme.replace("_", " "))
    save(fig, "fig06_lme_distributions.png")

    centers = t["table19_lme_centers.csv"]
    whole = lme.loc[
        lme.scheme.eq("whole_lme") & lme.model.isin(["v11_random_forest", "gmm_experts"])
    ].pivot(index="lme_id", columns="model", values="rmse")
    mapped = centers.merge(
        (whole.gmm_experts - whole.v11_random_forest).rename("delta_rmse"), on="lme_id"
    )
    fig, ax = plt.subplots(figsize=(11, 4.8))
    scale = np.nanmax(np.abs(mapped.delta_rmse))
    scatter = ax.scatter(
        mapped.longitude,
        mapped.latitude,
        c=mapped.delta_rmse,
        cmap="RdBu_r",
        vmin=-scale,
        vmax=scale,
        s=30 + 3 * np.sqrt(mapped.cruises),
        edgecolor="black",
        linewidth=0.25,
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-70, 80)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.grid(alpha=0.2)
    ax.set_title("Whole-LME transfer: GMM minus random forest", loc="left", fontweight="bold")
    fig.colorbar(scatter, ax=ax, label="delta RMSE")
    save(fig, "fig07_failure_map.png")

    caribbean = lme.loc[
        lme.lme_id.eq(12)
        & lme.model.isin(["v11_random_forest", "catboost_direct", "lme_adapters", "gmm_experts"])
    ]
    caribbean_pivot = caribbean.pivot(index="scheme", columns="model", values="rmse").loc[
        ["cruise", "spatial", "forward"]
    ]
    fig, ax = plt.subplots(figsize=(9, 4.7))
    caribbean_pivot.plot(
        kind="bar", ax=ax, color=[COLORS[column] for column in caribbean_pivot.columns]
    )
    prior = pd.Series({"cruise": 9.538, "spatial": 10.033, "forward": 13.571})
    ax.plot(
        range(3),
        prior.loc[caribbean_pivot.index],
        color="black",
        marker="o",
        label="Issue 26 candidate",
    )
    ax.tick_params(axis="x", rotation=15)
    style(ax, "Caribbean Sea (LME 12)")
    ax.legend(fontsize=8)
    save(fig, "fig08_caribbean.png")

    qualification = t["table14_lme_qualification.csv"]
    counts = qualification.groupby("model").qualified.sum().reindex(MODELS)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(counts.index, counts, color=[COLORS[model] for model in counts.index])
    ax.tick_params(axis="x", rotation=25)
    style(ax, "Development-only qualified LME counts", "qualified LMEs")
    save(fig, "fig09_qualified_lmes.png")

    extremes = t["table09_extreme_metrics.csv"]
    extreme = extremes.loc[
        extremes.scheme.eq("cruise")
        & extremes.model.isin(["v11_random_forest", "gmm_experts", "lme_adapters"])
    ]
    grouped_bars(
        extreme.pivot(index="extreme_band", columns="model", values="rmse"),
        ["v11_random_forest", "lme_adapters", "gmm_experts"],
        "Cruise-held-out fCO2 extremes",
        "fig10_extremes.png",
    )

    annual = t["table08_annual_diagnostics.csv"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for model in ["v11_random_forest", "lme_adapters", "gmm_experts"]:
        subset = annual.loc[annual.scheme.eq("cruise") & annual.model.eq(model)]
        ax.plot(subset.yr, subset.bias, marker="o", markersize=3, label=model, color=COLORS[model])
    ax.axhline(0, color="black", linewidth=0.8)
    style(ax, "Annual cruise-OOF bias", "bias (microatmospheres)")
    ax.legend(fontsize=8)
    save(fig, "fig11_annual_bias.png")

    intervals = t["table10_interval_metrics.csv"]
    selected = intervals.loc[
        intervals.model.isin(["v11_random_forest", "lme_adapters", "gmm_experts"])
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, column, nominal in zip(axes, ["coverage50", "coverage90"], [0.5, 0.9], strict=True):
        selected.pivot(index="scheme", columns="model", values=column)[
            ["v11_random_forest", "lme_adapters", "gmm_experts"]
        ].plot(
            kind="bar",
            ax=ax,
            color=[COLORS["v11_random_forest"], COLORS["lme_adapters"], COLORS["gmm_experts"]],
        )
        ax.axhline(nominal, color="black", linestyle="--")
        ax.tick_params(axis="x", rotation=20)
        style(ax, column, "empirical coverage")
    save(fig, "fig12_interval_coverage.png")

    risk = t["table11_risk_coverage.csv"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for model in ["v11_random_forest", "lme_adapters", "gmm_experts"]:
        subset = risk.loc[risk.scheme.eq("forward") & risk.model.eq(model)]
        ax.plot(
            100 * subset.retained_fraction,
            subset.rmse,
            marker="o",
            label=model,
            color=COLORS[model],
        )
    style(ax, "Forward risk-coverage")
    ax.set_xlabel("retained fraction (%)")
    ax.legend(fontsize=8)
    save(fig, "fig13_risk_coverage.png")

    bootstrap = t["table12_paired_bootstrap.csv"]
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{row.candidate}\n{row.scheme}" for row in bootstrap.itertuples()]
    mean = bootstrap.mean_delta_mse.to_numpy()
    ax.errorbar(
        np.arange(len(mean)),
        mean,
        yerr=[mean - bootstrap.ci_low_delta_mse, bootstrap.ci_high_delta_mse - mean],
        fmt="o",
        capsize=4,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    style(ax, "Paired cruise bootstrap", "candidate - random forest MSE")
    save(fig, "fig14_bootstrap.png")

    gate = t["table17_selection_gate.csv"]
    gate_count = gate.groupby("candidate").passed.sum().sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(gate_count.index, gate_count, color=[COLORS[model] for model in gate_count.index])
    ax.axvline(10, color="black", linestyle="--", label="all gates")
    style(ax, "Frozen promotion gate", "passed checks of 10")
    ax.legend()
    save(fig, "fig15_selection_gate.png")


def write_documents(t: dict[str, pd.DataFrame]) -> None:
    native = t["table01_native_metrics.csv"].set_index(["scheme", "model"])
    equal = t["table02_cruise_equal_metrics.csv"].set_index(["scheme", "model"])
    intervals = t["table10_interval_metrics.csv"].set_index(["scheme", "model"])
    trends = t["table21_trend_slopes.csv"].set_index(["scheme", "model"])
    forward = t["table22_forward_period_metrics.csv"].set_index(["scheme", "model"])
    rows = []
    for model in MODELS:
        rows.append(
            f"| {model} | {native.loc[('cruise', model), 'rmse']:.2f} | {native.loc[('spatial', model), 'rmse']:.2f} | {native.loc[('whole_lme', model), 'rmse']:.2f} | {native.loc[('forward', model), 'rmse']:.2f} |"
        )
    core_table = (
        "| model | cruise | spatial | whole-LME | forward |\n| --- | ---: | ---: | ---: | ---: |\n"
        + "\n".join(rows)
    )
    captions = "\n\n".join(
        f"### Figure {index}\n\n![{caption}](figures/{name})\n\n{caption}"
        for index, (name, caption) in enumerate(CAPTIONS.items(), 1)
    )
    report = (
        f"""# P2.5 global coastal fCO2 baseline and regional-sharing bake-off

Issue: #41

Decision: **promote no global fCO2 model; retain the honest v1.x random forest as the strict interpolation baseline and carry residual-sharing models only into the temporal/connectivity experiment**

## Scientific question and permitted claim

This experiment asks whether target residualization, strict cross-fitted SSS, fixed-LME adapters, or probabilistic environmental experts improve on an honestly retrained v1.x random forest and the frozen Issue #54 CatBoost. It supports a development-stage architecture decision. It does not authorize global product release, claim independent validation, or use the historical 2004-2005 comparison as model-selection evidence.

## Data, splits, and leakage controls

All candidates used the same 887,811 SOCAT fCO2 records from 5,143 cruises, 54 LMEs, and 1993-2025. Five-fold cruise, spatial-block, strict whole-LME, and forward-time validation were preserved from Issue #54. Cruises touching a held spatial block or LME were removed from training.

Observed SOCAT salinity was forbidden as an fCO2 input. Within every outer fold, SSS for fCO2 training rows was generated by two-fold cruise cross-fitting using only that outer fold's training subset; held-row SSS used a model fitted to all outer-training SSS labels. This produced 16 fold-specific SSS feature partitions. The old locked labels, external-independent data, and 2026 provisional records remained sealed.

## Candidate models and training

The matrix contained a fold-fitted seasonal/trend background, an honest 96-tree v1.x random forest, the exact Issue #54 direct CatBoost, CatBoost residualized around the seasonal/trend background, the same residual model plus cross-fitted SSS, shrunk fixed-LME/month adapters, and eight soft GMM/environmental residual experts. The 112 final prediction partitions cover every model and outer fold. Large predictions and 176 model checkpoints remain outside Git.

## Main development results

{core_table}

The honest random forest is strongest in ordinary transfer: native cruise/spatial RMSE is 33.96/37.45 microatmospheres, with cruise-equal RMSE {equal.loc[("cruise", "v11_random_forest"), "cruise_equal_rmse"]:.2f}/{equal.loc[("spatial", "v11_random_forest"), "cruise_equal_rmse"]:.2f}. Every modern candidate has a positive cruise- and spatial-level paired MSE difference relative to it; all 95 percent bootstrap intervals exclude zero in the wrong direction. No candidate supplies the required five-percent improvement.

The advantage reverses partly in time. LME adapters and GMM experts reduce pooled forward RMSE from 47.55 to 46.21 and 46.45 microatmospheres. In 2022-2025, GMM reaches {forward.loc[("forward_2022_2025", "gmm_experts"), "rmse"]:.2f}, compared with {forward.loc[("forward_2022_2025", "v11_random_forest"), "rmse"]:.2f} for the random forest. The random forest's cruise annual bias slope is {trends.loc[("cruise", "v11_random_forest"), "bias_slope_uatm_per_year"]:.3f} microatmospheres per year, while GMM is {trends.loc[("cruise", "gmm_experts"), "bias_slope_uatm_per_year"]:.3f}; explicit residual/trend structure better follows the temporal rise. Nevertheless, all late-forward RMSE values exceed the frozen 40-microatmosphere minimum.

Cross-fitted SSS consistently but modestly improves residual CatBoost: cruise/spatial/whole-LME/forward RMSE changes from 36.93/39.20/57.43/47.46 to 36.45/38.58/57.28/46.57. This confirms useful SSS information without supporting observed-SSS leakage or making SSS a sufficient solution.

No model preserves the prior Caribbean candidate's 9.54/10.03/13.57 cruise/spatial/forward RMSE simultaneously. Simple development thresholds qualify 22 LMEs for GMM versus 18 for the random forest, including four new IDs, but forward interval coverage for GMM is only {intervals.loc[("forward", "gmm_experts"), "coverage90"]:.3f} at nominal 90 percent and {intervals.loc[("forward", "gmm_experts"), "coverage50"]:.3f} at nominal 50 percent. Those counts therefore cannot be interpreted as trustworthy product coverage.

## Decision and limitations

No modern candidate passes the frozen promotion contract. The honest v1.x random forest remains the strict interpolation baseline, not a release candidate: it still misses the 30-microatmosphere cruise and 40-microatmosphere forward ceilings and develops a large late negative bias. Residual/GMM sharing is retained only as the temporal-transfer branch for Issue #42 because it improves trend and forward behavior while losing ordinary and whole-LME skill.

The next experiment must test temporal context and coastal connectivity against these complementary baselines. A valid successor must preserve random-forest cruise/spatial skill while recovering the residual models' trend and late-forward behavior. Locked and external-independent labels must remain sealed until one architecture passes the complete development and calibration gate.

## Figure and table index

{captions}

### Tables

"""
        + "\n".join(
            f"- `{name}`: machine-readable source data used by this report." for name in sorted(t)
        )
        + "\n"
    )
    (ARCHIVE / "REPORT.md").write_text(report, encoding="utf-8")
    (ARCHIVE / "README.md").write_text(
        "# p2_fco2_global_bakeoff_v2.3\n\nReviewer-ready archive for Issue #41. See `REPORT.md`; all figures have source CSVs and captions. Large prediction partitions, SSS stacks, and checkpoints remain outside Git.\n",
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
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)


def write_manifest() -> None:
    tracked = [
        ARCHIVE / "README.md",
        ARCHIVE / "REPORT.md",
        ARCHIVE / "CAPTIONS.md",
        *sorted(TABLES.glob("*.csv")),
        *sorted(FIGURES.glob("*.png")),
    ]
    source_tables = [path.name for path in sorted(TABLES.glob("*.csv"))]
    manifest = {
        "experiment_id": "p2_fco2_global_bakeoff_v2.3",
        "training_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "data_manifest_sha256": "0f4e9f37ddd65d5abb0ebd92715a8521703a7778375d05c1a6055254203d0401",
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "1993-2025 frozen development observations with cruise, spatial, whole-LME, and forward-time OOF evaluation",
        "decision": "no model promoted; retain honest v1.x RF as interpolation baseline and residual sharing as temporal challenger",
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
    (ARCHIVE / "archive_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


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
