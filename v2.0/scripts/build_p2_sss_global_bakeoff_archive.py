"""Build the reviewer-ready archive for the Issue #40 SSS bake-off."""

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
EXTERNAL = Path(r"C:\backup\phd\data\processed\recad_v2_3\p2_sss_global_bakeoff_v2.3")
ARCHIVE = ROOT / "docs/experiment_archive/p2_sss_global_bakeoff_v2.3"
TABLES = ARCHIVE / "tables"
FIGURES = ARCHIVE / "figures"
ANALYSIS = ROOT / "scripts/analyze_p2_sss_global_bakeoff_v2_3.py"
RUNNER = ROOT / "scripts/run_p2_sss_global_bakeoff_v2_3.py"
CONFIG = ROOT / "configs/frozen/sss_global_bakeoff_contract_v2.3.json"
MODEL_SOURCE = ROOT / "src/recad/evaluate/sss_bakeoff.py"
MODELS = ["raw_glorys", "robust_regional", "catboost_global", "gmm_experts", "top2_experts", "st96"]
COLORS = dict(
    zip(MODELS, ["#9aa3ad", "#bb8b55", "#3f78a8", "#2b9a72", "#8557a5", "#c85d62"], strict=True)
)


CAPTIONS = {
    "fig01_rmse_matrix.png": "Figure 1. Held-out SSS RMSE for all six model families and four validation designs; GMM experts lead ordinary cruise, spatial, and forward tests, while raw GLORYS is best under unseen whole-LME transfer.",
    "fig02_relative_effects.png": "Figure 2. Relative RMSE change from the frozen global CatBoost baseline; positive values mean improvement and expose the GMM model's small supported-domain gain alongside its whole-LME penalty.",
    "fig03_forward_periods.png": "Figure 3. Strict forward-time RMSE separated into 2019-2021 and 2022-2025; every learned model remains above the 1.2 PSU late-forward minimum target.",
    "fig04_lme_distribution.png": "Figure 4. Per-LME RMSE distributions for cruise, spatial, and whole-LME validation; pooled gains do not remove the long regional error tail.",
    "fig05_salinity_strata.png": "Figure 5. Cruise-held-out RMSE by observed salinity band; the largest errors occur at 20-30 PSU and the GMM improvement there is modest.",
    "fig06_coast_strata.png": "Figure 6. Cruise-held-out RMSE across distance-to-land bands, showing how performance changes from nearshore cells to the 200-400 km product margin.",
    "fig07_front_strata.png": "Figure 7. Cruise-held-out RMSE across local monthly GLORYS SSS-gradient bands; sharper background fronts remain harder and are diagnosed without using target labels as predictors.",
    "fig08_lme17.png": "Figure 8. LME 17 RMSE under all validation designs; GMM experts do not improve spatial, whole-LME, or forward transfer in this previously identified low-salinity failure region.",
    "fig09_interval_calibration.png": "Figure 9. Leakage-safe empirical 50% and 90% interval coverage; forward 90% coverage falls below nominal for both CatBoost and GMM, so uncertainty is not release-ready.",
    "fig10_risk_coverage.png": "Figure 10. RMSE after retaining observations with the narrowest predicted 90% intervals; uncertainty ranking separates easier from harder cases but does not establish unsampled-grid reliability.",
    "fig11_bootstrap.png": "Figure 11. Cruise-level paired bootstrap MSE contrasts against global CatBoost; GMM gains are statistically consistent in cruise and spatial tests, whereas ST96 and top-two experts degrade.",
    "fig12_gate.png": "Figure 12. Frozen GMM selection gate outcome; failure of the 5% cruise threshold and whole-LME non-degradation rule prevents model promotion despite several positive diagnostics.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style(ax: Any, title: str, ylabel: str = "RMSE (PSU)") -> None:
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
    if len(result) < 15:
        raise RuntimeError(f"expected at least 15 source tables, found {len(result)}")
    return result


def build_figures(t: dict[str, pd.DataFrame]) -> None:
    pooled = t["table01_pooled_metrics.csv"]
    pivot = pooled.pivot(index="model", columns="scheme", values="rmse").loc[
        MODELS, ["cruise", "spatial", "whole_lme", "forward"]
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    image = ax.imshow(pivot, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(4), ["cruise", "spatial", "whole LME", "forward"])
    ax.set_yticks(range(6), MODELS)
    for row in range(6):
        for column in range(4):
            ax.text(
                column, row, f"{pivot.iloc[row, column]:.3f}", ha="center", va="center", fontsize=8
            )
    ax.set_title("Global coastal SSS model-family bake-off", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, label="RMSE (PSU)")
    save(fig, "fig01_rmse_matrix.png")

    effects = t["table13_effects_vs_catboost.csv"]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.15
    x = np.arange(4)
    for offset, model in enumerate([m for m in MODELS if m != "catboost_global"]):
        subset = (
            effects.loc[effects.model.eq(model)]
            .set_index("scheme")
            .loc[["cruise", "spatial", "whole_lme", "forward"]]
        )
        ax.bar(
            x + (offset - 2) * width,
            100 * subset.relative_rmse_improvement,
            width,
            label=model,
            color=COLORS[model],
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, ["cruise", "spatial", "whole LME", "forward"])
    style(ax, "Change relative to global CatBoost", "RMSE improvement (%)")
    ax.legend(fontsize=8, ncol=3)
    save(fig, "fig02_relative_effects.png")

    forward = t["table10_forward_period_metrics.csv"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    periods = ["forward_2019_2021", "forward_2022_2025"]
    x = np.arange(2)
    for offset, model in enumerate(MODELS):
        values = forward.loc[forward.model.eq(model)].set_index("scheme").loc[periods, "rmse"]
        ax.bar(x + (offset - 2.5) * 0.13, values, 0.13, color=COLORS[model], label=model)
    ax.axhline(1.2, color="black", linestyle="--", linewidth=1, label="minimum target")
    ax.set_xticks(x, ["2019-2021", "2022-2025"])
    style(ax, "Strict forward-time performance")
    ax.legend(fontsize=7, ncol=4)
    save(fig, "fig03_forward_periods.png")

    lme = t["table03_lme_metrics.csv"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    for ax, scheme in zip(axes, ["cruise", "spatial", "whole_lme"], strict=True):
        values = [
            lme.loc[lme.scheme.eq(scheme) & lme.model.eq(model), "rmse"].to_numpy()
            for model in ["raw_glorys", "catboost_global", "gmm_experts"]
        ]
        ax.boxplot(values, tick_labels=["raw", "CatBoost", "GMM"], showfliers=False)
        style(ax, scheme.replace("_", " "))
    save(fig, "fig04_lme_distribution.png")

    strata = t["table05_stratified_metrics.csv"]
    for stratum_type, filename, title in (
        ("salinity_band", "fig05_salinity_strata.png", "Observed salinity strata"),
        ("coast_band", "fig06_coast_strata.png", "Distance-to-land strata"),
        ("front_band", "fig07_front_strata.png", "Local GLORYS SSS-gradient strata"),
    ):
        subset = strata.loc[
            strata.scheme.eq("cruise")
            & strata.stratum_type.eq(stratum_type)
            & strata.model.isin(["raw_glorys", "catboost_global", "gmm_experts"])
        ]
        pivot = subset.pivot(index="stratum", columns="model", values="rmse")
        fig, ax = plt.subplots(figsize=(9, 4.6))
        pivot[[c for c in ["raw_glorys", "catboost_global", "gmm_experts"] if c in pivot]].plot(
            kind="bar",
            ax=ax,
            color=[
                COLORS[c] for c in ["raw_glorys", "catboost_global", "gmm_experts"] if c in pivot
            ],
        )
        ax.tick_params(axis="x", rotation=20)
        style(ax, title)
        ax.legend(fontsize=8)
        save(fig, filename)

    lme17 = lme.loc[
        lme.lme_id.eq(17) & lme.model.isin(["raw_glorys", "catboost_global", "gmm_experts"])
    ]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    lme17.pivot(index="scheme", columns="model", values="rmse")[
        ["raw_glorys", "catboost_global", "gmm_experts"]
    ].plot(
        kind="bar",
        ax=ax,
        color=[COLORS[m] for m in ["raw_glorys", "catboost_global", "gmm_experts"]],
    )
    ax.tick_params(axis="x", rotation=20)
    style(ax, "LME 17 remains a failure region")
    save(fig, "fig08_lme17.png")

    intervals = t["table07_interval_metrics.csv"]
    subset = intervals.loc[intervals.model.isin(["catboost_global", "gmm_experts"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for ax, column, target in zip(axes, ["coverage50", "coverage90"], [0.5, 0.9], strict=True):
        subset.pivot(index="scheme", columns="model", values=column)[
            ["catboost_global", "gmm_experts"]
        ].plot(kind="bar", ax=ax, color=[COLORS["catboost_global"], COLORS["gmm_experts"]])
        ax.axhline(target, color="black", linestyle="--")
        ax.tick_params(axis="x", rotation=20)
        style(ax, column, "empirical coverage")
    save(fig, "fig09_interval_calibration.png")

    risk = t["table08_risk_coverage.csv"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for model in ["catboost_global", "gmm_experts"]:
        subset = risk.loc[risk.scheme.eq("forward") & risk.model.eq(model)]
        ax.plot(
            100 * subset.retained_fraction,
            subset.rmse,
            marker="o",
            label=model,
            color=COLORS[model],
        )
    style(ax, "Forward-time risk-coverage", "RMSE (PSU)")
    ax.set_xlabel("retained fraction (%)")
    ax.legend()
    save(fig, "fig10_risk_coverage.png")

    bootstrap = t["table09_paired_bootstrap.csv"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
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
    style(ax, "Paired cruise bootstrap", "candidate - CatBoost MSE")
    save(fig, "fig11_bootstrap.png")

    gate = t["table14_selection_gate.csv"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    values = gate.passed.astype(int)
    ax.barh(gate.gate, values, color=np.where(values.eq(1), "#2b9a72", "#c85d62"))
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 1], ["fail", "pass"])
    style(ax, "GMM promotion gate", "")
    save(fig, "fig12_gate.png")


def write_documents(t: dict[str, pd.DataFrame]) -> None:
    pooled = t["table01_pooled_metrics.csv"].set_index(["scheme", "model"])
    cruise_equal = t["table02_cruise_equal_metrics.csv"].set_index(["scheme", "model"])
    forward = t["table10_forward_period_metrics.csv"].set_index(["scheme", "model"])
    intervals = t["table07_interval_metrics.csv"].set_index(["scheme", "model"])
    gate = t["table14_selection_gate.csv"]
    core = []
    for model in MODELS:
        row = {"model": model}
        for scheme in ["cruise", "spatial", "whole_lme", "forward"]:
            row[scheme] = pooled.loc[(scheme, model), "rmse"]
        core.append(row)
    core_frame = pd.DataFrame(core)
    core_table = (
        "| model | cruise | spatial | whole_lme | forward |\n| --- | ---: | ---: | ---: | ---: |\n"
        + "\n".join(
            f"| {row.model} | {row.cruise:.3f} | {row.spatial:.3f} | {row.whole_lme:.3f} | {row.forward:.3f} |"
            for row in core_frame.itertuples()
        )
    )
    captions = "\n\n".join(
        f"### Figure {index}\n\n![{caption}](figures/{name})\n\n{caption}"
        for index, (name, caption) in enumerate(CAPTIONS.items(), 1)
    )
    report = (
        f"""# P2.4 global coastal SSS model-family bake-off

Issue: #40

Decision: **do not promote a new global SSS model; retain global CatBoost as the frozen baseline and carry GMM experts only as a support-aware challenger**

## Scientific question and permitted claim

This experiment asks whether regional correction, soft environmental experts, or a single-target ST96 transformer improves the frozen Issue #54 background-SSS residual baseline under the same observations, predictors, splits, and output coverage. It supports a development-stage architecture decision. It does not provide independent validation, authorize a global product, or establish accuracy in unobserved coastal regimes.

## Data, splits, and leakage controls

All candidates used the same 750,040 SOCAT SSS records from 4,615 cruises, 54 LMEs, and 1993-2025. Validation used five-fold held-cruise, spatial-block, strict whole-LME, and forward-time designs. Cruises touching a held spatial block or LME were removed from fitting. Background-field gaps used the same-fold direct A2 fallback. Empirical uncertainty widths were fitted from other folds; forward intervals used only cruise OOF records through 2018. The old Issue #25 locked set, 2026 provisional rows, and external-independent labels remained unopened.

Local front strength was computed from the monthly GLORYS SSS difference to four adjacent water nodes divided by grid distance. Distance-to-land bands diagnose cross-shore position; no target value enters either diagnostic predictor.

## Candidate models and training

The bake-off compared raw GLORYS, fold-fitted robust regional correction, frozen global CatBoost residual correction, eight probabilistic environmental/GMM experts shrunk 50% toward global CatBoost, eight top-two soft-gated experts, and a grouped-token ST96 transformer. All learned models used the same 16 retained Issue #54 inputs and identical outer folds. Training produced 96 prediction partitions and 32 learned checkpoints.

## Main development results

{core_table}

GMM experts achieved pooled RMSE 0.864 PSU in cruise validation and 0.918 PSU in spatial validation, compared with 0.907 and 0.973 PSU for global CatBoost. The pooled changes are 4.76% and 5.63%; 2,000 cruise-level paired bootstrap replicates placed both MSE contrasts below zero. Under the frozen primary cruise-equal metric, cruise RMSE changed from {cruise_equal.loc[("cruise", "catboost_global"), "cruise_equal_rmse"]:.3f} to {cruise_equal.loc[("cruise", "gmm_experts"), "cruise_equal_rmse"]:.3f} PSU, a 4.61% gain that remains below the 5% gate. Strict forward RMSE improved from 1.446 to 1.422 PSU, but the 2022-2025 value remained {forward.loc[("forward_2022_2025", "gmm_experts"), "rmse"]:.3f} PSU, above the 1.2 PSU minimum target.

The decisive failure is geographic transfer. Whole-LME RMSE rose from 2.470 PSU for CatBoost to 2.620 PSU for GMM, while uncorrected GLORYS was 1.740 PSU. In LME 17, GMM RMSE was 5.584 spatially, 6.136 for whole-LME, and 7.104 PSU forward; each is slightly worse than CatBoost. The 20-30 PSU cruise stratum also remained difficult at 3.360 PSU. These failures show that denser-region gains cannot justify unconditional global correction.

Intervals are calibrated in ordinary OOF settings, but forward 90% coverage was only {intervals.loc[("forward", "gmm_experts"), "coverage90"]:.3f} for GMM and {intervals.loc[("forward", "catboost_global"), "coverage90"]:.3f} for CatBoost. Width-based risk ranking is informative, yet it does not prove calibration on unsampled grids.

## Decision and limitations

GMM fails two frozen promotion gates: cruise improvement is below 5%, and whole-LME performance degrades. It also fails the specific no-degradation requirement in LME 17. ST96 and top-two experts are worse than global CatBoost in ordinary tests and collapse more severely in whole-LME transfer. Model size is therefore not the present bottleneck.

Issue #40 closes with no unconditional global SSS production model selected. Global CatBoost remains the reproducible baseline for the next reliability stage; GMM experts may be evaluated as a support-aware challenger only where observation support is explicit. Outside qualified support, the system must retain the background or suppress the correction. A new blind global audit and external-independent observations remain required before release.

Gate result: {int(gate.passed.sum())}/{len(gate)} checks passed. Locked test opened: **no**. External independent validation opened: **no**.

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
        "# p2_sss_global_bakeoff_v2.3\n\nReviewer-ready archive for Issue #40. See `REPORT.md`; all figures have source CSVs and captions. Large prediction partitions and checkpoints remain outside Git.\n",
        encoding="utf-8",
    )
    (ARCHIVE / "CAPTIONS.md").write_text(
        "# Figure captions\n\n"
        + "\n\n".join(f"## {name}\n\n{caption}" for name, caption in CAPTIONS.items())
        + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    tracked = [
        ARCHIVE / "README.md",
        ARCHIVE / "REPORT.md",
        ARCHIVE / "CAPTIONS.md",
        *sorted(TABLES.glob("*.csv")),
        *sorted(FIGURES.glob("*.png")),
    ]
    all_tables = [path.name for path in sorted(TABLES.glob("*.csv"))]
    manifest = {
        "experiment_id": "p2_sss_global_bakeoff_v2.3",
        "training_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "data_manifest_sha256": "0f4e9f37ddd65d5abb0ebd92715a8521703a7778375d05c1a6055254203d0401",
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "1993-2025 frozen development observations; cruise, spatial, whole-LME, and forward-time OOF only",
        "decision": "no model promoted; retain CatBoost as baseline and GMM as support-aware challenger",
        "figure_source_data": dict.fromkeys(CAPTIONS, all_tables),
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


def normalize_text_files() -> None:
    """Freeze portable LF bytes before recording archive hashes."""
    for pattern in ("*.md", "tables/*.csv"):
        for path in ARCHIVE.glob(pattern):
            content = path.read_text(encoding="utf-8")
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)


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
