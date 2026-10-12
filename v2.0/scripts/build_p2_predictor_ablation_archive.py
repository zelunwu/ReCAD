"""Build the reviewer archive and frozen decision for Issue #54."""

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
EXTERNAL = Path(r"C:\backup\phd\data\processed\recad_v2_3\p2_predictor_ablation_v2.3")
ARCHIVE = ROOT / "docs/experiment_archive/p2_predictor_ablation_v2.3"
TABLES = ARCHIVE / "tables"
FIGURES = ARCHIVE / "figures"
CONFIG = ROOT / "configs/frozen/predictor_ablation_decision_v2.3.json"
RUNNER = ROOT / "scripts/run_p2_predictor_ablation_v2_3.py"

COLORS = {
    "A0_geo_time": "#b9c2cc",
    "A1_geometry": "#8fa8bd",
    "A2_environment": "#4e83ac",
    "A3_background_sss": "#2e9b70",
    "A4_adt": "#bd8c55",
    "A5_sss_adt": "#7859a6",
    "A6_missing_robust": "#c75b62",
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


def style(ax: Any, title: str, ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)


def read_external(name: str) -> pd.DataFrame:
    return pd.read_csv(EXTERNAL / "tables" / name)


def copy_and_derive_tables() -> dict[str, pd.DataFrame]:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    result: dict[str, pd.DataFrame] = {}
    for path in sorted((EXTERNAL / "tables").glob("*.csv")):
        destination = TABLES / path.name
        shutil.copyfile(path, destination)
        result[path.name] = pd.read_csv(destination)

    log = pd.read_csv(EXTERNAL / "training_log_current.csv")
    log["target_mode"] = log.target_mode.fillna("direct")
    log.to_csv(TABLES / "table10_training_log.csv", index=False)
    result["table10_training_log.csv"] = log

    matchup = pd.read_parquet(
        EXTERNAL / "socat_predictor_matchup_v2.3.parquet",
        columns=[
            "group_key",
            "yr",
            "lme_id",
            "salinity",
            "fco2",
            "sss_bg_available",
            "adt_available",
        ],
    )
    scope = []
    for target, column in (("SSS", "salinity"), ("fCO2", "fco2")):
        rows = matchup.loc[matchup[column].notna()]
        scope.append(
            {
                "target": target,
                "rows": len(rows),
                "cruises": rows.group_key.nunique(),
                "lmes": rows.lme_id.nunique(),
                "first_year": rows.yr.min(),
                "last_year": rows.yr.max(),
                "sss_background_available_fraction": rows.sss_bg_available.mean(),
                "adt_available_fraction": rows.adt_available.mean(),
            }
        )
    scope_frame = pd.DataFrame(scope)
    scope_frame.to_csv(TABLES / "table11_data_scope.csv", index=False)
    result["table11_data_scope.csv"] = scope_frame

    lme = result["table02_lme_metrics.csv"]
    lme_summary = (
        lme.groupby(["target", "scheme", "variant"], observed=True)
        .agg(
            lme_count=("lme_id", "nunique"),
            macro_lme_rmse=("rmse", "mean"),
            median_lme_rmse=("rmse", "median"),
            worst_lme_rmse=("rmse", "max"),
        )
        .reset_index()
    )
    lme_summary.to_csv(TABLES / "table12_lme_macro_worst.csv", index=False)
    result["table12_lme_macro_worst.csv"] = lme_summary

    gates = pd.DataFrame(
        [
            {"gate": "median relative RMSE improvement", "threshold": ">= 0.5%"},
            {"gate": "cruise and spatial RMSE", "threshold": "both improve"},
            {
                "gate": "whole-LME and forward periods",
                "threshold": "no degradation over 1%",
            },
            {"gate": "strict common support", "threshold": "cruise and spatial improve"},
            {
                "gate": "paired cruise bootstrap",
                "threshold": "95% upper CI of delta MSE < 0",
            },
            {"gate": "coverage", "threshold": "no row removal for optional fields"},
        ]
    )
    gates.to_csv(TABLES / "table13_acceptance_gates.csv", index=False)
    result["table13_acceptance_gates.csv"] = gates

    excluded = pd.DataFrame(
        [
            {
                "candidate": name,
                "reason": reason,
                "status": "not tested in Issue 54",
            }
            for name, reason in (
                ("chlorophyll-a", "Issue 52 did not qualify a frozen global monthly source"),
                ("mixed-layer depth", "Issue 52 did not qualify a frozen global monthly source"),
                ("river/freshwater proxy", "no deployable global monthly field was frozen"),
                ("upwelling/circulation proxy", "no deployable global monthly field was frozen"),
            )
        ]
    )
    excluded.to_csv(TABLES / "table14_excluded_candidates.csv", index=False)
    result["table14_excluded_candidates.csv"] = excluded

    annual_rows = []
    selected = {"SSS": "A3_background_sss", "fCO2": "A2_environment"}
    for target, variant in selected.items():
        paths = sorted(
            (EXTERNAL / "fold_predictions").glob(f"{target}_cruise_fold*_{variant}.parquet")
        )
        frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        frame["error"] = frame.prediction - frame.truth
        annual = frame.groupby("yr", observed=True).agg(
            rows=("truth", "size"),
            cruises=("group_key", "nunique"),
            bias=("error", "mean"),
            mse=("error", lambda values: float(np.mean(values**2))),
        )
        annual["rmse"] = np.sqrt(annual.mse)
        annual["bias_jump"] = annual.bias.diff()
        annual["target"] = target
        annual["variant"] = variant
        annual_rows.extend(annual.reset_index().to_dict("records"))
    annual_frame = pd.DataFrame(annual_rows)
    annual_frame.to_csv(TABLES / "table15_annual_transition_diagnostic.csv", index=False)
    result["table15_annual_transition_diagnostic.csv"] = annual_frame
    return result


def build_figures(tables: dict[str, pd.DataFrame]) -> None:
    metrics = tables["table01_pooled_metrics.csv"]
    variants = tables["table08_variant_definitions.csv"].variant.tolist()
    schemes = ["cruise", "spatial", "whole_lme", "forward_2019_2021", "forward_2022_2025"]
    for target, filename, unit in (
        ("SSS", "fig01_sss_rmse_matrix.png", "RMSE (PSU)"),
        ("fCO2", "fig02_fco2_rmse_matrix.png", "RMSE (µatm)"),
    ):
        subset = metrics.loc[(metrics.target == target) & (metrics.support == "all")]
        pivot = subset.pivot(index="variant", columns="scheme", values="rmse").loc[
            variants, schemes
        ]
        normalized = pivot.div(pivot.loc["A2_environment"], axis=1)
        fig, ax = plt.subplots(figsize=(10, 5.2))
        image = ax.imshow(normalized, cmap="RdYlGn_r", vmin=0.85, vmax=1.15, aspect="auto")
        ax.set_xticks(range(len(schemes)), schemes, rotation=25, ha="right")
        ax.set_yticks(range(len(variants)), variants)
        for row in range(len(variants)):
            for column in range(len(schemes)):
                ax.text(
                    column,
                    row,
                    f"{pivot.iloc[row, column]:.2f}"
                    if target == "fCO2"
                    else f"{pivot.iloc[row, column]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        ax.set_title(
            f"{target}: held-out {unit} (color relative to A2)", loc="left", fontweight="bold"
        )
        fig.colorbar(image, ax=ax, label="RMSE / A2 RMSE")
        save(fig, filename)

    effects = tables["table06_effects.csv"]
    short_scheme = {
        "cruise": "CR",
        "spatial": "SP",
        "whole_lme": "LME",
        "forward_2019_2021": "F19",
        "forward_2022_2025": "F22",
    }
    short_group = {
        "background_sss": "BG-SSS",
        "adt": "ADT",
        "missing_robustness": "dropout",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 7.5))
    for ax, target in zip(axes, ("SSS", "fCO2"), strict=True):
        subset = effects.loc[effects.target == target]
        labels = [
            f"{short_group[row.group]} / {short_scheme[row.scheme]}" for row in subset.itertuples()
        ]
        values = subset.relative_rmse_improvement.to_numpy() * 100
        ax.barh(labels, values, color=np.where(values >= 0, "#2e9b70", "#c75b62"))
        ax.axvline(0, color="black", linewidth=0.8)
        style(ax, target, "")
        ax.set_xlabel("RMSE improvement (%)")
        ax.tick_params(axis="y", labelsize=8)
    save(fig, "fig03_optional_effects.png")

    primary = effects.loc[effects.scheme.isin(["cruise", "spatial"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, target in zip(axes, ("SSS", "fCO2"), strict=True):
        subset = primary.loc[primary.target == target]
        x = np.arange(len(subset))
        ax.bar(x - 0.18, subset.relative_rmse_improvement * 100, 0.36, label="all support")
        ax.bar(
            x + 0.18,
            subset.common_relative_rmse_improvement * 100,
            0.36,
            label="strict common",
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(
            x,
            [f"{short_group[r.group]}\n{short_scheme[r.scheme]}" for r in subset.itertuples()],
            rotation=20,
        )
        style(ax, target, "RMSE improvement (%)")
        ax.legend(fontsize=8)
    save(fig, "fig04_common_support.png")

    bootstrap = tables["table05_paired_bootstrap.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, target in zip(axes, ("SSS", "fCO2"), strict=True):
        subset = bootstrap.loc[bootstrap.target == target]
        x = np.arange(len(subset))
        mean = subset.mean_delta_mse.to_numpy()
        lower = subset.ci_low_delta_mse.to_numpy()
        upper = subset.ci_high_delta_mse.to_numpy()
        ax.errorbar(x, mean, yerr=[mean - lower, upper - mean], fmt="o", capsize=4)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, [f"{r.group}\n{r.scheme}" for r in subset.itertuples()], rotation=25)
        style(ax, target, "candidate - reference MSE")
    save(fig, "fig05_bootstrap_intervals.png")

    lme = tables["table02_lme_metrics.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, target in zip(axes, ("SSS", "fCO2"), strict=True):
        keep = (
            ["A2_environment", "A3_background_sss"]
            if target == "SSS"
            else ["A0_geo_time", "A2_environment"]
        )
        values = [
            lme.loc[
                (lme.target == target) & (lme.scheme == "whole_lme") & (lme.variant == variant),
                "rmse",
            ].to_numpy()
            for variant in keep
        ]
        ax.boxplot(values, tick_labels=keep, showfliers=False)
        ax.tick_params(axis="x", rotation=15)
        style(ax, target, "LME RMSE")
    save(fig, "fig06_lme_distributions.png")

    worst = tables["table12_lme_macro_worst.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, target, keep in zip(
        axes,
        ("SSS", "fCO2"),
        (
            ["A2_environment", "A3_background_sss"],
            ["A0_geo_time", "A2_environment"],
        ),
        strict=True,
    ):
        selected_rows = (
            worst.loc[
                (worst.target == target) & (worst.scheme == "whole_lme") & worst.variant.isin(keep)
            ]
            .set_index("variant")
            .loc[keep]
        )
        ax.bar(keep, selected_rows.worst_lme_rmse, color="#9b6a9e")
        ax.tick_params(axis="x", rotation=15)
        style(ax, target, "worst LME RMSE")
    fig.suptitle("Worst whole-LME error remains the deployment warning", fontweight="bold")
    save(fig, "fig07_worst_lme.png")

    time_block = tables["table03_time_block_metrics.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, target, variant in zip(
        axes, ("SSS", "fCO2"), ("A3_background_sss", "A2_environment"), strict=True
    ):
        subset = time_block.loc[
            (time_block.target == target)
            & (time_block.scheme == "cruise")
            & (time_block.variant == variant)
        ]
        ax.bar(subset.time_block, subset.rmse, color="#4e83ac")
        ax.tick_params(axis="x", rotation=20)
        style(ax, f"{target}: {variant}", "RMSE")
    save(fig, "fig08_time_blocks.png")

    forward = metrics.loc[
        (metrics.support == "all")
        & metrics.scheme.isin(["forward_2019_2021", "forward_2022_2025"])
        & metrics.variant.isin(["A2_environment", "A3_background_sss", "A4_adt"])
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, target in zip(axes, ("SSS", "fCO2"), strict=True):
        subset = forward.loc[forward.target == target]
        pivot = subset.pivot(index="variant", columns="scheme", values="rmse")
        pivot.plot(kind="bar", ax=ax, color=["#5b8db8", "#d18b47"])
        ax.tick_params(axis="x", rotation=20)
        style(ax, target, "RMSE")
        ax.legend(fontsize=8)
    save(fig, "fig09_forward_periods.png")

    annual = tables["table15_annual_transition_diagnostic.csv"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for ax, target in zip(axes, ("SSS", "fCO2"), strict=True):
        subset = annual.loc[annual.target == target]
        ax.plot(subset.yr, subset.bias, marker="o", markersize=3, linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.8)
        style(ax, f"{target}: annual cruise-OOF bias", "bias")
    axes[-1].set_xlabel("year")
    save(fig, "fig10_annual_bias.png")

    missing = tables["table04_missing_strata_metrics.csv"]
    chosen = missing.loc[
        (missing.scheme == "cruise")
        & (
            ((missing.target == "SSS") & (missing.variant == "A3_background_sss"))
            | ((missing.target == "fCO2") & (missing.variant == "A2_environment"))
        )
    ].copy()
    chosen["stratum"] = (
        "SSS=" + chosen.sss_bg_available.astype(str) + ", ADT=" + chosen.adt_available.astype(str)
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, target in zip(axes, ("SSS", "fCO2"), strict=True):
        subset = chosen.loc[chosen.target == target]
        ax.bar(subset.stratum, subset.rmse, color="#7c8e5a")
        ax.tick_params(axis="x", rotation=20)
        style(ax, target, "RMSE")
    save(fig, "fig11_missing_strata.png")

    decision = tables["table07_retained_decision.csv"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    pivot = decision.pivot(index="target", columns="group", values="retained").astype(int)
    image = ax.imshow(pivot, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=20)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            ax.text(
                column,
                row,
                "retain" if pivot.iloc[row, column] else "reject",
                ha="center",
                va="center",
            )
    ax.set_title("Frozen target-specific predictor decisions", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, ticks=[0, 1])
    save(fig, "fig12_retained_decision.png")


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    def render(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"
        return str(value)

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def build_report(tables: dict[str, pd.DataFrame], captions: dict[str, str]) -> str:
    metrics = tables["table01_pooled_metrics.csv"]
    decision = tables["table07_retained_decision.csv"]
    key = metrics.loc[
        (metrics.support == "all")
        & metrics.scheme.isin(
            ["cruise", "spatial", "whole_lme", "forward_2019_2021", "forward_2022_2025"]
        )
        & (
            ((metrics.target == "SSS") & (metrics.variant == "A3_background_sss"))
            | ((metrics.target == "fCO2") & (metrics.variant == "A2_environment"))
        ),
        ["target", "variant", "scheme", "n", "rmse", "mae", "bias", "r2"],
    ]
    scope = tables["table11_data_scope.csv"]
    body = f"""# P2.3d controlled SSS and fCO2 predictor ablations

Issue: #54

Decision: **retain background-SSS residual correction for SSS; retain no optional field for fCO2; reject ADT for both targets**

## Scientific question and permitted claim

This experiment tests which deployable predictor groups improve SSS and fCO2 under identical frozen observations, folds, geometry, preprocessing rules, and CatBoost budgets. It supports target-specific predictor selection for the later model-family bake-offs. It does not select the final production architecture, establish independent-test skill, or claim reliable performance outside observation-supported coastal regimes.

## Data, splits, and leakage controls

{markdown_table(scope)}

The frozen 1993-2025 SOCAT cache was intersected once with the Issue #53 coastal-node cube. Every variant used the same target rows. Cruise folds, spatial blocks, whole-LME folds, and forward-time partitions were group-safe: any cruise touching a held partition was excluded from fitting. The 2026 provisional records, locked test, and external-independent labels remained unopened. Optional missing fields stayed as NaN plus explicit availability flags; no candidate gained rows by dropping missing cases.

The original Issue #54 matrix named chlorophyll-a, mixed-layer depth, river inputs, and circulation proxies. Issue #52 did not qualify global monthly deployable fields for those groups, so testing them here would have changed the frozen data contract. Their exclusion is archived in Table 14. The realized sequence was geotime, geometry, mandatory SST/wind/atmospheric-xCO2, background SSS, optional ADT, their combination, and training-only field dropout.

## Candidate models and training

All 224 fits used CatBoost with 300 trees, depth 8, learning rate 0.05, fixed seeds, GPU execution, and weights that first equalized LMEs and then cruises. For SSS variants containing the background field, the model predicted the observation-minus-background residual wherever background SSS existed and fell back to the same-fold A2 direct model where it did not. fCO2 never used observed SOCAT salinity as a predictor; it used only the frozen background SSS field. A6 applied deterministic training-only dropout (5% background SSS, 15% ADT).

The retention gate required at least 0.5% median RMSE improvement, positive cruise and spatial effects, no greater than 1% degradation in whole-LME or either forward period, positive strict-common-support effects, and a negative 95% upper confidence bound from 2,000 cruise-level paired bootstrap replicates.

## Main development results

{markdown_table(key)}

Background-SSS residual correction passed every gate for SSS. Relative to A2, cruise RMSE changed from 0.962 to 0.907 PSU, spatial RMSE from 1.058 to 0.973 PSU, whole-LME RMSE from 8.334 to 2.470 PSU, 2019-2021 forward RMSE from 1.760 to 1.227 PSU, and 2022-2025 forward RMSE from 1.956 to 1.580 PSU. These values remain development estimates; the large whole-LME tail shows that a global average cannot be treated as universal reliability.

For fCO2, background SSS gave small pooled gains (cruise 36.484 to 36.070 µatm; spatial 39.001 to 38.469 µatm), but the cruise-level bootstrap interval crossed zero, so it failed the reproducibility gate. ADT failed for both targets and sometimes degraded cruise or forward performance. Missing-input dropout did not pass consistently. The frozen optional-input decision is therefore SSS-background residual correction for SSS and no optional field for fCO2.

Against the Issue #37 minimum development ceilings, the retained SSS input passes cruise (0.907 < 1.0 PSU) and spatial (0.973 < 1.2 PSU) but fails late forward time (1.580 > 1.2 PSU). The retained fCO2 input fails cruise (36.484 > 30 µatm) and both forward periods (44.236 and 50.326 > 40 µatm), while spatial RMSE narrowly passes its 40 µatm ceiling. Predictor selection alone therefore does not meet the release contract; #40 and #41 must improve temporal transfer and tail-region behavior.

{markdown_table(decision)}

## Decision and limitations

Issue #54 passes as a controlled negative-and-positive ablation archive. The input contract for #40 SSS is A2 plus the masked background-SSS residual branch and same-fold fallback. The input contract for #41 fCO2 is A2 only: geotime, frozen coastal geometry, SST, wind speed, and atmospheric xCO2. ADT remains diagnostic and must not gate product coverage. Background SSS may be revisited for fCO2 only with new evidence or a preregistered interaction model; the present small pooled improvement is insufficient.

The decisive results are development OOF estimates rather than independent validation. Whole-LME errors are much larger than ordinary cruise/spatial scores, particularly for SSS without its background field. Observation density is highly uneven among LMEs, annual bias changes include sampling-composition changes, and the single-source-era cube provides no natural source-transition experiment. Reliability flags and fixed-coverage independent tests remain mandatory downstream.

## Figure and table index

"""
    for index, (name, caption) in enumerate(captions.items(), start=1):
        body += f"### Figure {index}\n\n![{caption}](figures/{name})\n\n{caption}\n\n"
    body += "### Tables\n\n"
    for name in sorted(tables):
        body += f"- `{name}`: machine-readable source data used by this report.\n"
    return body


def main() -> int:
    if ARCHIVE.exists():
        shutil.rmtree(ARCHIVE)
    tables = copy_and_derive_tables()
    build_figures(tables)
    captions = {
        "fig01_sss_rmse_matrix.png": "Figure 1. SSS held-out RMSE for every feature variant and validation scheme; cell colors are normalized to the mandatory-environment A2 baseline while annotations retain absolute PSU values.",
        "fig02_fco2_rmse_matrix.png": "Figure 2. fCO2 held-out RMSE for every feature variant and validation scheme; pooled gains from background SSS are small compared with the variation among validation designs.",
        "fig03_optional_effects.png": "Figure 3. Relative RMSE effects of background SSS, ADT, and training-only missing-field dropout across all five decision schemes; positive bars denote lower candidate error.",
        "fig04_common_support.png": "Figure 4. Optional-field effects on all observations and the identical strict common-support subset, preventing coverage reduction from masquerading as predictor skill.",
        "fig05_bootstrap_intervals.png": "Figure 5. Cruise-level paired bootstrap mean squared-error contrasts with 95% intervals; an optional group passes this gate only when the complete interval is below zero in cruise and spatial validation.",
        "fig06_lme_distributions.png": "Figure 6. Distribution of per-LME errors under whole-LME validation; background-SSS residual correction sharply reduces SSS transfer error but substantial regional heterogeneity remains.",
        "fig07_worst_lme.png": "Figure 7. Worst per-LME RMSE under whole-LME validation for the principal baseline and retained variants, emphasizing the tail risk hidden by pooled global metrics.",
        "fig08_time_blocks.png": "Figure 8. Cruise-OOF RMSE in four frozen time blocks for the retained SSS input and frozen fCO2 input, documenting temporal rather than only pooled behavior.",
        "fig09_forward_periods.png": "Figure 9. Strict forward-time RMSE in 2019-2021 and 2022-2025 for the mandatory baseline and principal optional candidates; later years are harder for both targets.",
        "fig10_annual_bias.png": "Figure 10. Annual cruise-OOF bias for the retained target-specific inputs; year-to-year changes mix model drift with changing observational composition and are not interpreted as source discontinuities.",
        "fig11_missing_strata.png": "Figure 11. Cruise-OOF RMSE by background-SSS and ADT availability strata; unavailable SSS uses the frozen same-fold A2 fallback and ADT absence never removes a prediction.",
        "fig12_retained_decision.png": "Figure 12. Frozen target-specific gate outcome: background-SSS residual correction is retained only for SSS, while ADT and missing-field dropout are rejected for both targets.",
    }
    report = build_report(tables, captions)
    (ARCHIVE / "REPORT.md").write_text(report, encoding="utf-8")
    (ARCHIVE / "README.md").write_text(
        "# Issue 54 reviewer archive\n\nSee `REPORT.md`; large OOF predictions remain in the external processed-data directory recorded by the manifest.\n",
        encoding="utf-8",
    )
    captions_text = "# Figure captions\n\n" + "\n\n".join(
        f"## {name}\n\n{caption}" for name, caption in captions.items()
    )
    (ARCHIVE / "CAPTIONS.md").write_text(captions_text + "\n", encoding="utf-8")

    # Hash the LF-normalized bytes stored by Git so a clean Linux checkout and
    # a Windows checkout with core.autocrlf verify identically.
    for path in [
        ARCHIVE / "README.md",
        ARCHIVE / "REPORT.md",
        ARCHIVE / "CAPTIONS.md",
        *sorted(TABLES.glob("*.csv")),
    ]:
        path.write_bytes(path.read_text(encoding="utf-8").encode("utf-8"))

    decisions = tables["table07_retained_decision.csv"]
    frozen = {
        "version": "2.3",
        "issue": 54,
        "experiment_id": "p2_predictor_ablation_v2.3",
        "point_model": {
            "family": "CatBoostRegressor",
            "iterations": 300,
            "depth": 8,
            "learning_rate": 0.05,
            "lme_then_cruise_balanced_weights": True,
        },
        "sss_retained_input": "A3_background_sss",
        "sss_contract": "A2 direct fallback plus background-SSS residual correction when available",
        "fco2_retained_input": "A2_environment",
        "fco2_contract": "geotime, coastal geometry, SST, wind speed, atmospheric xCO2",
        "adt_retained": False,
        "decision_rows": decisions.to_dict("records"),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "large_output_root": str(EXTERNAL),
        "matchup_sha256": sha256(EXTERNAL / "socat_predictor_matchup_v2.3.parquet"),
    }
    CONFIG.write_text(json.dumps(frozen, indent=2), encoding="utf-8")

    tracked = [ARCHIVE / "README.md", ARCHIVE / "REPORT.md", ARCHIVE / "CAPTIONS.md"]
    tracked += sorted(FIGURES.glob("*.png")) + sorted(TABLES.glob("*.csv"))
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    figure_source = {
        name: [
            "table01_pooled_metrics.csv",
            "table06_effects.csv",
            "table05_paired_bootstrap.csv",
            "table02_lme_metrics.csv",
            "table03_time_block_metrics.csv",
            "table04_missing_strata_metrics.csv",
            "table07_retained_decision.csv",
            "table15_annual_transition_diagnostic.csv",
        ]
        for name in captions
    }
    manifest = {
        "experiment_id": "p2_predictor_ablation_v2.3",
        "training_git_commit": git_commit,
        "data_manifest_sha256": sha256(
            Path(r"C:\backup\phd\data\processed\recad_v2_3\global_coastal_predictor_cube_v2.3.nc")
        ),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "1993-2025 frozen development observations with cruise, spatial, whole-LME, and forward-time OOF evaluation",
        "decision": "retain SSS background-residual input; reject optional inputs for fCO2 and reject ADT for both targets",
        "figure_source_data": figure_source,
        "figure_captions": captions,
        "files_sha256": {
            str(path.relative_to(ARCHIVE)).replace("\\", "/"): sha256(path) for path in tracked
        },
        "archive_builder_sha256": sha256(Path(__file__)),
        "analysis_script_sha256": sha256(RUNNER),
        "source_artifacts_sha256": {
            str(path): sha256(path)
            for path in (
                Path(
                    r"C:\backup\phd\data\processed\recad_v2_3\global_coastal_predictor_cube_v2.3.nc"
                ),
                Path(r"C:\backup\phd\data\processed\recad_v2_3\global_socat_cache_v2.3.parquet"),
                Path(
                    r"C:\backup\phd\data\processed\recad_v2_3\global_group_split_manifest_v2.3.parquet"
                ),
                EXTERNAL / "socat_predictor_matchup_v2.3.parquet",
            )
        },
        "large_outputs": {
            "root": str(EXTERNAL),
            "prediction_files": len(list((EXTERNAL / "fold_predictions").glob("*.parquet"))),
            "matchup_rows": int(tables["table11_data_scope.csv"]["rows"].max()),
        },
    }
    (ARCHIVE / "archive_manifest.json").write_bytes(json.dumps(manifest, indent=2).encode("utf-8"))
    print(f"built {ARCHIVE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
