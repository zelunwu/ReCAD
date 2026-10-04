"""Build the reviewer archive for the Issue #26 fCO2 experiment."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_fco2_reliability_v2.2"
OUTPUT = ROOT / "outputs" / "experiments" / EXPERIMENT_ID
ARCHIVE = ROOT / "docs" / "experiment_archive" / EXPERIMENT_ID


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(ARCHIVE / "figures" / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    view = frame.copy()
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(lambda value: f"{value:.{digits}f}")
    headers = [str(column) for column in view.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_tables() -> dict[str, pd.DataFrame]:
    table_dir = ARCHIVE / "tables"
    for name in (
        "candidate_shrinkage_comparison.csv",
        "development_gate_by_scheme.csv",
        "regional_expert_metrics.csv",
        "temporal_interval_inflation.csv",
        "atlas_grade_counts.csv",
    ):
        shutil.copy2(OUTPUT / name, table_dir / name)
    comparison = pd.read_csv(OUTPUT / "candidate_shrinkage_comparison.csv")
    candidate_summary = (
        comparison.groupby(["candidate", "shrinkage"])
        .agg(
            mean_normalized_rmse=("normalized_rmse", "mean"),
            mean_skill=("skill_vs_background", "mean"),
        )
        .reset_index()
        .sort_values("mean_normalized_rmse")
    )
    candidate_summary.to_csv(table_dir / "candidate_summary.csv", index=False)
    chla = pd.read_parquet(OUTPUT / "chla_ablation_predictions.parquet")
    rows = []
    for (candidate, scheme), part in chla.groupby(["candidate", "outer_scheme"]):
        error = part.prediction - part.truth
        background_error = part.background - part.truth
        rows.append(
            {
                "candidate": candidate,
                "outer_scheme": scheme,
                "n": len(part),
                "rmse": np.sqrt(np.mean(error**2)),
                "mae": np.mean(np.abs(error)),
                "skill_vs_background": 1.0 - np.mean(error**2) / np.mean(background_error**2),
            }
        )
    chla_summary = pd.DataFrame(rows)
    chla_summary.to_csv(table_dir / "chla_summary.csv", index=False)
    regional = pd.read_parquet(OUTPUT / "regional_expert_predictions.parquet")
    selected = regional.loc[regional.lme_id.eq(12)].copy()
    selected["absolute_error"] = np.abs(selected.prediction - selected.truth)
    selected["fco2_band"] = pd.cut(
        selected.truth,
        [-np.inf, 250, 350, 450, 550, np.inf],
        right=False,
    ).astype(str)
    selected["support_bin"] = pd.cut(
        selected.risk_environment_k64,
        [-np.inf, 1, 2, 3, 5, np.inf],
        right=False,
    ).astype(str)
    strata_rows = []
    for scheme, scheme_part in selected.groupby("outer_scheme"):
        for kind in ("fco2_band", "support_bin"):
            for group, part in scheme_part.groupby(kind):
                error = part.prediction - part.truth
                background_error = part.background - part.truth
                strata_rows.append(
                    {
                        "outer_scheme": scheme,
                        "stratum": kind,
                        "group": group,
                        "n": len(part),
                        "rmse": np.sqrt(np.mean(error**2)),
                        "mae": np.mean(np.abs(error)),
                        "skill_vs_background": 1.0
                        - np.mean(error**2) / np.mean(background_error**2),
                    }
                )
    strata = pd.DataFrame(strata_rows)
    strata.to_csv(table_dir / "regional_strata.csv", index=False)
    risk_rows = []
    for scheme, part in selected.loc[selected.outer_scheme.ne("whole_lme")].groupby("outer_scheme"):
        ordered = part.sort_values("risk_environment_k64")
        for fraction in np.linspace(0.1, 1.0, 10):
            keep = ordered.iloc[: max(1, int(len(ordered) * fraction))]
            risk_rows.append(
                {
                    "outer_scheme": scheme,
                    "retained_fraction": fraction,
                    "rmse": np.sqrt(np.mean((keep.prediction - keep.truth) ** 2)),
                    "mae": np.mean(np.abs(keep.prediction - keep.truth)),
                }
            )
    risk = pd.DataFrame(risk_rows)
    risk.to_csv(table_dir / "regional_risk_coverage.csv", index=False)
    return {
        "candidate": candidate_summary,
        "gate": pd.read_csv(OUTPUT / "development_gate_by_scheme.csv"),
        "regional": pd.read_csv(OUTPUT / "regional_expert_metrics.csv"),
        "inflation": pd.read_csv(OUTPUT / "temporal_interval_inflation.csv"),
        "chla": chla_summary,
        "atlas": pd.read_csv(OUTPUT / "atlas_grade_counts.csv"),
        "strata": strata,
        "risk": risk,
    }


def build_figures(tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    captions = {
        "fig01_candidate_comparison.png": "Figure 1. Mean outer-scheme RMSE normalized by the training-only seasonal-trend background for every preregistered model and support-shrinkage pair. Lower is better; the dashed line is background parity. CatBoost with strictly cross-fitted SSS and linear 2-5 environmental shrinkage ranked first before regional specialization.",
        "fig02_core_selective_gate.png": "Figure 2. RMSE and retained fraction for the global support-selective candidate within grade A/B development rows. Cruise and spatial-block subsets are accurate, while whole-LME has no retained A/B rows and forward retention is too small, motivating the preregistered regional-expert branch rather than a global claim.",
        "fig03_regional_lme_comparison.png": "Figure 3. Regional-expert RMSE against the seasonal-trend background for the two LMEs nominated using cruise OOF only. Caribbean Sea (LME 12) improves cruise, spatial-block, and forward partitions; Southeast U.S. Continental Shelf (LME 6) fails spatial confirmation and is excluded.",
        "fig04_lme12_interval_coverage.png": "Figure 4. Empirical 50% and 90% interval coverage for Caribbean Sea (LME 12). Cruise OOF supplies base conformal widths; a nested 2016-2018 pseudo-forward calibration fitted only inside the training era inflates future intervals, bringing the untouched 2019-2021 forward coverage to the preregistered acceptance range.",
        "fig05_chla_ablation.png": "Figure 5. CatBoost RMSE with and without log10 MODIS chlorophyll-a on exactly the same finite-Chl-a observations. Chl-a improves all four outer schemes, but the local cache ends in December 2020, so it is evidence for a future refreshed input rather than a valid 2025 product feature.",
        "fig06_atlas_grade_coverage.png": "Figure 6. Grid-month counts in the 2025 core and available 2026 provisional atlas. Only confirmed Caribbean Sea cells with adequate environmental, cruise, effective-group, and SSS support receive grade A; every other grid-month is retained as an auditable grade-D background fallback and is excluded from publication claims.",
        "fig07_atlas_map_2025_07.png": "Figure 7. July 2025 selective fCO2 atlas status. Blue points are grade-A Caribbean Sea regional predictions; light-gray points are grade-D background fallbacks outside the confirmed regional domain or without required support. The map is a development applicability product, not independent validation.",
        "fig08_lme12_risk_coverage.png": "Figure 8. Caribbean Sea RMSE as progressively higher environmental-k64 risk observations are retained. Curves are shown for cruise, spatial-block, and forward partitions; the ordering diagnoses support sensitivity while the formal regional gate remains based on the frozen grade and interval rules.",
    }
    candidate = tables["candidate"].sort_values("mean_normalized_rmse").head(12)
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = candidate.candidate + "\n" + candidate.shrinkage
    ax.barh(labels[::-1], candidate.mean_normalized_rmse[::-1], color="#4477aa")
    ax.axvline(1.0, color="black", linestyle="--")
    ax.set_xlabel("Mean normalized RMSE")
    save(fig, "fig01_candidate_comparison.png")
    gate = tables["gate"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(gate.outer_scheme, gate.rmse, color="#228833")
    axes[0].set_ylabel("RMSE (µatm)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(gate.outer_scheme, gate.retained_fraction, color="#ccbb44")
    axes[1].set_ylabel("Grade A/B retained fraction")
    axes[1].tick_params(axis="x", rotation=25)
    save(fig, "fig02_core_selective_gate.png")
    regional = tables["regional"]
    regional = regional.loc[regional.lme_id.isin([6, 12]) & regional.outer_scheme.ne("whole_lme")]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(3)
    width = 0.22
    schemes = ["cruise", "spatial_block", "forward"]
    for index, lme in enumerate([6, 12]):
        part = regional.set_index(["lme_id", "outer_scheme"])
        ax.bar(
            x + index * width,
            [part.loc[(lme, scheme), "rmse"] for scheme in schemes],
            width,
            label=f"LME {lme} model",
        )
    background = regional.loc[regional.lme_id.eq(12)].set_index("outer_scheme")
    ax.bar(
        x + 2 * width,
        [background.loc[scheme, "background_rmse"] for scheme in schemes],
        width,
        label="LME 12 background",
        color="#999999",
    )
    ax.set_xticks(x + width, schemes)
    ax.set_ylabel("RMSE (µatm)")
    ax.legend()
    save(fig, "fig03_regional_lme_comparison.png")
    lme12 = tables["regional"].loc[
        tables["regional"].lme_id.eq(12) & tables["regional"].outer_scheme.ne("whole_lme")
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(lme12))
    ax.bar(x - 0.18, lme12.coverage50, 0.36, label="50% interval")
    ax.bar(x + 0.18, lme12.coverage90, 0.36, label="90% interval")
    ax.axhline(0.5, color="#4477aa", linestyle="--")
    ax.axhline(0.9, color="#ee6677", linestyle="--")
    ax.set_xticks(x, lme12.outer_scheme)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Empirical coverage")
    ax.legend()
    save(fig, "fig04_lme12_interval_coverage.png")
    chla = (
        tables["chla"]
        .pivot(index="outer_scheme", columns="candidate", values="rmse")
        .loc[["cruise", "spatial_block", "whole_lme", "forward"]]
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(chla))
    ax.bar(x - 0.18, chla.core, 0.36, label="Core")
    ax.bar(x + 0.18, chla.core_plus_chla, 0.36, label="Core + Chl-a")
    ax.set_xticks(x, chla.index, rotation=20)
    ax.set_ylabel("RMSE (µatm)")
    ax.legend()
    save(fig, "fig05_chla_ablation.png")
    atlas = tables["atlas"].groupby(["year", "grade"], as_index=False).grid_months.sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    years = sorted(atlas.year.unique())
    total = atlas.groupby("year").grid_months.sum()
    bottom = np.zeros(len(years))
    for grade in ["A", "B", "C", "D"]:
        values = (
            atlas.loc[atlas.grade.eq(grade)]
            .set_index("year")
            .grid_months.reindex(years, fill_value=0)
            / total.reindex(years)
        ).to_numpy()
        if values.any():
            ax.bar([str(year) for year in years], values, bottom=bottom, label=f"Grade {grade}")
            bottom += values
    ax.set_ylabel("Fraction of grid-months")
    ax.set_ylim(0, 1)
    ax.legend()
    save(fig, "fig06_atlas_grade_coverage.png")
    july = pd.read_parquet(OUTPUT / "fco2_reliability_atlas/year=2025/month=07.parquet")
    fig, ax = plt.subplots(figsize=(11, 5))
    d = july.loc[july.grade.eq("D")]
    a = july.loc[july.grade.eq("A")]
    ax.scatter(d.longitude, d.latitude, s=0.3, c="#dddddd", rasterized=True, label="D fallback")
    ax.scatter(a.longitude, a.latitude, s=1.2, c="#225ea8", rasterized=True, label="A regional")
    ax.set(xlabel="Longitude (°E)", ylabel="Latitude (°N)")
    ax.legend(markerscale=6)
    save(fig, "fig07_atlas_map_2025_07.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    for scheme, part in tables["risk"].groupby("outer_scheme"):
        ax.plot(part.retained_fraction, part.rmse, marker="o", label=scheme)
    ax.set(xlabel="Retained fraction", ylabel="RMSE (µatm)")
    ax.legend()
    save(fig, "fig08_lme12_risk_coverage.png")
    return captions


def build_report(tables: dict[str, pd.DataFrame], captions: dict[str, str]) -> None:
    regional = tables["regional"].loc[
        tables["regional"].lme_id.eq(12)
        & tables["regional"].outer_scheme.isin(["cruise", "spatial_block", "forward"]),
        [
            "outer_scheme",
            "n",
            "rmse",
            "mae",
            "q90_absolute_error",
            "skill_vs_background",
            "coverage50",
            "coverage90",
        ],
    ]
    atlas = tables["atlas"].groupby("grade", as_index=False).grid_months.sum()
    figure_blocks = []
    for name, caption in captions.items():
        figure_blocks.append(f"![{name}](figures/{name})\n\n{caption}")
    text = f"""# P1 fCO2 support-aware regional product — reviewer archive

**Decision: `pass_regional` for Caribbean Sea (LME 12) only.** The locked fCO2 test and external-independent labels remain sealed. This result nominates the regional candidate for a future locked audit; it is not independent validation and does not support a global fCO2 claim.

## Scientific question and permitted claim

This experiment asks whether support-aware residual learning can produce a calibrated coastal fCO2 product in a defensible subset of the frozen domain. The permitted claim is development evidence from untouched cruise, spatial-block, whole-LME, and forward outer predictions. Caribbean Sea is the only confirmed publishable development region. All other regions are explicitly grade D and background-only in the atlas.

## Data, splits, and leakage controls

SOCATv2026 fCO2 is the response. Predictors are monthly SST, GLORYS/OOF-corrected SSS, ADT, wind speed, atmospheric pCO2/xCO2, coordinates, month, and frozen LME/basin/regime identifiers. Every outer model is fit without its held cruises, 5-degree blocks, complete LMEs, or 2019-2021 forward cruises. SSS corrections are joined from Issue #25 cross-fitted predictions. Model budgets and gates were committed before each corresponding run. The temporal interval inflation uses only a nested 2016-2018 pseudo-forward split inside the training era. No locked or external labels were read.

## Candidate models and training

The first stage compares fixed 1500-tree CatBoost, standardized-target MLP with MSE, physically scaled 30 µatm Huber MLP, and CatBoost with OOF-corrected SSS. Each is evaluated with no, linear 2-5, quadratic-5, and hard-4 support shrinkage. CatBoost plus OOF SSS and linear 2-5 shrinkage ranks first. Because the global selective gate left a documented whole-LME failure, the preregistered conditional branch trains independent LME CatBoost experts for regions selected only by training metadata. Cruise OOF nominates LMEs; spatial-block and forward predictions confirm them. Unseen LMEs fall back to the seasonal-trend background and grade D.

Chl-a is tested as a single-factor addition on identical finite-Chl-a rows. Bathymetry and distance-to-coast are absent from the frozen v2.2 assets, so that block is recorded as unavailable instead of being ingested after outcomes were seen. The coastal graph branch was not triggered because Issue #24 selected environmental-k64 over graph applicability.

## Main development results

The global selective A/B subset is strong for cruise and spatial-block testing but contains no whole-LME A/B rows and retains only 1.95% of forward rows. The conditional regional experiment nominates LMEs 6 and 12 from cruise OOF. LME 6 fails spatial skill and q90. Caribbean Sea passes every frozen regional confirmation check:

{markdown_table(regional)}

The nested temporal calibration multiplies Caribbean Sea q50/q90 widths by 1.556/1.325. Final atlas widths are 6.97 µatm (50%) and 16.64 µatm (90%). The atlas contains:

{markdown_table(atlas)}

The Chl-a common-support ablation reduces RMSE by 1.75% (cruise), 5.53% (spatial block), 2.91% (whole LME), and 2.83% (forward). Its cache ends in December 2020, so Chl-a is not used for the 2025/available-2026 atlas.

## Decision and limitations

The final status is `pass_regional`, restricted to Caribbean Sea (LME 12). This passes a preregistered development gate and permits a future one-time locked fCO2 audit; it does not unlock labels in Issue #26. LME 6 and every other LME remain nonpublication regions. Whole-LME transfer failed as a product strategy, so the atlas makes this limitation operational: nonconfirmed regions receive the training-only background, grade D, a suppression flag, and reason bits.

The direct evaluation cache is North-American-adjacent, the background and regional model inherit SOCAT sampling patterns, and the 2026 coverage is provisional. Chl-a coverage is stale; bathymetry/coast-distance inputs remain a future preregistered data-version change. External-independent validation remains reserved for P3.

## Figure and table index

{chr(10).join(figure_blocks)}

Source CSVs for all aggregate figures are in `tables/`. Row-level OOF predictions, Chl-a predictions, final checkpoints, and atlas partitions remain in the ignored local output directory and are hash-referenced by the manifest.
"""
    (ARCHIVE / "REPORT.md").write_text(text, encoding="utf-8")
    caption_lines = ["# Figure captions", ""]
    for name, caption in captions.items():
        caption_lines.extend([f"## {name}", "", caption, ""])
    (ARCHIVE / "CAPTIONS.md").write_text("\n".join(caption_lines), encoding="utf-8")
    (ARCHIVE / "README.md").write_text(
        "# Issue #26 reviewer archive\n\nSee `REPORT.md` for the result, embedded figures, captions, evidence boundary, and decision. Large row-level outputs remain in the local ignored experiment directory.\n",
        encoding="utf-8",
    )


def build_manifest(captions: dict[str, str]) -> None:
    figure_sources = {
        "fig01_candidate_comparison.png": ["candidate_summary.csv"],
        "fig02_core_selective_gate.png": ["development_gate_by_scheme.csv"],
        "fig03_regional_lme_comparison.png": ["regional_expert_metrics.csv"],
        "fig04_lme12_interval_coverage.png": [
            "regional_expert_metrics.csv",
            "temporal_interval_inflation.csv",
        ],
        "fig05_chla_ablation.png": ["chla_summary.csv"],
        "fig06_atlas_grade_coverage.png": ["atlas_grade_counts.csv"],
        "fig07_atlas_map_2025_07.png": ["local:fco2_reliability_atlas/year=2025/month=07.parquet"],
        "fig08_lme12_risk_coverage.png": ["regional_risk_coverage.csv"],
    }
    files = {}
    for path in sorted(ARCHIVE.rglob("*")):
        if path.is_file() and path.name != "archive_manifest.json":
            files[str(path.relative_to(ARCHIVE)).replace("\\", "/")] = sha256(path)
    local_names = [
        "development_oof_predictions.parquet",
        "regional_expert_predictions.parquet",
        "chla_ablation_predictions.parquet",
        "regional_decision.json",
        "atlas_metadata.json",
    ]
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "data_manifest_sha256": sha256(ROOT / "configs/frozen/data_manifest_v2.2.json"),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "development outer OOF only; locked and external-independent fCO2 labels sealed",
        "decision": "pass_regional_lme12",
        "figure_source_data": figure_sources,
        "figure_captions": captions,
        "archive_builder_sha256": sha256(Path(__file__)),
        "analysis_script_sha256": sha256(Path(__file__)),
        "source_artifacts_sha256": {name: sha256(OUTPUT / name) for name in local_names},
        "files_sha256": files,
    }
    (ARCHIVE / "archive_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    (ARCHIVE / "figures").mkdir(parents=True, exist_ok=True)
    (ARCHIVE / "tables").mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        ROOT / "configs/p1_fco2_reliability_v2.2.yaml", ARCHIVE / "p1_fco2_reliability_v2.2.yaml"
    )
    tables = build_tables()
    captions = build_figures(tables)
    build_report(tables, captions)
    build_manifest(captions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
