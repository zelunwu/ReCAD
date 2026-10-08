"""Build the Issue #37 literature benchmark and improvement-contract archive."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from recad.governance.benchmark_contract import (
    load_benchmark_contract,
    validate_benchmark_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/frozen/benchmark_contract_v2.3.json"
DATA_MANIFEST = ROOT / "configs/frozen/data_manifest_v2.2.json"
ARCHIVE = ROOT / "docs/experiment_archive/p2_benchmark_freeze_v2.3"
TABLES = ARCHIVE / "tables"
FIGURES = ARCHIVE / "figures"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, value: str) -> None:
    path.write_bytes(value.encode("utf-8"))


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    normalized = [
        {
            key: json.dumps(value) if isinstance(value, (list, dict)) else value
            for key, value in row.items()
        }
        for row in rows
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(normalized[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(normalized)
    write_text(TABLES / name, buffer.getvalue())


def build_tables(contract: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    literature = contract["literature_benchmarks"]
    gates = [{"gate": key, "value": value} for key, value in contract["shared_gates"].items()]
    v1_rows = [{"field": key, "value": value} for key, value in contract["v1_retraining"].items()]
    conversion = [
        {"field": key, "value": value} for key, value in contract["target_conversion"].items()
    ]
    bootstrap = [
        {"field": key, "value": value} for key, value in contract["paired_bootstrap"].items()
    ]
    dimensions = [
        "target_basis",
        "resolution",
        "period",
        "support",
        "split",
        "calibration",
        "uncertainty",
    ]
    panel_matrix = []
    levels = {
        "native_context": [1, 1, 1, 0, 0, 0, 0],
        "common_grid": [1, 1, 1, 0, 0, 0, 0],
        "common_support": [1, 1, 1, 1, 0, 0, 1],
        "strict_method": [1, 1, 1, 1, 1, 1, 1],
    }
    for panel, values in levels.items():
        for dimension, controlled in zip(dimensions, values, strict=True):
            panel_matrix.append({"panel": panel, "dimension": dimension, "controlled": controlled})
    baselines = [
        {
            "target": "fCO2",
            "baseline": "training-fold seasonal-trend background",
            "role": "minimum positive-skill reference",
            "must_recompute": True,
        },
        {
            "target": "fCO2",
            "baseline": "strict retrained v1.x RF",
            "role": "primary causal method comparison",
            "must_recompute": True,
        },
        {
            "target": "fCO2",
            "baseline": "CatBoost/GBDT residual",
            "role": "strong same-split point baseline",
            "must_recompute": True,
        },
        {
            "target": "SSS",
            "baseline": "GLORYS12V1",
            "role": "operational background reference",
            "must_recompute": True,
        },
        {
            "target": "SSS",
            "baseline": "CatBoost/GBDT residual correction",
            "role": "strong same-split point baseline",
            "must_recompute": True,
        },
        {
            "target": "SSS",
            "baseline": "ESA CCI and SMAP",
            "role": "common-support external product context",
            "must_recompute": True,
        },
    ]
    acceptance = [
        {
            "check": "literature facts record basis, resolution, period, and validation",
            "passed": True,
        },
        {"check": "native and common-grid scores cannot select models", "passed": True},
        {"check": "common-support and strict-method panels are frozen", "passed": True},
        {"check": "2004-2005 is descriptive only", "passed": True},
        {"check": "minimum, competitive, and ideal targets are monotonic", "passed": True},
        {"check": "uncertainty and fixed-coverage gates are frozen", "passed": True},
        {"check": "paired grouped bootstrap procedure is frozen", "passed": True},
        {"check": "pCO2/fCO2 conversion is mandatory", "passed": True},
    ]
    sources = [
        {
            "source": row["source"],
            "url": row["url"],
            "accessed": "2026-10-08",
            "fact_type": "primary publication or official product documentation",
        }
        for row in literature
    ]
    return {
        "table01_literature_benchmarks.csv": literature,
        "table02_comparison_panels.csv": contract["comparison_panels"],
        "table03_performance_targets.csv": contract["performance_targets"],
        "table04_shared_gates.csv": gates,
        "table05_v1_retraining_contract.csv": v1_rows,
        "table06_target_conversion.csv": conversion,
        "table07_paired_bootstrap.csv": bootstrap,
        "table08_panel_comparability.csv": panel_matrix,
        "table09_baseline_roles.csv": baselines,
        "table10_acceptance_checks.csv": acceptance,
        "table11_source_catalog.csv": sources,
    }


def style(ax: Any, title: str, ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_figures(tables: dict[str, list[dict[str, Any]]]) -> None:
    literature = tables["table01_literature_benchmarks.csv"]
    fco2 = [row for row in literature if row["target"] == "fCO2" and row["value"] is not None]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.barh([row["product"] for row in fco2], [row["value"] for row in fco2], color="#5b8db8")
    ax.invert_yaxis()
    style(
        ax,
        "Published fCO2/pCO2 errors are context under different protocols",
        "RMSE or RMSD (uatm)",
    )
    save(fig, "fig01_fco2_literature_context.png")

    sss = [row for row in literature if row["target"] == "SSS" and row["value"] is not None]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([row["product"] for row in sss], [row["value"] for row in sss], color="#3e9f77")
    ax.tick_params(axis="x", rotation=20)
    style(ax, "Regional SSS literature context", "RMSE (PSU)")
    save(fig, "fig02_sss_literature_context.png")

    matrix = tables["table08_panel_comparability.csv"]
    panels = ["native_context", "common_grid", "common_support", "strict_method"]
    dimensions = [
        "target_basis",
        "resolution",
        "period",
        "support",
        "split",
        "calibration",
        "uncertainty",
    ]
    values = np.array(
        [
            [
                next(
                    row["controlled"]
                    for row in matrix
                    if row["panel"] == panel and row["dimension"] == dimension
                )
                for dimension in dimensions
            ]
            for panel in panels
        ]
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(dimensions)), dimensions, rotation=25, ha="right")
    ax.set_yticks(range(len(panels)), panels)
    style(ax, "Which comparison panels control each source of unfairness")
    save(fig, "fig03_panel_comparability.png")

    targets = tables["table03_performance_targets.csv"]
    tier_colors = {"minimum": "#a8b8c8", "competitive": "#4c86b6", "ideal": "#174f7a"}
    for target, name, unit in (
        ("fCO2", "fig04_fco2_targets.png", "RMSE ceiling (uatm)"),
        ("SSS", "fig05_sss_targets.png", "RMSE ceiling (PSU)"),
    ):
        rows = [row for row in targets if row["target"] == target]
        schemes = ["cruise", "spatial", "forward"]
        x = np.arange(3)
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for index, tier in enumerate(("minimum", "competitive", "ideal")):
            ax.bar(
                x + (index - 1) * 0.24,
                [
                    next(
                        row["rmse_ceiling"]
                        for row in rows
                        if row["scheme"] == scheme and row["tier"] == tier
                    )
                    for scheme in schemes
                ],
                0.24,
                label=tier,
                color=tier_colors[tier],
            )
        ax.set_xticks(x, schemes)
        ax.legend()
        style(ax, f"Frozen {target} absolute research targets", unit)
        save(fig, name)

    panels = tables["table02_comparison_panels.csv"]
    fig, ax = plt.subplots(figsize=(9, 4))
    levels = [1, 2, 3, 4]
    ax.step(levels, levels, where="mid", color="#6f42c1", linewidth=3)
    ax.scatter(levels, levels, s=180, color="#6f42c1")
    ax.set_xticks(levels, [row["id"] for row in panels], rotation=15)
    ax.set_yticks(levels, ["context", "formatted", "paired product", "causal method"])
    style(ax, "Evidence strength of the four benchmark panels")
    save(fig, "fig06_panel_hierarchy.png")

    baselines = tables["table09_baseline_roles.csv"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(baselines))
    ax.barh(
        y,
        [1 if row["must_recompute"] else 0 for row in baselines],
        color=["#2d7f5e" if row["target"] == "SSS" else "#346fa3" for row in baselines],
    )
    ax.set_yticks(y, [f"{row['target']}: {row['baseline']}" for row in baselines], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 1], ["published value", "recompute on common split"])
    style(ax, "Every decision baseline must be recomputed")
    save(fig, "fig07_baseline_roles.png")

    gates = [
        "paired improvement",
        "absolute RMSE",
        "positive skill",
        "worst LME",
        "50% coverage",
        "90% coverage",
        "fixed coverage",
        "WIS/CRPS",
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(gates, [1] * len(gates), color="#d08a32")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 1], ["not required", "required"])
    style(ax, "Minimum gate is multidimensional, not one RMSE")
    save(fig, "fig08_acceptance_dimensions.png")


CAPTIONS = {
    "fig01_fco2_literature_context.png": "Figure 1. Published fCO2 or pCO2 errors collected as scientific context. Bars are not a ranking because domains, target basis, observation support, calibration, and validation designs differ.",
    "fig02_sss_literature_context.png": "Figure 2. Regional coastal SSS errors reported for recent machine-learning studies. These satellite-era regional values motivate the tiers but cannot substitute for global cruise, spatial, and forward evaluation.",
    "fig03_panel_comparability.png": "Figure 3. Controlled dimensions in the four frozen panels. Only the strict-method panel controls target, grid, period, support, split, calibration, and uncertainty well enough to select a model.",
    "fig04_fco2_targets.png": "Figure 4. Frozen absolute fCO2 RMSE ceilings for minimum, competitive, and ideal evidence under cruise, spatial-block, and forward-time outer evaluation.",
    "fig05_sss_targets.png": "Figure 5. Frozen absolute SSS RMSE ceilings for minimum, competitive, and ideal evidence; spatial and forward transfer receive looser ceilings than new-cruise interpolation.",
    "fig06_panel_hierarchy.png": "Figure 6. Benchmark evidence hierarchy from native publication context through a common format and paired product comparison to the identical-data strict method comparison.",
    "fig07_baseline_roles.png": "Figure 7. Required decision baselines for SSS and fCO2. Every baseline must be recomputed on the same held observations rather than copied from a publication table.",
    "fig08_acceptance_dimensions.png": "Figure 8. The minimum advancement gate requires paired improvement, absolute accuracy, regional robustness, calibrated uncertainty, and fixed-coverage risk simultaneously.",
}


def build_documents() -> None:
    write_text(
        ARCHIVE / "CAPTIONS.md",
        "# Figure captions\n\n"
        + "\n\n".join(f"## {name}\n\n{caption}" for name, caption in CAPTIONS.items())
        + "\n",
    )
    write_text(
        ARCHIVE / "README.md",
        "# P2 benchmark freeze v2.3\n\nReviewer-ready archive for Issue #37. It freezes literature context, comparison panels, the strict v1.x improvement contract, uncertainty gates, and grouped paired-bootstrap rules. No model was trained and no locked or external label was opened.\n",
    )
    blocks = "\n\n".join(
        f"![{name}](figures/{name})\n\n{caption}" for name, caption in CAPTIONS.items()
    )
    report = f"""# Issue #37 literature benchmark and v1.x improvement contract

## Scientific question and permitted claim

What must ReCAD v2 demonstrate before it can be called better than ReCAD v1.x or competitive with modern coastal products? The answer is a frozen, target-specific contract. Published native scores are context. Model advancement requires an identical-data strict comparison against a fold-pure v1.x replica and the strongest same-split baseline, followed by common-support product comparison.

## Data, splits, and leakage controls

This governance experiment uses primary publications, official product documentation, and immutable ReCAD archives; it opens no observation labels. The published v1.1 random-test RMSE of 17.642 µatm and 2004-2005 RMSE of 28.975 µatm remain historical because full-period local calibration used those labels. The strict v1.x replica must use the original seven inputs, a 300-tree bagged RF, and training-fold-only calibration under cruise, spatial-block, and forward splits.

Four panels are frozen: native publication context; products standardized to 0.25° monthly with all available support; a paired common-observation intersection; and the identical-data strict method panel. pCO2 products are converted consistently with observations at matched SST/SSS using frozen PyCO2SYS settings before comparison with fCO2.

## Candidate models and training

No candidate was trained here. Future SSS candidates must beat GLORYS12V1 and the strongest same-split residual baseline. Future fCO2 candidates must beat the training-only seasonal-trend background, strict retrained v1.x RF, and strongest CatBoost/GBDT residual baseline. STTransformer, MoE, graph, and pretraining candidates receive no special allowance.

## Main development results

The literature review confirms that direct numerical ranking would be invalid. Roobaert et al. report 29 µatm for a global coastal reconstruction; CMEMS-LSCE reports coastal RMSD 27.6 µatm under reconstruction-month exclusion; RFR-LMEs uses random five-fold grid-cell validation; Duke et al. report 42.9 µatm under regional EXPOCODE withholding; Cho et al. report global-ocean 10-fold RMSE 13.57 µatm and external errors above 20 µatm. Recent regional SSS studies report 0.51-0.92 PSU, but do not establish a global coastal gate.

The minimum fCO2 ceilings are 30 µatm for cruise and 40 µatm for spatial/forward transfer. Competitive ceilings are 25/35/35; ideal ceilings are 20/30/30. SSS minimum ceilings are 1.0/1.2/1.2 PSU; competitive 0.8/1.0/1.0; ideal 0.6/0.8/0.8. Each tier additionally requires at least 5/10/15% paired improvement, positive skill in every scheme, no supported-LME RMSE ratio above 1.10, and no major scheme regression above 2%.

## Decision and limitations

Decision: **`benchmark_contract_frozen`**. Literature scores do not select models. A minimum pass also requires 50% coverage in [0.45, 0.55], 90% coverage in [0.85, 0.95], WIS and CRPS where samples exist, and improved risk at 80% retained coverage. Paired uncertainty uses 2,000 cruise-level bootstrap replicates, with a spatial-block variant and macro-region stratification. These thresholds are research gates, not claims that a specific public product is inferior; final product ranking awaits #38 observations and #40/#41 predictions on common support.

## Figure and table index

Tables 1-11 contain the literature catalog, four comparison panels, target tiers, shared gates, v1.x replica, conversion policy, bootstrap, comparability matrix, baseline roles, acceptance checks, and source catalog.

{blocks}

## Primary references

- Roobaert et al. (2024), https://doi.org/10.5194/essd-16-421-2024
- Chau et al. (2024), https://doi.org/10.5194/essd-16-121-2024
- Sharp et al. (2024), https://doi.org/10.1038/s41597-024-03530-7
- Duke et al. (2024), https://doi.org/10.1029/2024JC021134
- Cho et al. (2026), https://doi.org/10.1016/j.apor.2026.105210
- Jung et al. (2025), https://doi.org/10.1016/j.marpolbul.2025.118462
- Sung et al. (2025), https://doi.org/10.1016/j.jag.2025.104427
- Copernicus GLORYS12V1, https://doi.org/10.48670/moi-00021
- ESA CCI SSS, https://climate.esa.int/en/projects/sea-surface-salinity/
- NASA JPL SMAP CAP v5, https://podaac.jpl.nasa.gov/dataset/SMAP_JPL_L3_SSS_CAP_8DAY-RUNNINGMEAN_V5
"""
    write_text(ARCHIVE / "REPORT.md", report)


def build_manifest() -> None:
    tracked = [ARCHIVE / name for name in ("README.md", "REPORT.md", "CAPTIONS.md")]
    tracked += sorted(TABLES.glob("*.csv")) + sorted(FIGURES.glob("*.png"))
    sources = [
        "configs/frozen/benchmark_contract_v2.3.json",
        "configs/frozen/product_scope_v2.3.json",
        "docs/experiment_archive/p1_fco2_v11_parity_v2.2/REPORT.md",
        "docs/experiment_archive/p1_fco2_v11_parity_v2.2/archive_manifest.json",
        "docs/experiment_archive/p2_scope_reconciliation_v2.3/REPORT.md",
    ]
    script = Path(__file__).resolve()
    figure_sources = {
        "fig01_fco2_literature_context.png": ["table01_literature_benchmarks.csv"],
        "fig02_sss_literature_context.png": ["table01_literature_benchmarks.csv"],
        "fig03_panel_comparability.png": ["table08_panel_comparability.csv"],
        "fig04_fco2_targets.png": ["table03_performance_targets.csv"],
        "fig05_sss_targets.png": ["table03_performance_targets.csv"],
        "fig06_panel_hierarchy.png": ["table02_comparison_panels.csv"],
        "fig07_baseline_roles.png": ["table09_baseline_roles.csv"],
        "fig08_acceptance_dimensions.png": [
            "table03_performance_targets.csv",
            "table04_shared_gates.csv",
        ],
    }
    manifest = {
        "experiment_id": "p2_benchmark_freeze_v2.3",
        "training_git_commit": subprocess.check_output(
            ["git", "merge-base", "HEAD", "origin/main"], cwd=ROOT, text=True
        ).strip(),
        "data_manifest_sha256": sha256(DATA_MANIFEST),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "primary literature, official product metadata, and immutable ReCAD governance archives; no labels opened",
        "decision": "benchmark_contract_frozen",
        "figure_source_data": figure_sources,
        "figure_captions": CAPTIONS,
        "files_sha256": {
            str(path.relative_to(ARCHIVE)).replace("\\", "/"): sha256(path) for path in tracked
        },
        "archive_builder_sha256": sha256(script),
        "analysis_script_sha256": sha256(script),
        "source_artifacts_sha256": {relative: sha256(ROOT / relative) for relative in sources},
    }
    write_text(ARCHIVE / "archive_manifest.json", json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    contract = load_benchmark_contract(CONTRACT_PATH)
    errors = validate_benchmark_contract(contract)
    if errors:
        raise ValueError("Invalid benchmark contract:\n- " + "\n- ".join(errors))
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    tables = build_tables(contract)
    for name, rows in tables.items():
        write_csv(name, rows)
    build_figures(tables)
    build_documents()
    build_manifest()
    print(
        json.dumps(
            {
                "archive": str(ARCHIVE),
                "tables": len(tables),
                "figures": len(CAPTIONS),
                "decision": "benchmark_contract_frozen",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
