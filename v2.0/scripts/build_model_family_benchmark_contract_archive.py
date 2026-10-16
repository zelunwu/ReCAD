"""Build the reviewer archive for the Issue 88 benchmark preregistration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recad.evaluate.benchmark_contract import load_benchmark_contract, primary_cell_budget

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/frozen/model_family_benchmark_v2.5.in"
ARCHIVE = ROOT / "docs/experiment_archive/p2_model_family_benchmark_v2.5"
FIGURES = ARCHIVE / "figures"
TABLES = ARCHIVE / "tables"
VALIDATOR = ROOT / "scripts/validate_model_family_benchmark.py"


CAPTIONS = {
    "fig01_family_taxonomy.png": (
        "Figure 1. The frozen benchmark contains 14 model-family columns spanning linear, "
        "bagged-tree, boosted-tree, tabular-neural, and spatiotemporal-neural approaches. SOM "
        "and GRaCER are excluded from the columns because they are spatial-organization rows."
    ),
    "fig02_search_budgets.png": (
        "Figure 2. Every family receives a preregistered minimum and maximum optimization budget. "
        "A family may stop after the minimum only when the search and capacity tails meet the same "
        "empirical-convergence threshold; reaching the maximum without convergence is preserved as "
        "optimization_incomplete."
    ),
    "fig03_seed_requirements.png": (
        "Figure 3. Tree and linear families require at least three completed seeds, while neural "
        "families require five. Seed counts are convergence requirements and cannot be reduced after "
        "outer-fold results are inspected."
    ),
    "fig04_protocol_weights.png": (
        "Figure 4. The frozen primary score gives the largest weight to whole-region transfer, then "
        "buffered geography, with complete-cruise, forward-time, and early-era evidence retained. "
        "Every protocol uses cruise-equal RMSE normalized by its matched strict control."
    ),
    "fig05_fidelity_schedule.png": (
        "Figure 5. Every family follows the same 120,000-row, 300,000-row, and full-data fidelity "
        "sequence. The lower fidelities allocate search effort; only a completed full-data stage can "
        "be declared empirically converged."
    ),
    "fig06_convergence_gates.png": (
        "Figure 6. Advancement requires all convergence gates simultaneously: minimum trials, the "
        "full eligible data, the family-specific seed count, a flat final search tail, and a flat "
        "capacity tail. A failed gate remains visible rather than being converted into a poor score."
    ),
    "fig07_primary_cell_budget.png": (
        "Figure 7. The roadmap bounds the scientific comparison at 68 primary target-architecture "
        "cells: 28 global cells, at most 30 additional spatial cells for five advancing families per "
        "target, and at most 10 target-representation cells. Folds, seeds, and trials are repetitions."
    ),
    "fig08_information_policy.png": (
        "Figure 8. Main-ranking information is deliberately restricted. All families receive the "
        "same eligible rows within a target and deployment-valid nested SSS for fCO2; external target "
        "products, external pretraining, one-time independent labels, and 2026 labels are excluded."
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_text_sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def write_text(path: Path, text: str) -> None:
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def source_tables() -> dict[str, pd.DataFrame]:
    contract = load_benchmark_contract(CONFIG)
    families = pd.DataFrame(
        [
            {
                "family": item.name,
                "kind": item.kind,
                "representation": item.representation,
                "backend": item.backend,
                "neural": item.neural,
                "minimum_trials": item.minimum_trials,
                "maximum_trials": item.maximum_trials,
                "minimum_seeds": item.minimum_seeds,
                "search_parameters": len(item.search_space),
            }
            for item in contract.family_contracts.values()
        ]
    )
    protocols = pd.DataFrame(
        [(name, contract.protocol_weights[name]) for name in contract.protocols],
        columns=["protocol", "weight"],
    )
    fidelity = pd.DataFrame(
        [(index + 1, value, index == 2) for index, value in enumerate(contract.fidelity_rows)],
        columns=["stage", "eligible_rows", "required_for_convergence"],
    )
    convergence = pd.DataFrame(
        [
            ("minimum trials completed", "family-specific", "required"),
            ("full eligible rows completed", "100%", "required"),
            ("minimum seeds completed", "3 non-neural; 5 neural", "required"),
            ("final 20% search improvement", "<0.25%", "required"),
            ("capacity-tail improvement", "<0.25%", "required"),
        ],
        columns=["gate", "threshold", "status"],
    )
    selection = pd.DataFrame(
        [
            (1, "weighted geometric mean of normalized cruise-equal RMSE"),
            (2, "whole-region normalized cruise-equal RMSE"),
            (3, "worst-region normalized cruise-equal RMSE"),
            (4, "total wall seconds"),
            (5, "stable family name"),
        ],
        columns=["order", "criterion"],
    )
    invariants = pd.DataFrame(
        [
            ("same eligible rows within target", True),
            ("same corrected outer identities", True),
            ("nested deployable SSS for fCO2", True),
            ("external target products", False),
            ("external pretraining in main ranking", False),
            ("one-time independent labels", False),
            ("2026 labels", False),
            ("family-specific external subsample", False),
        ],
        columns=["information_or_rule", "allowed_or_required"],
    )
    literature = pd.DataFrame(
        [
            (
                "Random Forest / boosted trees",
                "NIES-ML3 and recent pCO2 mapping",
                "https://www.nies.go.jp/doi/10.17595/20220311.001-e.html",
            ),
            ("SOM plus FFNN", "ULB coastal product", "https://doi.org/10.5194/essd-16-421-2024"),
            (
                "Random cluster ensemble",
                "OceanSODA-ETHZ GRaCER",
                "https://doi.org/10.5194/essd-13-777-2021",
            ),
            (
                "Residual target plus FFNN",
                "OceanSODA-ETHZv2",
                "https://doi.org/10.1029/2024GB008127",
            ),
            (
                "U-Net plus ConvLSTM",
                "physics-informed transfer reconstruction",
                "https://doi.org/10.1038/s41597-024-03959-w",
            ),
            ("Vision Transformer", "SJTU AViT", "https://doi.org/10.5194/essd-17-6071-2025"),
            ("TabM", "ICLR 2025 tabular baseline", "https://openreview.net/forum?id=sdFefxP4gn"),
            ("TabPFN", "tabular foundation model", "https://doi.org/10.1038/s41586-024-08328-6"),
        ],
        columns=["family_or_structure", "coverage_role", "source"],
    )
    return {
        "table01_model_families.csv": families,
        "table02_protocol_weights.csv": protocols,
        "table03_fidelity_schedule.csv": fidelity,
        "table04_convergence_gates.csv": convergence,
        "table05_selection_order.csv": selection,
        "table06_primary_cell_budget.csv": primary_cell_budget(contract),
        "table07_information_invariants.csv": invariants,
        "table08_literature_coverage.csv": literature,
    }


def _save(figure: plt.Figure, name: str) -> None:
    figure.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close(figure)


def build_figures(tables: dict[str, pd.DataFrame]) -> None:
    families = tables["table01_model_families.csv"]
    figure, axis = plt.subplots(figsize=(10, 5))
    families.kind.value_counts().sort_values().plot.barh(ax=axis, color="#377eb8")
    axis.set(xlabel="Frozen family columns", ylabel="Model class", title="Benchmark model taxonomy")
    _save(figure, "fig01_family_taxonomy.png")

    figure, axis = plt.subplots(figsize=(12, 6))
    positions = np.arange(len(families))
    axis.barh(positions, families.maximum_trials, color="#a6cee3", label="maximum")
    axis.barh(positions, families.minimum_trials, color="#1f78b4", label="minimum")
    axis.set_yticks(positions, families.family)
    axis.set(xlabel="Optimization trials", title="Preregistered family-specific search budgets")
    axis.legend()
    _save(figure, "fig02_search_budgets.png")

    figure, axis = plt.subplots(figsize=(11, 5))
    colors = np.where(families.neural, "#e31a1c", "#33a02c")
    axis.bar(families.family, families.minimum_seeds, color=colors)
    axis.tick_params(axis="x", rotation=55)
    axis.set(ylabel="Required completed seeds", title="Seed requirement by family")
    _save(figure, "fig03_seed_requirements.png")

    protocols = tables["table02_protocol_weights.csv"]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(protocols.protocol, protocols.weight, color="#6a3d9a")
    axis.tick_params(axis="x", rotation=25)
    axis.set(
        ylabel="Primary-score weight", title="Frozen development-protocol weights", ylim=(0, 0.35)
    )
    _save(figure, "fig04_protocol_weights.png")

    fidelity = tables["table03_fidelity_schedule.csv"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(fidelity.stage, [120000, 300000, 750040], marker="o", linewidth=3, color="#ff7f00")
    axis.set(
        xticks=fidelity.stage,
        xlabel="Fidelity stage",
        ylabel="Rows (illustrative SSS full size)",
        title="Common multi-fidelity schedule",
    )
    axis.text(3, 750040, "FULL", ha="left", va="bottom")
    _save(figure, "fig05_fidelity_schedule.png")

    convergence = tables["table04_convergence_gates.csv"]
    figure, axis = plt.subplots(figsize=(11, 4.8))
    axis.barh(convergence.gate, np.ones(len(convergence)), color="#1b9e77")
    for index, threshold in enumerate(convergence.threshold):
        axis.text(0.5, index, threshold, ha="center", va="center", color="white", weight="bold")
    axis.set(xlim=(0, 1), xticks=[], title="All empirical-convergence gates must pass")
    _save(figure, "fig06_convergence_gates.png")

    budget = tables["table06_primary_cell_budget.csv"].iloc[:3]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(budget.stage, budget.primary_cells, color=["#1f78b4", "#33a02c", "#ff7f00"])
    axis.tick_params(axis="x", rotation=15)
    axis.set(ylabel="Primary target-architecture cells", title="Bounded P2.9 scientific matrix")
    axis.text(1.5, 31, "Total upper bound = 68", ha="center", weight="bold")
    _save(figure, "fig07_primary_cell_budget.png")

    invariants = tables["table07_information_invariants.csv"]
    figure, axis = plt.subplots(figsize=(11, 5.5))
    values = invariants.allowed_or_required.astype(int)
    colors = np.where(values == 1, "#1a9850", "#d73027")
    axis.barh(invariants.information_or_rule, np.ones(len(invariants)), color=colors)
    for index, value in enumerate(values):
        axis.text(
            0.5,
            index,
            "required" if value else "prohibited",
            ha="center",
            va="center",
            color="white",
            weight="bold",
        )
    axis.set(xlim=(0, 1), xticks=[], title="Main-ranking information policy")
    _save(figure, "fig08_information_policy.png")


def report_text(tables: dict[str, pd.DataFrame]) -> str:
    contract = load_benchmark_contract(CONFIG)
    family_rows = "\n".join(
        f"| {row.family} | {row.kind} | {row.representation} | {row.minimum_trials}-{row.maximum_trials} | {row.minimum_seeds} |"
        for row in tables["table01_model_families.csv"].itertuples()
    )
    figures = "\n\n".join(
        f"![{caption}](figures/{name})\n\n{caption}" for name, caption in CAPTIONS.items()
    )
    return f"""# Fair model-family and optimization benchmark preregistration

## Scientific question and permitted claim

Which globally applied model family best reconstructs coastal SSS and fCO2 when every family receives the same eligible observations, predictors, corrected outer identities, nested upstream lineage, and an empirically converged family-specific optimization? This Issue 88 archive freezes the comparison before new benchmark results are inspected. It contains no model-performance result and promotes no model.

## Data, splits, and leakage controls

The benchmark reuses the Issue-61 development identities for complete cruises, buffered geographic blocks, whole regions, forward time, and early era. Within a target, all families receive identical eligible rows. fCO2 receives only outer-nested deployment-valid SSS predictions. External pCO2/fCO2 products, external target pretraining, one-time independent labels, and 2026 labels are prohibited from the main ranking. TabPFN receives the full eligible pool; internal estimator sampling is part of its frozen model configuration rather than a special external subset.

## Candidate models and training

| Family | Class | Representation | Trial range | Seeds |
|---|---|---|---:|---:|
{family_rows}

Every family follows 120,000-row, 300,000-row, and full-data fidelity stages. It must complete at least {contract.family_contracts["ridge"].minimum_seeds} seeds, or five for neural families. Empirical convergence requires a complete full-data stage and less than {contract.maximum_tail_relative_improvement:.2%} improvement in both the final {contract.tail_fraction:.0%} of the search and the final capacity tail. A family that exhausts its maximum budget without satisfying these gates remains `optimization_incomplete` and cannot advance.

## Main development results

The machine decision is `benchmark_contract_frozen_no_models_ranked`. Fourteen model-family columns and five outer protocols are fixed. The top five eligible, empirically converged families per target will advance by the same deterministic score: the weighted geometric mean of matched-control-normalized cruise-equal RMSE across all five protocols. Whole-region ratio, worst-region ratio, wall time, and stable family name are frozen tie breakers. Missing protocols, missing secondary metrics, or incomplete optimization make a family ineligible rather than assigning an artificial poor rank.

The complete roadmap is bounded at 68 primary scientific cells. Hyperparameter trials, seeds, and folds are repetitions inside a cell. SOM and GRaCER remain spatial-organization rows for later issues and are not additional predictor-family columns.

{figures}

## Decision and limitations

Decision: `freeze_fair_model_family_benchmark_before_training`. The contract resolves the earlier unfairness in which Random Forest lacked the same optimization opportunity and TabPFN was proposed for a unique 50,000-row subset. It does not claim that the mathematical global optimum can be proven; it defines auditable empirical convergence under finite resource ceilings. Optional packages and large neural implementations must pass their preregistered smoke paths before P2.9b starts. A dependency or resource failure must remain an explicit infeasible or incomplete result and cannot be replaced after seeing another family's score.

The ranking selects five columns for controlled regionalization; it does not authorize different model families in different regions. The later shared-partition issues retain one family per target architecture and one target-independent partition for SSS and fCO2.

## Figure and table index

- Figures 1-8 correspond one-to-one to Tables 1-8 in `tables/`.
- Tables 1-8 contain the model taxonomy, protocol weights, fidelity schedule, convergence gates, selection order, bounded cell budget, information invariants, and literature coverage.
- Frozen input: `configs/frozen/model_family_benchmark_v2.5.in`.
- Machine validator: `scripts/validate_model_family_benchmark.py`.
"""


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    tables = source_tables()
    for name, table in tables.items():
        write_text(TABLES / name, table.to_csv(index=False, lineterminator="\n"))
    build_figures(tables)
    report = report_text(tables)
    captions = (
        "# Figure captions\n\n"
        + "\n\n".join(f"## {name}\n\n{caption}" for name, caption in CAPTIONS.items())
        + "\n"
    )
    readme = """# Issue 88 fair model-family benchmark contract

This reviewer archive freezes the model-family columns, data invariants, optimization-convergence rules, deterministic advancement score, and bounded experiment size before P2.9b training. Rebuild with `python scripts/build_model_family_benchmark_contract_archive.py` and verify with `python scripts/verify_experiment_archive.py docs/experiment_archive/p2_model_family_benchmark_v2.5`.
"""
    write_text(ARCHIVE / "REPORT.md", report)
    write_text(ARCHIVE / "CAPTIONS.md", captions)
    write_text(ARCHIVE / "README.md", readme)

    tracked = [ARCHIVE / "README.md", ARCHIVE / "REPORT.md", ARCHIVE / "CAPTIONS.md"]
    tracked += sorted(FIGURES.glob("*.png")) + sorted(TABLES.glob("*.csv"))
    figure_sources = {
        "fig01_family_taxonomy.png": ["table01_model_families.csv"],
        "fig02_search_budgets.png": ["table01_model_families.csv"],
        "fig03_seed_requirements.png": ["table01_model_families.csv"],
        "fig04_protocol_weights.png": ["table02_protocol_weights.csv"],
        "fig05_fidelity_schedule.png": ["table03_fidelity_schedule.csv"],
        "fig06_convergence_gates.png": ["table04_convergence_gates.csv"],
        "fig07_primary_cell_budget.png": ["table06_primary_cell_budget.csv"],
        "fig08_information_policy.png": ["table07_information_invariants.csv"],
    }
    manifest = {
        "experiment_id": "p2_model_family_benchmark_v2.5",
        "training_git_commit": "not_applicable_preregistration",
        "data_manifest_sha256": tracked_text_sha256(CONFIG),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "methodology and optimization preregistration; no new model result",
        "decision": {
            "machine_decision": "benchmark_contract_frozen_no_models_ranked",
            "family_count": 14,
            "target_count": 2,
            "primary_cell_upper_bound": 68,
            "models_ranked": False,
        },
        "figure_source_data": figure_sources,
        "figure_captions": CAPTIONS,
        "files_sha256": {
            str(path.relative_to(ARCHIVE)).replace("\\", "/"): sha256(path) for path in tracked
        },
        "archive_builder_sha256": tracked_text_sha256(Path(__file__)),
        "analysis_script_sha256": tracked_text_sha256(VALIDATOR),
        "source_artifacts_sha256": {
            str(CONFIG.relative_to(ROOT)).replace("\\", "/"): tracked_text_sha256(CONFIG)
        },
    }
    write_text(ARCHIVE / "archive_manifest.json", json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
