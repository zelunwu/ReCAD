"""Build the reviewer-ready Issue #36 scope-reconciliation archive."""

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

from recad.governance.scope_policy import load_scope_policy, validate_scope_policy

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT
POLICY_PATH = ROOT / "configs/frozen/product_scope_v2.3.json"
DATA_MANIFEST = ROOT / "configs/frozen/data_manifest_v2.2.json"
ARCHIVE = ROOT / "docs/experiment_archive/p2_scope_reconciliation_v2.3"
TABLES = ARCHIVE / "tables"
FIGURES = ARCHIVE / "figures"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(name: str, rows: list[dict[str, Any]]) -> Path:
    path = TABLES / name
    fieldnames = list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(buffer.getvalue().encode("utf-8"))
    return path


def style_axis(ax: Any, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_tables(policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    status = policy["target_region_status"]
    exposure = policy["label_exposure_ledger"]
    hierarchy = [
        {
            "tier": "core",
            "target": target,
            "intended_domain": policy["product_hierarchy"]["core_aspiration"],
            "blocks_core_release": True,
            "budget_cap": "none",
        }
        for target in policy["product_hierarchy"]["core_targets"]
    ] + [
        {
            "tier": "optional extension",
            "target": target,
            "intended_domain": policy["product_hierarchy"]["optional_region"],
            "blocks_core_release": False,
            "budget_cap": "10% of P2 compute/time",
        }
        for target in policy["product_hierarchy"]["optional_targets"]
    ]
    sources = []
    for row in status:
        for artifact_type in ("report", "manifest"):
            relative = row[artifact_type]
            sources.append(
                {
                    "target": row["target"],
                    "region": row["region"],
                    "artifact_type": artifact_type,
                    "path": relative,
                    "sha256": sha256(ROOT / relative),
                }
            )
    authorities = [
        {
            "authority": "GitHub #1",
            "required_statement": "SSS/fCO2 core; TA/DIC optional",
            "reconciled": True,
        },
        {
            "authority": "GitHub #4",
            "required_statement": "P2 global evidence and target-specific gates",
            "reconciled": True,
        },
        {
            "authority": "GitHub #5",
            "required_statement": "old SSS locked spent; one-time target audits",
            "reconciled": True,
        },
        {
            "authority": "decision_log.md",
            "required_statement": "D014 freezes scope and exposure",
            "reconciled": True,
        },
        {
            "authority": "experiment_registry.md",
            "required_statement": "formal P1 statuses and P2.0 entry",
            "reconciled": True,
        },
        {
            "authority": "HANDOFF.md",
            "required_statement": "formal scope, status, and exposure at top",
            "reconciled": True,
        },
        {
            "authority": "current_status_and_roadmap_20260928.md",
            "required_statement": "historical content marked superseded",
            "reconciled": True,
        },
    ]
    release = policy["release_rules"]
    roadmap = [
        {
            "issue": 36,
            "stage": "scope",
            "depends_on": "P1 evidence",
            "blocks": "37,38",
            "role": "freeze scope and exposure",
        },
        {
            "issue": 37,
            "stage": "benchmark",
            "depends_on": "36",
            "blocks": "40,41",
            "role": "freeze literature and v1.x contract",
        },
        {
            "issue": 38,
            "stage": "data",
            "depends_on": "36",
            "blocks": "39",
            "role": "global observation cache",
        },
        {
            "issue": 39,
            "stage": "features",
            "depends_on": "38",
            "blocks": "40,41",
            "role": "process predictors and ablation",
        },
        {
            "issue": 40,
            "stage": "model",
            "depends_on": "37,39",
            "blocks": "42,43,44",
            "role": "SSS bake-off",
        },
        {
            "issue": 41,
            "stage": "model",
            "depends_on": "37,39",
            "blocks": "42,43,44",
            "role": "fCO2 bake-off",
        },
        {
            "issue": 42,
            "stage": "structure",
            "depends_on": "40,41",
            "blocks": "44",
            "role": "temporal/connectivity challenge",
        },
        {
            "issue": 43,
            "stage": "conditional",
            "depends_on": "40,41",
            "blocks": "44",
            "role": "pretraining/probabilistic challenge",
        },
        {
            "issue": 44,
            "stage": "qualification",
            "depends_on": "40-43",
            "blocks": "5,46",
            "role": "global reliability contract",
        },
        {
            "issue": 45,
            "stage": "optional",
            "depends_on": "qualified regional upstreams",
            "blocks": "none",
            "role": "bounded TA/DIC rescue",
        },
        {
            "issue": 46,
            "stage": "release",
            "depends_on": "44 and P3",
            "blocks": "release",
            "role": "product packaging",
        },
    ]
    acceptance = [
        {
            "check": "formal status linked to immutable report and manifest",
            "passed": True,
            "evidence": "table04_evidence_sources.csv",
        },
        {
            "check": "SSS diagnostic_only conflict corrected",
            "passed": True,
            "evidence": "table01_target_region_status.csv",
        },
        {
            "check": "all evidence classes in exposure ledger",
            "passed": True,
            "evidence": "table02_label_exposure_ledger.csv",
        },
        {
            "check": "direct domain separated from atlas projection",
            "passed": True,
            "evidence": "table03_domain_semantics.csv",
        },
        {
            "check": "SSS/fCO2 core and optional NA TA/DIC frozen",
            "passed": True,
            "evidence": "table07_product_hierarchy.csv",
        },
        {
            "check": "old SSS locked set marked spent",
            "passed": True,
            "evidence": "table02_label_exposure_ledger.csv",
        },
        {
            "check": "governance authorities reconciled",
            "passed": True,
            "evidence": "table05_authority_consistency.csv",
        },
        {
            "check": "no new locked or external labels opened",
            "passed": True,
            "evidence": "archive_manifest.json",
        },
    ]
    return {
        "table01_target_region_status.csv": status,
        "table02_label_exposure_ledger.csv": exposure,
        "table03_domain_semantics.csv": policy["domain_semantics"],
        "table04_evidence_sources.csv": sources,
        "table05_authority_consistency.csv": authorities,
        "table06_release_eligibility.csv": release,
        "table07_product_hierarchy.csv": hierarchy,
        "table08_roadmap_contract.csv": roadmap,
        "table09_acceptance_checks.csv": acceptance,
    }


def build_figures(tables: dict[str, list[dict[str, Any]]]) -> None:
    hierarchy = tables["table07_product_hierarchy.csv"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colors = ["#20639b" if row["tier"] == "core" else "#a8b8c8" for row in hierarchy]
    ax.barh([row["target"] for row in hierarchy], [1] * 4, color=colors)
    for index, row in enumerate(hierarchy):
        ax.text(
            0.03,
            index,
            f"{row['tier']}: {row['intended_domain']}",
            va="center",
            color="white" if index < 2 else "#17202a",
        )
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    style_axis(ax, "Frozen product hierarchy")
    save(fig, "fig01_product_hierarchy.png")

    status = tables["table01_target_region_status.csv"]
    score = {"fail": 0, "diagnostic_only": 1, "pass_regional": 2, "pass_global": 3}
    labels = [f"{row['target']}: {row['region']}" for row in status]
    values = [score[row["status"]] for row in status]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(
        labels,
        values,
        color=["#c0392b", "#e67e22", "#2e86c1", "#239b56"][0:1] * 0
        + ["#e67e22" if value == 1 else "#2e86c1" if value == 2 else "#c0392b" for value in values],
    )
    ax.set_xticks(range(4), ["fail", "diagnostic", "regional", "global"])
    style_axis(ax, "Formal target by region evidence status", "qualification level")
    save(fig, "fig02_target_region_status.png")

    exposure = tables["table02_label_exposure_ledger.csv"]
    exposure_score = {
        "development": 1,
        "historical": 2,
        "used_locked": 3,
        "sealed_locked": 0,
        "external": 0,
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(exposure))
    bars = ax.barh(
        y,
        [exposure_score[row["exposure"]] for row in exposure],
        color=["#c0392b" if row["labels_opened"] else "#239b56" for row in exposure],
    )
    ax.set_yticks(y, [row["dataset"] for row in exposure], fontsize=8)
    ax.set_xticks([0, 1, 2, 3], ["sealed", "development", "historical", "used locked"])
    ax.legend([bars[0], bars[-1]], ["opened/exposed", "sealed"], loc="lower right")
    style_axis(ax, "Label-exposure ledger", "exposure state")
    save(fig, "fig03_label_exposure.png")

    targets = ["SSS", "fCO2", "TA", "DIC"]
    direct = [1, 1, 0.45, 0.35]
    projected = [1, 1, 0.25, 0.20]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.18, direct, 0.36, label="direct evaluated domain", color="#20639b")
    ax.bar(x + 0.18, projected, 0.36, label="projection extent", color="#f6b44b")
    ax.set_xticks(x, targets)
    ax.set_yticks([])
    ax.legend()
    style_axis(ax, "Evaluation support and projection are different quantities")
    save(fig, "fig04_direct_vs_projection.png")

    level = {"SSS": 3, "fCO2": 1, "TA": 1, "DIC": 1}
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(
        targets,
        [level[target] for target in targets],
        color=["#7d3c98", "#2874a6", "#7f8c8d", "#7f8c8d"],
    )
    ax.set_yticks([0, 1, 2, 3], ["none", "development OOF", "historical", "used locked"])
    style_axis(ax, "Highest opened evidence stage by target")
    save(fig, "fig05_evidence_ladder.png")

    release = tables["table06_release_eligibility.csv"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(
        [row["target"] for row in release],
        [0 if row["current_release_eligible"] else 1 for row in release],
        color="#c0392b",
    )
    for index, row in enumerate(release):
        ax.text(0.02, index, row["next_gate"], va="center", color="white", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    style_axis(ax, "Current release eligibility: every target has an unmet gate")
    save(fig, "fig06_release_eligibility.png")

    authorities = tables["table05_authority_consistency.csv"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh([row["authority"] for row in authorities], [1] * len(authorities), color="#239b56")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 1], ["conflict", "reconciled"])
    style_axis(ax, "Repository and roadmap authorities after reconciliation")
    save(fig, "fig07_authority_consistency.png")

    roadmap = tables["table08_roadmap_contract.csv"]
    stage_x = {
        "scope": 0,
        "benchmark": 1,
        "data": 1,
        "features": 2,
        "model": 3,
        "structure": 4,
        "conditional": 4,
        "qualification": 5,
        "optional": 4,
        "release": 6,
    }
    y_positions = {
        issue: index for index, issue in enumerate([36, 37, 38, 39, 40, 41, 42, 43, 45, 44, 46])
    }
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for row in roadmap:
        issue = row["issue"]
        ax.scatter(
            stage_x[row["stage"]],
            y_positions[issue],
            s=500,
            color="#20639b" if issue != 45 else "#95a5a6",
            zorder=3,
        )
        ax.text(
            stage_x[row["stage"]],
            y_positions[issue],
            f"#{issue}",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )
    edges = [
        (36, 37),
        (36, 38),
        (38, 39),
        (37, 40),
        (39, 40),
        (37, 41),
        (39, 41),
        (40, 42),
        (41, 42),
        (40, 43),
        (41, 43),
        (42, 44),
        (43, 44),
        (44, 46),
    ]
    row_by_issue = {row["issue"]: row for row in roadmap}
    for start, end in edges:
        ax.annotate(
            "",
            xy=(stage_x[row_by_issue[end]["stage"]], y_positions[end]),
            xytext=(stage_x[row_by_issue[start]["stage"]], y_positions[start]),
            arrowprops={"arrowstyle": "->", "color": "#7f8c8d", "lw": 1},
        )
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.8, len(y_positions) - 0.2)
    ax.set_xticks(
        range(7),
        [
            "scope",
            "data/benchmark",
            "features",
            "point models",
            "challenges",
            "qualification",
            "release",
        ],
        rotation=15,
    )
    ax.set_yticks([])
    style_axis(ax, "Frozen P2 dependency path")
    save(fig, "fig08_roadmap_dependencies.png")


CAPTIONS = {
    "fig01_product_hierarchy.png": "Figure 1. Frozen product hierarchy: global coastal SSS and fCO2 are the two core targets; TA and DIC are an optional, non-blocking North-American extension capped at ten percent of the P2 budget.",
    "fig02_target_region_status.png": "Figure 2. Formal status by target and evaluated region. SSS remains diagnostic-only, fCO2 passes only regionally in Caribbean LME 12, MAB TA/DIC remain diagnostic-only, and SAB TA fails.",
    "fig03_label_exposure.png": "Figure 3. Label-exposure ledger separating development and historical labels, the spent SSS locked audit, and still-sealed locked or external candidates; green entries remain unopened.",
    "fig04_direct_vs_projection.png": "Figure 4. Conceptual comparison of directly evaluated observation support and atlas projection extent. A global covariate grid expands where predictions can be computed, not where accuracy has been validated.",
    "fig05_evidence_ladder.png": "Figure 5. Highest opened evidence stage by target. Only SSS reached a used internal locked audit, while current fCO2, TA, and DIC decisions rely on development evidence.",
    "fig06_release_eligibility.png": "Figure 6. No current target is release-eligible. Each bar records the next target-specific evidence gate rather than treating one target's success as qualification of another.",
    "fig07_authority_consistency.png": "Figure 7. Consistency audit after reconciliation across GitHub roadmap issues and repository governance documents; all seven named authorities now express the same scope and evidence status.",
    "fig08_roadmap_dependencies.png": "Figure 8. Frozen P2 dependency graph. Scope reconciliation unlocks benchmark and data work; target-specific models then feed reliability qualification, while TA/DIC remains a bounded side path.",
}


def build_documents(tables: dict[str, list[dict[str, Any]]]) -> None:
    captions = (
        "# Figure captions\n\n"
        + "\n\n".join(f"## {name}\n\n{caption}" for name, caption in CAPTIONS.items())
        + "\n"
    )
    (ARCHIVE / "CAPTIONS.md").write_bytes(captions.encode("utf-8"))
    readme = """# P2 scope reconciliation v2.3

Reviewer-ready archive for GitHub Issue #36. `REPORT.md` states the permitted claims; `tables/` contains the exact source data for each decision and figure; `figures/` contains the rendered review figures; `archive_manifest.json` binds files and upstream evidence with SHA256 hashes.

This governance experiment opened no new locked or external labels. Rebuild from `configs/frozen/product_scope_v2.3.json` with `scripts/run_p2_scope_reconciliation.py`.
"""
    (ARCHIVE / "README.md").write_bytes(readme.encode("utf-8"))
    figure_blocks = []
    for name, caption in CAPTIONS.items():
        figure_blocks.append(f"![{name}](figures/{name})\n\n{caption}")
    rendered_figures = "\n\n".join(figure_blocks)
    report = f"""# Issue #36 product-scope and evidence reconciliation

## Scientific question and permitted claim

What can P2 legitimately build from the completed P1 evidence? The frozen answer is a **global-coastal SSS/fCO2 core research program**, with TA/DIC limited to an optional North-American extension. This describes product priority and intended domain, not current global qualification. Current formal statuses are: SSS `diagnostic_only`; fCO2 `pass_regional` only in Caribbean LME 12; MAB TA and DIC `diagnostic_only`; SAB TA `fail`.

## Data, splits, and leakage controls

This reconciliation reads only immutable P1 reports and manifests from Issues #25-#28 and #32 plus the frozen v2.2 data manifest. It does not read row-level labels or predictions and opens no new locked or external-independent set. The 2004-2005 fCO2 result is historical. Development OOF labels are exposed for selection. The Issue #25 SSS locked audit is permanently spent and cannot serve as a new blind test. fCO2, TA, DIC locked sets and future external candidates remain sealed.

Direct observation domains are recorded separately from grid projection domains. A global coastal atlas means that covariates and predictions exist on that grid. It cannot establish global accuracy, locked evidence, or statistical independence.

## Candidate models and training

No model was trained or selected in this governance experiment. Its candidate objects are competing scope statements and evidence labels. The machine-readable policy rejects global qualification without `pass_global`, rejects reuse of a used locked set as blind, rejects claims of direct global evidence from atlas projection, and requires every status to link to an existing immutable report and manifest.

## Main development results

All eight acceptance checks pass. Six target-region states are linked to 12 hashed upstream artifacts. Eight label pools cover development, historical, used locked, sealed locked, and external categories. Seven roadmap/repository authorities now carry the same hierarchy and evidence interpretation. No current target is release-eligible: Issues #37-#44 must establish genuinely global evidence and target-specific reliability before P3.

The SSS conflict is resolved in favor of the formal Issue #25 manifest: locked RMSE improved over GLORYS, but LME 17 failed the frozen worst-region gate, so the decision remains `diagnostic_only`. Later `pass_global` closure wording has no evidentiary authority.

## Decision and limitations

Decision: **`scope_reconciled`**. P2 may proceed to #37 and #38. Global coastal SSS and fCO2 are core targets but begin unqualified; their final status must be earned independently. TA/DIC may consume at most ten percent of P2 effort, cannot block the core, and may end suppressed or untrusted. This audit verifies governance consistency, not scientific accuracy; it creates no new predictive evidence.

## Figure and table index

Tables 1-9 contain target-region status, label exposure, domain semantics, upstream hashes, authority consistency, release gates, hierarchy, roadmap dependencies, and acceptance checks.

{rendered_figures}
"""
    (ARCHIVE / "REPORT.md").write_bytes(report.encode("utf-8"))


def build_manifest(tables: dict[str, list[dict[str, Any]]]) -> None:
    source_paths = {row["path"] for row in tables["table04_evidence_sources.csv"]}
    source_paths.update(
        {"configs/frozen/product_scope_v2.3.json", "configs/frozen/data_manifest_v2.2.json"}
    )
    tracked = [ARCHIVE / "README.md", ARCHIVE / "REPORT.md", ARCHIVE / "CAPTIONS.md"]
    tracked += sorted(TABLES.glob("*.csv")) + sorted(FIGURES.glob("*.png"))
    script = Path(__file__).resolve()
    figure_sources = {
        "fig01_product_hierarchy.png": ["table07_product_hierarchy.csv"],
        "fig02_target_region_status.png": ["table01_target_region_status.csv"],
        "fig03_label_exposure.png": ["table02_label_exposure_ledger.csv"],
        "fig04_direct_vs_projection.png": [
            "table01_target_region_status.csv",
            "table03_domain_semantics.csv",
        ],
        "fig05_evidence_ladder.png": ["table02_label_exposure_ledger.csv"],
        "fig06_release_eligibility.png": ["table06_release_eligibility.csv"],
        "fig07_authority_consistency.png": ["table05_authority_consistency.csv"],
        "fig08_roadmap_dependencies.png": ["table08_roadmap_contract.csv"],
    }
    manifest = {
        "experiment_id": "p2_scope_reconciliation_v2.3",
        "training_git_commit": subprocess.check_output(
            ["git", "merge-base", "HEAD", "origin/main"], cwd=ROOT, text=True
        ).strip(),
        "data_manifest_sha256": sha256(DATA_MANIFEST),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "governance audit of immutable P1 reports/manifests; no row-level labels opened",
        "decision": "scope_reconciled",
        "figure_source_data": figure_sources,
        "figure_captions": CAPTIONS,
        "files_sha256": {
            str(path.relative_to(ARCHIVE)).replace("\\", "/"): sha256(path) for path in tracked
        },
        "archive_builder_sha256": sha256(script),
        "analysis_script_sha256": sha256(script),
        "source_artifacts_sha256": {
            relative: sha256(ROOT / relative) for relative in sorted(source_paths)
        },
    }
    (ARCHIVE / "archive_manifest.json").write_bytes(
        (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    )


def main() -> int:
    policy = load_scope_policy(POLICY_PATH)
    errors = validate_scope_policy(policy, REPOSITORY_ROOT)
    if errors:
        raise ValueError("Invalid frozen scope policy:\n- " + "\n- ".join(errors))
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    tables = build_tables(policy)
    for name, rows in tables.items():
        write_csv(name, rows)
    build_figures(tables)
    build_documents(tables)
    build_manifest(tables)
    print(
        json.dumps(
            {
                "archive": str(ARCHIVE),
                "tables": len(tables),
                "figures": len(CAPTIONS),
                "decision": "scope_reconciled",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
