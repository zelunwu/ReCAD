"""Build the reviewer-facing preregistration archive for Issue 81."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from recad.evaluate.error_tier_contract import load_error_flag_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "frozen" / "absolute_error_flag_contract_v2.5.in"
ARCHIVE = ROOT / "docs" / "experiment_archive" / "p2_absolute_error_flag_contract_v2.5"
FIGURES = ARCHIVE / "figures"
TABLES = ARCHIVE / "tables"
VALIDATOR = ROOT / "scripts" / "validate_error_flag_contract.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tracked_text(path: Path) -> str:
    """Hash tracked text in the repository's canonical LF representation."""

    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def write_utf8_lf(path: Path, text: str) -> None:
    """Write deterministic UTF-8 text without platform newline conversion."""

    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def decision_table() -> pd.DataFrame:
    """Return the frozen ordering of product flag decisions."""

    return pd.DataFrame(
        [
            ("Outside declared domain", "fill", "fill", "NA", "fill"),
            ("Invalid or failed numerical prediction", "fill", "N", "VI", "fill"),
            ("Error evidence unavailable", "retain", "U", "VU", "fill"),
            ("Applicable uncertainty audit failed", "retain", "U", "VF", "fill"),
            ("Q90 upper 95% <= T_A", "retain", "A", "VD or VT", "publish"),
            ("T_A < Q90 upper 95% <= T_B", "retain", "B", "VD or VT", "publish"),
            ("T_B < Q90 upper 95% <= T_C", "retain", "C", "VD or VT", "publish"),
            ("Q90 upper 95% > T_C", "retain", "D", "VD or VT", "publish"),
        ],
        columns=["condition", "prediction", "error_tier", "evidence", "scientific_error_fields"],
    )


def build_figure(table: pd.DataFrame) -> None:
    colors = {
        "fill": "#f3f3f3",
        "N": "#555555",
        "U": "#9e9e9e",
        "A": "#1a9850",
        "B": "#91cf60",
        "C": "#fee08b",
        "D": "#d73027",
    }
    figure, axis = plt.subplots(figsize=(12, 6.8))
    axis.axis("off")
    rows = table[["condition", "error_tier", "evidence", "scientific_error_fields"]].values
    visual = axis.table(
        cellText=rows,
        colLabels=["Ordered condition", "Tier", "Evidence", "Error fields"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.58, 0.09, 0.14, 0.16],
        bbox=[0.01, 0.02, 0.98, 0.9],
    )
    visual.auto_set_font_size(False)
    visual.set_fontsize(10)
    visual.scale(1, 1.5)
    for column in range(4):
        visual[(0, column)].set_facecolor("#d9eaf7")
        visual[(0, column)].set_text_props(weight="bold")
    for row, tier in enumerate(table.error_tier, start=1):
        visual[(row, 1)].set_facecolor(colors[tier])
        visual[(row, 1)].set_text_props(
            color="white" if tier in {"N", "D"} else "black", weight="bold", ha="center"
        )
    axis.set_title(
        "Absolute reconstruction-error tier and evidence decision contract",
        fontsize=14,
        weight="bold",
        pad=10,
    )
    figure.savefig(FIGURES / "fig01_error_tier_decision_contract.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def build_report() -> tuple[str, str, str]:
    contract = load_error_flag_contract(CONFIG)
    caption = (
        "Figure 1. Ordered Issue 81 decision contract. A-D require a calibrated physical-error "
        "estimate and eligible direct or transfer evidence. U retains a numerical reconstruction "
        "while explicitly withholding an unsupported error claim; D means supported evidence of "
        "large error rather than unknown error."
    )
    threshold_rows = "\n".join(
        f"| {name} | {profile.unit} | UNSET | UNSET | UNSET |"
        for name, profile in contract.targets.items()
    )
    report = f"""# Absolute reconstruction-error flag contract preregistration

## Scientific question and permitted claim

Can ReCAD replace model-relative risk percentiles with fixed physical reconstruction-error tiers while preserving an explicit unknown state where error is not identifiable? This archive freezes the method and schema only. It does not assign A-D, because the scientific tolerances remain deliberately unset.

## Data, splits, and leakage controls

The future residual experiment will keep whole cruises intact and separately evaluate buffered spatial blocks, whole-region transfer, forward time, and early-era reconstruction. Every target-derived transformation, point model, upstream SSS estimate, error model, and calibration object must remain inside the appropriate training partition. The already opened Issue 5 labels cannot tune thresholds.

## Candidate models and training

No point model is selected or trained in this preregistration. The contract is model-agnostic. A future conditional-error model may estimate the 90th-percentile absolute grid-month error in physical units, but its score cannot directly define a tier. Measurement, representation, input, fitting, structural, and transfer errors remain a versioned error ledger; covariance and double-counting must be audited.

## Main development results

The methodology is frozen as `{contract.status}`. The primary statistic is `{contract.primary_statistic}`: the one-sided {contract.confidence_level:.0%} confidence upper bound for the conditional {contract.absolute_error_quantile:.0%} absolute grid-month reconstruction-error quantile. Production tier assignment fails closed until all thresholds are frozen.

| Target | Unit | T_A | T_B | T_C |
|---|---|---:|---:|---:|
{threshold_rows}

![Absolute reconstruction-error tier and evidence decision contract.](figures/fig01_error_tier_decision_contract.png)

{caption}

## Decision and limitations

Decision: `freeze_methodology_and_schema; block_production_grading_until_threshold_derivation`. A-D are supported physical-error tiers. U means that a numerical reconstruction exists but its error is not estimable or its applicable calibration failed. N means no valid prediction. VD and VT are eligible evidence states; VU and VF cannot receive A-D. Basin identity cannot directly change a tier. Current v2.4 percentile flags remain legacy relative applicability ranks and cannot be mechanically translated.

The target grid-month mean and observation representativeness treatment still require a dedicated experiment. Physical thresholds must be derived from preregistered scientific uses before final audit. No class occupancy is required, and all cells may legitimately receive the same tier or U.

## Figure and table index

- Figure 1: ordered decision contract; source `tables/table01_error_tier_decision_contract.csv`.
- Table 1: machine-readable decision conditions, output tiers, evidence states, and error-field policy.
- Frozen input: `configs/frozen/absolute_error_flag_contract_v2.5.in`.
"""
    captions = f"# Figure captions\n\n## fig01_error_tier_decision_contract.png\n\n{caption}\n"
    readme = """# Issue 81 absolute-error flag contract

This preregistration archive freezes the model-agnostic error-tier schema and decision order. Run `python scripts/build_absolute_error_flag_contract_archive.py`, then verify the archive with `python scripts/verify_experiment_archive.py docs/experiment_archive/p2_absolute_error_flag_contract_v2.5`.

Physical A/B/C/D thresholds are intentionally unset until their intended-use derivation is reviewed. The production validator fails when `--require-thresholds` is used before that freeze.
"""
    return report, captions, readme


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    table = decision_table()
    write_utf8_lf(
        TABLES / "table01_error_tier_decision_contract.csv",
        table.to_csv(index=False, lineterminator="\n"),
    )
    build_figure(table)
    report, captions, readme = build_report()
    write_utf8_lf(ARCHIVE / "REPORT.md", report)
    write_utf8_lf(ARCHIVE / "CAPTIONS.md", captions)
    write_utf8_lf(ARCHIVE / "README.md", readme)

    tracked = [
        ARCHIVE / "README.md",
        ARCHIVE / "REPORT.md",
        ARCHIVE / "CAPTIONS.md",
        FIGURES / "fig01_error_tier_decision_contract.png",
        TABLES / "table01_error_tier_decision_contract.csv",
    ]
    manifest = {
        "experiment_id": "p2_absolute_error_flag_contract_v2.5",
        "training_git_commit": "not_applicable_preregistration",
        "data_manifest_sha256": sha256_tracked_text(CONFIG),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "methodology and schema preregistration; no model result",
        "decision": {
            "machine_decision": "freeze_methodology_and_block_unset_thresholds",
            "thresholds_frozen": False,
            "production_grading_allowed": False,
        },
        "figure_source_data": {
            "fig01_error_tier_decision_contract.png": ["table01_error_tier_decision_contract.csv"]
        },
        "figure_captions": {
            "fig01_error_tier_decision_contract.png": (
                "Figure 1. Ordered Issue 81 decision contract. A-D require a calibrated physical-error "
                "estimate and eligible direct or transfer evidence. U retains a numerical reconstruction "
                "while explicitly withholding an unsupported error claim; D means supported evidence of "
                "large error rather than unknown error."
            )
        },
        "files_sha256": {
            str(path.relative_to(ARCHIVE)).replace("\\", "/"): sha256(path) for path in tracked
        },
        "archive_builder_sha256": sha256_tracked_text(Path(__file__)),
        "analysis_script_sha256": sha256_tracked_text(VALIDATOR),
        "source_artifacts_sha256": {
            str(CONFIG.relative_to(ROOT)).replace("\\", "/"): sha256_tracked_text(CONFIG)
        },
    }
    write_utf8_lf(ARCHIVE / "archive_manifest.json", json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
