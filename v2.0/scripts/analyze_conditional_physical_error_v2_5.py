"""Analyze and archive Issue 81 conditional physical-error calibration."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recad.evaluate.conditional_error import error_bound_metrics, load_conditional_error_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/frozen/conditional_physical_error_v2.5.in"
ARCHIVE = ROOT / "docs/experiment_archive/p2_conditional_physical_error_v2.5"
COLORS = {"SSS": "#4472C4", "fCO2": "#C55A11"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode()).hexdigest()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value.replace("\r\n", "\n").encode())


def write_table(frame: pd.DataFrame, name: str) -> None:
    write_text(ARCHIVE / "tables" / name, frame.to_csv(index=False, lineterminator="\n"))


def decisions(contract) -> pd.DataFrame:
    rows = []
    for target in contract.targets:
        rows.extend(
            json.loads(
                (contract.output_root / "tables" / f"{target}_decisions.json").read_text(
                    encoding="utf-8"
                )
            )
        )
    return pd.DataFrame(rows)


def selected_oof(contract, selection: pd.DataFrame) -> tuple[pd.DataFrame, list[Path]]:
    frames = []
    paths = []
    for row in selection.itertuples(index=False):
        for fold in range(contract.group_folds):
            path = (
                contract.output_root
                / "partitions"
                / f"{row.target}_{row.protocol}_{row.selected_candidate}_fold{fold}.parquet"
            )
            frames.append(pd.read_parquet(path))
            paths.append(path)
    return pd.concat(frames, ignore_index=True), paths


def group_metrics(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for identity, cohort in frame.groupby(keys, observed=True):
        if not isinstance(identity, tuple):
            identity = (identity,)
        metrics = error_bound_metrics(cohort)
        adequate = metrics["rows"] >= 500 and metrics["groups"] >= 10
        rows.append(
            {
                **dict(zip(keys, identity, strict=True)),
                **metrics,
                "adequately_sampled": adequate,
                "cohort_qualified": bool(
                    adequate
                    and metrics["row_coverage"] >= 0.85
                    and metrics["group_q90_coverage"] >= 0.85
                ),
            }
        )
    return pd.DataFrame(rows)


def build_tables(contract) -> tuple[dict[str, pd.DataFrame], list[Path]]:
    candidate = pd.concat(
        [
            pd.read_csv(contract.output_root / "tables" / f"{target}_candidate_summary.csv")
            for target in contract.targets
        ],
        ignore_index=True,
    )
    fold = pd.concat(
        [
            pd.read_csv(contract.output_root / "tables" / f"{target}_fold_metrics.csv")
            for target in contract.targets
        ],
        ignore_index=True,
    )
    selection = decisions(contract)
    oof, paths = selected_oof(contract, selection)
    region = group_metrics(oof, ["target", "scheme", "ocean_region_id"])
    support_rows = []
    for field, edges in contract.stratified_edges.items():
        binned = oof.assign(support_bin=pd.cut(oof[field], edges, right=False).astype(str))
        table = group_metrics(binned, ["target", "scheme", "support_bin"])
        table.insert(2, "support_field", field)
        support_rows.append(table)
    support = pd.concat(support_rows, ignore_index=True)
    reliability_rows = []
    for (target, scheme), cohort in oof.groupby(["target", "scheme"], observed=True):
        rank = pd.qcut(cohort.error_bound.rank(method="first"), 10, labels=False)
        for bin_id, values in cohort.groupby(rank, observed=True):
            metrics = error_bound_metrics(values)
            reliability_rows.append(
                {
                    "target": target,
                    "protocol": scheme,
                    "bound_decile": int(bin_id),
                    "mean_predicted_bound": float(values.error_bound.mean()),
                    "observed_q90_absolute_error": float(
                        np.quantile(values.absolute_error, 0.9, method="higher")
                    ),
                    **metrics,
                }
            )
    reliability = pd.DataFrame(reliability_rows)
    sources = pd.DataFrame(
        [
            {
                "path": str(path).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in paths
        ]
    )
    model_paths = sorted((contract.output_root / "models").glob("*"))
    models = pd.DataFrame(
        [
            {
                "path": str(path).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in model_paths
            if path.is_file()
        ]
    )
    adequate_region = region.loc[region.adequately_sampled]
    adequate_support = support.loc[support.adequately_sampled]
    decision = pd.DataFrame(
        [
            {
                "decision": "retain_protocol_specific_conditional_error_estimators_thresholds_unset",
                "all_protocols_selected": bool(selection.calibration_passed.all()),
                "adequate_region_cohorts": len(adequate_region),
                "qualified_region_cohorts": int(adequate_region.cohort_qualified.sum()),
                "adequate_support_cohorts": len(adequate_support),
                "qualified_support_cohorts": int(adequate_support.cohort_qualified.sum()),
                "thresholds_remain_unset": True,
                "grades_emitted": False,
                "independent_labels_opened": False,
            }
        ]
    )
    return {
        "table01_candidate_summary.csv": candidate,
        "table02_fold_metrics.csv": fold,
        "table03_selection.csv": selection,
        "table04_region_calibration.csv": region,
        "table05_support_calibration.csv": support,
        "table06_reliability_bins.csv": reliability,
        "table07_selected_partitions.csv": sources,
        "table08_model_artifacts.csv": models,
        "table09_decision.csv": decision,
    }, paths + [path for path in model_paths if path.is_file()]


def build_figures(tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    figure_dir = ARCHIVE / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary = tables["table01_candidate_summary.csv"]
    captions = {}
    metrics = [
        (
            "fig01_candidate_mean_bound.png",
            "mean_bound",
            "Mean physical error bound",
            "Figure 1. Mean calibrated physical error bound for every candidate and transfer protocol. Targets are shown in separate panels because their units differ.",
        ),
        (
            "fig02_candidate_row_coverage.png",
            "row_coverage",
            "Row coverage",
            "Figure 2. Observation-level coverage of each candidate's physical q90 upper bound. The dashed 0.90 line is the frozen marginal target.",
        ),
        (
            "fig03_candidate_group_coverage.png",
            "group_q90_coverage",
            "Cruise q90 coverage",
            "Figure 3. Fraction of complete cruises whose within-cruise 90th-percentile absolute error is covered. Cruises, rather than dense rows, are the evidence units.",
        ),
    ]
    for name, metric, ylabel, caption in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
            selected = summary.loc[summary.target.eq(target)]
            pivot = selected.pivot(index="protocol", columns="candidate", values=metric)
            pivot.plot(kind="bar", ax=axis)
            if "coverage" in metric:
                axis.axhline(0.9, color="black", linestyle="--", linewidth=1)
            axis.set(title=target, ylabel=ylabel, xlabel="Protocol")
            axis.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(figure_dir / name, dpi=180)
        plt.close(fig)
        captions[name] = caption
    fold = tables["table02_fold_metrics.csv"]
    selection = tables["table03_selection.csv"][["target", "protocol", "selected_candidate"]]
    selected_fold = fold.merge(selection, on=["target", "protocol"])
    selected_fold = selected_fold.loc[selected_fold.candidate.eq(selected_fold.selected_candidate)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        data = selected_fold.loc[selected_fold.target.eq(target)]
        for protocol, values in data.groupby("protocol"):
            axis.plot(values.fold, values.row_coverage, marker="o", label=protocol)
        axis.axhline(0.9, color="black", linestyle="--", linewidth=1)
        axis.set(title=target, xlabel="Error-model fold", ylabel="Row coverage")
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig04_selected_fold_coverage.png", dpi=180)
    plt.close(fig)
    captions["fig04_selected_fold_coverage.png"] = (
        "Figure 4. Fold-specific row coverage for the selected estimator in each protocol. No fold is removed when it performs worse than the pooled result."
    )
    reliability = tables["table06_reliability_bins.csv"]
    for index, target in enumerate(("SSS", "fCO2"), start=5):
        fig, axis = plt.subplots(figsize=(6.5, 5))
        selected = reliability.loc[reliability.target.eq(target)]
        for protocol, values in selected.groupby("protocol"):
            axis.plot(
                values.mean_predicted_bound,
                values.observed_q90_absolute_error,
                marker="o",
                label=protocol,
            )
        maximum = max(
            selected.mean_predicted_bound.max(), selected.observed_q90_absolute_error.max()
        )
        axis.plot([0, maximum], [0, maximum], color="black", linestyle="--")
        axis.set(
            xlabel="Mean predicted upper bound",
            ylabel="Observed q90 absolute error",
            title=f"{target} reliability by bound decile",
        )
        axis.legend(fontsize=7)
        fig.tight_layout()
        name = f"fig0{index}_{target.lower()}_reliability.png"
        fig.savefig(figure_dir / name, dpi=180)
        plt.close(fig)
        captions[name] = (
            f"Figure {index}. {target} observed 90th-percentile absolute error versus the mean calibrated upper bound in each predicted-bound decile. Points below the diagonal are conservative."
        )
    region = tables["table04_region_calibration.csv"]
    adequate = region.loc[region.adequately_sampled]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        values = adequate.loc[adequate.target.eq(target)]
        axis.hist(values.row_coverage, bins=np.linspace(0, 1, 21), color=COLORS[target])
        axis.axvline(0.85, color="black", linestyle="--")
        axis.set(title=target, xlabel="Adequate region-protocol cohort coverage", ylabel="Cohorts")
    fig.tight_layout()
    fig.savefig(figure_dir / "fig07_region_coverage.png", dpi=180)
    plt.close(fig)
    captions["fig07_region_coverage.png"] = (
        "Figure 7. Distribution of row coverage across adequately sampled ocean-region by protocol cohorts. Cohorts below 0.85 are retained as failed evidence rather than hidden by pooling."
    )
    support = tables["table05_support_calibration.csv"]
    adequate = support.loc[support.adequately_sampled]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        values = adequate.loc[adequate.target.eq(target)]
        axis.hist(values.row_coverage, bins=np.linspace(0, 1, 21), color=COLORS[target])
        axis.axvline(0.85, color="black", linestyle="--")
        axis.set(title=target, xlabel="Adequate support-stratum coverage", ylabel="Cohorts")
    fig.tight_layout()
    fig.savefig(figure_dir / "fig08_support_coverage.png", dpi=180)
    plt.close(fig)
    captions["fig08_support_coverage.png"] = (
        "Figure 8. Distribution of coverage across adequately sampled fixed geographic, environmental, cruise-diversity, and temporal-support strata. Failed strata require VF or U in later product mapping."
    )
    return captions


def main() -> None:
    contract = load_conditional_error_contract(CONFIG)
    tables, external = build_tables(contract)
    for name, table in tables.items():
        write_table(table, name)
    captions = build_figures(tables)
    selected = tables["table03_selection.csv"]
    summary = tables["table01_candidate_summary.csv"]
    bullets = []
    for row in selected.itertuples(index=False):
        metric = summary.loc[
            summary.target.eq(row.target)
            & summary.protocol.eq(row.protocol)
            & summary.candidate.eq(row.selected_candidate)
        ].iloc[0]
        bullets.append(
            f"- {row.target} / {row.protocol}: {row.selected_candidate}; mean bound {metric.mean_bound:.3f}, row coverage {metric.row_coverage:.3f}, cruise-q90 coverage {metric.group_q90_coverage:.3f}."
        )
    decision = tables["table09_decision.csv"].iloc[0]
    blocks = "\n\n".join(
        f"![{caption}](figures/{name})\n\n{caption}" for name, caption in captions.items()
    )
    report = f"""# Conditional physical-error estimation

## Scientific question and permitted claim

This experiment asks whether deployment-available diagnostics can estimate a conservative conditional 90th-percentile absolute reconstruction error in PSU or microatmospheres. It supports development calibration on observed held cruises. It does not define intended-use thresholds, publish A-D grades, or validate unsupported cells.

## Data, splits, and leakage controls

The five residual protocols remain separate. Inside each target-protocol bank, complete cruises are hashed into five error-model folds. Each evaluation uses three folds for fitting, one disjoint fold for calibration, and one disjoint fold for testing. The calibration offset is the distribution-free one-sided 95% upper order statistic across cruise-level q90 residual scores. Basin is excluded from model features. No independent or 2026 label is opened.

## Candidate models and training

The frozen candidates are a constant physical q90, an interpretable fixed-bin support model, and a CatBoost conditional quantile model. All consume the same deployment-available prediction, disagreement, support, location, and season fields. Candidate selection requires pooled row coverage at least 0.90, every fold at least 0.88, cruise-q90 coverage at least 0.90, then minimizes mean physical bound. Protocols are never pooled; future deployment must use their upper envelope.

## Main development results

{chr(10).join(bullets)}

All ten target-protocol cells select an eligible estimator. SSS selects CatBoost in all protocols. fCO2 selects the simpler fixed-bin estimator for cruise, geographic, and whole-region transfer, and CatBoost only for forward and early-era transfer. Model complexity is therefore not treated as evidence quality.

{blocks}

## Decision and limitations

Decision: retain the ten protocol-specific estimators as development calibration objects. {int(decision.qualified_region_cohorts)} of {int(decision.adequate_region_cohorts)} adequately sampled region-protocol cohorts and {int(decision.qualified_support_cohorts)} of {int(decision.adequate_support_cohorts)} adequately sampled fixed-support cohorts pass the 0.85 cohort audit. Failed cohorts must later receive VF or U; aggregate calibration cannot override them. T_A, T_B, and T_C remain UNSET for both targets, so no A-D grade is emitted.

These bounds describe end-to-end errors represented by observed held cruises. They do not identify Antarctica or another unsupported cell solely from model spread. Measurement and grid-month representativeness components remain separate and must be bounded without double-counting the held-out residual.

## Figure and table index

- Tables 1-3: candidate metrics, fold metrics, and selected protocol-specific estimators.
- Tables 4-6: region calibration, support calibration, and reliability bins.
- Tables 7-9: selected partition hashes, model artifact hashes, and the frozen decision.
- Figures 1-8: candidate sharpness and coverage, fold stability, reliability, and cohort qualification.
"""
    write_text(ARCHIVE / "REPORT.md", report)
    write_text(
        ARCHIVE / "README.md",
        "# Issue 81 conditional physical-error estimation\n\nStart with [REPORT.md](REPORT.md). Row-level outputs and models remain outside Git.\n",
    )
    write_text(
        ARCHIVE / "CAPTIONS.md",
        "# Figure captions\n\n"
        + "\n\n".join(f"## {name}\n\n{caption}" for name, caption in captions.items())
        + "\n",
    )
    tracked = [ARCHIVE / name for name in ("README.md", "REPORT.md", "CAPTIONS.md")]
    tracked += sorted((ARCHIVE / "figures").glob("*.png"))
    tracked += sorted((ARCHIVE / "tables").glob("*.csv"))
    manifest = {
        "experiment_id": "p2_conditional_physical_error_v2.5",
        "training_git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "data_manifest_sha256": text_sha256(CONFIG),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "development-only protocol-specific conditional physical-error calibration",
        "decision": "retain_protocol_specific_estimators_thresholds_unset",
        "figure_source_data": {
            "fig01_candidate_mean_bound.png": ["table01_candidate_summary.csv"],
            "fig02_candidate_row_coverage.png": ["table01_candidate_summary.csv"],
            "fig03_candidate_group_coverage.png": ["table01_candidate_summary.csv"],
            "fig04_selected_fold_coverage.png": [
                "table02_fold_metrics.csv",
                "table03_selection.csv",
            ],
            "fig05_sss_reliability.png": ["table06_reliability_bins.csv"],
            "fig06_fco2_reliability.png": ["table06_reliability_bins.csv"],
            "fig07_region_coverage.png": ["table04_region_calibration.csv"],
            "fig08_support_coverage.png": ["table05_support_calibration.csv"],
        },
        "figure_captions": captions,
        "files_sha256": {
            str(path.relative_to(ARCHIVE)).replace("\\", "/"): sha256(path) for path in tracked
        },
        "archive_builder_sha256": text_sha256(Path(__file__)),
        "analysis_script_sha256": text_sha256(
            ROOT / "scripts/run_conditional_physical_error_v2_5.py"
        ),
        "source_artifacts_sha256": {
            str(CONFIG.relative_to(ROOT)).replace("\\", "/"): text_sha256(CONFIG),
            **{str(path).replace("\\", "/"): sha256(path) for path in external},
        },
    }
    write_text(ARCHIVE / "archive_manifest.json", json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
