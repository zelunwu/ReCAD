"""Build separated deployment-matched residual banks for Issue 81."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd

from recad.data.ocean_regions import load_ocean_region_contract
from recad.evaluate.residual_bank import (
    ResidualBankContract,
    empirical_quantile,
    hierarchical_group_quantile_bound,
    load_residual_bank_contract,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "frozen" / "end_to_end_residual_bank_v2.5.in"
REGION_CONFIG = ROOT / "configs" / "frozen" / "ocean_region_provenance_v2.5.in"
ARCHIVE = ROOT / "docs" / "experiment_archive" / "p2_end_to_end_residual_bank_v2.5"
COLORS = {"SSS": "#2166ac", "fCO2": "#b2182b"}
SUPPORT_FIELDS = (
    "geographic_km",
    "environment_k16",
    "unique_cruises",
    "time_extrapolation_years",
)
KEEP_COLUMNS = (
    "record_id",
    "group_key",
    "yr",
    "mon",
    "coastal_node",
    "latitude",
    "longitude",
    "truth",
    "prediction",
    "model_disagreement",
    "background_disagreement",
    "geographic_km",
    "environment_k16",
    "unique_cruises",
    "effective_groups",
    "support_entropy",
    "time_extrapolation_years",
    "missing_mandatory_input",
    "predicted_error_scale",
    "width50",
    "width90",
    "selected_interval_method",
    "target",
    "scheme",
    "fold",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def write_table(frame: pd.DataFrame, path: Path) -> None:
    write_text(path, frame.to_csv(index=False, lineterminator="\n"))


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def load_region_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with netCDF4.Dataset(path) as dataset:
        return (
            np.asarray(dataset.variables["ocean_region_id"][:], dtype=np.int8),
            np.asarray(dataset.variables["macro_basin_id"][:], dtype=np.int8),
        )


def metric_row(frame: pd.DataFrame, contract: ResidualBankContract) -> dict[str, float | int]:
    error = frame.residual.to_numpy(float)
    absolute = np.abs(error)
    truth = frame.truth.to_numpy(float)
    prediction = frame.prediction.to_numpy(float)
    finite = np.isfinite(error) & np.isfinite(truth) & np.isfinite(prediction)
    error = error[finite]
    absolute = absolute[finite]
    truth = truth[finite]
    prediction = prediction[finite]
    hierarchy = hierarchical_group_quantile_bound(
        absolute,
        frame.loc[finite, "group_key"].to_numpy(),
        quantile=contract.quantile,
        confidence=contract.confidence,
        minimum_groups=contract.minimum_groups_for_finite_bound,
    )
    denominator = np.sum((truth - truth.mean()) ** 2)
    width90 = frame.loc[finite, "width90"].to_numpy(float)
    width_finite = np.isfinite(width90)
    return {
        "rows": len(error),
        "groups": int(frame.loc[finite, "group_key"].nunique()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(absolute)),
        "bias": float(np.mean(error)),
        "r2": float(1 - np.sum(error**2) / denominator) if denominator > 0 else float("nan"),
        "record_q90_absolute_error": empirical_quantile(absolute, contract.quantile),
        "cruise_q90_absolute_error": hierarchy["group_q90"],
        "cruise_q90_upper95": hierarchy["group_q90_upper95"],
        "upper_order_rank": hierarchy["upper_order_rank"],
        "coverage90": (
            float(np.mean(absolute[width_finite] <= width90[width_finite]))
            if width_finite.any()
            else float("nan")
        ),
        "mean_width90": float(np.mean(width90[width_finite]))
        if width_finite.any()
        else float("nan"),
    }


def load_bank(
    target: str,
    protocol: str,
    contract: ResidualBankContract,
    ocean_region: np.ndarray,
    macro_basin: np.ndarray,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    pattern = contract.partition_pattern.format(target=target, protocol=protocol)
    paths = sorted(contract.partition_root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no partitions match {contract.partition_root / pattern}")
    frames = []
    sources = []
    for path in paths:
        frame = pd.read_parquet(path)
        missing = sorted(set(KEEP_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"{path.name} lacks required columns: {missing}")
        if set(frame.target.astype(str)) != {target} or set(frame.scheme.astype(str)) != {protocol}:
            raise ValueError(f"partition identity mismatch: {path}")
        frames.append(frame.loc[:, KEEP_COLUMNS].copy())
        sources.append(
            {
                "target": target,
                "protocol": protocol,
                "fold": int(frame.fold.iloc[0]),
                "path": str(path).replace("\\", "/"),
                "rows": len(frame),
                "sha256": sha256(path),
            }
        )
    bank = pd.concat(frames, ignore_index=True)
    bank["residual"] = bank.prediction - bank.truth
    bank["absolute_error"] = bank.residual.abs()
    node = bank[contract.node_column].to_numpy(np.int64)
    valid_node = (node >= 0) & (node < len(ocean_region))
    bank["ocean_region_id"] = np.where(
        valid_node, ocean_region[np.clip(node, 0, len(ocean_region) - 1)], -1
    )
    bank["macro_basin_id"] = np.where(
        valid_node, macro_basin[np.clip(node, 0, len(macro_basin) - 1)], -1
    )
    bank["evaluation_id"] = (
        bank.target.astype(str)
        + ":"
        + bank.scheme.astype(str)
        + ":"
        + bank.fold.astype(str)
        + ":"
        + bank.record_id.astype(str)
    )
    return bank, sources


def summarize_support(
    bank: pd.DataFrame, target: str, protocol: str, contract: ResidualBankContract
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for field in SUPPORT_FIELDS:
        edges = contract.strata_edges[field]
        labels = [f"[{left:g},{right:g})" for left, right in itertools.pairwise(edges)]
        bins = pd.cut(bank[field], bins=edges, labels=labels, right=False, include_lowest=True)
        for label, cohort in bank.groupby(bins, observed=True):
            metrics = metric_row(cohort, contract)
            rows.append(
                {
                    "target": target,
                    "protocol": protocol,
                    "support_field": field,
                    "support_bin": str(label),
                    "cohort_eligible": (
                        metrics["rows"] >= contract.minimum_rows_per_cohort
                        and metrics["groups"] >= contract.minimum_groups_per_cohort
                    ),
                    **metrics,
                }
            )
    return rows


def build_tables(contract: ResidualBankContract) -> tuple[dict[str, pd.DataFrame], list[Path]]:
    output = contract.output_root
    bank_dir = output / "banks"
    bank_dir.mkdir(parents=True, exist_ok=True)
    ocean_region, macro_basin = load_region_arrays(contract.geographic_provenance)
    region_contract = load_ocean_region_contract(REGION_CONFIG)
    region_names = {region.region_id: region.canonical_name for region in region_contract.regions}
    summary_rows = []
    fold_rows = []
    region_rows = []
    support_rows = []
    source_rows = []
    duplicate_rows = []
    outputs = []
    completeness_rows = []
    for target in contract.targets:
        for protocol in contract.required_protocols:
            available = protocol in contract.available_protocols
            pattern = contract.partition_pattern.format(target=target, protocol=protocol)
            paths = sorted(contract.partition_root.glob(pattern)) if available else []
            completeness_rows.append(
                {
                    "target": target,
                    "protocol": protocol,
                    "required": True,
                    "available": bool(paths),
                    "partition_files": len(paths),
                    "blocks_final_grading": not bool(paths),
                }
            )
            if not paths:
                continue
            bank, sources = load_bank(target, protocol, contract, ocean_region, macro_basin)
            source_rows.extend(sources)
            duplicates = int(bank.record_id.duplicated(keep=False).sum())
            duplicate_rows.append(
                {
                    "target": target,
                    "protocol": protocol,
                    "rows": len(bank),
                    "unique_record_ids": bank.record_id.nunique(),
                    "rows_with_duplicate_record_id": duplicates,
                    "unique_evaluation_ids": bank.evaluation_id.nunique(),
                }
            )
            if bank.evaluation_id.duplicated().any():
                raise ValueError(f"duplicate evaluation identities in {target}/{protocol}")
            bank_path = bank_dir / f"{target}_{protocol}_residual_bank.parquet"
            bank.to_parquet(bank_path, index=False)
            outputs.append(bank_path)
            summary_rows.append(
                {"target": target, "protocol": protocol, **metric_row(bank, contract)}
            )
            for fold, cohort in bank.groupby("fold"):
                fold_rows.append(
                    {
                        "target": target,
                        "protocol": protocol,
                        "fold": int(fold),
                        **metric_row(cohort, contract),
                    }
                )
            for region_id, cohort in bank.groupby("ocean_region_id"):
                region_rows.append(
                    {
                        "target": target,
                        "protocol": protocol,
                        "ocean_region_id": int(region_id),
                        "ocean_region": region_names.get(int(region_id), "Unresolved"),
                        **metric_row(cohort, contract),
                    }
                )
            support_rows.extend(summarize_support(bank, target, protocol, contract))
    completeness = pd.DataFrame(completeness_rows)
    decision = pd.DataFrame(
        [
            {
                "decision": "residual_banks_reusable_but_contract_incomplete",
                "final_grading_allowed": False,
                "missing_required_protocols": ",".join(
                    sorted(set(contract.required_protocols) - set(contract.available_protocols))
                ),
                "thresholds_remain_unset": True,
                "basin_changes_grade": False,
                "independent_labels_opened": False,
            }
        ]
    )
    fields = pd.DataFrame(
        [
            ("residual", "prediction minus truth in target physical units"),
            ("absolute_error", "absolute deployment-matched held-out residual"),
            ("record_q90_absolute_error", "observation-equal q90 absolute error"),
            ("cruise_q90_absolute_error", "q90 across within-cruise q90 absolute errors"),
            (
                "cruise_q90_upper95",
                "distribution-free one-sided 95% upper bound for the cruise-level q90",
            ),
            ("coverage90", "fraction of held errors no larger than the frozen Issue-44 width90"),
            ("ocean_region_id", "GOaS v1 provenance only; never a quality switch"),
        ],
        columns=["field", "definition"],
    )
    tables = {
        "table01_protocol_metrics.csv": pd.DataFrame(summary_rows),
        "table02_fold_metrics.csv": pd.DataFrame(fold_rows),
        "table03_region_metrics.csv": pd.DataFrame(region_rows),
        "table04_support_strata_metrics.csv": pd.DataFrame(support_rows),
        "table05_protocol_completeness.csv": completeness,
        "table06_source_partitions.csv": pd.DataFrame(source_rows),
        "table07_duplicate_audit.csv": pd.DataFrame(duplicate_rows),
        "table08_decision.csv": decision,
        "table09_field_dictionary.csv": fields,
    }
    table_dir = output / "tables"
    for name, frame in tables.items():
        path = table_dir / name
        write_table(frame, path)
        outputs.append(path)
    return tables, outputs


def grouped_bars(
    table: pd.DataFrame,
    value: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    protocols = list(dict.fromkeys(table.protocol))
    x = np.arange(len(protocols))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 5.2))
    for offset, target in zip((-0.5, 0.5), ("SSS", "fCO2"), strict=True):
        part = table.loc[table.target.eq(target)].set_index("protocol").reindex(protocols)
        axis.bar(x + offset * width, part[value], width, label=target, color=COLORS[target])
    axis.set_xticks(x, protocols, rotation=20, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title, weight="bold")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def target_bars(
    table: pd.DataFrame,
    value: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    protocols = list(dict.fromkeys(table.protocol))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        part = table.loc[table.target.eq(target)].set_index("protocol").reindex(protocols)
        axis.bar(protocols, part[value], color=COLORS[target])
        axis.set_title(target)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(title, weight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def support_plot(table: pd.DataFrame, field: str, title: str, path: Path) -> None:
    subset = table.loc[table.support_field.eq(field) & table.protocol.eq("geographic")].copy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        part = subset.loc[subset.target.eq(target)]
        axis.plot(
            np.arange(len(part)),
            part.record_q90_absolute_error,
            marker="o",
            color=COLORS[target],
        )
        axis.set_xticks(np.arange(len(part)), part.support_bin, rotation=35, ha="right")
        axis.set_title(target)
        axis.set_ylabel("Record q90 absolute error")
        axis.grid(alpha=0.25)
    figure.suptitle(title, weight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_figures(tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    figures = ARCHIVE / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    protocol = tables["table01_protocol_metrics.csv"]
    target_bars(
        protocol,
        "record_q90_absolute_error",
        "Absolute error (target units)",
        "Observation-equal q90 absolute error by transfer protocol",
        figures / "fig01_record_q90.png",
    )
    target_bars(
        protocol,
        "cruise_q90_upper95",
        "Absolute error (target units)",
        "Hierarchical cruise q90 one-sided upper bound",
        figures / "fig02_hierarchical_upper.png",
    )
    grouped_bars(
        protocol,
        "coverage90",
        "Empirical coverage",
        "Frozen Issue-44 nominal 90% interval coverage",
        figures / "fig03_coverage90.png",
    )
    fold = tables["table02_fold_metrics.csv"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        part = fold.loc[fold.target.eq(target)]
        for protocol, cohort in part.groupby("protocol"):
            axis.plot(cohort.fold, cohort.record_q90_absolute_error, marker="o", label=protocol)
        axis.set_title(target)
        axis.set_xlabel("Outer fold")
        axis.set_ylabel("Record q90 absolute error")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("Outer-fold variability is retained rather than pooled away", weight="bold")
    figure.tight_layout()
    figure.savefig(figures / "fig04_fold_variability.png", dpi=180)
    plt.close(figure)
    support = tables["table04_support_strata_metrics.csv"]
    support_plot(
        support,
        "geographic_km",
        "Geographic holdout error by distance to training support",
        figures / "fig05_geographic_support.png",
    )
    support_plot(
        support,
        "environment_k16",
        "Geographic holdout error by environmental novelty",
        figures / "fig06_environment_support.png",
    )
    support_plot(
        support,
        "unique_cruises",
        "Geographic holdout error by nearby unique-cruise support",
        figures / "fig07_cruise_support.png",
    )
    completeness = tables["table05_protocol_completeness.csv"].copy()
    matrix = completeness.pivot(index="target", columns="protocol", values="available").astype(int)
    matrix = matrix.reindex(
        columns=["cruise", "geographic", "whole_region", "forward", "early_era"]
    )
    figure, axis = plt.subplots(figsize=(9, 2.8))
    image = axis.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=20, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            axis.text(
                column, row, "available" if matrix.iloc[row, column] else "missing", ha="center"
            )
    axis.set_title("Required residual-bank protocol completeness", weight="bold")
    figure.colorbar(image, ax=axis, ticks=[0, 1], label="availability")
    figure.tight_layout()
    figure.savefig(figures / "fig08_protocol_completeness.png", dpi=180)
    plt.close(figure)
    return {
        "fig01_record_q90.png": "Figure 1. Observation-equal 90th-percentile absolute held-out error for the frozen SSS and fCO2 candidates, kept separate across complete-cruise, buffered-geographic, whole-region, and forward-time protocols. Different physical units are shown in separate colored series and must not be compared across targets.",
        "fig02_hierarchical_upper.png": "Figure 2. Distribution-free one-sided 95% upper bound for the 90th percentile across within-cruise 90th-percentile absolute errors. Complete cruises are evidence units, so dense rows from one cruise cannot create artificial confidence.",
        "fig03_coverage90.png": "Figure 3. Empirical coverage of the nominal 90% Issue-44 intervals on the identical frozen predictions. Shifted protocols remain visibly below nominal coverage; this residual-bank phase does not repair or conceal that negative result.",
        "fig04_fold_variability.png": "Figure 4. Fold-specific record-level 90th-percentile absolute error. Variation across held geographic blocks, regions, and time windows remains explicit instead of being erased by a single pooled score.",
        "fig05_geographic_support.png": "Figure 5. Buffered-geographic held-out error stratified by distance to training support using fixed kilometer bins from the external .in contract. Empty or weak cohorts are not used to redefine bin boundaries.",
        "fig06_environment_support.png": "Figure 6. Buffered-geographic held-out error stratified by standardized environmental k-nearest-neighbour distance. The diagnostic is in physical error units; the novelty score itself is not a quality grade.",
        "fig07_cruise_support.png": "Figure 7. Buffered-geographic held-out error stratified by the count of nearby unique cruises. Cruise support is a deployment diagnostic and cannot by itself assign an A-D tier.",
        "fig08_protocol_completeness.png": "Figure 8. Completeness of the five protocol banks required by the frozen absolute-error contract. Early-era evidence is missing for both targets, so final A-D grading remains blocked even though the other four banks are reusable.",
    }


def build_archive(
    contract: ResidualBankContract,
    config: Path,
    tables: dict[str, pd.DataFrame],
    outputs: list[Path],
) -> None:
    table_dir = ARCHIVE / "tables"
    for name, table in tables.items():
        write_table(table, table_dir / name)
    captions = build_figures(tables)
    summary = tables["table01_protocol_metrics.csv"]
    lines = []
    for _, row in summary.iterrows():
        lines.append(
            f"- {row.target} / {row.protocol}: RMSE {row.rmse:.3f}, record q90 {row.record_q90_absolute_error:.3f}, hierarchical upper {row.cruise_q90_upper95:.3f}, coverage90 {row.coverage90:.3f}."
        )
    blocks = []
    for name, caption in captions.items():
        blocks.append(f"![{caption}](figures/{name})\n\n{caption}")
    report = f"""# Deployment-matched end-to-end residual banks

## Scientific question and permitted claim

This experiment asks what deployment-matched end-to-end residual evidence is available for physical SSS and fCO2 error estimation under the frozen final candidates. The permitted claim is limited to residual-bank construction and an evidence-gap audit. It cannot define intended-use thresholds, publish A-D grades, or validate observation-free waters.

## Data, splits, and leakage controls

This Issue 81 phase reuses the exact corrected Issue-61 outer predictions archived by Issue 44. Raw residual banks remain separate for complete-cruise, buffered-geographic plus 100-km exclusion, whole-region, and past-to-future protocols. It opens no new labels. Complete cruises remain the independent evidence units. Every source partition is hashed, record and evaluation identities are audited, and no duplicate identity was found. Basin identity is attached from GOaS v1 for reporting only and cannot alter a grade.

## Candidate models and training

The point predictions are frozen: the Issue-44 three-seed SSS CatBoost background-residual candidate and the fCO2 inner-OOF-weighted CatBoost/F4 convex ensemble. This experiment does not fit or select a point model, refit a calibration model, or tune a threshold. The saved outer predictions therefore isolate error analysis from model selection.

The primary feasibility statistic is hierarchical: compute the 90th-percentile absolute error inside each complete cruise, then take the 90th percentile across cruises and a distribution-free one-sided 95% order-statistic upper bound. This prevents dense rows within one cruise from manufacturing precision. Observation-equal q90, fold variability, region strata, support strata, RMSE, MAE, bias, R2, width and coverage remain diagnostics. These development residuals cannot define the intended-use `T_A`, `T_B`, or `T_C` thresholds.

## Main development results

{chr(10).join(lines)}

The four available residual banks are reusable, but the frozen contract also requires an early-era test in which later observations fit the model and earlier cruises are held out. That bank does not exist in Issues 61-44. Final grading therefore fails closed. No protocol is pooled into a global error estimate, and no error is asserted for observation-free cells from ensemble spread alone.

{chr(10).join([block + chr(10) for block in blocks])}

## Decision and limitations

Decision: retain the four separated banks as development evidence and run the missing early-era prediction experiment before conditional error calibration. After all five banks exist, fit and cross-fit candidate conditional physical-error estimators, test their calibration by protocol and cohort, and only then evaluate intended-use threshold profiles. The current v2.4 product remains unchanged.

The hierarchical bound is deliberately stricter than observation-equal error and describes transfer across observed cruises; it is not yet an area-time-weighted global product error. Observed locations cannot by themselves identify error in a coast with no target observations or validated analogue. Measurement and representativeness components are not separately estimated in this phase and must be added without double-counting the end-to-end residual.

## Figure and table index

- Table 1: protocol metrics.
- Table 2: outer-fold metrics.
- Table 3: GOaS v1 region reporting strata.
- Table 4: fixed support-diagnostic strata.
- Table 5: required-protocol completeness.
- Table 6: every source partition with SHA256.
- Table 7: duplicate identity audit.
- Table 8: frozen decision.
- Table 9: field dictionary.
"""
    write_text(ARCHIVE / "REPORT.md", report)
    write_text(
        ARCHIVE / "CAPTIONS.md",
        "# Figure captions\n\n"
        + "\n\n".join(f"## {name}\n\n{caption}" for name, caption in captions.items())
        + "\n",
    )
    write_text(
        ARCHIVE / "README.md",
        "# Issue 81 end-to-end residual banks\n\nStart with [REPORT.md](REPORT.md). Row-level residual banks remain outside Git.\n",
    )
    tracked = [ARCHIVE / "README.md", ARCHIVE / "REPORT.md", ARCHIVE / "CAPTIONS.md"]
    tracked += sorted((ARCHIVE / "figures").glob("*.png"))
    tracked += sorted(table_dir.glob("*.csv"))
    figure_sources = {
        "fig01_record_q90.png": ["table01_protocol_metrics.csv"],
        "fig02_hierarchical_upper.png": ["table01_protocol_metrics.csv"],
        "fig03_coverage90.png": ["table01_protocol_metrics.csv"],
        "fig04_fold_variability.png": ["table02_fold_metrics.csv"],
        "fig05_geographic_support.png": ["table04_support_strata_metrics.csv"],
        "fig06_environment_support.png": ["table04_support_strata_metrics.csv"],
        "fig07_cruise_support.png": ["table04_support_strata_metrics.csv"],
        "fig08_protocol_completeness.png": ["table05_protocol_completeness.csv"],
    }
    external_hashes = {str(path).replace("\\", "/"): sha256(path) for path in outputs}
    external_hashes[str(contract.geographic_provenance).replace("\\", "/")] = sha256(
        contract.geographic_provenance
    )
    manifest = {
        "experiment_id": "p2_end_to_end_residual_bank_v2.5",
        "training_git_commit": git_commit(),
        "data_manifest_sha256": tracked_text_sha256(config),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "development-only deployment-matched residual banks on frozen corrected outer predictions",
        "decision": "residual_banks_reusable_but_early_era_missing_and_final_grading_blocked",
        "figure_source_data": figure_sources,
        "figure_captions": captions,
        "files_sha256": {
            str(path.relative_to(ARCHIVE)).replace("\\", "/"): sha256(path) for path in tracked
        },
        "archive_builder_sha256": tracked_text_sha256(Path(__file__)),
        "analysis_script_sha256": tracked_text_sha256(Path(__file__)),
        "source_artifacts_sha256": {
            str(config.relative_to(ROOT)).replace("\\", "/"): tracked_text_sha256(config),
            **external_hashes,
        },
    }
    write_text(ARCHIVE / "archive_manifest.json", json.dumps(manifest, indent=2) + "\n")
    write_text(
        contract.output_root / "external_artifact_manifest.json",
        json.dumps(external_hashes, indent=2) + "\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_residual_bank_contract(args.config)
    tables, outputs = build_tables(contract)
    build_archive(contract, args.config, tables, outputs)
    decision = tables["table08_decision.csv"].iloc[0].to_dict()
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
