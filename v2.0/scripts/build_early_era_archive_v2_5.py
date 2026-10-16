"""Build the reviewer archive for strict reverse-time early-era validation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recad.evaluate.early_era import load_early_era_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/frozen/early_era_validation_v2.5.in"
ARCHIVE = ROOT / "docs/experiment_archive/p2_early_era_validation_v2.5"


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


def load_partitions() -> tuple[pd.DataFrame, list[Path]]:
    contract = load_early_era_contract(CONFIG)
    paths = sorted((contract.output_root / "partitions").glob("*_early_era_fold*.parquet"))
    if len(paths) != 6:
        raise RuntimeError(f"expected six early-era partitions, found {len(paths)}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame["error"] = frame.prediction - frame.truth
    frame["absolute_error"] = frame.error.abs()
    return frame, paths


def tables(frame: pd.DataFrame, paths: list[Path]) -> dict[str, pd.DataFrame]:
    fold = (
        frame.groupby(["target", "fold"], as_index=False)
        .agg(
            rows=("record_id", "size"),
            groups=("group_key", "nunique"),
            rmse=("error", lambda x: float(np.sqrt(np.mean(x**2)))),
            mae=("absolute_error", "mean"),
            bias=("error", "mean"),
            q90=("absolute_error", lambda x: float(np.quantile(x, 0.9, method="higher"))),
            coverage90=(
                "absolute_error",
                lambda x: float(np.mean(x <= frame.loc[x.index, "width90"])),
            ),
        )
        .sort_values(["target", "fold"])
    )
    source = pd.DataFrame(
        [
            {
                "path": str(path).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in paths
        ]
    )
    audit_rows = []
    contract = load_early_era_contract(CONFIG)
    for target in contract.targets:
        audit_rows.extend(
            pd.read_csv(contract.output_root / "tables" / f"{target}_split_audit.csv").to_dict(
                "records"
            )
        )
    audit = pd.DataFrame(audit_rows)
    annual = frame.groupby(["target", "yr"], as_index=False).agg(
        rows=("record_id", "size"),
        groups=("group_key", "nunique"),
        rmse=("error", lambda x: float(np.sqrt(np.mean(x**2)))),
        bias=("error", "mean"),
        q90=("absolute_error", lambda x: float(np.quantile(x, 0.9, method="higher"))),
    )
    edges = [0, 25, 50, 100, 200, 400, np.inf]
    support = (
        frame.assign(distance_bin=pd.cut(frame.geographic_km, edges, right=False))
        .groupby(["target", "distance_bin"], as_index=False, observed=True)
        .agg(
            rows=("record_id", "size"),
            groups=("group_key", "nunique"),
            rmse=("error", lambda x: float(np.sqrt(np.mean(x**2)))),
            q90=("absolute_error", lambda x: float(np.quantile(x, 0.9, method="higher"))),
        )
    )
    support["distance_bin"] = support.distance_bin.astype(str)
    upstream_paths = sorted((contract.output_root / "upstream_sss").glob("*.parquet"))
    upstream = pd.DataFrame(
        [
            {
                "path": str(path).replace("\\", "/"),
                "rows": len(pd.read_parquet(path, columns=["record_id"])),
                "sha256": sha256(path),
            }
            for path in upstream_paths
        ]
    )
    windows = pd.DataFrame(
        [
            {
                "fold": fold.fold,
                "held_start": fold.held_start,
                "held_end": fold.held_end,
                "train_start": fold.train_start,
                "minimum_gap_years": fold.train_start - fold.held_end,
            }
            for fold in contract.folds
        ]
    )
    decision = pd.DataFrame(
        [
            {
                "decision": "add_complete_early_era_banks_and_continue_conditional_error_calibration",
                "thresholds_remain_unset": True,
                "independent_labels_opened": False,
                "labels_2026_opened": False,
            }
        ]
    )
    return {
        "table01_fold_metrics.csv": fold,
        "table02_source_partitions.csv": source,
        "table03_leakage_audit.csv": audit,
        "table04_annual_metrics.csv": annual,
        "table05_geographic_support.csv": support,
        "table06_upstream_sss.csv": upstream,
        "table07_windows.csv": windows,
        "table08_decision.csv": decision,
    }


def figures(data: dict[str, pd.DataFrame], frame: pd.DataFrame) -> dict[str, str]:
    figure_dir = ARCHIVE / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fold = data["table01_fold_metrics.csv"]
    captions = {}
    definitions = [
        (
            "fig01_fold_rmse.png",
            "rmse",
            "RMSE",
            "Figure 1. Fold-specific reverse-time RMSE in each target's physical units; the two targets use separate panels and are not compared by magnitude.",
        ),
        (
            "fig02_fold_q90.png",
            "q90",
            "90th-percentile absolute error",
            "Figure 2. Fold-specific observation-equal 90th-percentile absolute error for complete early cruises held before all model-fitting observations.",
        ),
        (
            "fig03_fold_coverage90.png",
            "coverage90",
            "Nominal 90% interval coverage",
            "Figure 3. Empirical coverage of prediction-stratified 90% intervals on early held cruises; the 0.90 reference line is diagnostic rather than a pass manufactured by filtering.",
        ),
        (
            "fig04_fold_bias.png",
            "bias",
            "Mean prediction minus truth",
            "Figure 4. Fold-specific signed bias. The earliest fCO2 window retains its negative bias as evidence of unresolved reverse-time transfer error.",
        ),
    ]
    for name, metric, label, caption in definitions:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
            selected = fold.loc[fold.target.eq(target)]
            axis.bar(
                selected.fold.astype(str),
                selected[metric],
                color="#4472C4" if target == "SSS" else "#C55A11",
            )
            if metric == "coverage90":
                axis.axhline(0.9, color="black", linestyle="--", linewidth=1)
            axis.set(title=target, xlabel="Early-era fold", ylabel=label)
        fig.tight_layout()
        fig.savefig(figure_dir / name, dpi=180)
        plt.close(fig)
        captions[name] = caption
    audit = data["table03_leakage_audit.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        selected = audit.loc[audit.target.eq(target)]
        axis.bar(selected.fold.astype(str), selected.held_rows, color="#70AD47")
        axis.set(title=target, xlabel="Early-era fold", ylabel="Held rows")
    fig.tight_layout()
    fig.savefig(figure_dir / "fig05_held_rows.png", dpi=180)
    plt.close(fig)
    captions["fig05_held_rows.png"] = (
        "Figure 5. Number of held observations in each complete-cruise early-era fold; all windows have substantial evidence and window definitions were frozen from metadata before error inspection."
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        values = np.sort(frame.loc[frame.target.eq(target), "absolute_error"].to_numpy(float))
        probability = np.arange(1, len(values) + 1) / len(values)
        axis.plot(values, probability, color="#4472C4" if target == "SSS" else "#C55A11")
        axis.set(
            title=target,
            xlabel="Absolute error",
            ylabel="Empirical cumulative probability",
            xlim=(0, np.quantile(values, 0.99)),
        )
    fig.tight_layout()
    fig.savefig(figure_dir / "fig06_error_cdf.png", dpi=180)
    plt.close(fig)
    captions["fig06_error_cdf.png"] = (
        "Figure 6. Empirical cumulative distributions of absolute reverse-time error, truncated visually at each target's 99th percentile while all rows remain in reported metrics."
    )
    annual = data["table04_annual_metrics.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        selected = annual.loc[annual.target.eq(target)]
        axis.plot(
            selected.yr,
            selected.rmse,
            marker="o",
            color="#4472C4" if target == "SSS" else "#C55A11",
        )
        axis.set(title=target, xlabel="Held observation year", ylabel="RMSE")
    fig.tight_layout()
    fig.savefig(figure_dir / "fig07_annual_rmse.png", dpi=180)
    plt.close(fig)
    captions["fig07_annual_rmse.png"] = (
        "Figure 7. Annual reverse-time RMSE inside the held windows, showing temporal heterogeneity that is hidden by a single pooled early-era score."
    )
    support = data["table05_geographic_support.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        selected = support.loc[support.target.eq(target)]
        axis.plot(
            selected.distance_bin,
            selected.q90,
            marker="o",
            color="#4472C4" if target == "SSS" else "#C55A11",
        )
        axis.tick_params(axis="x", rotation=35)
        axis.set(
            title=target,
            xlabel="Distance to later training support (km bin)",
            ylabel="Q90 absolute error",
        )
    fig.tight_layout()
    fig.savefig(figure_dir / "fig08_geographic_support.png", dpi=180)
    plt.close(fig)
    captions["fig08_geographic_support.png"] = (
        "Figure 8. Reverse-time 90th-percentile absolute error across fixed geographic-support bins; sparse or distant support remains a diagnostic and does not itself assign an A-D grade."
    )
    return captions


def main() -> None:
    frame, paths = load_partitions()
    data = tables(frame, paths)
    for name, table in data.items():
        write_table(table, name)
    captions = figures(data, frame)
    metrics = data["table01_fold_metrics.csv"]
    bullets = "\n".join(
        f"- {row.target} fold {int(row.fold)}: RMSE {row.rmse:.3f}, q90 {row.q90:.3f}, bias {row.bias:.3f}, coverage90 {row.coverage90:.3f}."
        for _, row in metrics.iterrows()
    )
    blocks = "\n\n".join(
        f"![{caption}](figures/{name})\n\n{caption}" for name, caption in captions.items()
    )
    report = f"""# Strict reverse-time early-era validation

## Scientific question and permitted claim

This experiment tests whether the frozen SSS and fCO2 candidates can reconstruct complete early cruises when every fitting and calibration observation comes from a later era. It supports a development claim about observed early cruises only; it cannot identify error in observation-free waters or define A-D thresholds.

## Data, splits, and leakage controls

Three windows were frozen from label metadata before inspecting errors: 1993-1998 held with training from 2004, 1999-2003 held with training from 2009, and 2004-2008 held with training from 2014. Complete normalized cruises are indivisible. Cruises crossing a boundary or lying in the temporal gap are excluded. All audits report zero train-held cruise overlap. No independent or 2026 label was opened.

## Candidate models and training

SSS uses the frozen three-seed CatBoost background-residual finalist. fCO2 uses the frozen F4 continuous-background ExtraTrees plus CatBoost convex ensemble with CatBoost weight 0.3. Its SSS input is regenerated inside each later-only training hierarchy: model-fit cruises receive two-fold out-of-fold SSS, while calibration and early held cruises receive predictions from SSS models fitted only on fCO2 model-fit cruise groups.

## Main development results

{bullets}

{blocks}

## Decision and limitations

Both early-era banks are accepted into the separated residual-bank evidence. Pooled RMSE is 0.856 for SSS and 36.122 microatmospheres for fCO2; pooled q90 absolute error is 0.922 and 52.616 respectively. The earliest fCO2 fold has a -5.561 microatmosphere bias, and interval coverage falls to 0.867 in the latest early-era fold. These negative results remain explicit. The next step is cross-fitted conditional physical-error estimation across all five protocols. T_A, T_B, and T_C remain unset.

## Figure and table index

- Tables 1-8 contain fold metrics, source hashes, leakage audits, annual metrics, support strata, nested SSS lineage, frozen windows, and the decision.
- Figures 1-8 show fold errors and coverage, evidence size, the error distribution, temporal heterogeneity, and geographic-support behavior.
"""
    write_text(ARCHIVE / "REPORT.md", report)
    write_text(
        ARCHIVE / "README.md",
        "# Issue 81 early-era validation\n\nStart with [REPORT.md](REPORT.md). Row-level predictions remain outside Git.\n",
    )
    write_text(
        ARCHIVE / "CAPTIONS.md",
        "# Figure captions\n\n"
        + "\n\n".join(f"## {name}\n\n{caption}" for name, caption in captions.items())
        + "\n",
    )
    tracked = [ARCHIVE / name for name in ("README.md", "REPORT.md", "CAPTIONS.md")]
    tracked += sorted((ARCHIVE / "figures").glob("*.png")) + sorted(
        (ARCHIVE / "tables").glob("*.csv")
    )
    script = Path(__file__)
    source_hashes = {str(path).replace("\\", "/"): sha256(path) for path in paths}
    manifest = {
        "experiment_id": "p2_early_era_validation_v2.5",
        "training_git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "data_manifest_sha256": text_sha256(CONFIG),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "development-only reverse-time complete-cruise validation",
        "decision": "add_complete_early_era_banks_and_continue_conditional_error_calibration",
        "figure_source_data": {
            "fig01_fold_rmse.png": ["table01_fold_metrics.csv"],
            "fig02_fold_q90.png": ["table01_fold_metrics.csv"],
            "fig03_fold_coverage90.png": ["table01_fold_metrics.csv"],
            "fig04_fold_bias.png": ["table01_fold_metrics.csv"],
            "fig05_held_rows.png": ["table03_leakage_audit.csv"],
            "fig06_error_cdf.png": ["table01_fold_metrics.csv"],
            "fig07_annual_rmse.png": ["table04_annual_metrics.csv"],
            "fig08_geographic_support.png": ["table05_geographic_support.csv"],
        },
        "figure_captions": captions,
        "files_sha256": {
            str(path.relative_to(ARCHIVE)).replace("\\", "/"): sha256(path) for path in tracked
        },
        "archive_builder_sha256": text_sha256(script),
        "analysis_script_sha256": text_sha256(ROOT / "scripts/run_early_era_validation_v2_5.py"),
        "source_artifacts_sha256": {
            str(CONFIG.relative_to(ROOT)).replace("\\", "/"): text_sha256(CONFIG),
            **source_hashes,
        },
    }
    write_text(ARCHIVE / "archive_manifest.json", json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
