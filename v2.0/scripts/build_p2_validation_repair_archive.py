"""Build the reviewer archive for the Issue 61 validation repair."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_4\p2_validation_repair_v2.4")
ARCHIVE = ROOT / "docs/experiment_archive/p2_validation_repair_v2.4"
FIGURES = ARCHIVE / "figures"
TABLES = ARCHIVE / "tables"

MODEL_LABELS = {
    "catboost_background_residual": "CatBoost residual",
    "raw_glorys": "Raw GLORYS",
    "direct_catboost": "Direct CatBoost",
    "modernized_rf": "Modern RF",
    "strict_v1_rf": "Strict v1 RF",
}
SCHEMES = ["cruise", "geographic", "whole_region", "forward"]
COLORS = {
    "CatBoost residual": "#0072B2",
    "Raw GLORYS": "#999999",
    "Direct CatBoost": "#009E73",
    "Modern RF": "#E69F00",
    "Strict v1 RF": "#CC79A7",
}


def sha256(path: Path) -> str:
    """Return the SHA256 of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_lf(path: Path) -> None:
    """Write a text artifact with repository-stable LF line endings."""

    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    path.write_bytes(content)


def save_figure(figure: plt.Figure, name: str) -> None:
    """Save a publication-readable PNG."""

    figure.tight_layout()
    figure.savefig(FIGURES / name, dpi=220, bbox_inches="tight")
    plt.close(figure)


def grouped_bars(
    frame: pd.DataFrame,
    *,
    target: str,
    metric: str,
    ylabel: str,
    name: str,
    references: list[float | None] | None = None,
) -> None:
    """Plot one metric by validation design and model."""

    selected = frame.loc[frame.target == target].copy()
    selected["model_label"] = selected.model.map(MODEL_LABELS)
    models = list(dict.fromkeys(selected.model_label))
    figure, axis = plt.subplots(figsize=(9.2, 5.2))
    width = 0.78 / len(models)
    x = np.arange(len(SCHEMES))
    for offset, model in enumerate(models):
        values = (
            selected.loc[selected.model_label == model].set_index("scheme")[metric].reindex(SCHEMES)
        )
        axis.bar(
            x + (offset - (len(models) - 1) / 2) * width,
            values,
            width,
            label=model,
            color=COLORS[model],
        )
    if references is not None:
        for index, reference in enumerate(references):
            if reference is None:
                continue
            axis.hlines(
                reference,
                index - 0.43,
                index + 0.43,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="scheme gate" if index == 0 else None,
            )
    axis.set_xticks(x, [value.replace("_", " ") for value in SCHEMES])
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=min(3, len(models)))
    save_figure(figure, name)


def copy_tables() -> dict[str, pd.DataFrame]:
    """Copy exact analysis tables into the reviewer archive."""

    source_names = [
        "corrected_native_metrics.csv",
        "corrected_common_grid_metrics.csv",
        "corrected_risk_coverage.csv",
        "corrected_gate_matrix.csv",
        "model_coverage.csv",
        "archive_correction_matrix.csv",
        "exposure_summary.csv",
    ]
    frames: dict[str, pd.DataFrame] = {}
    for name in source_names:
        shutil.copy2(OUTPUT / "tables" / name, TABLES / name)
        frames[name] = pd.read_csv(TABLES / name)
    split_frames = []
    for target in ("SSS", "fCO2"):
        part = pd.read_csv(OUTPUT / "splits" / f"{target}_split_audit.csv")
        split_frames.append(part)
    frames["split_audit.csv"] = pd.concat(split_frames, ignore_index=True)
    frames["split_audit.csv"].to_csv(TABLES / "split_audit.csv", index=False)
    shutil.copy2(OUTPUT / "training_log.csv", TABLES / "training_log.csv")
    frames["training_log.csv"] = pd.read_csv(TABLES / "training_log.csv")
    return frames


def build_figures(frames: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    """Build all figures and return their source-table mappings."""

    metrics = frames["corrected_native_metrics.csv"]
    grouped_bars(
        metrics,
        target="SSS",
        metric="rmse",
        ylabel="SSS RMSE (psu)",
        name="fig01_sss_rmse.png",
        references=[1.0, 1.2, None, 1.2],
    )
    grouped_bars(
        metrics,
        target="fCO2",
        metric="rmse",
        ylabel="fCO2 RMSE (µatm)",
        name="fig02_fco2_rmse.png",
        references=[30.0, 40.0, None, 40.0],
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        selected = metrics.loc[metrics.target == target].copy()
        selected["model_label"] = selected.model.map(MODEL_LABELS)
        for model, part in selected.groupby("model_label"):
            axis.plot(
                [SCHEMES.index(value) for value in part.scheme],
                part.r2,
                "o-",
                label=model,
                color=COLORS[model],
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(range(4), [value.replace("_", "\n") for value in SCHEMES])
        axis.set_title(target)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Held-out R²")
    axes[1].legend(frameon=False, fontsize=8)
    save_figure(figure, "fig03_r2.png")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        selected = metrics.loc[metrics.target == target].copy()
        selected["model_label"] = selected.model.map(MODEL_LABELS)
        for model, part in selected.groupby("model_label"):
            axis.plot(
                [SCHEMES.index(value) for value in part.scheme],
                part.cruise_equal_rmse,
                "o-",
                label=model,
                color=COLORS[model],
            )
        axis.set_xticks(range(4), [value.replace("_", "\n") for value in SCHEMES])
        axis.set_title(target)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Cruise-equal RMSE")
    axes[1].legend(frameon=False, fontsize=8)
    save_figure(figure, "fig04_cruise_equal_rmse.png")
    for metric, nominal, number in (("coverage50", 0.5, 5), ("coverage90", 0.9, 6)):
        figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
        for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
            selected = metrics.loc[metrics.target == target].copy()
            selected["model_label"] = selected.model.map(MODEL_LABELS)
            for model, part in selected.groupby("model_label"):
                axis.plot(
                    [SCHEMES.index(value) for value in part.scheme],
                    part[metric],
                    "o-",
                    label=model,
                    color=COLORS[model],
                )
            axis.axhline(nominal, color="black", linestyle="--", linewidth=1)
            axis.set_xticks(range(4), [value.replace("_", "\n") for value in SCHEMES])
            axis.set_title(target)
            axis.grid(alpha=0.25)
        axes[0].set_ylabel(f"Empirical {int(nominal * 100)}% interval coverage")
        axes[1].legend(frameon=False, fontsize=8)
        save_figure(figure, f"fig{number:02d}_{metric}.png")
    grouped_bars(
        metrics,
        target="fCO2",
        metric="mean_width90",
        ylabel="Mean 90% interval width (µatm)",
        name="fig07_fco2_interval_width.png",
    )
    risk = frames["corrected_risk_coverage.csv"].copy()
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        selected = risk.loc[(risk.target == target) & (risk.scheme == "geographic")].copy()
        selected["model_label"] = selected.model.map(MODEL_LABELS)
        for model, part in selected.groupby("model_label"):
            axis.plot(
                part.retained_fraction,
                part.rmse,
                "o-",
                label=model,
                color=COLORS[model],
            )
        axis.set_title(f"{target}: geographic holdout")
        axis.set_xlabel("Fraction retained by narrowest intervals")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("RMSE")
    axes[1].legend(frameon=False, fontsize=8)
    save_figure(figure, "fig08_risk_coverage.png")
    coverage = frames["model_coverage.csv"].copy()
    coverage["model_label"] = coverage.model.map(MODEL_LABELS)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        selected = coverage.loc[coverage.target == target]
        for model, part in selected.groupby("model_label"):
            axis.plot(
                [SCHEMES.index(value) for value in part.scheme],
                part.eligible_fraction,
                "o-",
                label=model,
                color=COLORS[model],
            )
        axis.set_xticks(range(4), [value.replace("_", "\n") for value in SCHEMES])
        axis.set_title(target)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Eligible prediction fraction")
    axes[1].legend(frameon=False, fontsize=8)
    save_figure(figure, "fig09_prediction_coverage.png")
    splits = frames["split_audit.csv"]
    summary = splits.groupby(["target", "scheme"], as_index=False).agg(
        fit_rows=("fit_rows", "mean"), held_rows=("held_rows", "mean")
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        part = summary.loc[summary.target == target].set_index("scheme").reindex(SCHEMES)
        axis.bar(np.arange(4) - 0.18, part.fit_rows, 0.36, label="fit", color="#0072B2")
        axis.bar(np.arange(4) + 0.18, part.held_rows, 0.36, label="held", color="#D55E00")
        axis.set_xticks(range(4), [value.replace("_", "\n") for value in SCHEMES])
        axis.set_title(target)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean rows per fold")
    axes[1].legend(frameon=False)
    save_figure(figure, "fig10_split_sizes.png")
    leakage_columns = [
        "group_overlap",
        "record_overlap",
        "fit_calibration_group_overlap",
        "forbidden_time_rows",
    ]
    leakage = splits.groupby(["target", "scheme"])[leakage_columns].sum()
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    image = axis.imshow(leakage.to_numpy(float), cmap="Reds", vmin=0, vmax=1)
    axis.set_yticks(range(len(leakage)), [" / ".join(value) for value in leakage.index])
    axis.set_xticks(
        range(len(leakage_columns)), [value.replace("_", "\n") for value in leakage_columns]
    )
    for row in range(len(leakage)):
        for column in range(len(leakage_columns)):
            axis.text(column, row, int(leakage.iloc[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis, label="leaking rows/groups")
    save_figure(figure, "fig11_leakage_audit.png")
    gates = frames["corrected_gate_matrix.csv"].copy()
    gate_columns = ["rmse_pass", "coverage50_pass", "coverage90_pass"]
    gates["passed"] = gates[gate_columns].sum(axis=1)
    gates["applicable"] = gates[gate_columns].notna().sum(axis=1)
    gates["label"] = gates.target + " / " + gates.scheme + " / " + gates.model.map(MODEL_LABELS)
    figure, axis = plt.subplots(figsize=(9.5, 8.5))
    colors = gates.passed.map({0: "#D55E00", 1: "#E69F00", 2: "#F0E442", 3: "#009E73"})
    axis.barh(np.arange(len(gates)), gates.passed, color=colors)
    axis.set_yticks(range(len(gates)), gates.label)
    axis.set_xlim(0, 3.1)
    axis.set_xlabel("Passed applicable gates (RMSE, 50% coverage, 90% coverage)")
    axis.grid(axis="x", alpha=0.25)
    save_figure(figure, "fig12_gate_matrix.png")
    return {
        "fig01_sss_rmse.png": ["corrected_native_metrics.csv"],
        "fig02_fco2_rmse.png": ["corrected_native_metrics.csv"],
        "fig03_r2.png": ["corrected_native_metrics.csv"],
        "fig04_cruise_equal_rmse.png": ["corrected_native_metrics.csv"],
        "fig05_coverage50.png": ["corrected_native_metrics.csv"],
        "fig06_coverage90.png": ["corrected_native_metrics.csv"],
        "fig07_fco2_interval_width.png": ["corrected_native_metrics.csv"],
        "fig08_risk_coverage.png": ["corrected_risk_coverage.csv"],
        "fig09_prediction_coverage.png": ["model_coverage.csv"],
        "fig10_split_sizes.png": ["split_audit.csv"],
        "fig11_leakage_audit.png": ["split_audit.csv"],
        "fig12_gate_matrix.png": ["corrected_gate_matrix.csv"],
    }


def captions() -> dict[str, str]:
    """Return self-contained captions for every archived figure."""

    return {
        "fig01_sss_rmse.png": "Figure 1. Pooled held-out SSS RMSE across four leakage-repaired validation designs. CatBoost correction improves on raw GLORYS for cruise, geographic, and forward validation, but degrades under complete-region transfer and misses the 1.2 psu geographic and forward gate.",
        "fig02_fco2_rmse.png": "Figure 2. Pooled held-out fCO2 RMSE across the strict v1 RF replica, modernized RF, and direct CatBoost. Modern RF leads cruise and forward validation, while direct CatBoost is least poor for geographic and whole-region extrapolation; every preregistered RMSE gate fails.",
        "fig03_r2.png": "Figure 3. Held-out R-squared for SSS and fCO2 under the four repaired validation designs. Ordinary cruise validation remains optimistic relative to geographic and whole-region transfer, especially for fCO2 where geographic R-squared falls to about 0.31 for the best model.",
        "fig04_cruise_equal_rmse.png": "Figure 4. Cruise-equal RMSE gives every held group equal weight instead of allowing dense cruises to dominate. The ranking broadly confirms the pooled metrics while exposing the larger error experienced by sparsely sampled groups.",
        "fig05_coverage50.png": "Figure 5. Empirical coverage of nominal 50% prediction intervals calibrated only inside each outer training partition. Cruise coverage is near nominal, but geographic, whole-region, and forward transfer frequently under-cover, so these intervals are not yet deployable uncertainty estimates.",
        "fig06_coverage90.png": "Figure 6. Empirical coverage of nominal 90% prediction intervals calibrated without held-out labels. Several strict-transfer settings fall outside the preregistered 0.85-0.95 acceptance range, demonstrating distribution shift that internal calibration cannot repair by itself.",
        "fig07_fco2_interval_width.png": "Figure 7. Mean width of the nested 90% fCO2 intervals for each model and validation design. Width must be read together with Figure 6: a narrower interval is useful only when its held-out coverage remains calibrated.",
        "fig08_risk_coverage.png": "Figure 8. Geographic-holdout RMSE after retaining observations with the narrowest predicted 90% intervals. The curves test whether interval width ranks error risk; they do not establish accuracy at coastal grid cells with no observations.",
        "fig09_prediction_coverage.png": "Figure 9. Fraction of held-out observations receiving an eligible prediction. The strict v1 RF loses roughly 2.8-2.9% of fCO2 rows through its complete-case feature contract, whereas the modern models and residual SSS path retain all rows.",
        "fig10_split_sizes.png": "Figure 10. Mean fit and held-out row counts per fold after complete-group exclusion and, for geographic folds, the 100 km spatial buffer. Smaller geographic training sets are a deliberate consequence of leakage control rather than failed data loading.",
        "fig11_leakage_audit.png": "Figure 11. Summed leakage diagnostics across every target and validation design. All recorded group overlap, record overlap, fit-calibration group overlap, and forbidden future-time counts are exactly zero.",
        "fig12_gate_matrix.png": "Figure 12. Count of passed applicable preregistered gates for each target, validation design, and model; whole-region has two applicable coverage gates because no absolute RMSE ceiling was frozen for that scheme. Passing a subset does not promote a model; Issue 61 unblocks fair model-family screening while retaining all baselines as development evidence only.",
    }


def build_report(caption_map: dict[str, str], metrics: pd.DataFrame) -> str:
    """Return the reviewer-facing Markdown report."""

    key = metrics.set_index(["target", "scheme", "model"])
    sss_cruise = key.loc[("SSS", "cruise", "catboost_background_residual")]
    sss_geo = key.loc[("SSS", "geographic", "catboost_background_residual")]
    sss_region = key.loc[("SSS", "whole_region", "catboost_background_residual")]
    f_cruise = key.loc[("fCO2", "cruise", "modernized_rf")]
    f_geo = key.loc[("fCO2", "geographic", "direct_catboost")]
    figure_text = []
    for number, (name, caption) in enumerate(caption_map.items(), start=1):
        figure_text.append(f"### Figure {number}\n\n![{caption}](figures/{name})\n\n{caption}")
    figure_section = "\n\n".join(figure_text)
    return f"""# P2 validation repair v2.4

## Scientific question and permitted claim

Issue #61 asks whether the existing SSS and fCO2 baselines survive validation that excludes complete cruises, actual geographic blocks plus a 100 km buffer, complete biogeographic regions, and future periods. The permitted claim is limited to corrected development evidence. No new locked-test or external-independent labels were opened, and no global production model is promoted here.

## Data, splits, and leakage controls

The frozen 1993-2025 matchup contains 750,040 SSS labels and 887,811 fCO2 labels. Five cruise folds, five 5-degree geographic folds, five whole-region folds, and three rolling-forward windows were evaluated. Geographic fitting removes complete held groups and every candidate grid cell within 100 km of held coordinates. Every outer fit is divided by complete group into 80% model fitting and 20% interval calibration. The fitted model is never refit after seeing calibration residuals. Cross-target SSS used by fCO2 is generated entirely inside the permitted outer-fit lineage. All audited group, record, calibration-group, and future-time overlap counts are zero.

The calibration plan was clarified before formal training: a single deterministic grouped 80/20 split replaced an early two-fold cross-conformal sketch so that no target could enter another target through an indirectly reused calibration group. This amendment is recorded in the experiment record and protocol.

## Candidate models and training

SSS compares raw GLORYS with a 300-iteration CatBoost residual correction that falls back safely when background SSS is absent. fCO2 compares (1) an exact v1-style 300-tree complete-case RF using its original seven predictors and background SSS, (2) a modernized 96-tree RF using nested cross-fitted SSS and expanded coastal predictors, and (3) direct CatBoost with the same leakage-safe lineage. These are corrected baselines; broader Transformer, expert, and neural-family screening remains assigned to Issue #62.

## Main development results

SSS CatBoost reaches RMSE {sss_cruise.rmse:.3f} psu and R² {sss_cruise.r2:.3f} under cruise holdout, but weakens to {sss_geo.rmse:.3f} psu under geographic holdout and {sss_region.rmse:.3f} psu under whole-region holdout. Raw GLORYS is 1.721 psu overall and becomes preferable to CatBoost for complete-region transfer. The repaired result therefore supports SSS correction where observational support is represented, but rejects a claim of uniformly accurate unsampled-region extrapolation.

For fCO2, modern RF is best under cruise holdout at RMSE {f_cruise.rmse:.2f} µatm and R² {f_cruise.r2:.3f}, missing the 30 µatm gate. Direct CatBoost is best under geographic holdout at RMSE {f_geo.rmse:.2f} µatm and R² {f_geo.r2:.3f}; both RF variants exceed 62 µatm there. The corrected evidence does not support immediate global-model promotion.

Nested intervals are close to nominal under cruise holdout but frequently under-cover during geographic, whole-region, and forward transfer. Interval width can still be studied as a risk-ranking feature, but cannot yet be published as a calibrated grid-cell uncertainty product.

## Decision and limitations

Issue #61 succeeds as a validation repair and fails as a promotion gate. It unblocks Issue #62 because exposure lineage, strict spatial splitting, forward splitting, nested calibration, exact-v1 identity, and common-grid comparisons now have reusable implementations. The next experiment must compare model families on these exact folds and must optimize for strict geographic and forward performance rather than ordinary cruise performance alone.

These numbers are development estimates from observed locations. They are not an independent product validation and do not measure error at never-observed grid cells. Whole-region folds combine LMEs with fixed neutral regions where LME assignment is absent. No TA or DIC conclusion is changed by this experiment.

The prior `p2_sss_global_bakeoff_v2.3`, `p2_fco2_global_bakeoff_v2.3`, and `p2_context_connectivity_v2.3` archives remain useful development records, but their sealed-label, strict-spatial, family-wide rejection, and deployable-uncertainty interpretations are superseded by this repair. Correction notes in those archive folders point here.

## Figure and table index

{figure_section}

Exact source CSVs are in `tables/`. `archive_manifest.json` maps every figure to its source table and hashes all tracked files. Large row-level predictions, nested SSS predictions, split indices, and checkpoints remain outside Git under `{OUTPUT}` and are referenced by hash.
"""


def main() -> int:
    """Build report, figures, captions, and provenance manifest."""

    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    frames = copy_tables()
    mappings = build_figures(frames)
    caption_map = captions()
    caption_text = "# Figure captions\n\n" + "\n\n".join(
        f"## {name}\n\n{caption}" for name, caption in caption_map.items()
    )
    (ARCHIVE / "CAPTIONS.md").write_text(caption_text + "\n", encoding="utf-8")
    report = build_report(caption_map, frames["corrected_native_metrics.csv"])
    (ARCHIVE / "REPORT.md").write_text(report, encoding="utf-8")
    (ARCHIVE / "README.md").write_text(
        "# Issue 61 reviewer archive\n\nStart with [REPORT.md](REPORT.md). Rebuild with "
        "`python scripts/build_p2_validation_repair_archive.py` and verify with "
        "`python scripts/verify_experiment_archive.py "
        "docs/experiment_archive/p2_validation_repair_v2.4`.\n",
        encoding="utf-8",
    )
    tracked = [ARCHIVE / "README.md", ARCHIVE / "REPORT.md", ARCHIVE / "CAPTIONS.md"]
    tracked += sorted(FIGURES.glob("*.png")) + sorted(TABLES.glob("*.csv"))
    for path in tracked:
        if path.suffix in {".md", ".csv"}:
            normalize_lf(path)
    relative_hashes = {
        str(path.relative_to(ARCHIVE)).replace("\\", "/"): sha256(path) for path in tracked
    }
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_artifacts = [
        OUTPUT / "data_manifest.json",
        OUTPUT / "protocol.json",
        OUTPUT / "decision.json",
        OUTPUT / "splits/SSS_split_indices.npz",
        OUTPUT / "splits/fCO2_split_indices.npz",
    ]
    manifest = {
        "experiment_id": "p2_validation_repair_v2.4",
        "training_git_commit": git_commit,
        "data_manifest_sha256": sha256(OUTPUT / "data_manifest.json"),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "1993-2025 frozen development labels; repaired outer holdouts only",
        "decision": "validation repair accepted; no production model promoted; Issue 62 unblocked",
        "figure_source_data": mappings,
        "figure_captions": caption_map,
        "files_sha256": relative_hashes,
        "archive_builder_sha256": sha256(Path(__file__)),
        "analysis_script_sha256": sha256(ROOT / "scripts/run_p2_validation_repair_v2_4.py"),
        "source_artifacts_sha256": {str(path): sha256(path) for path in source_artifacts},
    }
    (ARCHIVE / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
