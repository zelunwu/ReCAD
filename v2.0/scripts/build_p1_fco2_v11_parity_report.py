"""Build the reviewer archive for the ReCAD v1.1 parity experiment."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recad.viz.regions import region_masks, v11_region_keys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/experiments/p1_fco2_v11_parity_v2.2"
ISSUE26 = Path("D:/proj_personal/PhD/ReCAD/v2.0/outputs/experiments/p1_fco2_reliability_v2.2")
ARCHIVE = ROOT / "docs/experiment_archive/p1_fco2_v11_parity_v2.2"
REFERENCE_ORDER = ["GStL & GB", "SS", "GoME", "MAB", "SAB", "GoMX"]
KEY_TO_DISPLAY = {
    "GStL": "GStL & GB",
    "SS": "SS",
    "GoMe": "GoME",
    "MAB": "MAB",
    "SAB": "SAB",
    "GoMx": "GoMX",
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_table() -> pd.DataFrame:
    path = ISSUE26 / "development_oof_predictions.parquet"
    data = pd.read_parquet(path)
    latitudes = np.sort(data.latitude.unique())
    longitudes = np.sort(data.longitude.unique())
    masks = region_masks(longitudes, latitudes)
    yi = pd.Index(latitudes).get_indexer(data.latitude)
    xi = pd.Index(longitudes).get_indexer(data.longitude)
    data["region_key"] = "unassigned"
    for key in v11_region_keys()[:-1]:
        data.loc[masks[key][yi, xi], "region_key"] = key
    data = data.loc[data.region_key.ne("unassigned")]
    rows = []
    for support, selected in (
        ("all", data),
        ("grade_AB", data.loc[data.grade.isin(["A", "B"])]),
    ):
        for (scheme, region), part in selected.groupby(["outer_scheme", "region_key"]):
            error = part.prediction - part.truth
            background_error = part.background - part.truth
            rmse = float(np.sqrt(np.mean(error**2)))
            background_rmse = float(np.sqrt(np.mean(background_error**2)))
            rows.append(
                {
                    "support": support,
                    "outer_scheme": scheme,
                    "region_key": region,
                    "Region": KEY_TO_DISPLAY[region],
                    "n": len(part),
                    "rmse": rmse,
                    "mae": float(np.mean(np.abs(error))),
                    "q90_absolute_error": float(np.quantile(np.abs(error), 0.9)),
                    "background_rmse": background_rmse,
                    "skill_vs_background": 1.0 - rmse / background_rmse,
                }
            )
    return pd.DataFrame(rows)


def plot_historical_aggregate(combined: pd.DataFrame, destination: Path) -> None:
    subset = combined.loc[
        combined.Region.eq("NAACOM") & combined.Type.isin(["Test", "Validation"])
    ].copy()
    order = [
        "Published v1.1",
        "v1 RF replica",
        "Direct CatBoost",
        "Global balanced",
        "Regional experts",
    ]
    subset["candidate"] = pd.Categorical(subset.candidate, order, ordered=True)
    subset = subset.sort_values(["candidate", "Type"])
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(order))
    width = 0.36
    for offset, split, color in (
        (-width / 2, "Test", "#4477AA"),
        (width / 2, "Validation", "#CC6677"),
    ):
        values = subset.loc[subset.Type.eq(split)].set_index("candidate").reindex(order).RMSE
        bars = ax.bar(x + offset, values, width, label=split, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x, order, rotation=18, ha="right")
    ax.set_ylabel("RMSE (µatm)")
    ax.set_title("Historical v1.1 comparison: random Test versus 2004-2005 Validation")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=220)
    plt.close(fig)


def plot_regional_validation(combined: pd.DataFrame, destination: Path) -> None:
    subset = combined.loc[
        combined.Region.isin(REFERENCE_ORDER) & combined.Type.eq("Validation")
    ].copy()
    candidates = ["Published v1.1", "Global balanced", "Regional experts"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(REFERENCE_ORDER))
    width = 0.25
    colors = ["#999999", "#4477AA", "#EE7733"]
    for index, (candidate, color) in enumerate(zip(candidates, colors, strict=True)):
        values = (
            subset.loc[subset.candidate.eq(candidate)]
            .set_index("Region")
            .reindex(REFERENCE_ORDER)
            .RMSE
        )
        ax.bar(x + (index - 1) * width, values, width, label=candidate, color=color)
    ax.set_xticks(x, REFERENCE_ORDER)
    ax.set_ylabel("2004-2005 RMSE (µatm)")
    ax.set_title("Historical validation by published v1.1 region")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=220)
    plt.close(fig)


def plot_tradeoff(combined: pd.DataFrame, destination: Path) -> None:
    subset = combined.loc[combined.Region.eq("NAACOM")]
    wide = subset.pivot(index="candidate", columns="Type", values="RMSE")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for candidate, row in wide.dropna(subset=["Test", "Validation"]).iterrows():
        ax.scatter(row.Test, row.Validation, s=80)
        ax.annotate(
            candidate, (row.Test, row.Validation), xytext=(5, 5), textcoords="offset points"
        )
    ax.axvline(17.641706522771837, color="#777777", linestyle="--", linewidth=1)
    ax.axhline(28.974568004074396, color="#777777", linestyle="--", linewidth=1)
    ax.fill_between([0, 17.641706522771837], 0, 28.974568004074396, alpha=0.08)
    ax.set_xlim(14, 24)
    ax.set_ylim(27, 37)
    ax.set_xlabel("Random Test RMSE (µatm)")
    ax.set_ylabel("2004-2005 Validation RMSE (µatm)")
    ax.set_title("No honest candidate crosses both historical thresholds")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=220)
    plt.close(fig)


def plot_strict_heatmap(strict: pd.DataFrame, destination: Path) -> None:
    subset = strict.loc[
        strict.support.eq("grade_AB")
        & strict.outer_scheme.isin(["cruise", "spatial_block", "forward"])
    ]
    values = subset.pivot(index="Region", columns="outer_scheme", values="skill_vs_background")
    values = values.reindex(REFERENCE_ORDER).reindex(columns=["cruise", "spatial_block", "forward"])
    matrix = values.to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    image = ax.imshow(matrix, vmin=-0.2, vmax=0.35, cmap="RdBu", aspect="auto")
    ax.set_xticks(range(3), ["Cruise", "Spatial block", "Forward"])
    ax.set_yticks(range(len(REFERENCE_ORDER)), REFERENCE_ORDER)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                "no rows" if np.isnan(value) else f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )
    fig.colorbar(image, ax=ax, label="Skill versus training-only background")
    ax.set_title("Strict outer-fold skill on grade A/B rows in v1.1 regions")
    fig.tight_layout()
    fig.savefig(destination, dpi=220)
    plt.close(fig)


def main() -> int:
    figures = ARCHIVE / "figures"
    tables = ARCHIVE / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    historical = pd.read_csv(OUTPUT / "historical_metrics.csv")
    phase_b = pd.read_csv(OUTPUT / "phase_b_metrics.csv")
    reference = pd.read_csv(OUTPUT / "published_v11_reference.csv")
    reference.insert(0, "candidate", "Published v1.1")
    historical.candidate = historical.candidate.map(
        {"global_balanced": "Global balanced", "regional_experts": "Regional experts"}
    )
    phase_b.candidate = phase_b.candidate.map(
        {"v1_rf_replica": "v1 RF replica", "direct_catboost": "Direct CatBoost"}
    )
    combined = pd.concat([reference, historical, phase_b], ignore_index=True)
    combined.to_csv(tables / "historical_metrics_combined.csv", index=False)

    strict = strict_table()
    strict.to_csv(tables / "strict_outer_metrics_by_v11_region.csv", index=False)
    strict_schemes = {"cruise", "spatial_block", "forward"}
    strict_ab = strict.loc[strict.support.eq("grade_AB")]
    region_gate = []
    published = reference.loc[reference.Type.eq("Validation")].set_index("Region").RMSE
    for region in REFERENCE_ORDER:
        part = strict_ab.loc[strict_ab.Region.eq(region)]
        schemes = set(part.outer_scheme)
        passed = (
            schemes.issuperset(strict_schemes)
            and bool((part.skill_vs_background > 0).all())
            and bool((part.rmse <= published[region]).all())
            and bool((part.q90_absolute_error <= 35).all())
        )
        region_gate.append(
            {
                "Region": region,
                "required_schemes_present": schemes.issuperset(strict_schemes),
                "passed": passed,
            }
        )
    gates = pd.DataFrame(region_gate)
    gates.to_csv(tables / "strict_region_gate.csv", index=False)

    plot_historical_aggregate(combined, figures / "fig01_historical_aggregate.png")
    plot_regional_validation(combined, figures / "fig02_historical_regions.png")
    plot_tradeoff(combined, figures / "fig03_random_temporal_tradeoff.png")
    plot_strict_heatmap(strict, figures / "fig04_strict_outer_skill.png")

    for name in (
        "historical_decision.json",
        "phase_b_decision.json",
        "protocol.json",
        "published_v11_reference.csv",
    ):
        shutil.copy2(OUTPUT / name, tables / name)

    captions = {
        "fig01_historical_aggregate.png": "Figure 1. Aggregate historical comparison on the original v1.1 random Test and 2004-2005 Validation masks. The exact RF replica wins the random split but fails temporal transfer; the global balanced residual model nearly reproduces the published Validation value but misses the random Test threshold.",
        "fig02_historical_regions.png": "Figure 2. Historical 2004-2005 RMSE by the six published regions. No Stage-A architecture uniformly improves the published table. The table is comparison-only because the v1.1 local calibration used SOCAT labels across the full period.",
        "fig03_random_temporal_tradeoff.png": "Figure 3. Random-split and 2004-2005 errors for each honest candidate. Dashed lines show the published v1.1 values; the shaded lower-left quadrant is the preregistered dual-parity region. No candidate enters it without label-informed post-calibration.",
        "fig04_strict_outer_skill.png": "Figure 4. Skill versus the training-only seasonal-trend background on grade A/B rows under untouched cruise, spatial-block, and forward outer folds, remapped to the original v1.1 regions. Missing forward cells mean no A/B observations survived the support rule. No region passes all required schemes and error gates.",
    }
    caption_text = "# Figure captions\n\n" + "\n\n".join(
        f"## {name}\n\n{caption}" for name, caption in captions.items()
    )
    (ARCHIVE / "CAPTIONS.md").write_text(caption_text + "\n", encoding="utf-8")

    report = """# ReCAD v1.1 parity and strict-transfer audit

**Decision: historical dual parity fails, and no original v1.1 region passes the strict product gate.** Current locked fCO2 and external-independent labels remained sealed. The published v1.1 table is retained as a historical comparison, not as independent evidence.

## Why this experiment was needed

Issue #32 tests whether the v2 design can reproduce and exceed ReCAD v1.1 on its original North American Atlantic domain before claiming a global product. Two questions are separated: whether a model can beat the published random and 2004-2005 numbers, and whether it transfers across cruises, spatial blocks, and future years without using held labels.

## Protocol and models

Stage A was committed before training. It uses exact v1.1 geometric regions, the historical masks through 2021, three seeds, a training-only regional/month seasonal-trend background, and two CatBoost residual candidates: one globally balanced model and six hard regional experts. Stage B was registered after Stage A failed. It reconstructs the exact seven v1 inputs—longitude, latitude, month, SSS, SST, ADT, and atmospheric pCO2—and compares a 300-tree bagged RF replica with direct CatBoost. Neither stage performs label-informed local calibration.

## Historical comparison

| candidate | random Test RMSE | 2004-2005 RMSE | interpretation |
| --- | ---: | ---: | --- |
| Published v1.1 | 17.642 | 28.975 | Historical reported product after local calibration |
| v1 RF replica | 15.964 | 35.507 | Excellent interpolation; poor temporal transfer |
| Direct CatBoost | 19.793 | 30.456 | Balanced but misses both historical thresholds |
| Global balanced residual | 22.536 | 29.353 | Closest honest temporal reproduction |
| Hard regional experts | 20.365 | 30.950 | Helps random Test, harms temporal transfer |

The dual-parity gate requires RMSE below both 17.642 and 28.975 µatm. No honest candidate passes. Blending the RF and residual candidates also produces a continuous tradeoff rather than a point below both thresholds.

![Historical aggregate comparison](figures/fig01_historical_aggregate.png)

Figure 1. Aggregate historical comparison on the original v1.1 random Test and 2004-2005 Validation masks. The exact RF replica wins the random split but fails temporal transfer; the global balanced residual model nearly reproduces the published Validation value but misses the random Test threshold.

![Historical regional comparison](figures/fig02_historical_regions.png)

Figure 2. Historical 2004-2005 RMSE by the six published regions. No Stage-A architecture uniformly improves the published table. The table is comparison-only because the v1.1 local calibration used SOCAT labels across the full period.

![Random and temporal tradeoff](figures/fig03_random_temporal_tradeoff.png)

Figure 3. Random-split and 2004-2005 errors for each honest candidate. Dashed lines show the published v1.1 values; the shaded lower-left quadrant is the preregistered dual-parity region. No candidate enters it without label-informed post-calibration.

## v1.1 method audit

The archived v1 code first trains a 300-tree bagged ensemble with minimum leaf size 1 and all seven predictors at every split (`v1/trainRFER7.m`). The reconstruction notebook then fits local linear coefficients between the RF product and SOCAT across the full time series before calculating the reported split metrics (`v1/Data_Calibrate_SOMFNN_coastalv2.ipynb`, calibration cells 25-31 and metric cells 39-41). Because 2004-2005 SOCAT participates in that calibration, the published “Validation” result is not independent of the final calibrated product. This does not invalidate the old product, but it prevents using 28.975 µatm as proof of out-of-time generalization.

## Strict transfer audit

The current Issue #26 global support-aware candidate was remapped to the same six regions using untouched outer predictions. Grade A/B rows must be present in cruise, spatial-block, and forward schemes, have positive background skill, stay below the corresponding published regional RMSE, and keep q90 absolute error at or below 35 µatm. No region passes. SAB is promising on cruise and spatial-block rows but has no surviving forward A/B rows; GoMX has forward support but misses its very stringent published regional threshold in spatial transfer.

![Strict outer-fold skill](figures/fig04_strict_outer_skill.png)

Figure 4. Skill versus the training-only seasonal-trend background on grade A/B rows under untouched cruise, spatial-block, and forward outer folds, remapped to the original v1.1 regions. Missing forward cells mean no A/B observations survived the support rule. No region passes all required schemes and error gates.

## Decision and next experiment

Issue #32 closes as a negative but decisive benchmark. The exact old RF is retained as an interpolation baseline, while the global balanced residual model is retained as the temporal-transfer baseline. Neither is a publishable global architecture. The next global experiment must train a global backbone with soft regional adapters or mixture-of-experts, optimize macro-region and worst-region loss, and use strictly cross-fitted SSS. Model selection must use cruise, spatial-block, and forward outer folds; the old random split and 2004-2005 table remain descriptive only. Current locked and external-independent labels stay sealed until that candidate passes the development gates.

## Reproducibility

Aggregate source tables, decisions, protocol hashes, and figure captions are stored with this report. Row-level predictions and checkpoints remain in the ignored experiment output and are referenced by hashes in `archive_manifest.json`.
"""
    (ARCHIVE / "REPORT.md").write_text(report, encoding="utf-8")

    tracked_files = [path for path in ARCHIVE.rglob("*") if path.is_file()]
    source_files = [
        OUTPUT / "historical_predictions.parquet",
        OUTPUT / "phase_b_predictions.parquet",
        ISSUE26 / "development_oof_predictions.parquet",
    ]
    manifest = {
        "experiment_id": "p1_fco2_v11_parity_v2.2",
        "decision": "fail_global_parity",
        "locked_fco2_opened": False,
        "external_independent_opened": False,
        "figures": captions,
        "tracked_files": {
            str(path.relative_to(ARCHIVE)): file_hash(path) for path in tracked_files
        },
        "ignored_source_files": {
            str(path): {"bytes": path.stat().st_size, "sha256": file_hash(path)}
            for path in source_files
        },
    }
    (ARCHIVE / "archive_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
