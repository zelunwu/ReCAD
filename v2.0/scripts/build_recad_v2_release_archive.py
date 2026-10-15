"""Build the reviewer-ready archive for the conditional ReCAD v2.4 release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_4\recad_v2_release_v2.4")
ARCHIVE = ROOT / "docs/experiment_archive/recad_v2_release_v2.4"
TARGETS = ("SSS", "fCO2")
UNITS = {"SSS": "PSU", "fCO2": "µatm"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save(figure: plt.Figure, name: str) -> None:
    figure.savefig(ARCHIVE / "figures" / name, dpi=180, bbox_inches="tight")
    plt.close(figure)


def product_path(target: str) -> Path:
    return OUTPUT / "products" / f"ReCAD_v2.4_{target}_1993_2025.nc"


def main() -> None:
    for directory in (ARCHIVE / "figures", ARCHIVE / "tables"):
        directory.mkdir(parents=True, exist_ok=True)
    coverage = pd.read_csv(OUTPUT / "tables/release_status_by_month.csv")
    summary = pd.read_csv(OUTPUT / "tables/release_status_summary.csv")
    checks = pd.read_csv(OUTPUT / "tables/release_checks.csv")
    comparison = pd.read_csv(OUTPUT / "tables/v11_common_metrics.csv")
    common = pd.read_parquet(OUTPUT / "fco2_v11_common_predictions.parquet")
    common = common.loc[common.v1_eligible].copy()

    coverage.to_csv(ARCHIVE / "tables/table01_release_status_by_month.csv", index=False)
    summary.to_csv(ARCHIVE / "tables/table02_release_status_summary.csv", index=False)
    checks.to_csv(ARCHIVE / "tables/table03_release_checks.csv", index=False)
    comparison.to_csv(ARCHIVE / "tables/table04_v11_common_metrics.csv", index=False)
    pd.read_csv(OUTPUT / "tables/v11_common_025_predictions.csv").to_csv(
        ARCHIVE / "tables/table05_v11_common_025_predictions.csv", index=False
    )

    benchmark = json.loads(
        (ROOT / "configs/frozen/benchmark_contract_v2.3.json").read_text(encoding="utf-8")
    )
    benchmark_rows = [
        row
        for row in benchmark["literature_benchmarks"]
        if row["product"]
        in {
            "ReCAD v1.1 published",
            "ReCAD v1.1 published 2004-2005",
            "Northeast Pacific high-resolution NN",
        }
    ]
    for row in benchmark_rows:
        row["comparison_role"] = "context_only_not_common_support"
    pd.DataFrame(benchmark_rows).to_csv(
        ARCHIVE / "tables/table06_literature_context.csv", index=False
    )

    release = json.loads((OUTPUT / "release_manifest.json").read_text(encoding="utf-8"))
    decisions = []
    for target in TARGETS:
        decisions.extend(
            [
                {
                    "target": target,
                    "layer": "trusted",
                    "scope": release["trusted_scope"][target],
                },
                {
                    "target": target,
                    "layer": "diagnostic",
                    "scope": release["diagnostic_scope"][target],
                },
                {
                    "target": target,
                    "layer": "suppressed",
                    "scope": release["suppressed_scope"][target],
                },
            ]
        )
    pd.DataFrame(decisions).to_csv(ARCHIVE / "tables/table07_release_scope.csv", index=False)

    variable_rows = []
    map_sources = []
    for target in TARGETS:
        with netCDF4.Dataset(product_path(target)) as dataset:
            for name, variable in dataset.variables.items():
                variable_rows.append(
                    {
                        "target": target,
                        "variable": name,
                        "dtype": str(variable.dtype),
                        "dimensions": ";".join(variable.dimensions),
                        "units": getattr(variable, "units", ""),
                        "long_name": getattr(variable, "long_name", ""),
                    }
                )
            latitude = np.asarray(dataset.variables["latitude"][:])
            longitude = np.asarray(dataset.variables["longitude"][:])
            status = np.asarray(dataset.variables["status"][-1, :])
            sample = np.arange(0, len(latitude), 4)
            figure, axis = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
            plot = axis.scatter(
                longitude[sample],
                latitude[sample],
                c=status[sample],
                s=1,
                cmap=ListedColormap(["#1a9850", "#fee08b", "#d73027"]),
                vmin=1,
                vmax=3,
            )
            axis.set(xlabel="Longitude", ylabel="Latitude", xlim=(0, 360), ylim=(-90, 90))
            colorbar = figure.colorbar(plot, ax=axis, ticks=[1, 2, 3])
            colorbar.ax.set_yticklabels(["trusted", "diagnostic", "suppressed"])
            save(figure, f"fig_{target.lower()}_01_status_map.png")
            map_sources.append(
                {
                    "target": target,
                    "time_index": len(dataset.dimensions["time"]) - 1,
                    "sample_step": 4,
                    "source_product": product_path(target).name,
                }
            )
    pd.DataFrame(variable_rows).to_csv(
        ARCHIVE / "tables/table08_variable_dictionary.csv", index=False
    )
    pd.DataFrame(map_sources).to_csv(ARCHIVE / "tables/table09_map_source.csv", index=False)

    for target in TARGETS:
        target_coverage = coverage.loc[coverage.target.eq(target)].copy()
        figure, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
        for status, color in (
            ("trusted", "#1a9850"),
            ("diagnostic", "#fee08b"),
            ("suppressed", "#d73027"),
        ):
            part = target_coverage.loc[target_coverage.status.eq(status)]
            axis.plot(part.time_index, part.area_fraction, label=status, color=color)
        axis.set(xlabel="Month index since 1993-01", ylabel="Area fraction", ylim=(0, 1))
        axis.legend(frameon=False, ncol=3)
        save(figure, f"fig_{target.lower()}_02_area_timeseries.png")

        target_summary = summary.loc[summary.target.eq(target)].set_index("status")
        target_summary = target_summary.reindex(["trusted", "diagnostic", "suppressed"])
        figure, axis = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
        axis.bar(
            target_summary.index,
            target_summary.mean_area_fraction,
            color=["#1a9850", "#fee08b", "#d73027"],
        )
        axis.set(xlabel="Release layer", ylabel="Mean area fraction", ylim=(0, 1))
        save(figure, f"fig_{target.lower()}_03_area_summary.png")

    order = ["native_1over8_common", "common_0.25_cruise"]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x = np.arange(len(order))
    width = 0.35
    for offset, (model, label, color) in enumerate(
        (
            ("v1_strict_prediction", "strict v1.x RF", "#d95f02"),
            ("v2_prediction", "ReCAD v2.4", "#1b9e77"),
        )
    ):
        values = (
            comparison.loc[comparison.model.eq(model)].set_index("resolution").reindex(order).rmse
        )
        axis.bar(x + (offset - 0.5) * width, values, width, label=label, color=color)
    axis.set(
        xticks=x,
        xticklabels=["native 1/8°", "common 0.25°"],
        ylabel="Held-cruise fCO2 RMSE (µatm)",
    )
    axis.legend(frameon=False)
    save(figure, "fig_07_v11_rmse_comparison.png")

    for number, (column, title) in enumerate(
        (
            ("v1_strict_prediction", "strict v1.x RF"),
            ("v2_prediction", "ReCAD v2.4"),
        ),
        start=8,
    ):
        figure, axis = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
        plot = axis.hexbin(common.truth, common[column], gridsize=55, mincnt=1, bins="log")
        low = float(min(common.truth.min(), common[column].min()))
        high = float(max(common.truth.max(), common[column].max()))
        axis.plot([low, high], [low, high], color="black", linewidth=1)
        axis.set(xlabel="Observed fCO2 (µatm)", ylabel=f"{title} (µatm)")
        figure.colorbar(plot, ax=axis, label="log density")
        save(figure, f"fig_{number:02d}_{column}.png")

    group = (
        common.assign(
            v1_squared=(common.v1_strict_prediction - common.truth) ** 2,
            v2_squared=(common.v2_prediction - common.truth) ** 2,
        )
        .groupby("group_key", as_index=False)
        .agg(
            v1_rmse=("v1_squared", lambda values: np.sqrt(values.mean())),
            v2_rmse=("v2_squared", lambda values: np.sqrt(values.mean())),
        )
    )
    group["rmse_improvement"] = group.v1_rmse - group.v2_rmse
    group.to_csv(ARCHIVE / "tables/table10_v11_per_cruise_improvement.csv", index=False)
    group = group.sort_values("rmse_improvement")
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.bar(
        np.arange(len(group)),
        group.rmse_improvement,
        color=np.where(group.rmse_improvement >= 0, "#1b9e77", "#d95f02"),
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="Held cruise ranked by v2 improvement", ylabel="v1 RMSE - v2 RMSE (µatm)")
    save(figure, "fig_10_per_cruise_improvement.png")

    artifacts = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file():
            artifacts.append(
                {
                    "path": str(path.relative_to(OUTPUT)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(artifacts).to_csv(
        ARCHIVE / "tables/table11_external_artifact_hashes.csv", index=False
    )

    captions = {
        "fig_sss_01_status_map.png": "Figure 1. December 2025 SSS release status on the native coastal nodes. Trusted, diagnostic, and suppressed layers remain geographically explicit instead of hiding difficult cells.",
        "fig_fco2_01_status_map.png": "Figure 2. December 2025 fCO2 release status on the native coastal nodes. Pacific A/B cells are diagnostic under the P3 regional gate, while grade D remains suppressed globally.",
        "fig_sss_02_area_timeseries.png": "Figure 3. Monthly area fractions of the three SSS release layers from 1993 through 2025. The trusted layer remains bounded near one fifth of coastal area rather than expanding after validation.",
        "fig_fco2_02_area_timeseries.png": "Figure 4. Monthly area fractions of the three fCO2 release layers from 1993 through 2025. Pacific regional restriction reduces trusted area and increases diagnostic coverage.",
        "fig_sss_03_area_summary.png": "Figure 5. Mean 1993-2025 SSS area fraction by release layer. Most complete-domain cells remain suppressed because geographic availability is not equivalent to validated reliability.",
        "fig_fco2_03_area_summary.png": "Figure 6. Mean 1993-2025 fCO2 area fraction by release layer after the Pacific A/B downgrade. Trusted area is deliberately smaller than the Issue-44 development mask.",
        "fig_07_v11_rmse_comparison.png": "Figure 7. Strict v1.x Random Forest and ReCAD v2.4 fCO2 RMSE on identical complete-cruise rows at native 1/8° and cruise-preserving 0.25° aggregation. v2 improves both comparisons by more than 25%.",
        "fig_08_v1_strict_prediction.png": "Figure 8. Strict v1.x Random Forest predictions against observations on the 7,907 common held-cruise rows. This control omits the published full-period local calibration that would leak future context.",
        "fig_09_v2_prediction.png": "Figure 9. ReCAD v2.4 predictions against observations on the same 7,907 held-cruise rows used by the strict v1.x control. The shared support makes the model comparison directly auditable.",
        "fig_10_per_cruise_improvement.png": "Figure 10. Per-cruise fCO2 RMSE improvement of v2 over strict v1.x on common rows. Positive bars favor v2; remaining negative cruises are preserved rather than removed from the release evidence.",
    }
    report = [
        "# ReCAD v2.4 conditional release",
        "",
        "## Scientific question and permitted claim",
        "",
        "Issue #46 converts the P3-qualified target/region/grade pairs into a versioned 1993-2025 release and tests whether fCO2 improves on a strict v1.x Random Forest under identical complete-cruise exclusion and common rows. The permitted claim is a flagged global-coastal product with same-source transfer evidence. It is not an unconditional global-accuracy or cross-dataset claim.",
        "",
        "## Data, splits, and leakage controls",
        "",
        "The release copies the immutable Issue-44 full-domain predictions and applies the Issue-5 decision without retraining or changing estimates. SSS A/B is trusted across represented basins; fCO2 A/B is trusted outside the coarse Pacific basin. C is diagnostic and D suppressed. TA/DIC and provisional 2026 are absent. The strict v1.x comparison trains after removing the same 50 cruises and is evaluated on 7,907 identical eligible rows; its 0.25° panel preserves cruise identity.",
        "",
        "## Candidate models and training",
        "",
        "The released estimates are the frozen Issue-62 SSS CatBoost background-residual ensemble and Issue-64 fCO2 continuous-background ExtraTrees/CatBoost convex ensemble. Product arrays contain the estimate, 50/90 widths, grade, status, reason bits, support diagnostics, basin ID, and separate trusted/diagnostic prediction variables. No post-P3 target tuning occurs.",
        "",
        "## Main development results",
        "",
        "SSS trusted area averages 20.61% and fCO2 trusted area 12.28%; diagnostic fractions are 8.91% and 18.61%, while 70.47% and 69.11% remain suppressed. Every product has 396 months and 838,661 coastal nodes, with zero release-layer consistency mismatch.",
        "",
        "On common native rows, strict v1.x fCO2 RMSE is 41.100 µatm and v2 is 30.325 µatm, a 26.2% improvement. At cruise-preserving common 0.25°, RMSE is 42.424 versus 31.014 µatm, a 26.9% improvement. These pass the frozen 5% gate. Published v1.1 random-test and 2004-2005 values and the Northeast Pacific literature value remain context only because their labels, calibration, target basis, and domains differ.",
        "",
    ]
    for name, caption in captions.items():
        report.extend([f"![{caption}](figures/{name})", "", caption, ""])
    report.extend(
        [
            "## Decision and limitations",
            "",
            "Decision: **`release_complete_flagged_scope`**. Publish SSS/fCO2 estimates only together with grade, status, interval width, reason bits, support, and provenance. Use `trusted_prediction` for qualified analysis, `diagnostic_prediction` for hypothesis generation, and never silently fill suppressed values. Pacific fCO2 A/B remains diagnostic. TA/DIC and 2026 remain outside this release.",
            "",
            "The products contain globally complete model estimates but only 20.6%/12.3% mean trusted area. P3 evidence is from SOCAT v2026 complete-cruise withholding and does not prove transfer to another observing system. Strict v1.x is a fair method-family control, not an exact reconstruction of its published full-period locally calibrated product. Literature scores are therefore not ranked against the new release.",
            "",
            "## Figure and table index",
            "",
            "Figures 1-6 show release geography and area coverage. Figures 7-10 show common-support v1.x comparisons. Tables 1-3 store release coverage and checks; Tables 4-5 store common comparison metrics and predictions; Table 6 stores literature context; Tables 7-10 store scope, variables, map provenance, and per-cruise comparison; Table 11 hashes every external artifact.",
            "",
        ]
    )
    (ARCHIVE / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (ARCHIVE / "README.md").write_text(
        "# ReCAD v2.4 release archive\n\nReviewer-ready Issue #46 evidence. The multi-GB NetCDF products remain outside Git and are pinned by Table 11 and `release_manifest.json`.\n",
        encoding="utf-8",
    )
    caption_lines = ["# Figure captions", ""]
    for name, caption in captions.items():
        caption_lines.extend([f"## {name}", "", caption, ""])
    (ARCHIVE / "CAPTIONS.md").write_text("\n".join(caption_lines), encoding="utf-8")

    tracked = ["README.md", "REPORT.md", "CAPTIONS.md"]
    tracked += [
        str(path.relative_to(ARCHIVE)).replace("\\", "/")
        for path in sorted((ARCHIVE / "figures").glob("*"))
    ]
    tracked += [
        str(path.relative_to(ARCHIVE)).replace("\\", "/")
        for path in sorted((ARCHIVE / "tables").glob("*.csv"))
    ]
    figure_sources = {
        "fig_sss_01_status_map.png": [
            "table09_map_source.csv",
            "local:ReCAD_v2.4_SSS_1993_2025.nc",
        ],
        "fig_fco2_01_status_map.png": [
            "table09_map_source.csv",
            "local:ReCAD_v2.4_fCO2_1993_2025.nc",
        ],
        "fig_sss_02_area_timeseries.png": ["table01_release_status_by_month.csv"],
        "fig_fco2_02_area_timeseries.png": ["table01_release_status_by_month.csv"],
        "fig_sss_03_area_summary.png": ["table02_release_status_summary.csv"],
        "fig_fco2_03_area_summary.png": ["table02_release_status_summary.csv"],
        "fig_07_v11_rmse_comparison.png": ["table04_v11_common_metrics.csv"],
        "fig_08_v1_strict_prediction.png": ["local:fco2_v11_common_predictions.parquet"],
        "fig_09_v2_prediction.png": ["local:fco2_v11_common_predictions.parquet"],
        "fig_10_per_cruise_improvement.png": ["table10_v11_per_cruise_improvement.csv"],
    }
    builder_hash = sha256(Path(__file__))
    manifest = {
        "experiment_id": "recad_v2_release_v2.4",
        "training_git_commit": "7f78ab372cea612b67ffea338690154edfab98d9",
        "data_manifest_sha256": sha256(ROOT / "configs/frozen/p3_cruise_holdout_manifest_v2.4.csv"),
        "locked_test_opened": True,
        "external_independent_opened": False,
        "evidence_scope": "flagged 1993-2025 release after one-time same-source complete-cruise validation; no cross-dataset claim",
        "decision": "release_complete_flagged_scope",
        "figure_source_data": figure_sources,
        "figure_captions": captions,
        "files_sha256": {relative: sha256(ARCHIVE / relative) for relative in tracked},
        "archive_builder_sha256": builder_hash,
        "analysis_script_sha256": sha256(ROOT / "scripts/build_recad_v2_release.py"),
        "source_artifacts_sha256": {
            "release_manifest": sha256(OUTPUT / "release_manifest.json"),
            "v11_common_comparison": sha256(OUTPUT / "v11_common_comparison.json"),
            "external_artifact_hash_table": sha256(
                ARCHIVE / "tables/table11_external_artifact_hashes.csv"
            ),
        },
    }
    (ARCHIVE / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
