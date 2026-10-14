"""Build the reviewer archive for the Issue 63 process-predictor ablation."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from recad.evaluate.context_connectivity import water_neighbors

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_4\p2_process_predictors_v2.4")
CUBE = Path(r"C:\backup\phd\data\processed\recad_v2_3\global_coastal_predictor_cube_v2.3.nc")
REPAIR = Path(r"C:\backup\phd\data\processed\recad_v2_4\p2_validation_repair_v2.4")
ARCHIVE = ROOT / "docs/experiment_archive/p2_process_predictors_v2.4"
FIGURES = ARCHIVE / "figures"
TABLES = ARCHIVE / "tables"
CONFIG = ROOT / "configs/p2_process_predictors_v2.4.json"
RUNNER = ROOT / "scripts/run_p2_process_predictors_v2_4.py"
FEATURE_BUILDER = ROOT / "scripts/build_p2_process_features_v2_4.py"
SOURCE = ROOT / "src/recad/evaluate/process_predictors.py"
SCHEMES = ["cruise", "geographic", "whole_region", "forward"]
LABELS = {
    "S0_base": "Baseline",
    "S1_dense_front": "SST/wind front",
    "S2_freshwater_proxy": "Freshwater proxy",
    "S3_optional_circulation": "ADT circulation",
    "F0_base": "Baseline",
    "F1_dense_front": "SST/wind front",
    "F2_freshwater_proxy": "Freshwater proxy",
    "F3_optional_circulation": "ADT circulation",
    "F4_continuous_background_residual": "Continuous background",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_lf(path: Path) -> None:
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))


def save_figure(figure: plt.Figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(FIGURES / name, dpi=220, bbox_inches="tight")
    plt.close(figure)


def load_pooled(target: str, variant: str, scheme: str, seed: int = 63024) -> pd.DataFrame:
    paths = sorted(
        (OUTPUT / "partitions").glob(f"{target}_{scheme}_fold*_{variant}_seed{seed}.parquet")
    )
    if not paths:
        raise FileNotFoundError((target, variant, scheme, seed))
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def common_mask(frame: pd.DataFrame, variant: str) -> np.ndarray:
    if "dense_front" in variant:
        return (
            frame.sst_front_neighbour_count.gt(0) & frame.wspd_front_neighbour_count.gt(0)
        ).to_numpy()
    if "freshwater" in variant:
        return (
            frame.sss_process_available.eq(1) & frame.sss_local_neighbour_count.gt(0)
        ).to_numpy()
    if "circulation" in variant:
        return (
            frame.adt_process_available.eq(1) & frame.adt_local_neighbour_count.gt(0)
        ).to_numpy()
    return np.ones(len(frame), dtype=bool)


def rmse(frame: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean((frame.prediction - frame.truth) ** 2)))


def prepare_tables() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    artifact_rows = []
    for kind, pattern in (
        ("feature_cache", "process_features.parquet"),
        ("partition", "partitions/*.parquet"),
        ("selection", "selection/*.json"),
        ("checkpoint", "checkpoints/*"),
    ):
        for path in sorted(OUTPUT.glob(pattern)):
            if path.is_file():
                artifact_rows.append(
                    {
                        "kind": kind,
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    pd.DataFrame(artifact_rows).to_csv(
        OUTPUT / "tables/external_artifact_manifest.csv", index=False
    )
    for name in (
        "screen_metrics.csv",
        "all_seed_metrics.csv",
        "stability_metrics.csv",
        "source_lineage_audit.csv",
        "product_domain_monthly_coverage.csv",
        "external_artifact_manifest.csv",
    ):
        source = OUTPUT / "tables" / name
        shutil.copy2(source, TABLES / name)
        frames[name] = pd.read_csv(source)

    decision = json.loads((OUTPUT / "retained_factors.json").read_text(encoding="utf-8"))
    decision_rows = []
    for target, target_decision in decision["decisions"].items():
        for factor in target_decision["factors"]:
            for scheme in factor["schemes"]:
                decision_rows.append(
                    {
                        "target": target,
                        "variant": factor["variant"],
                        "retained": factor["retained"],
                        **scheme,
                    }
                )
    frames["factor_decisions.csv"] = pd.DataFrame(decision_rows)
    frames["factor_decisions.csv"].to_csv(TABLES / "factor_decisions.csv", index=False)

    split = pd.concat(
        [
            pd.read_csv(REPAIR / "splits" / f"{target}_split_audit.csv").assign(target=target)
            for target in ("SSS", "fCO2")
        ],
        ignore_index=True,
    )
    frames["split_audit.csv"] = split
    split.to_csv(TABLES / "split_audit.csv", index=False)

    source_era_rows = []
    dataset = xr.open_dataset(CUBE, chunks=None, cache=False)
    for field in ("sst", "sss_bg", "adt", "wspd", "xco2air"):
        values = dataset[f"{field}_source_era"].values
        changes = int(np.count_nonzero(values[1:] != values[:-1]))
        source_era_rows.append(
            {
                "field": field,
                "unique_source_eras": len(np.unique(values)),
                "transition_count": changes,
                "first_era": int(values[0]),
                "last_era": int(values[-1]),
                "audit_result": "single_standardized_source_era"
                if changes == 0
                else "transition_requires_diagnostic",
            }
        )
    frames["source_era_audit.csv"] = pd.DataFrame(source_era_rows)
    frames["source_era_audit.csv"].to_csv(TABLES / "source_era_audit.csv", index=False)

    support_rows = []
    variants = {
        "SSS": ["S1_dense_front", "S2_freshwater_proxy", "S3_optional_circulation"],
        "fCO2": [
            "F1_dense_front",
            "F2_freshwater_proxy",
            "F3_optional_circulation",
            "F4_continuous_background_residual",
        ],
    }
    for target, target_variants in variants.items():
        baseline_name = "S0_base" if target == "SSS" else "F0_base"
        for variant in target_variants:
            for scheme in SCHEMES:
                baseline = load_pooled(target, baseline_name, scheme)
                candidate = load_pooled(target, variant, scheme)
                joined = baseline[["record_id", "truth", "prediction"]].merge(
                    candidate,
                    on=["record_id", "truth"],
                    suffixes=("_baseline", "_candidate"),
                    validate="one_to_one",
                )
                mask = common_mask(joined, variant)
                for view, selected in (
                    ("all_support", np.ones(len(joined), bool)),
                    ("common_support", mask),
                ):
                    part = joined.loc[selected]
                    support_rows.append(
                        {
                            "target": target,
                            "variant": variant,
                            "scheme": scheme,
                            "view": view,
                            "n": len(part),
                            "support_fraction": float(np.mean(selected)),
                            "baseline_rmse": float(
                                np.sqrt(np.mean((part.prediction_baseline - part.truth) ** 2))
                            ),
                            "candidate_rmse": float(
                                np.sqrt(np.mean((part.prediction_candidate - part.truth) ** 2))
                            ),
                        }
                    )
    frames["support_metrics.csv"] = pd.DataFrame(support_rows)
    frames["support_metrics.csv"].to_csv(TABLES / "support_metrics.csv", index=False)

    diagnostic_rows = []
    diagnostic_specs = {
        "S1_dense_front": "sst_front_mad",
        "S2_freshwater_proxy": "freshwater_deficit",
        "S3_optional_circulation": "adt_local_mad",
        "F1_dense_front": "sst_front_mad",
        "F2_freshwater_proxy": "freshwater_deficit",
        "F3_optional_circulation": "adt_local_mad",
    }
    for target in ("SSS", "fCO2"):
        for variant, feature in diagnostic_specs.items():
            if not variant.startswith("S" if target == "SSS" else "F"):
                continue
            frame = load_pooled(target, variant, "cruise")
            finite = frame[feature].notna()
            frame = frame.loc[finite].copy()
            frame["bin"] = pd.qcut(frame[feature], 5, duplicates="drop")
            for interval, part in frame.groupby("bin", observed=True):
                diagnostic_rows.append(
                    {
                        "target": target,
                        "variant": variant,
                        "feature": feature,
                        "bin": str(interval),
                        "feature_mean": float(part[feature].mean()),
                        "n": len(part),
                        "rmse": rmse(part),
                        "bias": float((part.prediction - part.truth).mean()),
                    }
                )
    frames["process_bin_diagnostics.csv"] = pd.DataFrame(diagnostic_rows)
    frames["process_bin_diagnostics.csv"].to_csv(
        TABLES / "process_bin_diagnostics.csv", index=False
    )

    boundary_rows = []
    for target, variants_to_check in {
        "SSS": ["S0_base", "S1_dense_front", "S3_optional_circulation"],
        "fCO2": ["F0_base", "F1_dense_front", "F4_continuous_background_residual"],
    }.items():
        baseline = load_pooled(target, variants_to_check[0], "cruise")
        node_lme = baseline.groupby("grid_flat", observed=True).lme_id.agg(
            lambda x: x.mode().iloc[0]
        )
        grid = np.sort(node_lme.index.to_numpy(np.int64))
        neighbours = water_neighbors(grid)
        lme = node_lme.reindex(grid).to_numpy()
        boundary = np.zeros(len(grid), dtype=bool)
        for slot in range(4):
            valid = neighbours[:, slot] >= 0
            boundary[valid] |= lme[valid] != lme[neighbours[valid, slot]]
        boundary_map = pd.Series(boundary, index=grid)
        for variant in variants_to_check:
            frame = load_pooled(target, variant, "cruise")
            frame["observed_lme_boundary"] = frame.grid_flat.map(boundary_map).fillna(False)
            for is_boundary, part in frame.groupby("observed_lme_boundary"):
                boundary_rows.append(
                    {
                        "target": target,
                        "variant": variant,
                        "observed_lme_boundary": bool(is_boundary),
                        "n": len(part),
                        "rmse": rmse(part),
                        "bias": float((part.prediction - part.truth).mean()),
                    }
                )
    frames["boundary_diagnostics.csv"] = pd.DataFrame(boundary_rows)
    frames["boundary_diagnostics.csv"].to_csv(TABLES / "boundary_diagnostics.csv", index=False)
    return frames


def grouped_bars(frame: pd.DataFrame, target: str, name: str) -> None:
    selected = frame.loc[frame.target == target]
    variants = list(dict.fromkeys(selected.variant))
    figure, axis = plt.subplots(figsize=(12, 5.5))
    width = 0.82 / len(variants)
    x = np.arange(len(SCHEMES))
    for offset, variant in enumerate(variants):
        values = selected.loc[selected.variant == variant].set_index("scheme").rmse.reindex(SCHEMES)
        axis.bar(
            x + (offset - (len(variants) - 1) / 2) * width, values, width, label=LABELS[variant]
        )
    axis.set_xticks(x, [value.replace("_", "\n") for value in SCHEMES])
    axis.set_ylabel("RMSE (PSU)" if target == "SSS" else "RMSE (µatm)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3)
    save_figure(figure, name)


def build_figures(frames: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    screen = frames["screen_metrics.csv"]
    grouped_bars(screen, "SSS", "fig01_sss_factor_rmse.png")
    grouped_bars(screen, "fCO2", "fig02_fco2_factor_rmse.png")

    decision = frames["factor_decisions.csv"].copy()
    pivot = decision.pivot(
        index=["target", "variant"], columns="scheme", values="relative_rmse_improvement"
    ).reindex(columns=SCHEMES)
    figure, axis = plt.subplots(figsize=(9.5, 6))
    image = axis.imshow(100 * pivot, cmap="RdBu", vmin=-5, vmax=5, aspect="auto")
    axis.set_yticks(range(len(pivot)), [f"{a} / {LABELS[b]}" for a, b in pivot.index])
    axis.set_xticks(range(len(SCHEMES)), [x.replace("_", "\n") for x in SCHEMES])
    figure.colorbar(image, ax=axis, label="RMSE improvement (%)")
    save_figure(figure, "fig03_relative_improvement.png")

    stability = frames["stability_metrics.csv"]
    selected = stability.loc[
        stability.variant.isin(["F0_base", "F4_continuous_background_residual"])
    ]
    figure, axis = plt.subplots(figsize=(9.5, 5.5))
    for variant, marker in (("F0_base", "o"), ("F4_continuous_background_residual", "s")):
        part = selected.loc[selected.variant == variant].set_index("scheme").reindex(SCHEMES)
        axis.errorbar(
            range(4),
            part.rmse_mean,
            yerr=part.rmse_std,
            marker=marker,
            capsize=4,
            label=LABELS[variant],
        )
    axis.set_xticks(range(4), [x.replace("_", "\n") for x in SCHEMES])
    axis.set_ylabel("fCO2 RMSE (µatm)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    save_figure(figure, "fig04_background_seed_stability.png")

    figure, axis = plt.subplots(figsize=(9.5, 5.5))
    for variant, marker in (("F0_base", "o"), ("F4_continuous_background_residual", "s")):
        part = selected.loc[selected.variant == variant].set_index("scheme").reindex(SCHEMES)
        axis.plot(range(4), part.coverage90_mean, marker=marker, label=LABELS[variant])
    axis.axhspan(0.85, 0.95, color="#009E73", alpha=0.12)
    axis.set_xticks(range(4), [x.replace("_", "\n") for x in SCHEMES])
    axis.set_ylabel("Empirical 90% coverage")
    axis.set_ylim(0.7, 1.0)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    save_figure(figure, "fig05_background_coverage.png")

    coverage = frames["product_domain_monthly_coverage.csv"]
    figure, axis = plt.subplots(figsize=(12, 5))
    date = pd.to_datetime({"year": coverage.year, "month": coverage.month, "day": 1})
    for field in ("sst", "wspd", "sss_bg", "adt"):
        axis.plot(date, 100 * coverage[f"{field}_area_fraction"], label=field)
    axis.set_ylabel("Product-domain area coverage (%)")
    axis.set_ylim(75, 101)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=4)
    save_figure(figure, "fig06_product_domain_coverage.png")

    support = frames["support_metrics.csv"].copy()
    support["improvement"] = (
        100 * (support.baseline_rmse - support.candidate_rmse) / support.baseline_rmse
    )
    selected_support = support.loc[support.scheme.isin(["geographic", "whole_region"])]
    figure, axis = plt.subplots(figsize=(12, 6))
    labels = [
        f"{row.target}\n{LABELS[row.variant]}\n{row.scheme}"
        for row in selected_support.itertuples()
    ]
    colors = selected_support["view"].map({"all_support": "#0072B2", "common_support": "#E69F00"})
    axis.bar(range(len(selected_support)), selected_support.improvement, color=colors)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(range(len(selected_support)), labels, rotation=70, ha="right")
    axis.set_ylabel("RMSE improvement (%)")
    save_figure(figure, "fig07_all_common_support.png")

    bins = frames["process_bin_diagnostics.csv"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, target in zip(axes, ("SSS", "fCO2"), strict=True):
        for variant, part in bins.loc[bins.target == target].groupby("variant"):
            axis.plot(part.feature_mean, part.rmse, marker="o", label=LABELS[variant])
        axis.set_title(target)
        axis.set_xlabel("Process-proxy bin mean")
        axis.set_ylabel("RMSE")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    save_figure(figure, "fig08_process_regimes.png")

    boundary = frames["boundary_diagnostics.csv"]
    boundary = boundary.loc[boundary.observed_lme_boundary]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    xlabels = [f"{row.target}\n{LABELS[row.variant]}" for row in boundary.itertuples()]
    axis.bar(range(len(boundary)), boundary.rmse, color="#56B4E9")
    axis.set_xticks(range(len(boundary)), xlabels, rotation=35, ha="right")
    axis.set_ylabel("RMSE on observed LME-boundary nodes")
    axis.grid(axis="y", alpha=0.25)
    save_figure(figure, "fig09_boundary_diagnostic.png")

    era = frames["source_era_audit.csv"]
    source = frames["source_lineage_audit.csv"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(era.field, era.transition_count, color="#009E73")
    axes[0].set_ylabel("Recorded source-era transitions")
    axes[0].set_ylim(0, 1)
    status_count = source.status.value_counts()
    axes[1].bar(status_count.index, status_count.values, color="#CC79A7")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_ylabel("Number of preregistered sources")
    save_figure(figure, "fig10_source_audit.png")

    return {
        "fig01_sss_factor_rmse.png": ["screen_metrics.csv"],
        "fig02_fco2_factor_rmse.png": ["screen_metrics.csv"],
        "fig03_relative_improvement.png": ["factor_decisions.csv"],
        "fig04_background_seed_stability.png": ["stability_metrics.csv"],
        "fig05_background_coverage.png": ["stability_metrics.csv"],
        "fig06_product_domain_coverage.png": ["product_domain_monthly_coverage.csv"],
        "fig07_all_common_support.png": ["support_metrics.csv"],
        "fig08_process_regimes.png": ["process_bin_diagnostics.csv"],
        "fig09_boundary_diagnostic.png": ["boundary_diagnostics.csv"],
        "fig10_source_audit.png": ["source_era_audit.csv", "source_lineage_audit.csv"],
    }


def captions() -> dict[str, str]:
    return {
        "fig01_sss_factor_rmse.png": "Figure 1. SSS held-out RMSE for the CatBoost baseline and each deployable process factor on the exact corrected cruise, geographic plus 100 km buffer, whole-region, and past-only forward folds; none of the three factors passes the preregistered two-scheme retention gate.",
        "fig02_fco2_factor_rmse.png": "Figure 2. fCO2 held-out RMSE for the ExtraTrees baseline, three deployable process proxies, and the continuous spherical-seasonal-trend background residual; only the continuous background passes the frozen screen retention rule.",
        "fig03_relative_improvement.png": "Figure 3. Single-seed RMSE improvement relative to each target's identical-fold baseline; red cells are regressions and blue cells are improvements, exposing split-specific gains that cannot justify unrestricted feature combinations.",
        "fig04_background_seed_stability.png": "Figure 4. Three-seed mean and standard deviation for the retained fCO2 continuous-background residual and its ExtraTrees baseline; the gain is small and stable, while geographic and whole-region error remain far above publication gates.",
        "fig05_background_coverage.png": "Figure 5. Empirical coverage of nominal 90% intervals for the retained fCO2 branch and baseline; the green band is the 0.85-0.95 target and shows that continuous-background gains do not repair spatial-shift under-coverage.",
        "fig06_product_domain_coverage.png": "Figure 6. Monthly area-weighted availability over all 838,661 frozen coastal product nodes from 1993-2025; SST and wind are complete, background SSS has explicit sparse gaps, and optional ADT has the largest coverage loss.",
        "fig07_all_common_support.png": "Figure 7. Geographic and whole-region RMSE improvement on all rows with deployable fallbacks and on factor-specific common support; reporting both views prevents apparent gains obtained by deleting difficult missing-field cases.",
        "fig08_process_regimes.png": "Figure 8. Cruise-holdout RMSE across quintiles of the tested front, freshwater, and circulation proxies; errors vary strongly by regime, but the added proxy does not deliver the required transferable aggregate improvement.",
        "fig09_boundary_diagnostic.png": "Figure 9. Cruise-holdout RMSE on observation-supported coastal nodes adjacent to a node assigned to another majority LME; the continuous fCO2 background introduces no discrete LME correction and does not create a new hard boundary artifact.",
        "fig10_source_audit.png": "Figure 10. Frozen-cube source-era and preregistered source-availability audit; all deployed cube fields use one standardized source era, while Chl-a, mixed-layer depth, and river discharge remain explicit unavailable negative results rather than fabricated proxies.",
    }


def report_text(caption_map: dict[str, str], frames: dict[str, pd.DataFrame]) -> str:
    stability = frames["stability_metrics.csv"].set_index(["target", "variant", "scheme"])
    base = stability.loc[("fCO2", "F0_base")]
    retained = stability.loc[("fCO2", "F4_continuous_background_residual")]
    results = "/".join(f"{retained.loc[s, 'rmse_mean']:.3f}" for s in SCHEMES)
    baseline = "/".join(f"{base.loc[s, 'rmse_mean']:.3f}" for s in SCHEMES)
    figures = []
    for index, (name, caption) in enumerate(caption_map.items(), start=1):
        figures.append(f"### Figure {index}\n\n![{caption}](figures/{name})\n\n{caption}")
    return f"""# P2 deployable process predictors and continuous background v2.4

## Scientific question and permitted claim

Issue #63 tests whether missing deployable process information or a discontinuous background explains the transfer failures that remained after the fair Issue 62 model-family screen. This is corrected-fold development evidence only. The experiment opened no new locked, external-independent, or 2026 labels and does not qualify a product.

## Data, splits, and leakage controls

Every branch uses the exact Issue 61 cruise, geographic plus 100 km buffer, whole-region, and past-only forward indices, the Issue 62 target-specific finalist family, the same six-configuration inner allowance, outer-training-only calibration, and outer-nested predicted SSS for fCO2. Each factor is added alone. Missing optional fields retain every row through a zero-valued contrast plus explicit availability or neighbour-count fallback. Both all-support and strict common-support results are archived.

## Candidate models and training

The deployable factors are same-month water-connected SST/wind contrasts, a background-SSS freshwater deficit and local contrast, and an optional ADT local contrast. The fCO2-only background is an outer-training-only Ridge surface over unit-sphere coordinates, annual harmonics, trend, and interactions, followed by an ExtraTrees residual. It contains no LME identity or held-label correction. Chl-a, mixed-layer depth, and river discharge were unavailable in the frozen workspace and are preserved as negative availability results; no synthetic substitute was renamed after them.

## Main development results

No SSS process factor passes the retention gate. The front and ADT proxies improve the one-seed whole-region RMSE by about 3.0% and 3.6%, respectively, but regress ordinary cruise or geographic transfer and fail the paired two-scheme rule. The background-SSS freshwater proxy regresses geographic and whole-region performance. These are retained negative results.

For fCO2, the continuous background residual is the sole retained development branch. Its three-seed cruise/geographic/whole-region/forward RMSE is **{results} µatm**, compared with **{baseline} µatm** for the matched ExtraTrees baseline. The improvement is small: roughly 1.5% on cruise, 0.8% on geographic after three seeds, 0.4% on whole-region, and 1.1% on forward. It passed the frozen screen because cruise and geographic exceeded 1% with paired grouped upper MSE differences below zero. The process proxies do not pass; the freshwater branch notably regresses geographic RMSE by 4.2%.

The retained branch remains a development component rather than a production promotion. Geographic and whole-region RMSE remain about 54.2 and 52.8 µatm. Empirical 90% coverage remains only 0.796 in geographic, 0.782 in whole-region, and 0.837 forward, so the uncertainty contract still fails under shift. Source-era arrays contain a single standardized era for every deployed cube field, and the continuous background has no hard LME offsets. Observation-supported boundary diagnostics show no newly introduced discontinuity, but they cannot prove behavior on coastlines with no observations.

## Decision and limitations

The machine decision is `retain_fco2_continuous_background_for_development_only`. SSS remains the Issue 62 CatBoost background-residual finalist without added Issue 63 process fields. fCO2 carries the continuous background residual forward as a controlled candidate, with the direct ExtraTrees path retained as the comparison and fallback. No tested single process factor earns combination rights. Therefore Issue 64 may test the already-triggered graph hypothesis only with a changed architecture or information path, and may test an uncertainty head specifically against the documented shifted under-coverage. It must not rerun Issue 42's failed four-neighbour ST96 implementation, and it has no evidence-based process-factor combination branch from Issue 63.

## Figure and table index

{chr(10).join(figures)}

Exact source CSVs are under `tables/`. Row-level predictions, selections, feature cache, and 1,080 checkpoint files remain outside Git under `{OUTPUT}`. `archive_manifest.json` maps every figure to its source tables and hashes the complete archive.
"""


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    frames = prepare_tables()
    mappings = build_figures(frames)
    caption_map = captions()
    (ARCHIVE / "REPORT.md").write_text(report_text(caption_map, frames), encoding="utf-8")
    (ARCHIVE / "CAPTIONS.md").write_text(
        "# Figure captions\n\n"
        + "\n\n".join(f"## {name}\n\n{caption}" for name, caption in caption_map.items())
        + "\n",
        encoding="utf-8",
    )
    (ARCHIVE / "README.md").write_text(
        "# Issue 63 reviewer archive\n\nStart with [REPORT.md](REPORT.md). Rebuild with "
        "`python scripts/build_p2_process_predictors_archive.py` and verify with "
        "`python scripts/verify_experiment_archive.py "
        "docs/experiment_archive/p2_process_predictors_v2.4`.\n",
        encoding="utf-8",
    )
    tracked = [ARCHIVE / "README.md", ARCHIVE / "REPORT.md", ARCHIVE / "CAPTIONS.md"]
    tracked += sorted(FIGURES.glob("*.png")) + sorted(TABLES.glob("*.csv"))
    for path in tracked:
        if path.suffix in {".md", ".csv"}:
            normalize_lf(path)
    training_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest = {
        "experiment_id": "p2_process_predictors_v2.4",
        "training_git_commit": training_commit,
        "data_manifest_sha256": sha256(OUTPUT / "feature_manifest.json"),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "provisional_2026_opened": False,
        "evidence_scope": "1993-2025 frozen development labels on corrected Issue 61 folds",
        "decision": {
            "status": "retain_fco2_continuous_background_for_development_only",
            "SSS_retained_factors": [],
            "fCO2_retained_factors": ["F4_continuous_background_residual"],
            "production_model_promoted": False,
        },
        "figure_source_data": mappings,
        "figure_captions": caption_map,
        "files_sha256": {
            str(path.relative_to(ARCHIVE)).replace("\\", "/"): sha256(path) for path in tracked
        },
        "archive_builder_sha256": sha256(Path(__file__)),
        "analysis_script_sha256": sha256(RUNNER),
        "source_artifacts_sha256": {
            str(path): sha256(path)
            for path in (
                CONFIG,
                RUNNER,
                FEATURE_BUILDER,
                SOURCE,
                OUTPUT / "feature_manifest.json",
                OUTPUT / "retained_factors.json",
                OUTPUT / "analysis.json",
                REPAIR / "splits/SSS_split_indices.npz",
                REPAIR / "splits/fCO2_split_indices.npz",
            )
        },
    }
    (ARCHIVE / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
