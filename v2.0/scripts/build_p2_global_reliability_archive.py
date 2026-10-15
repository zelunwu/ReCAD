"""Build the reviewer-ready Issue 44 reliability archive."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd

from build_p2_global_reliability_product_v2_4 import assign_contract
from recad.evaluate.reliability_contract import interval_wis

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_4\p2_global_reliability_contract_v2.4")
CUBE = Path(r"C:\backup\phd\data\processed\recad_v2_3\global_coastal_predictor_cube_v2.3.nc")
MATCHUP = Path(
    r"C:\backup\phd\data\processed\recad_v2_3\p2_predictor_ablation_v2.3"
    r"\socat_predictor_matchup_v2.3.parquet"
)
ARCHIVE = ROOT / "docs/experiment_archive/p2_global_reliability_contract_v2.4"
FIGURES = ARCHIVE / "figures"
TABLES = ARCHIVE / "tables"
SCHEMES = ["cruise", "geographic", "whole_region", "forward"]
COLORS = {"A": "#2166ac", "B": "#67a9cf", "C": "#fdae61", "D": "#b2182b"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_lf(path: Path) -> None:
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))


def pooled(target: str, scheme: str) -> pd.DataFrame:
    paths = sorted((OUTPUT / "partitions").glob(f"{target}_{scheme}_fold*.parquet"))
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def save(figure: plt.Figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(FIGURES / name, dpi=220, bbox_inches="tight")
    plt.close(figure)


def build_tables(contracts: dict) -> dict[str, pd.DataFrame]:
    metadata = pd.read_parquet(MATCHUP, columns=["record_id", "region_key", "lme_id"])
    metadata = metadata.drop_duplicates("record_id")
    interval_rows, risk_rows, grade_rows, annual_rows, region_rows = [], [], [], [], []
    frames = {}
    for target in ("SSS", "fCO2"):
        thresholds = [*contracts[target]["risk_thresholds"], np.inf]
        for scheme in SCHEMES:
            frame = pooled(target, scheme)
            frames[(target, scheme)] = frame
            error = frame.prediction.to_numpy(float) - frame.truth.to_numpy(float)
            absolute = np.abs(error)
            interval_rows.append(
                {
                    "target": target,
                    "scheme": scheme,
                    "rows": len(frame),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "mae": float(np.mean(absolute)),
                    "bias": float(np.mean(error)),
                    "coverage50": float(np.mean(absolute <= frame.width50)),
                    "coverage90": float(np.mean(absolute <= frame.width90)),
                    "mean_width50": float(np.mean(2 * frame.width50)),
                    "mean_width90": float(np.mean(2 * frame.width90)),
                    "wis": interval_wis(
                        frame.truth, frame.prediction, frame.width50, frame.width90
                    ),
                }
            )
            for fraction, threshold in zip((0.5, 0.75, 0.9, 1.0), thresholds, strict=True):
                selected = frame.predicted_error_scale.to_numpy(float) <= threshold
                risk_rows.append(
                    {
                        "target": target,
                        "scheme": scheme,
                        "nominal_cruise_risk_fraction": fraction,
                        "retained_fraction": float(selected.mean()),
                        "rows": int(selected.sum()),
                        "rmse": float(np.sqrt(np.mean(error[selected] ** 2))),
                        "mae": float(np.mean(absolute[selected])),
                    }
                )
            grade, status, reason = assign_contract(
                frame, frame.predicted_error_scale.to_numpy(float), contracts[target]
            )
            frame = frame.assign(grade=np.asarray(["", "A", "B", "C", "D"])[grade])
            frame["status_code"] = status
            frame["reason_bits"] = reason
            for label in ("A", "B", "C", "D", "AB"):
                selected = frame.grade.isin(["A", "B"]) if label == "AB" else frame.grade.eq(label)
                if not selected.any():
                    continue
                grade_rows.append(
                    {
                        "target": target,
                        "scheme": scheme,
                        "grade": label,
                        "rows": int(selected.sum()),
                        "fraction": float(selected.mean()),
                        "rmse": float(np.sqrt(np.mean(error[selected] ** 2))),
                        "mae": float(np.mean(absolute[selected])),
                        "coverage90": float(
                            np.mean(absolute[selected] <= frame.loc[selected, "width90"])
                        ),
                    }
                )
            for year, part in frame.groupby("yr", observed=True):
                local = part.prediction.to_numpy(float) - part.truth.to_numpy(float)
                annual_rows.append(
                    {
                        "target": target,
                        "scheme": scheme,
                        "year": int(year),
                        "rows": len(part),
                        "rmse": float(np.sqrt(np.mean(local**2))),
                        "coverage90": float(np.mean(np.abs(local) <= part.width90)),
                    }
                )
            joined = frame.merge(metadata, on="record_id", how="left", validate="many_to_one")
            for region, part in joined.groupby("region_key", observed=True):
                if len(part) < 200:
                    continue
                local = part.prediction.to_numpy(float) - part.truth.to_numpy(float)
                region_rows.append(
                    {
                        "target": target,
                        "scheme": scheme,
                        "region_key": str(region),
                        "rows": len(part),
                        "rmse": float(np.sqrt(np.mean(local**2))),
                        "coverage90": float(np.mean(np.abs(local) <= part.width90)),
                    }
                )
    result = {
        "outer_metrics.csv": pd.DataFrame(interval_rows),
        "fixed_coverage_risk.csv": pd.DataFrame(risk_rows),
        "observed_grade_metrics.csv": pd.DataFrame(grade_rows),
        "annual_calibration.csv": pd.DataFrame(annual_rows),
        "regional_calibration.csv": pd.DataFrame(region_rows),
    }
    method_rows = []
    for path in sorted((OUTPUT / "selection").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["method_scores"]:
            method_rows.append(
                {
                    "target": payload["target"],
                    "scheme": payload["scheme"],
                    "fold": payload["fold"],
                    "selected": row["method"] == payload["selected"]["method"],
                    **row,
                }
            )
    result["interval_method_audit.csv"] = pd.DataFrame(method_rows)
    return result


def product_tables() -> dict[str, pd.DataFrame]:
    monthly = pd.read_csv(OUTPUT / "tables/product_monthly_grade_coverage.csv")
    aggregate = (
        monthly.groupby(["target", "grade"], observed=True)
        .agg(node_months=("nodes", "sum"), mean_area_fraction=("area_fraction", "mean"))
        .reset_index()
    )
    annual = (
        monthly.groupby(["target", "year", "grade"], observed=True)
        .area_fraction.mean()
        .reset_index()
    )
    mask_rows = []
    with netCDF4.Dataset(CUBE) as source:
        area = np.asarray(source.variables["cell_area_km2"][:], float)
        total = area.sum()
        for time_index in range(len(source.dimensions["time"])):
            row = {
                "time_index": time_index,
                "year": int(source.variables["year"][time_index]),
                "month": int(source.variables["month"][time_index]),
            }
            for name in ("predictor_ready_mask", "strict_common_support_mask"):
                selected = np.asarray(source.variables[name][time_index, :], bool)
                row[f"{name}_area_fraction"] = float(area[selected].sum() / total)
            mask_rows.append(row)
    return {
        "product_monthly_grade_coverage.csv": monthly,
        "product_aggregate_grade_coverage.csv": aggregate,
        "product_annual_grade_coverage.csv": annual,
        "product_mask_coverage.csv": pd.DataFrame(mask_rows),
    }


def product_snapshot() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    reason_rows = []
    for target in ("SSS", "fCO2"):
        path = OUTPUT / "products" / f"{target}_reliability_v2.4.nc"
        with netCDF4.Dataset(path) as dataset:
            index = 390  # July 2025
            grade = np.asarray(dataset.variables["applicability_grade"][index, :])
            reason = np.asarray(dataset.variables["reason_bits"][index, :])
            area = np.asarray(dataset.variables["cell_area_km2"][:], float)
            frames.append(
                pd.DataFrame(
                    {
                        "target": target,
                        "longitude": np.where(
                            np.asarray(dataset.variables["longitude"][:]) > 180,
                            np.asarray(dataset.variables["longitude"][:]) - 360,
                            np.asarray(dataset.variables["longitude"][:]),
                        ),
                        "latitude": np.asarray(dataset.variables["latitude"][:]),
                        "grade": np.asarray(["", "A", "B", "C", "D"])[grade],
                    }
                )
            )
            for bit, name in (
                (1, "no_training_within_200km"),
                (16, "environmental_extrapolation"),
                (32, "low_group_diversity"),
                (64, "missing_predictor"),
                (512, "temporal_extrapolation"),
                (1024, "model_disagreement"),
            ):
                selected = (reason & bit) > 0
                reason_rows.append(
                    {
                        "target": target,
                        "reason_bit": bit,
                        "reason": name,
                        "nodes": int(selected.sum()),
                        "area_fraction": float(area[selected].sum() / area.sum()),
                    }
                )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(reason_rows)


def make_figures(tables: dict[str, pd.DataFrame], snapshot: pd.DataFrame) -> None:
    outer = tables["outer_metrics.csv"]
    pivot = outer.pivot(index="scheme", columns="target", values="rmse").loc[SCHEMES]
    ax = pivot.plot.bar(figsize=(8, 4), secondary_y="fCO2")
    ax.set_ylabel("SSS RMSE (PSU)")
    ax.right_ax.set_ylabel("fCO2 RMSE (µatm)")
    save(ax.figure, "fig01_outer_rmse.png")

    pivot = outer.pivot(index="scheme", columns="target", values="coverage90").loc[SCHEMES]
    ax = pivot.plot.bar(figsize=(8, 4))
    ax.axhspan(0.85, 0.95, color="#d9f0d3", alpha=0.5)
    ax.axhline(0.9, color="black", linestyle="--")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Empirical 90% coverage")
    save(ax.figure, "fig02_outer_coverage90.png")

    pivot = outer.pivot(index="scheme", columns="target", values="wis").loc[SCHEMES]
    ax = pivot.plot.bar(figsize=(8, 4), secondary_y="fCO2")
    ax.set_ylabel("SSS WIS")
    ax.right_ax.set_ylabel("fCO2 WIS")
    save(ax.figure, "fig03_outer_wis.png")

    risk = tables["fixed_coverage_risk.csv"]
    for number, target in ((4, "SSS"), (5, "fCO2")):
        fig, ax = plt.subplots(figsize=(7, 4))
        for scheme in SCHEMES:
            part = risk[(risk.target == target) & (risk.scheme == scheme)]
            ax.plot(part.retained_fraction, part.rmse, marker="o", label=scheme)
        ax.set_xlabel("Retained observation fraction")
        ax.set_ylabel("RMSE" + (" (µatm)" if target == "fCO2" else " (PSU)"))
        ax.legend()
        save(fig, f"fig0{number}_{target.lower()}_risk_coverage.png")

    grade = tables["observed_grade_metrics.csv"]
    ab = grade[grade.grade == "AB"].pivot(index="scheme", columns="target", values="rmse")
    ax = ab.reindex(SCHEMES).plot.bar(figsize=(8, 4), secondary_y="fCO2")
    ax.set_ylabel("SSS A/B RMSE (PSU)")
    ax.right_ax.set_ylabel("fCO2 A/B RMSE (µatm)")
    save(ax.figure, "fig06_ab_observed_rmse.png")

    product = tables["product_aggregate_grade_coverage.csv"]
    pivot = product.pivot(index="target", columns="grade", values="mean_area_fraction")
    ax = pivot[["A", "B", "C", "D"]].plot.bar(
        stacked=True, figsize=(7, 4), color=[COLORS[x] for x in ("A", "B", "C", "D")]
    )
    ax.set_ylabel("Mean area-weighted fraction")
    ax.legend(title="Grade")
    save(ax.figure, "fig07_product_grade_area.png")

    annual = tables["product_annual_grade_coverage.csv"]
    annual = annual[annual.grade.isin(["A", "B"])].groupby(["target", "year"]).area_fraction.sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    for target in ("SSS", "fCO2"):
        part = annual.loc[target]
        ax.plot(part.index, part.values, label=target)
    ax.set_ylabel("A/B area-weighted fraction")
    ax.set_xlabel("Year")
    ax.legend()
    save(fig, "fig08_trusted_area_by_year.png")

    for number, target in ((9, "SSS"), (10, "fCO2")):
        part = snapshot[snapshot.target == target]
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for grade_name in ("D", "C", "B", "A"):
            selected = part.grade.eq(grade_name)
            ax.scatter(
                part.loc[selected, "longitude"],
                part.loc[selected, "latitude"],
                s=0.12,
                c=COLORS[grade_name],
                label=grade_name,
                rasterized=True,
            )
        ax.set(xlim=(-180, 180), ylim=(-80, 80), xlabel="Longitude", ylabel="Latitude")
        ax.legend(markerscale=20, ncol=4)
        save(fig, f"fig{number:02d}_{target.lower()}_grade_map_2025_07.png")

    reasons = tables["product_reason_bits_2025_07.csv"]
    pivot = reasons.pivot(index="reason", columns="target", values="area_fraction")
    ax = pivot.plot.bar(figsize=(9, 4))
    ax.set_ylabel("Area-weighted fraction")
    save(ax.figure, "fig11_reason_bit_prevalence.png")

    region = tables["regional_calibration.csv"]
    top = region.groupby("region_key").rows.sum().nlargest(15).index
    part = region[(region.scheme == "cruise") & region.region_key.isin(top)]
    pivot = part.pivot(index="region_key", columns="target", values="coverage90")
    ax = pivot.sort_values("fCO2").plot.barh(figsize=(8, 6))
    ax.axvline(0.9, color="black", linestyle="--")
    ax.set_xlabel("Cruise-fold 90% coverage")
    save(ax.figure, "fig12_regional_calibration.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    calibration = json.loads((OUTPUT / "calibration_contract.json").read_text(encoding="utf-8"))
    tables = build_tables(calibration["contracts"])
    tables.update(product_tables())
    snapshot, reasons = product_snapshot()
    tables["product_reason_bits_2025_07.csv"] = reasons
    for name, frame in tables.items():
        frame.to_csv(TABLES / name, index=False)
    shutil.copy2(OUTPUT / "tables/SSS_partition_summary.csv", TABLES)
    shutil.copy2(OUTPUT / "tables/fCO2_partition_summary.csv", TABLES)
    make_figures(tables, snapshot)

    external = []
    for target, item in calibration["products"].items():
        path = Path(item["path"])
        external.append(
            {
                "kind": "product",
                "target": target,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": item["sha256"],
            }
        )
    for path in (OUTPUT / "calibration_contract.json", OUTPUT / "calibration_contract_frozen.json"):
        external.append(
            {
                "kind": "calibration",
                "target": "both",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(external).to_csv(TABLES / "external_artifact_manifest.csv", index=False)

    outer = tables["outer_metrics.csv"].set_index(["target", "scheme"])
    product = tables["product_aggregate_grade_coverage.csv"]
    trusted = product[product.grade.isin(["A", "B"])].groupby("target").mean_area_fraction.sum()
    suppressed = product[product.grade.eq("D")].set_index("target").mean_area_fraction
    shifted = outer.loc[(slice(None), ["geographic", "whole_region", "forward"]), "coverage90"]
    gate = bool((shifted >= 0.85).all() and (suppressed <= 0.75).all())
    decision = {
        "machine_decision": "freeze_applicability_contract_but_fail_global_interval_qualification",
        "acceptance_gate_passed": gate,
        "trusted_area_fraction": trusted.to_dict(),
        "suppressed_area_fraction": suppressed.to_dict(),
        "minimum_shifted_coverage90": float(shifted.min()),
        "independent_labels_opened": False,
        "labels_2026_opened": False,
    }
    (OUTPUT / "reliability_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )

    captions = {
        "fig01_outer_rmse.png": "Figure 1. Pooled corrected-fold point RMSE for the frozen SSS and fCO2 candidates. Geographic, whole-region, and forward partitions quantify transfer error rather than random-row interpolation.",
        "fig02_outer_coverage90.png": "Figure 2. Empirical 90% interval coverage on corrected outer folds. Cruise coverage is near nominal, while multiple shifted partitions fall below the frozen 0.85 lower bound.",
        "fig03_outer_wis.png": "Figure 3. Weighted interval score for the selected target-specific interval rules across all corrected validation schemes; lower values indicate sharper calibrated distributions.",
        "fig04_sss_risk_coverage.png": "Figure 4. SSS RMSE as observations are retained through fixed cruise-derived risk thresholds. Shifted regions retain fewer rows and remain harder even at matched risk cutoffs.",
        "fig05_fco2_risk_coverage.png": "Figure 5. fCO2 RMSE at fixed 50%, 75%, 90%, and complete cruise-derived risk envelopes. Risk ranking is useful in-distribution but weak under whole-region transfer.",
        "fig06_ab_observed_rmse.png": "Figure 6. RMSE among observations receiving frozen A/B status in each outer scheme. Geographic folds yield no A/B rows because the support envelope correctly abstains everywhere.",
        "fig07_product_grade_area.png": "Figure 7. Mean area-weighted A/B/C/D coverage across the complete 396-month by 838,661-node product geometry. Difficult areas remain present as C/D rather than being discarded.",
        "fig08_trusted_area_by_year.png": "Figure 8. Area-weighted A/B fraction from 1993 through 2025. Changes reflect predictor availability and distance from the temporal and environmental training envelope.",
        "fig09_sss_grade_map_2025_07.png": "Figure 9. Global coastal SSS applicability grades in July 2025. A/B is concentrated near demonstrated observational support; C is diagnostic and D is suppressed.",
        "fig10_fco2_grade_map_2025_07.png": "Figure 10. Global coastal fCO2 applicability grades in July 2025. The full geometry is retained, including unsupported coasts marked D rather than silently omitted.",
        "fig11_reason_bit_prevalence.png": "Figure 11. Area-weighted prevalence of each deployable reason bit in July 2025. Multiple bits may coexist, so fractions are diagnostic and do not sum to one.",
        "fig12_regional_calibration.png": "Figure 12. Cruise-fold 90% coverage in the fifteen observation regions with the largest row counts. Regional heterogeneity remains visible despite near-nominal pooled cruise coverage.",
    }
    report = f"""# Issue #44 — Global coastal calibration and applicability contract

## Scientific question and permitted claim

This experiment asks whether the frozen Issue 62 SSS candidate and Issue 64 fCO2 ensemble can receive leakage-safe uncertainty intervals and deployable A/B/C/D grid-month applicability flags over the complete 1993-2025 coastal domain. The result is a development-stage reliability contract and complete auditable projection. It is not independent qualification of unsampled coasts.

## Data, splits, and leakage controls

All point and calibration evidence uses the corrected Issue 61 cruise, geographic-plus-buffer, whole-region, and past-only forward folds. Interval method selection stayed inside outer training. Risk heads use only deployment-available support, environmental novelty, input availability, time distance, and model/background disagreement. Locked, external-independent, and 2026 labels remained sealed. The final products preserve all 332,109,756 grid-months per target.

## Candidate models and training

SSS is the three-seed CatBoost background-residual finalist. fCO2 is the continuous-background ExtraTrees/CatBoost convex ensemble with a frozen final CatBoost weight of 0.30 and final fCO2 training uses the Issue 61 leakage-safe cross-fitted SSS covariate. SSS intervals use support-risk-scaled conformal widths; fCO2 uses the prediction-stratified conformal control. Grades use cruise-derived 50/75/90% risk thresholds plus hard geographic, environmental, missing-input, temporal, group-diversity, and disagreement reason bits.

## Main development results

Cruise-fold RMSE is {outer.loc[("SSS", "cruise"), "rmse"]:.3f} PSU for SSS and {outer.loc[("fCO2", "cruise"), "rmse"]:.2f} µatm for fCO2. Cruise 90% coverage is {outer.loc[("SSS", "cruise"), "coverage90"]:.3f} and {outer.loc[("fCO2", "cruise"), "coverage90"]:.3f}. The minimum shifted-scheme coverage is {decision["minimum_shifted_coverage90"]:.3f}, below the frozen 0.85 gate. Complete-domain mean trusted area is {trusted["SSS"]:.1%} for SSS and {trusted["fCO2"]:.1%} for fCO2; suppressed area is {suppressed["SSS"]:.1%} and {suppressed["fCO2"]:.1%}, respectively. The suppressed-area limit passes, but global interval qualification fails.

## Decision and limitations

The A/B/C/D machinery and complete products are frozen for audit and the later one-time independent decision. Global predictive qualification does not pass because intervals under-cover geographic and whole-region shifts. A/B means supported by the development contract, not proven accurate in observation-free waters. C is diagnostic; D is suppressed. The product must retain these flags and may not publish unflagged global values. No difficult region was removed to improve a score.

## Figures

"""
    for name, caption in captions.items():
        report += f"![{caption}](figures/{name})\n\n{caption}\n\n"
    report += """## Figure and table index

Figures 1-12 cover point error, calibration, risk coverage, observed grades, complete-domain area, temporal drift, global maps, reason bits, and regional calibration. Tables contain outer metrics, fixed-coverage risk, observed-grade metrics, annual and regional calibration, interval-method audit, complete-product grades and masks, reason-bit prevalence, partition summaries, and external artifact hashes.
"""
    (ARCHIVE / "REPORT.md").write_text(report, encoding="utf-8")
    (ARCHIVE / "README.md").write_text(
        "# Issue 44 reviewer archive\n\nRun `python scripts/build_p2_global_reliability_archive.py`, then verify with `python scripts/verify_experiment_archive.py docs/experiment_archive/p2_global_reliability_contract_v2.4`. Multi-GB NetCDF products and checkpoints remain outside Git.\n",
        encoding="utf-8",
    )
    (ARCHIVE / "CAPTIONS.md").write_text(
        "# Figure captions\n\n"
        + "\n\n".join(f"## {name}\n\n{caption}" for name, caption in captions.items())
        + "\n",
        encoding="utf-8",
    )
    source_map = {
        "fig01_outer_rmse.png": ["outer_metrics.csv"],
        "fig02_outer_coverage90.png": ["outer_metrics.csv"],
        "fig03_outer_wis.png": ["outer_metrics.csv"],
        "fig04_sss_risk_coverage.png": ["fixed_coverage_risk.csv"],
        "fig05_fco2_risk_coverage.png": ["fixed_coverage_risk.csv"],
        "fig06_ab_observed_rmse.png": ["observed_grade_metrics.csv"],
        "fig07_product_grade_area.png": ["product_aggregate_grade_coverage.csv"],
        "fig08_trusted_area_by_year.png": ["product_annual_grade_coverage.csv"],
        "fig09_sss_grade_map_2025_07.png": ["local:SSS_reliability_v2.4.nc"],
        "fig10_fco2_grade_map_2025_07.png": ["local:fCO2_reliability_v2.4.nc"],
        "fig11_reason_bit_prevalence.png": ["product_reason_bits_2025_07.csv"],
        "fig12_regional_calibration.png": ["regional_calibration.csv"],
    }
    for path in [ARCHIVE / "README.md", ARCHIVE / "REPORT.md", ARCHIVE / "CAPTIONS.md"]:
        normalize_lf(path)
    for path in TABLES.glob("*.csv"):
        normalize_lf(path)
    tracked = [ARCHIVE / "README.md", ARCHIVE / "REPORT.md", ARCHIVE / "CAPTIONS.md"]
    tracked += sorted(FIGURES.glob("*.png")) + sorted(TABLES.glob("*.csv"))
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest = {
        "experiment_id": "p2_global_reliability_contract_v2.4",
        "training_git_commit": commit,
        "data_manifest_sha256": sha256(ROOT / "configs/p2_global_reliability_contract_v2.4.json"),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "provisional_2026_opened": False,
        "evidence_scope": "1993-2025 development labels on corrected Issue 61 folds and complete unlabeled product geometry",
        "decision": decision,
        "figure_source_data": source_map,
        "figure_captions": captions,
        "files_sha256": {
            str(path.relative_to(ARCHIVE)).replace("\\", "/"): sha256(path) for path in tracked
        },
        "archive_builder_sha256": sha256(Path(__file__)),
        "analysis_script_sha256": sha256(
            ROOT / "scripts/run_p2_global_reliability_contract_v2_4.py"
        ),
        "source_artifacts_sha256": {
            str(ROOT / "configs/p2_global_reliability_contract_v2.4.json"): sha256(
                ROOT / "configs/p2_global_reliability_contract_v2.4.json"
            ),
            str(OUTPUT / "calibration_contract.json"): sha256(OUTPUT / "calibration_contract.json"),
            str(CUBE): "0f4e9f37ddd65d5abb0ebd92715a8521703a7778375d05c1a6055254203d0401",
        },
    }
    (ARCHIVE / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
