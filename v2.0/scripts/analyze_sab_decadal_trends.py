"""Diagnose whether SAB surface TA and SSS decadal trends are identifiable."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import TheilSenRegressor

from recad.evaluate.p1_framework import FrozenManifest, P1DataGateway, Purpose, sha256

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_ta_sss_sab_decadal_trends_v2.2"
LOWER_LATITUDE = 28.45
UPPER_LATITUDE = 35.3


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sab_mask(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the canonical SAB without mutating the input."""
    selected = frame.loc[
        frame.lme_id.eq(6) & frame.latitude.ge(LOWER_LATITUDE) & frame.latitude.lt(UPPER_LATITUDE)
    ].copy()
    return selected.reset_index(drop=True)


def load_unsealed(gateway: P1DataGateway, target: str, columns: list[str]) -> pd.DataFrame:
    """Load train/development labels while leaving both sealed partitions untouched."""
    frames = [
        gateway.load_labels(target, purpose, columns=columns)
        for purpose in (Purpose.TRAIN, Purpose.SELECTION)
    ]
    return sab_mask(pd.concat(frames, ignore_index=True))


def design_matrix(
    frame: pd.DataFrame,
    *,
    adjust_season_space: bool,
    adjust_sss: bool = False,
) -> pd.DataFrame:
    """Build the registered low-order trend design."""
    year = frame.year.to_numpy(float)
    data: dict[str, np.ndarray] = {"trend_per_decade": (year - year.mean()) / 10.0}
    if adjust_season_space:
        month = frame.month.to_numpy(float)
        latitude = frame.latitude.to_numpy(float) - frame.latitude.mean()
        longitude = frame.longitude.to_numpy(float) - frame.longitude.mean()
        data.update(
            {
                "month_sin_1": np.sin(2 * np.pi * month / 12),
                "month_cos_1": np.cos(2 * np.pi * month / 12),
                "month_sin_2": np.sin(4 * np.pi * month / 12),
                "month_cos_2": np.cos(4 * np.pi * month / 12),
                "latitude": latitude,
                "longitude": longitude,
                "latitude_sq": latitude**2,
                "longitude_sq": longitude**2,
                "latitude_longitude": latitude * longitude,
            }
        )
    if adjust_sss:
        data["salinity"] = frame.salinity.to_numpy(float) - frame.salinity.mean()
    design = pd.DataFrame(data)
    design.insert(0, "const", 1.0)
    return design


def fit_clustered_trend(
    frame: pd.DataFrame,
    *,
    method: str,
    adjust_season_space: bool,
    adjust_sss: bool = False,
) -> dict[str, object]:
    """Fit a cruise-equal WLS trend with cruise-clustered covariance."""
    work = frame.copy()
    if adjust_sss:
        work = work.loc[work.salinity.notna()].copy()
    work = work.reset_index(drop=True)
    if work.group_key.nunique() < 4 or work.year.nunique() < 3:
        return {
            "method": method,
            "n": len(work),
            "cruises": work.group_key.nunique(),
            "years": work.year.nunique(),
            "slope_per_decade": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
        }
    counts = work.groupby("group_key").truth.transform("size")
    design = design_matrix(
        work,
        adjust_season_space=adjust_season_space,
        adjust_sss=adjust_sss,
    )
    x = design.to_numpy(float)
    y = work.truth.to_numpy(float)
    weights = 1.0 / counts.to_numpy(float)
    bread = np.linalg.pinv(x.T @ (weights[:, None] * x))
    coefficients = bread @ (x.T @ (weights * y))
    residuals = y - x @ coefficients
    groups = work.group_key.to_numpy()
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for group in pd.unique(groups):
        selected = groups == group
        score = x[selected].T @ (weights[selected] * residuals[selected])
        meat += np.outer(score, score)
    clusters = work.group_key.nunique()
    observations, parameters = x.shape
    correction = (clusters / (clusters - 1)) * ((observations - 1) / (observations - parameters))
    covariance = correction * bread @ meat @ bread
    trend_index = design.columns.get_loc("trend_per_decade")
    slope = coefficients[trend_index]
    standard_error = np.sqrt(max(covariance[trend_index, trend_index], 0.0))
    z_score = slope / standard_error if standard_error > 0 else np.inf
    return {
        "method": method,
        "n": len(work),
        "cruises": work.group_key.nunique(),
        "years": work.year.nunique(),
        "slope_per_decade": slope,
        "ci_low": slope - 1.96 * standard_error,
        "ci_high": slope + 1.96 * standard_error,
        "p_value": 2 * norm.sf(abs(z_score)),
    }


def cruise_robust_sensitivity(frame: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Return robust cruise-mean and leave-one-cruise-out raw slopes."""
    cruise = (
        frame.groupby("group_key", as_index=False)
        .agg(year=("year", "median"), value=("truth", "mean"))
        .sort_values("year")
    )
    theil = TheilSenRegressor(random_state=0).fit(cruise[["year"]], cruise.value)
    rows = [
        {
            "variable": variable,
            "omitted_cruise": "none_theil_sen",
            "slope_per_decade": float(theil.coef_[0] * 10),
        }
    ]
    for omitted in cruise.group_key:
        subset = cruise.loc[cruise.group_key.ne(omitted)]
        slope = np.polyfit(subset.year, subset.value, 1)[0] * 10
        rows.append(
            {
                "variable": variable,
                "omitted_cruise": omitted,
                "slope_per_decade": slope,
            }
        )
    return pd.DataFrame(rows)


def coverage_table(frame: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Summarize annual observational support."""
    table = frame.groupby("year", as_index=False).agg(
        records=("truth", "size"),
        cruises=("group_key", "nunique"),
        mean=("truth", "mean"),
        median=("truth", "median"),
        latitude_mean=("latitude", "mean"),
        longitude_mean=("longitude", "mean"),
    )
    table.insert(0, "variable", variable)
    return table


def decade_table(frame: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Return record- and cruise-equal summaries by calendar decade."""
    work = frame.assign(decade=(frame.year // 10) * 10)
    records = work.groupby("decade", as_index=False).agg(
        records=("truth", "size"),
        cruises=("group_key", "nunique"),
        years=("year", "nunique"),
        record_mean=("truth", "mean"),
        record_sd=("truth", "std"),
    )
    cruise = work.groupby(["decade", "group_key"], as_index=False).truth.mean()
    cruise = cruise.groupby("decade", as_index=False).agg(
        cruise_equal_mean=("truth", "mean"), cruise_equal_sd=("truth", "std")
    )
    table = records.merge(cruise, on="decade")
    table.insert(0, "variable", variable)
    return table


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def run_analysis(output: Path, archive: Path) -> dict[str, object]:
    """Run the diagnostic and build its reviewer archive."""
    output.mkdir(parents=True, exist_ok=True)
    figures, tables = archive / "figures", archive / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    manifest = FrozenManifest.load(ROOT / "configs/frozen/data_manifest_v2.2.json")
    validation = manifest.validate(hash_mode="full")
    gateway = P1DataGateway(manifest)
    ta = load_unsealed(gateway, "ta", ["salinity"])
    sss = load_unsealed(gateway, "sss", [])

    coverage = pd.concat([coverage_table(ta, "TA"), coverage_table(sss, "SSS")], ignore_index=True)
    decades = pd.concat([decade_table(ta, "TA"), decade_table(sss, "SSS")], ignore_index=True)
    trend_rows = []
    for variable, frame in (("TA", ta), ("SSS", sss)):
        for name, adjusted in (("cruise_equal_raw", False), ("season_space", True)):
            row = fit_clustered_trend(
                frame,
                method=name,
                adjust_season_space=adjusted,
            )
            row["variable"] = variable
            trend_rows.append(row)
    ta_sss = fit_clustered_trend(
        ta,
        method="season_space_plus_sss",
        adjust_season_space=True,
        adjust_sss=True,
    )
    ta_sss["variable"] = "TA"
    trend_rows.append(ta_sss)
    trends = pd.DataFrame(trend_rows)
    robust = pd.concat(
        [cruise_robust_sensitivity(ta, "TA"), cruise_robust_sensitivity(sss, "SSS")],
        ignore_index=True,
    )

    cutoff_rows = []
    for cutoff in (2021, 2022, 2023, 2024):
        row = fit_clustered_trend(
            sss.loc[sss.year.le(cutoff)],
            method="season_space",
            adjust_season_space=True,
        )
        row["end_year"] = cutoff
        cutoff_rows.append(row)
    cutoffs = pd.DataFrame(cutoff_rows)

    band_rows = []
    for lower, upper, label in (
        (28.45, 30.5, "28.45-30.5"),
        (30.5, 33.0, "30.5-33"),
        (33.0, 35.3, "33-35.3"),
    ):
        subset = sss.loc[sss.latitude.ge(lower) & sss.latitude.lt(upper)]
        row = fit_clustered_trend(
            subset,
            method="season_space",
            adjust_season_space=True,
        )
        row["latitude_band"] = label
        band_rows.append(row)
    bands = pd.DataFrame(band_rows)

    cruise_means = pd.concat(
        [
            frame.groupby("group_key", as_index=False)
            .agg(
                year=("year", "median"),
                month=("month", "median"),
                records=("truth", "size"),
                value=("truth", "mean"),
                latitude=("latitude", "mean"),
                longitude=("longitude", "mean"),
            )
            .assign(variable=variable)
            for variable, frame in (("TA", ta), ("SSS", sss))
        ],
        ignore_index=True,
    )
    scope = pd.DataFrame(
        [
            {
                "variable": variable,
                "records": len(frame),
                "cruises": frame.group_key.nunique(),
                "years_with_data": frame.year.nunique(),
                "first_year": frame.year.min(),
                "last_year": frame.year.max(),
                "maximum_single_year_fraction": frame.year.value_counts(normalize=True).max(),
            }
            for variable, frame in (("TA", ta), ("SSS", sss))
        ]
    )

    table_map = {
        "table01_scope.csv": scope,
        "table02_annual_coverage.csv": coverage,
        "table03_decade_summaries.csv": decades,
        "table04_trend_estimates.csv": trends,
        "table05_cruise_robust_sensitivity.csv": robust,
        "table06_sss_end_year_sensitivity.csv": cutoffs,
        "table07_sss_latitude_band_trends.csv": bands,
        "table08_cruise_means.csv": cruise_means,
    }
    for name, frame in table_map.items():
        write_csv(frame, tables / name)

    make_figures(coverage, decades, trends, robust, cutoffs, bands, cruise_means, figures)
    captions = figure_captions()
    report = build_report(scope, trends, robust, cutoffs, captions)
    (archive / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    (archive / "CAPTIONS.md").write_text(
        "# Figure captions\n\n"
        + "".join(f"## {name}\n\n{caption}\n\n" for name, caption in captions.items()),
        encoding="utf-8",
        newline="\n",
    )
    (archive / "README.md").write_text(
        "# SAB observational decadal-trend diagnostic\n\n"
        "Issue #19 diagnostic archive. Locked-test and external-independent labels remain sealed.\n",
        encoding="utf-8",
        newline="\n",
    )
    figure_sources = {
        "fig01_ta_annual_coverage.png": ["table02_annual_coverage.csv"],
        "fig02_ta_cruise_means.png": ["table08_cruise_means.csv"],
        "fig03_ta_method_sensitivity.png": [
            "table04_trend_estimates.csv",
            "table05_cruise_robust_sensitivity.csv",
        ],
        "fig04_sss_annual_coverage.png": ["table02_annual_coverage.csv"],
        "fig05_sss_cruise_means.png": ["table08_cruise_means.csv"],
        "fig06_sss_end_year_sensitivity.png": ["table06_sss_end_year_sensitivity.csv"],
        "fig07_decade_summaries.png": ["table03_decade_summaries.csv"],
        "fig08_sss_latitude_band_trends.png": ["table07_sss_latitude_band_trends.csv"],
    }
    tracked = [
        archive / "README.md",
        archive / "REPORT.md",
        archive / "CAPTIONS.md",
        *sorted(figures.glob("*.png")),
        *sorted(tables.glob("*.csv")),
    ]
    archive_manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_git_commit": "not_applicable_observational_diagnostic",
        "data_manifest_sha256": validation["manifest_sha256"],
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "SAB train/development observational sampling diagnostic",
        "decision": {"TA": "not_identifiable", "SSS": "diagnostic_only"},
        "archive_builder_sha256": file_hash(Path(__file__)),
        "analysis_script_sha256": file_hash(Path(__file__)),
        "source_artifacts_sha256": {
            "carbon_cache": manifest.content["artifacts_sha256"][
                str(manifest.cache_path("carbon"))
            ],
            "socat_cache": manifest.content["artifacts_sha256"][str(manifest.cache_path("socat"))],
        },
        "figure_source_data": figure_sources,
        "figure_captions": captions,
        "files_sha256": {
            str(path.relative_to(archive)).replace("\\", "/"): file_hash(path) for path in tracked
        },
    }
    (archive / "archive_manifest.json").write_text(
        json.dumps(archive_manifest, indent=2), encoding="utf-8", newline="\n"
    )
    protocol = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": archive_manifest["created_utc"],
        "data_manifest_sha256": sha256(manifest.path),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "scope": scope.to_dict(orient="records"),
        "decision": archive_manifest["decision"],
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return archive_manifest


def make_figures(
    coverage: pd.DataFrame,
    decades: pd.DataFrame,
    trends: pd.DataFrame,
    robust: pd.DataFrame,
    cutoffs: pd.DataFrame,
    bands: pd.DataFrame,
    cruise_means: pd.DataFrame,
    figures: Path,
    region_name: str = "SAB",
) -> None:
    for variable, prefix, unit in (("TA", "ta", "µmol kg⁻¹"), ("SSS", "sss", "PSU")):
        annual = coverage.loc[coverage.variable.eq(variable)]
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].bar(annual.year, annual.records, color="#0072B2")
        axes[0].set_ylabel("Records")
        axes[1].bar(annual.year, annual.cruises, color="#009E73")
        axes[1].set_ylabel("Cruises")
        axes[1].set_xlabel("Year")
        fig.tight_layout()
        save_figure(
            fig, figures / f"fig{'01' if variable == 'TA' else '04'}_{prefix}_annual_coverage.png"
        )

        cruise = cruise_means.loc[cruise_means.variable.eq(variable)]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        size = 25 + 12 * np.sqrt(cruise.records)
        ax.scatter(cruise.year, cruise.value, s=size, alpha=0.72, color="#D55E00")
        ax.set_xlabel("Year")
        ax.set_ylabel(f"Cruise-mean {variable} ({unit})")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        save_figure(
            fig, figures / f"fig{'02' if variable == 'TA' else '05'}_{prefix}_cruise_means.png"
        )

    ta_trends = trends.loc[trends.variable.eq("TA")].copy()
    ta_robust = robust.loc[robust.variable.eq("TA")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].errorbar(
        ta_trends.method,
        ta_trends.slope_per_decade,
        yerr=[
            ta_trends.slope_per_decade - ta_trends.ci_low,
            ta_trends.ci_high - ta_trends.slope_per_decade,
        ],
        fmt="o",
        capsize=4,
    )
    axes[0].axhline(0, color="black", lw=1)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("TA trend (µmol kg⁻¹ decade⁻¹)")
    loco = ta_robust.loc[ta_robust.omitted_cruise.ne("none_theil_sen")]
    axes[1].hist(loco.slope_per_decade, bins=8, color="#0072B2", alpha=0.8)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_xlabel("Leave-one-cruise-out raw slope")
    fig.tight_layout()
    save_figure(fig, figures / "fig03_ta_method_sensitivity.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        cutoffs.end_year,
        cutoffs.slope_per_decade,
        yerr=[
            cutoffs.slope_per_decade - cutoffs.ci_low,
            cutoffs.ci_high - cutoffs.slope_per_decade,
        ],
        marker="o",
        capsize=4,
    )
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Last included year")
    ax.set_ylabel("Adjusted SSS trend (PSU decade⁻¹)")
    fig.tight_layout()
    save_figure(fig, figures / "fig06_sss_end_year_sensitivity.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, variable, unit in zip(axes, ("TA", "SSS"), ("µmol kg⁻¹", "PSU"), strict=True):
        part = decades.loc[decades.variable.eq(variable)]
        ax.plot(part.decade, part.record_mean, marker="o", label="record-weighted")
        ax.plot(part.decade, part.cruise_equal_mean, marker="o", label="cruise-equal")
        ax.set_title(variable)
        ax.set_xlabel("Decade")
        ax.set_ylabel(unit)
        ax.legend()
    fig.tight_layout()
    save_figure(fig, figures / "fig07_decade_summaries.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        bands.latitude_band,
        bands.slope_per_decade,
        yerr=[bands.slope_per_decade - bands.ci_low, bands.ci_high - bands.slope_per_decade],
        fmt="o",
        capsize=4,
    )
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel(f"{region_name} latitude band (°N)")
    ax.set_ylabel("Adjusted SSS trend (PSU decade⁻¹)")
    fig.tight_layout()
    save_figure(fig, figures / "fig08_sss_latitude_band_trends.png")


def figure_captions() -> dict[str, str]:
    return {
        "fig01_ta_annual_coverage.png": "Figure 1. Annual SAB surface-TA record and cruise coverage in the unsealed train/development partitions. The series has only eight sampled years, and 2014 contributes 366 of 441 records.",
        "fig02_ta_cruise_means.png": "Figure 2. Cruise-mean TA against time; point area scales with within-cruise record count. Fourteen cruises do not provide repeated, spatially balanced coverage for a regional climate trend.",
        "fig03_ta_method_sensitivity.png": "Figure 3. TA trend estimates across adjustment choices and the leave-one-cruise-out raw-slope distribution. Large shifts among methods and sign changes under cruise omission diagnose non-identifiability.",
        "fig04_sss_annual_coverage.png": "Figure 4. Annual SOCAT in-situ SSS coverage in canonical SAB. Coverage is much denser than TA but remains strongly uneven among years and cruises.",
        "fig05_sss_cruise_means.png": "Figure 5. Cruise-mean SOCAT SSS against time; point area scales with record count. The scatter shows that sampling different coastal and offshore regimes is large relative to the fitted decadal change.",
        "fig06_sss_end_year_sensitivity.png": "Figure 6. Seasonally and spatially adjusted SSS trend as the final included year moves from 2021 to 2024. The negative estimate strengthens when sparse, fresher 2024 sampling is included.",
        "fig07_decade_summaries.png": "Figure 7. Record-weighted and cruise-equal means by calendar decade. These points summarize the sampled observations and must not be interpreted as spatially complete SAB decadal climatologies.",
        "fig08_sss_latitude_band_trends.png": "Figure 8. Adjusted SSS trends fitted separately in three SAB latitude bands. Heterogeneity and uncertainty across bands limit a single basin-wide trend claim.",
    }


def build_report(
    scope: pd.DataFrame,
    trends: pd.DataFrame,
    robust: pd.DataFrame,
    cutoffs: pd.DataFrame,
    captions: dict[str, str],
) -> str:
    ta = trends.set_index(["variable", "method"])
    sss_cut = cutoffs.set_index("end_year")
    ta_loco = robust.loc[
        robust.variable.eq("TA") & robust.omitted_cruise.ne("none_theil_sen"),
        "slope_per_decade",
    ]
    figure_blocks = "\n\n".join(
        f"![{name}](figures/{name})\n\n{caption}" for name, caption in captions.items()
    )
    ta_scope = scope.set_index("variable").loc["TA"]
    sss_scope = scope.set_index("variable").loc["SSS"]
    return f"""# SAB surface TA and SSS decadal-trend diagnostic

## Scientific question and permitted claim

This analysis asks whether the existing observations identify a decadal trend in canonical SAB (LME 6, 28.45-35.3°N). It may diagnose sampling adequacy and estimator sensitivity. It cannot establish a spatially complete climate trend, validate a product, or attribute a trend to a process.

## Data, splits, and leakage controls

TA uses primary QC=2 CODAP/GLODAP observations at 0-5 m. SSS uses SOCAT in-situ surface salinity. Only frozen train and development partitions were materialized through the P1 gateway; locked-test and external-independent labels remained sealed. TA has {int(ta_scope.records)} records from {int(ta_scope.cruises)} cruises in {int(ta_scope.years_with_data)} sampled years ({int(ta_scope.first_year)}-{int(ta_scope.last_year)}), with {100 * ta_scope.maximum_single_year_fraction:.1f}% of records in one year. SSS has {int(sss_scope.records)} records from {int(sss_scope.cruises)} cruises in {int(sss_scope.years_with_data)} years ({int(sss_scope.first_year)}-{int(sss_scope.last_year)}).

## Candidate models and training

No predictive model was trained. Trends use cruise-equal weighted least squares with cruise-clustered covariance. Registered sensitivities include a raw temporal fit, harmonic seasonal plus quadratic spatial adjustment, TA adjustment for collocated SSS, Theil-Sen regression on cruise means, leave-one-cruise-out raw slopes, SSS end-year truncation, and separate latitude bands.

## Main development results

TA is not identifiable as a regional decadal trend. The raw cruise-equal estimate is {ta.loc[("TA", "cruise_equal_raw")].slope_per_decade:.1f} µmol kg⁻¹ decade⁻¹ (95% CI {ta.loc[("TA", "cruise_equal_raw")].ci_low:.1f} to {ta.loc[("TA", "cruise_equal_raw")].ci_high:.1f}). Seasonal/spatial adjustment changes it to {ta.loc[("TA", "season_space")].slope_per_decade:.1f}; adding SSS changes it to {ta.loc[("TA", "season_space_plus_sss")].slope_per_decade:.1f}. Leave-one-cruise-out raw slopes span {ta_loco.min():.1f} to {ta_loco.max():.1f}, including both signs.

SSS supports a diagnostic negative estimate, not a robust climate claim. The seasonally/spatially adjusted slope is {ta.loc[("SSS", "season_space")].slope_per_decade:.3f} PSU decade⁻¹ (95% CI {ta.loc[("SSS", "season_space")].ci_low:.3f} to {ta.loc[("SSS", "season_space")].ci_high:.3f}). Its end-year sensitivity moves from {sss_cut.loc[2021].slope_per_decade:.3f} through {sss_cut.loc[2023].slope_per_decade:.3f} to {sss_cut.loc[2024].slope_per_decade:.3f}, demonstrating sensitivity to the final sparse sampling year.

## Decision and limitations

TA receives `not_identifiable`: too few cruises and years, severe 2014 concentration, and estimator/leave-one-cruise instability. SSS receives `diagnostic_only`: the adjusted estimate is weakly negative, but its confidence interval includes zero and its magnitude changes with end year and latitude band. A publishable trend requires a fixed gridded domain with monthly anomaly construction, explicit observation-error treatment, and preferably reconstruction/product ensembles evaluated against withheld stations or cruises. The present results describe only sampled, unsealed observations.

## Figure and table index

Every figure is backed by a CSV under `tables/`; hashes and mappings are in `archive_manifest.json`.

{figure_blocks}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/experiments" / EXPERIMENT_ID,
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "docs/experiment_archive" / EXPERIMENT_ID,
    )
    args = parser.parse_args()
    result = run_analysis(args.output, args.archive)
    print(json.dumps(result["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
