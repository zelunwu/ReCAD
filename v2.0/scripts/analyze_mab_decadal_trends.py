"""Run the frozen SAB trend diagnostic methodology in canonical MAB."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import analyze_sab_decadal_trends as core
from recad.evaluate.p1_framework import FrozenManifest, P1DataGateway, Purpose, sha256

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_ta_sss_mab_decadal_trends_v2.2"
LOWER_LATITUDE = 35.2
UPPER_LATITUDE = 41.75


def mab_mask(frame: pd.DataFrame) -> pd.DataFrame:
    """Select canonical MAB using its frozen inclusive northern endpoint."""
    selected = frame.loc[
        frame.lme_id.eq(7) & frame.latitude.ge(LOWER_LATITUDE) & frame.latitude.le(UPPER_LATITUDE)
    ].copy()
    return selected.reset_index(drop=True)


def load_unsealed(gateway: P1DataGateway, target: str, columns: list[str]) -> pd.DataFrame:
    frames = [
        gateway.load_labels(target, purpose, columns=columns)
        for purpose in (Purpose.TRAIN, Purpose.SELECTION)
    ]
    return mab_mask(pd.concat(frames, ignore_index=True))


def run_analysis(output: Path, archive: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    figures, tables = archive / "figures", archive / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    manifest = FrozenManifest.load(ROOT / "configs/frozen/data_manifest_v2.2.json")
    validation = manifest.validate(hash_mode="full")
    gateway = P1DataGateway(manifest)
    ta = load_unsealed(gateway, "ta", ["salinity", "source"])
    sss = load_unsealed(gateway, "sss", [])

    coverage = pd.concat(
        [core.coverage_table(ta, "TA"), core.coverage_table(sss, "SSS")],
        ignore_index=True,
    )
    decades = pd.concat(
        [core.decade_table(ta, "TA"), core.decade_table(sss, "SSS")],
        ignore_index=True,
    )
    trend_rows = []
    for variable, frame in (("TA", ta), ("SSS", sss)):
        for name, adjusted in (("cruise_equal_raw", False), ("season_space", True)):
            row = core.fit_clustered_trend(
                frame,
                method=name,
                adjust_season_space=adjusted,
            )
            row["variable"] = variable
            trend_rows.append(row)
    ta_sss = core.fit_clustered_trend(
        ta,
        method="season_space_plus_sss",
        adjust_season_space=True,
        adjust_sss=True,
    )
    ta_sss["variable"] = "TA"
    trend_rows.append(ta_sss)
    trends = pd.DataFrame(trend_rows)
    robust = pd.concat(
        [
            core.cruise_robust_sensitivity(ta, "TA"),
            core.cruise_robust_sensitivity(sss, "SSS"),
        ],
        ignore_index=True,
    )

    cutoff_rows = []
    for cutoff in (2022, 2023, 2024, 2025):
        row = core.fit_clustered_trend(
            sss.loc[sss.year.le(cutoff)],
            method="season_space",
            adjust_season_space=True,
        )
        row["end_year"] = cutoff
        cutoff_rows.append(row)
    cutoffs = pd.DataFrame(cutoff_rows)

    band_rows = []
    for lower, upper in ((35.2, 38.0), (38.0, 40.0), (40.0, 41.75)):
        upper_mask = sss.latitude.le(upper) if upper == UPPER_LATITUDE else sss.latitude.lt(upper)
        subset = sss.loc[sss.latitude.ge(lower) & upper_mask]
        row = core.fit_clustered_trend(
            subset,
            method="season_space",
            adjust_season_space=True,
        )
        row["latitude_band"] = f"{lower:g}-{upper:g}"
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
    source_composition = ta.groupby("source", as_index=False).agg(
        records=("truth", "size"),
        cruises=("group_key", "nunique"),
        years=("year", "nunique"),
        first_year=("year", "min"),
        last_year=("year", "max"),
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
        "table09_ta_source_composition.csv": source_composition,
    }
    for name, frame in table_map.items():
        core.write_csv(frame, tables / name)

    core.make_figures(
        coverage,
        decades,
        trends,
        robust,
        cutoffs,
        bands,
        cruise_means,
        figures,
        region_name="MAB",
    )
    captions = figure_captions()
    report = build_report(scope, trends, robust, cutoffs, bands, source_composition, captions)
    (archive / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    caption_text = "# Figure captions\n\n" + "".join(
        f"## {name}\n\n{caption}\n\n" for name, caption in captions.items()
    )
    (archive / "CAPTIONS.md").write_text(
        caption_text.rstrip() + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (archive / "README.md").write_text(
        "# MAB observational decadal-trend diagnostic\n\n"
        "Issue #21 archive. Locked-test and external-independent labels remain sealed.\n",
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
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_git_commit": "not_applicable_observational_diagnostic",
        "data_manifest_sha256": validation["manifest_sha256"],
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "MAB train/development observational sampling diagnostic",
        "decision": {"TA": "not_identifiable", "SSS": "diagnostic_only"},
        "archive_builder_sha256": core.file_hash(Path(__file__)),
        "analysis_script_sha256": core.file_hash(Path(__file__)),
        "source_artifacts_sha256": {
            "carbon_cache": manifest.content["artifacts_sha256"][
                str(manifest.cache_path("carbon"))
            ],
            "socat_cache": manifest.content["artifacts_sha256"][str(manifest.cache_path("socat"))],
        },
        "figure_source_data": figure_sources,
        "figure_captions": captions,
        "files_sha256": {
            str(path.relative_to(archive)).replace("\\", "/"): core.file_hash(path)
            for path in tracked
        },
    }
    (archive / "archive_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8", newline="\n"
    )
    protocol = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": result["created_utc"],
        "data_manifest_sha256": sha256(manifest.path),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "scope": scope.to_dict(orient="records"),
        "decision": result["decision"],
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return result


def figure_captions() -> dict[str, str]:
    return {
        "fig01_ta_annual_coverage.png": "Figure 1. Annual MAB surface-TA records and cruises in unsealed train/development. Coverage is broader than SAB but remains temporally uneven.",
        "fig02_ta_cruise_means.png": "Figure 2. Cruise-mean MAB TA through time, with point area proportional to record count. Spatial and seasonal cruise composition is large relative to temporal change.",
        "fig03_ta_method_sensitivity.png": "Figure 3. MAB TA trends across adjustment choices and leave-one-cruise raw slopes. Removing the raw decline after spatial/seasonal adjustment diagnoses sampling confounding.",
        "fig04_sss_annual_coverage.png": "Figure 4. Annual MAB SOCAT in-situ SSS cruise-grid-month coverage. Dense coverage still varies materially by year and cruise.",
        "fig05_sss_cruise_means.png": "Figure 5. Cruise-mean MAB SOCAT SSS through time. Cross-cruise spatial-regime scatter is much larger than the fitted basin-wide trend.",
        "fig06_sss_end_year_sensitivity.png": "Figure 6. Adjusted MAB SSS trend under end-year truncation. The estimate changes from weakly positive through 2023 to weakly negative through 2024-2025.",
        "fig07_decade_summaries.png": "Figure 7. Record-weighted and cruise-equal MAB observations by decade. They summarize sampling and are not spatially complete climatologies.",
        "fig08_sss_latitude_band_trends.png": "Figure 8. Adjusted MAB SSS trends by fixed latitude band. Positive central/northern estimates contrast with the weakly negative full-region estimate, exposing spatial-composition confounding.",
    }


def build_report(
    scope: pd.DataFrame,
    trends: pd.DataFrame,
    robust: pd.DataFrame,
    cutoffs: pd.DataFrame,
    bands: pd.DataFrame,
    sources: pd.DataFrame,
    captions: dict[str, str],
) -> str:
    metric = trends.set_index(["variable", "method"])
    ta_loco = robust.loc[
        robust.variable.eq("TA") & robust.omitted_cruise.ne("none_theil_sen"),
        "slope_per_decade",
    ]
    scope_by_variable = scope.set_index("variable")
    source_text = ", ".join(
        f"{row.source}: {int(row.records)} records/{int(row.cruises)} cruises"
        for row in sources.itertuples()
    )
    cutoff_text = ", ".join(
        f"{int(row.end_year)}: {row.slope_per_decade:+.3f}" for row in cutoffs.itertuples()
    )
    band_text = ", ".join(
        f"{row.latitude_band}°N: {row.slope_per_decade:+.3f}" for row in bands.itertuples()
    )
    figures = "\n\n".join(
        f"![{name}](figures/{name})\n\n{caption}" for name, caption in captions.items()
    )
    ta_scope, sss_scope = scope_by_variable.loc["TA"], scope_by_variable.loc["SSS"]
    return f"""# MAB surface TA and SSS decadal-trend diagnostic

## Scientific question and permitted claim

This experiment repeats Issue #19 without changing its method or decision rule in canonical MAB (LME 7, 35.2-41.75°N). It diagnoses observational trend identifiability; it does not establish a spatially complete climate trend or validate a product.

## Data, splits, and leakage controls

TA uses primary QC=2 surface observations: {source_text}. It has {int(ta_scope.records)} records, {int(ta_scope.cruises)} cruises and {int(ta_scope.years_with_data)} sampled years ({int(ta_scope.first_year)}-{int(ta_scope.last_year)}). SSS uses {int(sss_scope.records)} SOCAT cruise-grid-month rows from {int(sss_scope.cruises)} cruises and {int(sss_scope.years_with_data)} years ({int(sss_scope.first_year)}-{int(sss_scope.last_year)}). Only train/development were read through the gateway; locked-test and external-independent labels remained sealed.

## Candidate models and training

No prediction model was trained. The frozen SAB methods were reused unchanged: cruise-equal WLS with cruise-clustered covariance, raw and harmonic-season/quadratic-space adjustment, TA adjustment for SSS, Theil-Sen cruise means, leave-one-cruise-out raw slopes, SSS end-year truncation, and three fixed latitude bands.

## Main development results

TA is not identifiable. Its raw estimate is {metric.loc[("TA", "cruise_equal_raw")].slope_per_decade:+.1f} µmol kg⁻¹ decade⁻¹ (95% CI {metric.loc[("TA", "cruise_equal_raw")].ci_low:+.1f} to {metric.loc[("TA", "cruise_equal_raw")].ci_high:+.1f}), but seasonal/spatial adjustment changes it to {metric.loc[("TA", "season_space")].slope_per_decade:+.1f} and adding SSS changes it to {metric.loc[("TA", "season_space_plus_sss")].slope_per_decade:+.1f}. Leave-one-cruise raw slopes span {ta_loco.min():+.1f} to {ta_loco.max():+.1f}; the raw decline is sampling-confounded.

SSS is diagnostic only. Raw and adjusted full-MAB estimates are {metric.loc[("SSS", "cruise_equal_raw")].slope_per_decade:+.3f} and {metric.loc[("SSS", "season_space")].slope_per_decade:+.3f} PSU decade⁻¹; the adjusted 95% CI is {metric.loc[("SSS", "season_space")].ci_low:+.3f} to {metric.loc[("SSS", "season_space")].ci_high:+.3f}. End-year estimates are {cutoff_text}. Fixed-band adjusted slopes are {band_text}. The opposing full-region and band behavior is consistent with changing spatial sampling, not a uniform MAB trend.

## Decision and limitations

TA receives `not_identifiable`; SSS receives `diagnostic_only`. Neither scattered-observation series supports a basin-wide trend claim. A publishable estimate requires fixed-grid monthly anomalies, area weighting, temporal-correlation treatment, and a validated reconstruction ensemble.

## Figure and table index

Every figure is backed by a CSV under `tables/`; hashes and mappings are in `archive_manifest.json`.

{figures}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/experiments" / EXPERIMENT_ID)
    parser.add_argument(
        "--archive", type=Path, default=ROOT / "docs/experiment_archive" / EXPERIMENT_ID
    )
    args = parser.parse_args()
    result = run_analysis(args.output, args.archive)
    print(json.dumps(result["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
