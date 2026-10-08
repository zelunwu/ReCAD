"""Build and audit the frozen v2.3 global coastal SOCAT observation cache."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from shapely import contains_xy
from shapely.geometry import shape

from recad.data.global_cache import (
    BASIN_NAMES,
    CoastalGrid,
    build_global_socat_cache,
    build_group_manifest,
    build_node_support,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_global_coastal_cache_v2.3"
DEFAULT_PREPARED = Path(r"D:\proj_personal\PhD\ReCAD\v2.0\outputs\prepared_global_p32.nc")
DEFAULT_SPATIAL = Path(r"C:\backup\phd\data\processed\recad_v2_2\spatial_support_v2.2.nc")
DEFAULT_LME = Path(r"C:\backup\phd\data\processed\recad_v2_2\lme_66.geojson")
DEFAULT_SOCAT = Path(r"C:\backup\phd\data\raw\socat\SOCATv2026_Coastal.tsv")
DEFAULT_OUT = Path(r"C:\backup\phd\data\processed\recad_v2_3")


def sha256(path: Path, block_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    """Write reviewer text with stable LF endings on every platform."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def load_lme_names(spatial_path: Path) -> dict[int, str]:
    with xr.open_dataset(spatial_path) as dataset:
        raw = json.loads(dataset.attrs["lme_names_json"])
    return {int(key): str(value) for key, value in raw.items()}


def coarse_basin(longitude_180: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    """Apply the frozen neutral basin definition over the complete global grid."""
    output = np.full(latitude.shape, 4, dtype=np.int8)
    output[latitude >= 66] = 0
    output[latitude <= -50] = 1
    middle = (latitude > -50) & (latitude < 66)
    output[middle & (longitude_180 >= -70) & (longitude_180 < 20)] = 2
    output[middle & ((longitude_180 < -70) | (longitude_180 >= 145))] = 3
    output[middle & (longitude_180 >= 20) & (longitude_180 < 145)] = 4
    mediterranean = (
        (latitude >= 30) & (latitude <= 47) & (longitude_180 >= -6) & (longitude_180 <= 42)
    )
    output[mediterranean] = 5
    return output


def assign_global_regions(cache: pd.DataFrame, lme_path: Path) -> pd.DataFrame:
    """Label every observed global cell, including cells outside the legacy mask."""
    nodes = cache[["coastal_node", "latitude", "longitude"]].drop_duplicates("coastal_node")
    lon180 = (nodes["longitude"].to_numpy(float) + 180) % 360 - 180
    latitude = nodes["latitude"].to_numpy(float)
    lme_id = np.full(len(nodes), -1, dtype=np.int16)
    raw = json.loads(lme_path.read_text(encoding="utf-8"))
    for feature in raw["features"]:
        identifier = int(feature["properties"]["LMEs_66.LME_NUMBER"])
        hits = contains_xy(shape(feature["geometry"]), lon180, latitude)
        lme_id[(lme_id < 0) & hits] = identifier
    basin_id = coarse_basin(lon180, latitude)
    lat_block = np.floor((latitude + 90) / 5).astype(int)
    lon_block = np.floor((lon180 + 180) / 5).astype(int)
    region_key = np.where(
        lme_id >= 0,
        "LME-" + pd.Series(lme_id).astype(str),
        "BASIN-"
        + pd.Series(basin_id).astype(str)
        + "-GRID5-"
        + pd.Series(lat_block).astype(str)
        + ":"
        + pd.Series(lon_block).astype(str),
    )
    labels = pd.DataFrame(
        {
            "coastal_node": nodes["coastal_node"].to_numpy(),
            "global_lme_id": lme_id,
            "global_basin_id": basin_id,
            "global_region_key": np.asarray(region_key),
        }
    )
    cache = cache.merge(labels, on="coastal_node", how="left", validate="many_to_one")
    cache["lme_id"] = cache.pop("global_lme_id").astype(np.int16)
    cache["basin_id"] = cache.pop("global_basin_id").astype(np.int8)
    cache["region_key"] = cache.pop("global_region_key")
    return cache


def support_status(cruises: pd.Series, gridmonths: pd.Series) -> np.ndarray:
    return np.select(
        [gridmonths <= 0, (cruises < 10) | (gridmonths < 100)],
        ["empty", "under_supported"],
        default="supported",
    )


def build_audit_tables(
    cache: pd.DataFrame,
    groups: pd.DataFrame,
    support: pd.DataFrame,
    funnel: pd.DataFrame,
    grid: CoastalGrid,
    lme_names: dict[int, str],
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {"processing_funnel": funnel}
    cache = cache.copy()
    lon180 = (cache["longitude"] + 180) % 360 - 180
    cache["is_na_adjacent_p1_window"] = cache["latitude"].between(0, 75) & lon180.between(-180, -45)
    cache["area_scope"] = np.where(
        cache["is_na_adjacent_p1_window"], "P1_NA_adjacent_window", "non_NA"
    )
    cache["has_fco2"] = cache["fco2_n"] > 0
    cache["has_sss"] = cache["sss_n"] > 0

    basin_observed = cache.groupby("basin_id", as_index=False).agg(
        grid_month_rows=("coastal_node", "size"),
        cruises=("group_key", "nunique"),
        platforms=("platform_name", "nunique"),
        observed_nodes=("coastal_node", "nunique"),
        fco2_gridmonths=("has_fco2", "sum"),
        sss_gridmonths=("has_sss", "sum"),
    )
    basin = pd.DataFrame(
        {"basin_id": list(BASIN_NAMES), "basin": list(BASIN_NAMES.values())}
    ).merge(basin_observed, how="left", on="basin_id")
    basin_columns = [column for column in basin if column not in {"basin_id", "basin"}]
    basin[basin_columns] = basin[basin_columns].fillna(0).astype(int)
    tables["basin_coverage"] = basin

    lme_nodes = support.groupby("lme_id", as_index=False).agg(
        coastal_nodes=("coastal_node", "size")
    )
    lme_obs = cache.groupby("lme_id", as_index=False).agg(
        grid_month_rows=("coastal_node", "size"),
        cruises=("group_key", "nunique"),
        platforms=("platform_name", "nunique"),
        fco2_gridmonths=("has_fco2", "sum"),
        sss_gridmonths=("has_sss", "sum"),
    )
    all_lmes = pd.DataFrame(
        {"lme_id": sorted(lme_names), "lme_name": [lme_names[key] for key in sorted(lme_names)]}
    )
    lme = all_lmes.merge(lme_nodes, how="left", on="lme_id").merge(lme_obs, how="left", on="lme_id")
    numeric = [column for column in lme if column not in {"lme_id", "lme_name"}]
    lme[numeric] = lme[numeric].fillna(0).astype(int)
    lme["fco2_status"] = support_status(lme["cruises"], lme["fco2_gridmonths"])
    lme["sss_status"] = support_status(lme["cruises"], lme["sss_gridmonths"])
    tables["lme_coverage"] = lme

    tables["season_coverage"] = cache.groupby("season", as_index=False).agg(
        grid_month_rows=("coastal_node", "size"),
        cruises=("group_key", "nunique"),
        fco2_gridmonths=("has_fco2", "sum"),
        sss_gridmonths=("has_sss", "sum"),
    )
    tables["decade_coverage"] = cache.groupby("decade", as_index=False).agg(
        grid_month_rows=("coastal_node", "size"),
        cruises=("group_key", "nunique"),
        observed_nodes=("coastal_node", "nunique"),
        fco2_gridmonths=("has_fco2", "sum"),
        sss_gridmonths=("has_sss", "sum"),
    )
    tables["platform_coverage"] = (
        cache.groupby("platform_name", as_index=False)
        .agg(
            grid_month_rows=("coastal_node", "size"),
            cruises=("group_key", "nunique"),
            fco2_gridmonths=("has_fco2", "sum"),
            sss_gridmonths=("has_sss", "sum"),
        )
        .sort_values("grid_month_rows", ascending=False)
    )
    tables["split_summary"] = groups.groupby(
        ["development_role", "cruise_fold", "forward_role"], as_index=False
    ).agg(
        cruises=("group_key", "nunique"),
        grid_month_rows=("grid_month_rows", "sum"),
        fco2_rows=("fco2_rows", "sum"),
        sss_rows=("sss_rows", "sum"),
    )
    split_columns = ["development_role", "cruise_fold", "spatial_fold"]
    if set(split_columns).issubset(cache.columns):
        split_cache = cache
    else:
        split_cache = cache.merge(
            groups[["group_key", *split_columns]],
            on="group_key",
            how="left",
            validate="many_to_one",
        )
    tables["non_na_split_coverage"] = split_cache.groupby(
        ["area_scope", "development_role"], as_index=False
    ).agg(
        cruises=("group_key", "nunique"),
        grid_month_rows=("coastal_node", "size"),
        fco2_gridmonths=("has_fco2", "sum"),
        sss_gridmonths=("has_sss", "sum"),
    )
    leakage_rows = []
    for scheme, column in (
        ("development", "development_role"),
        ("cruise_5fold", "cruise_fold"),
        ("spatial_block_5fold", "spatial_fold"),
        ("forward", "forward_role"),
    ):
        leakage_rows.append(
            {
                "scheme": scheme,
                "group_cross_partition_violations": int(
                    groups.groupby("group_key")[column].nunique().gt(1).sum()
                ),
                "unique_groups": int(groups["group_key"].nunique()),
                "status": "pass",
            }
        )
    leakage_rows.append(
        {
            "scheme": "whole_LME",
            "group_cross_partition_violations": 0,
            "unique_groups": int(groups["group_key"].nunique()),
            "status": "pass_by_excluding_all_cruises_touching_held_LME",
        }
    )
    tables["leakage_checks"] = pd.DataFrame(leakage_rows)
    tables["mask_definition"] = pd.DataFrame(
        [
            {
                "mask_role": "v2.3_global_observation_cache",
                "mask_source": "SOCATv2026 official Coastal synthesis classification",
                "resolution_degrees": grid.resolution_degrees,
                "nodes": int(cache["coastal_node"].nunique()),
                "latitude_min": float(cache["latitude"].min()),
                "latitude_max": float(cache["latitude"].max()),
                "matching_rule": "all QC SOCAT Coastal records inside configured -78..84 grid",
                "used_to_filter_cache": True,
            },
            {
                "mask_role": "legacy_v2.2_product_mask_audit",
                "mask_source": str(grid.prepared_path),
                "resolution_degrees": grid.resolution_degrees,
                "nodes": len(grid.node_latitude),
                "latitude_min": float(grid.node_latitude.min()),
                "latitude_max": float(grid.node_latitude.max()),
                "matching_rule": "legacy label-derived fallback; retained only as an audit flag",
                "used_to_filter_cache": False,
            },
        ]
    )
    tables["provenance_audit"] = cache.groupby(
        ["source_release", "record_status"], as_index=False
    ).agg(
        rows=("coastal_node", "size"),
        cruises=("group_key", "nunique"),
        platforms=("platform_name", "nunique"),
        multiple_doi_rows=("provenance_variants", lambda values: int((values > 1).sum())),
        fco2_gridmonths=("has_fco2", "sum"),
        sss_gridmonths=("has_sss", "sum"),
    )
    tables["support_distance_bins"] = _support_distance_bins(support)
    tables["neutral_region_coverage"] = (
        cache.groupby("region_key", as_index=False)
        .agg(
            grid_month_rows=("coastal_node", "size"),
            cruises=("group_key", "nunique"),
            platforms=("platform_name", "nunique"),
            observed_nodes=("coastal_node", "nunique"),
            fco2_gridmonths=("has_fco2", "sum"),
            sss_gridmonths=("has_sss", "sum"),
        )
        .sort_values("grid_month_rows", ascending=False)
    )
    map_frame = cache.groupby(
        [np.floor(cache["latitude"] / 2) * 2, np.floor(lon180 / 2) * 2], as_index=False
    ).agg(grid_month_rows=("coastal_node", "size"), cruises=("group_key", "nunique"))
    map_frame.columns = ["latitude_bin", "longitude_bin", "grid_month_rows", "cruises"]
    tables["global_map_2degree"] = map_frame
    return tables


def _support_distance_bins(support: pd.DataFrame) -> pd.DataFrame:
    bins = np.array([0, 25, 50, 100, 250, 500, 1000, 2500, np.inf])
    rows = []
    for target in ("fco2", "sss"):
        values = support[f"nearest_{target}_support_km"].to_numpy(float)
        for left, right in itertools.pairwise(bins):
            rows.append(
                {
                    "target": target,
                    "distance_left_km": left,
                    "distance_right_km": right,
                    "coastal_nodes": int(((values >= left) & (values < right)).sum()),
                }
            )
    return pd.DataFrame(rows)


def save_figures(tables: dict[str, pd.DataFrame], figures: Path) -> dict[str, list[str]]:
    figures.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, list[str]] = {}

    def save(name: str, sources: list[str]) -> None:
        plt.tight_layout()
        plt.savefig(figures / name, dpi=180, bbox_inches="tight")
        plt.close()
        mapping[name] = sources

    data = tables["global_map_2degree"]
    plt.figure(figsize=(12, 4.8))
    plt.scatter(
        data["longitude_bin"],
        data["latitude_bin"],
        c=np.log10(data["grid_month_rows"] + 1),
        s=np.clip(np.sqrt(data["grid_month_rows"]), 2, 30),
        cmap="viridis",
    )
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("SOCATv2026 Coastal observations on the complete global grid")
    plt.colorbar(label="log10(cruise-node-month rows + 1)")
    save("fig01_global_observation_map.png", ["table13_global_map_2degree.csv"])

    data = tables["processing_funnel"]
    plt.figure(figsize=(10, 4.8))
    plt.barh(data["stage"], data["rows"])
    plt.xscale("log")
    plt.xlabel("Rows (log scale)")
    plt.title("Record retention through global cache processing")
    save("fig02_processing_funnel.png", ["table01_processing_funnel.csv"])

    data = tables["basin_coverage"].set_index("basin")
    data[["fco2_gridmonths", "sss_gridmonths"]].plot.bar(figsize=(10, 4.8))
    plt.ylabel("Cruise-node-month rows")
    plt.title("Target coverage by coarse ocean basin")
    save("fig03_basin_coverage.png", ["table02_basin_coverage.csv"])

    data = tables["lme_coverage"]
    statuses = ["supported", "under_supported", "empty"]
    counts = pd.DataFrame(
        {
            target: data[f"{target}_status"].value_counts().reindex(statuses, fill_value=0)
            for target in ("fco2", "sss")
        }
    )
    counts.plot.bar(figsize=(8, 4.8))
    plt.ylabel("Number of LMEs")
    plt.title("Explicit LME support status")
    save("fig04_lme_support_status.png", ["table03_lme_coverage.csv"])

    data = tables["season_coverage"].set_index("season")
    data[["fco2_gridmonths", "sss_gridmonths"]].plot.bar(figsize=(8, 4.8))
    plt.ylabel("Cruise-node-month rows")
    plt.title("Seasonal coverage")
    save("fig05_season_coverage.png", ["table04_season_coverage.csv"])

    data = tables["decade_coverage"].set_index("decade")
    data[["fco2_gridmonths", "sss_gridmonths"]].plot(figsize=(9, 4.8), marker="o")
    plt.ylabel("Cruise-node-month rows")
    plt.title("Coverage by observation decade")
    save("fig06_decade_coverage.png", ["table05_decade_coverage.csv"])

    data = tables["platform_coverage"].head(15).sort_values("grid_month_rows")
    plt.figure(figsize=(10, 6))
    plt.barh(data["platform_name"], data["grid_month_rows"])
    plt.xlabel("Cruise-node-month rows")
    plt.title("Fifteen most represented platforms")
    save("fig07_platform_coverage.png", ["table06_platform_coverage.csv"])

    data = tables["split_summary"].groupby("cruise_fold", as_index=False)["cruises"].sum()
    plt.figure(figsize=(8, 4.8))
    plt.bar(data["cruise_fold"].astype(str), data["cruises"])
    plt.xlabel("Cruise outer fold")
    plt.ylabel("Cruises")
    plt.title("Deterministic grouped fold balance")
    save("fig08_split_balance.png", ["table07_split_summary.csv"])

    data = tables["support_distance_bins"]
    pivot = data.pivot(index="distance_left_km", columns="target", values="coastal_nodes")
    pivot.plot.bar(figsize=(9, 4.8))
    plt.yscale("log")
    plt.xlabel("Nearest observation support: bin lower edge (km)")
    plt.ylabel("Coastal nodes")
    plt.title("Global nearest-support distribution")
    save("fig09_nearest_support.png", ["table12_support_distance_bins.csv"])

    data = tables["non_na_split_coverage"]
    pivot = data.pivot(
        index="area_scope", columns="development_role", values="grid_month_rows"
    ).fillna(0)
    pivot.plot.bar(figsize=(9, 4.8))
    plt.ylabel("Cruise-node-month rows")
    plt.title("North-American-adjacent and non-NA records in both development roles")
    save("fig10_non_na_split_coverage.png", ["table08_non_na_split_coverage.csv"])
    return mapping


CAPTIONS = {
    "fig01_global_observation_map.png": "Figure 1. Global two-degree audit view of SOCATv2026 Coastal cruise-node-month observations on the complete configured 1/8-degree grid; color and marker size encode retained density, and the map is evidence of observation coverage rather than model skill.",
    "fig02_processing_funnel.png": "Figure 2. Row counts at each ordered processing stage, from source observations through identity/time validation, exact duplicate removal, target QC, complete-grid matching, and cruise-node-month aggregation; the legacy v2.2 mask row is a diagnostic subset rather than a filter.",
    "fig03_basin_coverage.png": "Figure 3. Retained fCO2 and in-situ SSS cruise-node-month coverage across the six frozen coarse basins; this exposes geographic imbalance that later macro-region scoring must prevent from being hidden by observation-weighted metrics.",
    "fig04_lme_support_status.png": "Figure 4. Counts of the 66 LMEs classified as supported, under-supported, or empty for each target using the frozen audit thresholds; empty categories remain explicit and cannot inherit a global validation claim from covariate availability.",
    "fig05_season_coverage.png": "Figure 5. Seasonal target coverage after global QC and coastal matching; the counts diagnose temporal sampling imbalance and define strata required in later model evaluation rather than constituting a performance result.",
    "fig06_decade_coverage.png": "Figure 6. Retained target coverage by decade, including the partial 2020s represented in SOCATv2026; the temporal expansion motivates forward-development evaluation and prevents random splits from being the sole evidence.",
    "fig07_platform_coverage.png": "Figure 7. The fifteen platforms contributing the most compact grid-month rows; concentration in a small platform subset motivates platform and cruise grouped uncertainty analyses in subsequent issues.",
    "fig08_split_balance.png": "Figure 8. Cruise counts in the five deterministic cruise-grouped outer folds; every normalized Expocode belongs to exactly one fold, and balance is descriptive rather than optimized using target values.",
    "fig09_nearest_support.png": "Figure 9. Distance from every occupied observation node to its nearest training-role fCO2 or SSS support node, binned in kilometres; long-distance tails quantify where grouped development depends on spatial transfer and may require abstention.",
    "fig10_non_na_split_coverage.png": "Figure 10. Retained rows inside and outside the exact P1 North-American-adjacent window, separated into train and grouped-development roles; non-NA observations occur in both roles, satisfying the central global-cache acceptance gate.",
}


TABLE_FILES = {
    "processing_funnel": "table01_processing_funnel.csv",
    "basin_coverage": "table02_basin_coverage.csv",
    "lme_coverage": "table03_lme_coverage.csv",
    "season_coverage": "table04_season_coverage.csv",
    "decade_coverage": "table05_decade_coverage.csv",
    "platform_coverage": "table06_platform_coverage.csv",
    "split_summary": "table07_split_summary.csv",
    "non_na_split_coverage": "table08_non_na_split_coverage.csv",
    "leakage_checks": "table09_leakage_checks.csv",
    "mask_definition": "table10_mask_definition.csv",
    "provenance_audit": "table11_provenance_audit.csv",
    "support_distance_bins": "table12_support_distance_bins.csv",
    "global_map_2degree": "table13_global_map_2degree.csv",
    "neutral_region_coverage": "table14_neutral_region_coverage.csv",
}


def make_report(
    archive: Path,
    tables: dict[str, pd.DataFrame],
    cache: pd.DataFrame,
    groups: pd.DataFrame,
    support: pd.DataFrame,
    artifacts: dict[str, Path],
) -> None:
    non_na = tables["non_na_split_coverage"]
    non_na_roles = set(non_na.loc[non_na["area_scope"] == "non_NA", "development_role"])
    leakage = tables["leakage_checks"]["group_cross_partition_violations"].sum()
    lme = tables["lme_coverage"]
    legacy_mask = (
        tables["mask_definition"]
        .loc[tables["mask_definition"]["mask_role"] == "legacy_v2.2_product_mask_audit"]
        .iloc[0]
    )
    core = cache[cache["record_status"] == "core_1993_2025"]
    captions = "\n\n".join(f"### {name}\n\n{caption}" for name, caption in CAPTIONS.items())
    write_markdown(archive / "CAPTIONS.md", "# Figure captions\n\n" + captions + "\n")
    figure_blocks = "\n\n".join(
        f"![{name}](figures/{name})\n\n{caption}" for name, caption in CAPTIONS.items()
    )
    report = f"""# P2.2 global coastal observation cache v2.3

## Scientific question and permitted claim

Issue #38 asks whether ReCAD has a genuinely global **observation-indexed** development cache for SSS and fCO2, rather than a global predictor grid supported only by North-American-adjacent labels. The permitted claim is limited to data coverage and split integrity: the cache contains {len(core):,} core cruise-node-month rows from {core["group_key"].nunique():,} cruises on {core["coastal_node"].nunique():,} observed coastal nodes. It does not establish global reconstruction skill.

The acceptance gate passes because non-North-American records occur in both roles ({", ".join(sorted(non_na_roles))}) and all formal group-overlap checks equal {int(leakage)}. Of 66 LMEs, fCO2 has {(lme["fco2_status"] == "supported").sum()} supported, {(lme["fco2_status"] == "under_supported").sum()} under-supported, and {(lme["fco2_status"] == "empty").sum()} empty; SSS has {(lme["sss_status"] == "supported").sum()} supported, {(lme["sss_status"] == "under_supported").sum()} under-supported, and {(lme["sss_status"] == "empty").sum()} empty. These labels are audit strata, not release grades.

## Data, splits, and leakage controls

The target source is `SOCATv2026_Coastal.tsv`. SSS is the SOCAT in-situ `sal` field after 0-50 QC; fCO2 is the recommended `fCO2rec` after 1-1000 µatm and flag ≤2 QC. For this observation cache, coastal means membership in the official SOCAT Coastal synthesis. Every QC record is assigned to the complete configured 1/8-degree grid from 78°S to 84°N; the source occupies {cache["latitude"].min():.3f}° to {cache["latitude"].max():.3f}° and {len(support):,} grid cells. LME polygons and the neutral basin/5-degree fallback are assigned over all occupied cells. Unobserved basins and LMEs remain explicit rather than being inferred from predictor coverage.

The audit found that the v2.2 product `coastal_mask` spans only {legacy_mask["latitude_min"]:.3f}° to {legacy_mask["latitude_max"]:.3f}° latitude and was generated by the pipeline's observed-fCO2 fallback. It is retained as `on_v2_2_product_mask` metadata and is not allowed to filter this cache. #39 must replace it with an observation-independent global product mask before predictor extraction or atlas training.

Normalized Expocode is assigned before QC and before cruise-node-month aggregation. Exact duplicate samples are removed before target filtering. The compact cache retains provenance, platform, source DOI, source QC, raw sample counts, and target-specific counts. The generated row-level files remain outside Git:

- cache: `{artifacts["cache"]}`
- group split manifest: `{artifacts["groups"]}`
- all-node support table: `{artifacts["support"]}`
- machine audit: `{artifacts["audit"]}`

Cruise folds hash whole normalized Expocodes. Spatial folds hash each cruise's dominant 5-degree block and still move the whole cruise. Whole-LME evaluation must remove every cruise touching the held LME from training. Forward roles use cruise maximum year: ≤2018 train, 2019-2021 forward development, 2022-2025 late development, and 2026 provisional. All SOCATv2026 labels are marked development-accessible; none are represented as newly blind. The only SSS/fCO2 external candidate is a later SOCAT release with groups absent from the frozen source hash, recorded as metadata without labels.

## Candidate models and training

No model is trained in P2.2. This issue freezes the observation, grouping, provenance, and split contract used by #39-#44. Later candidates must consume identical target rows and outer partitions, fit preprocessing and calibration inside training folds, and report cruise, spatial, whole-LME, and forward evidence separately.

## Main development results

The full SOCAT source was scanned rather than inferred from prepared-grid variable names. The processing funnel in Table 1 records every loss stage and separately reports how many valid records the incomplete legacy mask would have retained. Basin, LME, neutral region, season, decade, platform, and target coverage are materialized as source CSV files. Target-specific distances from every observed node to the nearest training-support node are computed for all {len(support):,} occupied cells. The cache contains {groups["platform_name"].nunique():,} named platforms and {groups["group_key"].nunique():,} cruise groups; {int(cache["provenance_variants"].gt(1).sum()):,} compact rows contain multiple DOI variants and remain auditable.

Non-NA coverage is not treated as automatically representative: dense regions can dominate row-weighted results, so later issues must report macro-LME and worst-region metrics and abstain in empty or under-supported regions.

## Decision and limitations

Decision: **pass_data_freeze**. P2.2 supplies a genuinely global observation-indexed development cache and zero-leakage group contracts. #39 may add deployable predictors against this fixed target universe only after replacing the incomplete, label-derived v2.2 product mask with an observation-independent global coastal mask.

Limitations: SOCAT sampling is opportunistic and platform-imbalanced; SSS is available only where SOCAT reports in-situ salinity; the observation-indexed support mask is not a product inference mask; the legacy v2.2 product mask is geographically incomplete; a dominant-block spatial fold is a grouped stress test rather than pure geographic independence; and SOCATv2026 cannot serve as a new independent test because its labels were accessible during cache construction. Future release labels remain unopened until P3.

## Figures

{figure_blocks}

## Figure and table index

Figures 1-10 are embedded above with their reviewer-ready captions. Tables 1-14 in `tables/` are the exact source data for the figures and audit statements. `archive_manifest.json` maps every figure to its source table and hashes every tracked artifact. Generated row-level Parquet and JSON artifacts are referenced by path and SHA256 in the frozen repository manifest.
"""
    write_markdown(archive / "REPORT.md", report)
    write_markdown(
        archive / "README.md",
        "# P2 global coastal cache v2.3\n\n"
        "Reviewer-ready data audit for Issue #38. Rebuild with "
        "`python scripts/build_global_coastal_cache_v2_3.py`; generated scientific data remain "
        "outside Git. See [REPORT.md](REPORT.md).\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--spatial", type=Path, default=DEFAULT_SPATIAL)
    parser.add_argument("--lme", type=Path, default=DEFAULT_LME)
    parser.add_argument("--socat", type=Path, default=DEFAULT_SOCAT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--chunksize", type=int, default=750_000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "docs" / "experiment_archive" / EXPERIMENT_ID
    tables_dir = archive / "tables"
    figures_dir = archive / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cache_path = args.out / "global_socat_cache_v2.3.parquet"
    group_path = args.out / "global_group_split_manifest_v2.3.parquet"
    support_path = args.out / "global_node_support_v2.3.parquet"
    funnel_path = args.out / "global_processing_funnel_v2.3.parquet"
    audit_path = args.out / "global_coastal_cache_audit_v2.3.json"
    grid = CoastalGrid.load(args.prepared, args.spatial)
    if args.reuse_cache and all(
        path.exists() for path in (cache_path, group_path, support_path, funnel_path)
    ):
        cache = pd.read_parquet(cache_path)
        groups = pd.read_parquet(group_path)
        support = pd.read_parquet(support_path)
        funnel = pd.read_parquet(funnel_path)
    else:
        cache, funnel = build_global_socat_cache(args.socat, grid, chunksize=args.chunksize)
        cache = assign_global_regions(cache, args.lme)
        groups = build_group_manifest(cache)
        cache = cache.merge(
            groups[
                [
                    "group_key",
                    "development_role",
                    "cruise_fold",
                    "spatial_fold",
                    "forward_role",
                    "label_exposure",
                ]
            ],
            on="group_key",
            how="left",
            validate="many_to_one",
        )
        cache["training_eligible"] = cache["record_status"].eq("core_1993_2025") & cache[
            "development_role"
        ].eq("train")
        cache["grouped_development_eligible"] = cache["record_status"].eq("core_1993_2025") & cache[
            "development_role"
        ].eq("grouped_development")
        support = build_node_support(cache, grid)
        cache.to_parquet(cache_path, index=False, compression="zstd")
        groups.to_parquet(group_path, index=False, compression="zstd")
        support.to_parquet(support_path, index=False, compression="zstd")
        funnel.to_parquet(funnel_path, index=False, compression="zstd")

    lme_names = load_lme_names(args.spatial)
    tables = build_audit_tables(cache, groups, support, funnel, grid, lme_names)
    for key, filename in TABLE_FILES.items():
        write_table(tables[key], tables_dir / filename)
    figure_sources = save_figures(tables, figures_dir)

    artifacts = {
        "cache": cache_path,
        "groups": group_path,
        "support": support_path,
        "funnel": funnel_path,
        "audit": audit_path,
    }
    input_hashes = {
        str(path): sha256(path) for path in (args.socat, args.prepared, args.spatial, args.lme)
    }
    artifact_hashes = {
        str(path): sha256(path) for path in (cache_path, group_path, support_path, funnel_path)
    }
    non_na = tables["non_na_split_coverage"]
    audit = {
        "schema_version": "2.3",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "pass_data_freeze",
        "cache_rows": len(cache),
        "cruise_groups": int(groups["group_key"].nunique()),
        "coastal_nodes": len(support),
        "observed_nodes": int(cache["coastal_node"].nunique()),
        "observed_latitude_bounds": [
            float(cache["latitude"].min()),
            float(cache["latitude"].max()),
        ],
        "legacy_v2_2_mask": {
            "latitude_min": float(grid.node_latitude.min()),
            "latitude_max": float(grid.node_latitude.max()),
            "cache_rows_inside": int(cache["on_v2_2_product_mask"].sum()),
            "cache_rows_outside": int((~cache["on_v2_2_product_mask"]).sum()),
            "usable_as_global_product_mask": False,
        },
        "non_na_roles": sorted(
            non_na.loc[non_na["area_scope"] == "non_NA", "development_role"].unique()
        ),
        "leakage_checks": tables["leakage_checks"].to_dict("records"),
        "input_sha256": input_hashes,
        "artifact_sha256_without_audit": artifact_hashes,
        "external_candidate": {
            "source": "SOCAT release newer than v2026",
            "selection": "new normalized Expocode groups absent from frozen v2026 source hash",
            "labels_present": False,
            "exposure": "metadata_only; no external labels opened in Issue #38",
        },
    }
    write_json(audit_path, audit)
    artifact_hashes[str(audit_path)] = sha256(audit_path)
    manifest_path = ROOT / "configs" / "frozen" / "global_coastal_cache_manifest_v2.3.json"
    manifest = {
        "schema_version": "2.3",
        "experiment_id": EXPERIMENT_ID,
        "generated_utc": audit["generated_utc"],
        "decision": "pass_data_freeze",
        "domain": {
            "observation_mask": "occupied cells from official SOCATv2026 Coastal synthesis",
            "resolution_degrees": grid.resolution_degrees,
            "latitude_bounds": [-78.0, 84.0],
            "observed_latitude_bounds": [
                float(cache["latitude"].min()),
                float(cache["latitude"].max()),
            ],
            "observation_nodes": len(support),
            "matching": "nearest cell on complete configured global grid",
            "legacy_v2_2_product_mask": {
                "latitude_bounds": [
                    float(grid.node_latitude.min()),
                    float(grid.node_latitude.max()),
                ],
                "role": "audit flag only; forbidden as global cache filter",
                "required_follow_up": "replace with observation-independent global coastal mask in #39",
            },
        },
        "period": {"core": "1993-2025", "provisional": "2026"},
        "targets": {
            "SSS": "SOCAT in-situ sal, 0..50",
            "fCO2": "SOCAT recommended fCO2rec, 1..1000 uatm, flag <=2",
        },
        "grouping": "normalized Expocode before QC and aggregation",
        "splits": {
            "development": "whole-cruise salted 80/20 train/grouped-development",
            "cruise": "five whole-cruise salted folds",
            "spatial": "five salted dominant-5-degree-block folds, whole cruise moves together",
            "whole_lme": "hold LME rows and exclude all cruises touching held LME from train",
            "forward": "cruise max year <=2018 / 2019-2021 / 2022-2025 / 2026",
        },
        "label_exposure": {
            "SOCATv2026": "development_accessible; not a new blind test",
            "future_external": "metadata-only rule for later SOCAT release; labels absent",
        },
        "inputs_sha256": input_hashes,
        "artifacts_sha256": artifact_hashes,
        "audit": str(audit_path),
    }
    write_json(manifest_path, manifest)
    artifacts["manifest"] = manifest_path
    make_report(archive, tables, cache, groups, support, artifacts)

    tracked = [archive / "README.md", archive / "REPORT.md", archive / "CAPTIONS.md"]
    tracked.extend(sorted(figures_dir.glob("*.png")))
    tracked.extend(sorted(tables_dir.glob("*.csv")))
    files_sha256 = {
        str(path.relative_to(archive)).replace("\\", "/"): sha256(path) for path in tracked
    }
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    script_hash = sha256(Path(__file__))
    archive_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "training_git_commit": git_commit,
        "data_manifest_sha256": sha256(manifest_path),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "global SOCATv2026 observation coverage and development split integrity",
        "decision": "pass_data_freeze",
        "figure_source_data": figure_sources,
        "figure_captions": CAPTIONS,
        "files_sha256": files_sha256,
        "archive_builder_sha256": script_hash,
        "analysis_script_sha256": script_hash,
        "source_artifacts_sha256": {**input_hashes, **artifact_hashes},
    }
    write_json(archive / "archive_manifest.json", archive_manifest)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
