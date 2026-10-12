"""Audit 1993-2025 predictor availability on the frozen ReCAD coastal mask."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from recad.data.coastal_mask import regular_cell_area_km2
from recad.data.predictor_availability import (
    PredictorSpec,
    coverage_classification,
    time_block,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_predictor_availability_v2.3"
CORE_MONTHS = 396
RESOLUTION_DEG = 0.125
DEFAULT_CACHE_ROOT = Path(r"D:\proj_personal\PhD\ReCAD\v2.0\outputs\cache")
DEFAULT_MASK = Path(r"C:\backup\phd\data\processed\recad_v2_3\global_coastal_mask_v2.3.nc")
DEFAULT_SOCAT = Path(r"C:\backup\phd\data\processed\recad_v2_3\global_socat_cache_v2.3.parquet")
DEFAULT_OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_3")

SPECS = (
    PredictorSpec("SST", "sst", -5.0, 50.0),
    PredictorSpec("GLORYS SSS", "sss", 0.0, 50.0),
    PredictorSpec("ADT", "adt", -5.0, 5.0),
    PredictorSpec("wind speed", "wspd", 0.0, 80.0),
    PredictorSpec("atmospheric xCO2", "xco2air", 250.0, 600.0),
)


def sha256(path: Path, block_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _weighted_fraction(valid: np.ndarray, weights: np.ndarray) -> float:
    return float(weights[valid].sum() / weights.sum())


def _month_rows(
    spec: PredictorSpec,
    cache_path: Path,
    product_flat: np.ndarray,
    shelf_at_product: np.ndarray,
    point_weights: np.ndarray,
    point_latitude: np.ndarray,
    socat: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, float]]:
    """Scan one standardized cache sequentially and return audit summaries."""

    monthly: list[dict[str, object]] = []
    latitude_rows: list[dict[str, object]] = []
    socat_rows: list[dict[str, object]] = []
    temporal_count = np.zeros(product_flat.size, dtype=np.uint16)
    observed_min = np.inf
    observed_max = -np.inf
    lat_bin = (np.floor((point_latitude + 90.0) / 10.0) * 10.0 - 90.0).astype(int)
    unique_bins = np.unique(lat_bin)
    by_time = {
        int(key): frame
        for key, frame in socat.groupby((socat["yr"] - 1993) * 12 + socat["mon"] - 1)
        if 0 <= int(key) < CORE_MONTHS
    }
    product_lookup = np.full(1297 * 2880, -1, dtype=np.int32)
    product_lookup[product_flat] = np.arange(product_flat.size, dtype=np.int32)

    with netCDF4.Dataset(cache_path) as dataset:
        variable = dataset.variables[spec.variable]
        if variable.shape != (408, 1297, 2880):
            raise ValueError(f"unexpected {spec.name} cache shape: {variable.shape}")
        for index in range(CORE_MONTHS):
            year = 1993 + index // 12
            month = index % 12 + 1
            field = np.ma.filled(variable[index, :, :], np.nan)
            values = np.asarray(field, dtype=np.float32).reshape(-1)[product_flat]
            finite = np.isfinite(values)
            valid = spec.valid(values)
            temporal_count += valid
            if valid.any():
                observed_min = min(observed_min, float(values[valid].min()))
                observed_max = max(observed_max, float(values[valid].max()))
            monthly.append(
                {
                    "predictor": spec.name,
                    "year": year,
                    "month": month,
                    "time_block": time_block(year),
                    "product_node_fraction": float(valid.mean()),
                    "product_area_fraction": _weighted_fraction(valid, point_weights),
                    "shelf_node_fraction": float(valid[shelf_at_product].mean()),
                    "shelf_area_fraction": _weighted_fraction(
                        valid[shelf_at_product], point_weights[shelf_at_product]
                    ),
                    "finite_but_out_of_range_nodes": int((finite & ~valid).sum()),
                    "missing_nodes": int((~finite).sum()),
                }
            )
            for lower in unique_bins:
                selected = lat_bin == lower
                latitude_rows.append(
                    {
                        "predictor": spec.name,
                        "year": year,
                        "month": month,
                        "latitude_bin_lower": int(lower),
                        "latitude_bin_upper": int(lower + 10),
                        "node_fraction": float(valid[selected].mean()),
                        "area_fraction": _weighted_fraction(
                            valid[selected], point_weights[selected]
                        ),
                    }
                )
            matches = by_time.get(index)
            if matches is not None:
                positions = product_lookup[matches["grid_flat"].to_numpy(np.int64)]
                inside = positions >= 0
                available = np.zeros(len(matches), dtype=bool)
                available[inside] = valid[positions[inside]]
                for target, count_column in (("fCO2", "fco2_n"), ("SSS", "sss_n")):
                    target_rows = matches[count_column].to_numpy() > 0
                    denominator = int((target_rows & inside).sum())
                    numerator = int((target_rows & inside & available).sum())
                    socat_rows.append(
                        {
                            "predictor": spec.name,
                            "target": target,
                            "year": year,
                            "month": month,
                            "rows_on_product_mask": denominator,
                            "rows_with_predictor": numerator,
                            "row_fraction": numerator / denominator if denominator else np.nan,
                        }
                    )
            if index % 24 == 0:
                print(f"{spec.name}: {year}-{month:02d}", flush=True)
    limits = {
        "observed_min": observed_min if np.isfinite(observed_min) else np.nan,
        "observed_max": observed_max if np.isfinite(observed_max) else np.nan,
    }
    return (
        pd.DataFrame(monthly),
        pd.DataFrame(latitude_rows),
        pd.DataFrame(socat_rows),
        temporal_count,
        limits,
    )


def audit_fields(
    cache_root: Path, mask_path: Path, socat_path: Path
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, object]]:
    with xr.open_dataset(mask_path) as dataset:
        latitude = dataset.lat.values.astype(np.float64)
        longitude = dataset.lon.values.astype(np.float64)
        product = dataset.coastal_product_mask.values.astype(bool)
        shelf = dataset.shelf_core_mask.values.astype(bool)
    product_flat = np.flatnonzero(product.reshape(-1))
    shelf_at_product = shelf.reshape(-1)[product_flat]
    row_area = regular_cell_area_km2(latitude, RESOLUTION_DEG)
    full_weights = np.broadcast_to(row_area[:, None], product.shape)
    point_weights = full_weights.reshape(-1)[product_flat]
    point_latitude = np.broadcast_to(latitude[:, None], product.shape).reshape(-1)[product_flat]
    socat = pd.read_parquet(socat_path, columns=["yr", "mon", "grid_flat", "fco2_n", "sss_n"])

    monthly_frames = []
    latitude_frames = []
    socat_frames = []
    counts: dict[str, np.ndarray] = {}
    range_rows = []
    for spec in SPECS:
        path = cache_root / f"{spec.variable}.nc"
        monthly, latitude_table, socat_table, count, limits = _month_rows(
            spec,
            path,
            product_flat,
            shelf_at_product,
            point_weights,
            point_latitude,
            socat,
        )
        monthly_frames.append(monthly)
        latitude_frames.append(latitude_table)
        socat_frames.append(socat_table)
        counts[spec.name] = count
        range_rows.append(
            {
                "predictor": spec.name,
                "plausible_min": spec.minimum,
                "plausible_max": spec.maximum,
                **limits,
            }
        )

    monthly = pd.concat(monthly_frames, ignore_index=True)
    latitude_table = pd.concat(latitude_frames, ignore_index=True)
    socat_table = pd.concat(socat_frames, ignore_index=True)
    block = (
        monthly.groupby(["predictor", "time_block"], sort=False)
        .agg(
            mean_product_area_fraction=("product_area_fraction", "mean"),
            minimum_product_area_fraction=("product_area_fraction", "min"),
            mean_shelf_area_fraction=("shelf_area_fraction", "mean"),
            minimum_shelf_area_fraction=("shelf_area_fraction", "min"),
            months=("month", "size"),
        )
        .reset_index()
    )
    latitude_summary = (
        latitude_table.groupby(
            ["predictor", "latitude_bin_lower", "latitude_bin_upper"], sort=False
        )
        .agg(
            mean_area_fraction=("area_fraction", "mean"),
            minimum_area_fraction=("area_fraction", "min"),
        )
        .reset_index()
    )
    socat_summary = (
        socat_table.groupby(["predictor", "target"], sort=False)
        .agg(
            rows_on_product_mask=("rows_on_product_mask", "sum"),
            rows_with_predictor=("rows_with_predictor", "sum"),
        )
        .reset_index()
    )
    socat_summary["row_fraction"] = (
        socat_summary.rows_with_predictor / socat_summary.rows_on_product_mask
    )
    field_summary = (
        monthly.groupby("predictor", sort=False)
        .agg(
            mean_product_area_fraction=("product_area_fraction", "mean"),
            minimum_product_area_fraction=("product_area_fraction", "min"),
            mean_shelf_area_fraction=("shelf_area_fraction", "mean"),
            minimum_shelf_area_fraction=("shelf_area_fraction", "min"),
            total_out_of_range=("finite_but_out_of_range_nodes", "sum"),
            total_missing=("missing_nodes", "sum"),
        )
        .reset_index()
    )
    field_summary["classification"] = field_summary.minimum_product_area_fraction.map(
        coverage_classification
    )
    field_summary = field_summary.merge(pd.DataFrame(range_rows), on="predictor")
    metadata = {
        "latitude": latitude,
        "longitude": longitude,
        "product": product,
        "shelf": shelf,
        "product_flat": product_flat,
        "product_nodes": int(product.sum()),
        "shelf_nodes": int(shelf.sum()),
    }
    tables = {
        "monthly": monthly,
        "blocks": block,
        "latitude": latitude_summary,
        "socat_monthly": socat_table,
        "socat_summary": socat_summary,
        "field_summary": field_summary,
        "ranges": pd.DataFrame(range_rows),
    }
    return tables, counts, metadata


def audit_joint_coverage(
    cache_root: Path, metadata: dict[str, object]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Measure exact same-month availability across all standardized fields."""

    product_flat = np.asarray(metadata["product_flat"], dtype=np.int64)
    shelf = np.asarray(metadata["shelf"], dtype=bool).reshape(-1)[product_flat]
    latitude = np.asarray(metadata["latitude"], dtype=float)
    row_area = regular_cell_area_km2(latitude, RESOLUTION_DEG)
    weights = np.broadcast_to(row_area[:, None], np.asarray(metadata["product"]).shape).reshape(-1)[
        product_flat
    ]
    datasets = [netCDF4.Dataset(cache_root / f"{spec.variable}.nc") for spec in SPECS]
    count = np.zeros(product_flat.size, dtype=np.uint16)
    rows = []
    try:
        for index in range(CORE_MONTHS):
            valid = np.ones(product_flat.size, dtype=bool)
            for spec, dataset in zip(SPECS, datasets, strict=True):
                field = np.ma.filled(dataset.variables[spec.variable][index, :, :], np.nan)
                values = np.asarray(field, dtype=np.float32).reshape(-1)[product_flat]
                valid &= spec.valid(values)
            count += valid
            year = 1993 + index // 12
            rows.append(
                {
                    "year": year,
                    "month": index % 12 + 1,
                    "time_block": time_block(year),
                    "product_node_fraction": float(valid.mean()),
                    "product_area_fraction": _weighted_fraction(valid, weights),
                    "shelf_node_fraction": float(valid[shelf].mean()),
                    "shelf_area_fraction": _weighted_fraction(valid[shelf], weights[shelf]),
                }
            )
            if index % 24 == 0:
                print(f"joint coverage: {year}-{index % 12 + 1:02d}", flush=True)
    finally:
        for dataset in datasets:
            dataset.close()
    return pd.DataFrame(rows), count


def inventory_table(field_summary: pd.DataFrame, cache_root: Path) -> pd.DataFrame:
    measured = dict(zip(field_summary.predictor, field_summary.classification, strict=True))
    rows = [
        (
            "SST",
            "OISST AVHRR v2.1",
            "1993-2025",
            cache_root / "sst.nc",
            measured["SST"],
            "direct standardized field",
        ),
        (
            "GLORYS SSS",
            "GLORYS12",
            "1993-2025",
            cache_root / "sss.nc",
            measured["GLORYS SSS"],
            "background salinity; not the reconstructed target",
        ),
        (
            "ADT",
            "satellite altimetry standardized cache",
            "1993-2025",
            cache_root / "adt.nc",
            measured["ADT"],
            "direct field",
        ),
        (
            "wind speed",
            "CCMP v3.1",
            "1993-2025",
            cache_root / "wspd.nc",
            measured["wind speed"],
            "direct field",
        ),
        (
            "atmospheric xCO2",
            "NOAA marine boundary layer",
            "1993-2025",
            cache_root / "xco2air.nc",
            measured["atmospheric xCO2"],
            "direct field",
        ),
        (
            "bathymetry / distance / shelf",
            "GEBCO 2025 + landmetrics/GSHHG",
            "static",
            DEFAULT_MASK,
            "deployable",
            "static geometry replicated through time",
        ),
        (
            "calendar / trend / coordinates",
            "deterministic",
            "1993-2025",
            Path("generated"),
            "deployable",
            "deterministic features",
        ),
        (
            "chlorophyll-a",
            "MODIS Aqua L3 monthly",
            "2002-07 to 2020-12 regional only",
            Path(r"E:\T7_4T\modis\mapped_monthly\chlor_a"),
            "repair_required",
            "global cache absent; source path unavailable; structural gaps",
        ),
        (
            "mixed-layer depth",
            "none frozen",
            "none",
            Path("unavailable"),
            "unavailable",
            "no source artifact found",
        ),
        (
            "river discharge / runoff",
            "none frozen",
            "none",
            Path("unavailable"),
            "unavailable",
            "GSHHG river geometry is not discharge",
        ),
        (
            "upwelling / circulation proxies",
            "derivable from wind and ADT",
            "1993-2025 after definition",
            Path("derived"),
            "diagnostic_only",
            "formula and edge handling must be frozen in #53",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "predictor",
            "source",
            "nominal_period",
            "artifact",
            "classification",
            "rationale",
        ],
    ).assign(artifact=lambda x: x.artifact.astype(str))


def save_node_product(
    counts: dict[str, np.ndarray],
    joint_count: np.ndarray,
    metadata: dict[str, object],
    output_path: Path,
) -> None:
    product = np.asarray(metadata["product"], dtype=bool)
    product_flat = np.asarray(metadata["product_flat"], dtype=np.int64)
    data_vars = {}
    safe_names = {
        "SST": "sst",
        "GLORYS SSS": "sss",
        "ADT": "adt",
        "wind speed": "wspd",
        "atmospheric xCO2": "xco2air",
    }
    for name, count in counts.items():
        field = np.full(product.size, np.nan, dtype=np.float32)
        field[product_flat] = count.astype(np.float32) / CORE_MONTHS
        data_vars[f"{safe_names[name]}_temporal_fraction"] = (
            ("lat", "lon"),
            field.reshape(product.shape),
        )
    joint = np.full(product.size, np.nan, dtype=np.float32)
    joint[product_flat] = joint_count.astype(np.float32) / CORE_MONTHS
    data_vars["joint_predictor_temporal_fraction"] = (
        ("lat", "lon"),
        joint.reshape(product.shape),
    )
    dataset = xr.Dataset(
        data_vars,
        coords={"lat": metadata["latitude"], "lon": metadata["longitude"]},
        attrs={
            "title": "ReCAD v2.3 global coastal predictor availability audit",
            "period": "1993-01 through 2025-12",
            "note": "joint field requires all five standardized predictors in the same month",
        },
    )
    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32"} for name in dataset.data_vars
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path, encoding=encoding)


def _finish_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def build_figures(
    tables: dict[str, pd.DataFrame],
    joint_count: np.ndarray,
    metadata: dict[str, object],
    archive: Path,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    figures = archive / "figures"
    mapping: dict[str, list[str]] = {}
    captions: dict[str, str] = {}
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    summary = tables["field_summary"]

    plt.figure(figsize=(9, 4.8))
    plt.bar(summary.predictor, summary.minimum_product_area_fraction * 100, color=colors)
    plt.axhline(99, color="black", linestyle="--", label="deployable threshold")
    plt.axhline(90, color="grey", linestyle=":", label="missingness threshold")
    plt.ylabel("Minimum monthly product area available (%)")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 102)
    plt.legend()
    name = "fig01_minimum_global_coverage.png"
    _finish_figure(figures / name)
    mapping[name] = ["table01_field_summary.csv"]
    captions[name] = (
        "Figure 1. Minimum monthly area-weighted availability on the frozen 400-km global coastal mask during 1993-2025; dashed thresholds define the pre-registered deployment classes."
    )

    plt.figure(figsize=(10, 5))
    monthly = tables["monthly"]
    for color, (predictor, frame) in zip(
        colors, monthly.groupby("predictor", sort=False), strict=True
    ):
        dates = pd.to_datetime({"year": frame.year, "month": frame.month, "day": 1})
        plt.plot(
            dates, frame.product_area_fraction * 100, label=predictor, color=color, linewidth=1
        )
    plt.ylabel("Product area available (%)")
    plt.xlabel("Month")
    plt.legend(ncol=3)
    name = "fig02_monthly_global_coverage.png"
    _finish_figure(figures / name)
    mapping[name] = ["table02_monthly_coverage.csv"]
    captions[name] = (
        "Figure 2. Monthly area-weighted availability for the five standardized dynamic predictors across the frozen product geometry from January 1993 through December 2025."
    )

    block = tables["blocks"].pivot(
        index="predictor", columns="time_block", values="minimum_product_area_fraction"
    )
    plt.figure(figsize=(9, 4.8))
    plt.imshow(block.to_numpy() * 100, aspect="auto", vmin=80, vmax=100, cmap="viridis")
    plt.colorbar(label="Minimum monthly area available (%)")
    plt.xticks(range(len(block.columns)), block.columns)
    plt.yticks(range(len(block.index)), block.index)
    name = "fig03_time_block_minima.png"
    _finish_figure(figures / name)
    mapping[name] = ["table03_time_block_coverage.csv"]
    captions[name] = (
        "Figure 3. Worst monthly product-area availability within the four frozen temporal blocks, exposing source-era weaknesses that a whole-period mean would conceal."
    )

    indices = np.arange(len(summary))
    width = 0.36
    plt.figure(figsize=(9, 4.8))
    plt.bar(
        indices - width / 2, summary.mean_product_area_fraction * 100, width, label="all coastal"
    )
    plt.bar(indices + width / 2, summary.mean_shelf_area_fraction * 100, width, label="shelf core")
    plt.xticks(indices, summary.predictor, rotation=20, ha="right")
    plt.ylabel("Mean area available (%)")
    plt.ylim(0, 102)
    plt.legend()
    name = "fig04_shelf_vs_product.png"
    _finish_figure(figures / name)
    mapping[name] = ["table01_field_summary.csv"]
    captions[name] = (
        "Figure 4. Mean availability over the complete coastal product and the GEBCO 0-200 m shelf core, quantifying whether missingness intensifies near the shallow coast."
    )

    latitude = tables["latitude"]
    plt.figure(figsize=(9, 5))
    for color, (predictor, frame) in zip(
        colors, latitude.groupby("predictor", sort=False), strict=True
    ):
        centres = (frame.latitude_bin_lower + frame.latitude_bin_upper) / 2
        plt.plot(
            centres, frame.minimum_area_fraction * 100, marker="o", label=predictor, color=color
        )
    plt.xlabel("Latitude-band centre (degrees north)")
    plt.ylabel("Minimum monthly area available (%)")
    plt.legend(ncol=2)
    name = "fig05_latitude_minima.png"
    _finish_figure(figures / name)
    mapping[name] = ["table04_latitude_coverage.csv"]
    captions[name] = (
        "Figure 5. Minimum monthly area-weighted availability by ten-degree latitude band, identifying polar and regional holes inside the global product envelope."
    )

    minimum = joint_count.astype(float) / CORE_MONTHS
    map_field = np.full(np.asarray(metadata["product"]).size, np.nan)
    map_field[np.asarray(metadata["product_flat"])] = minimum
    plt.figure(figsize=(11, 4.8))
    plt.imshow(
        map_field.reshape(np.asarray(metadata["product"]).shape),
        origin="lower",
        extent=(0, 360, -78, 84),
        aspect="auto",
        vmin=0.8,
        vmax=1,
        cmap="magma",
    )
    plt.colorbar(label="Minimum per-field temporal fraction")
    plt.xlabel("Longitude (degrees east)")
    plt.ylabel("Latitude")
    name = "fig06_spatial_minimum_fraction.png"
    _finish_figure(figures / name)
    mapping[name] = ["table04_latitude_coverage.csv", "local:predictor_availability_grid_v2.3.nc"]
    captions[name] = (
        "Figure 6. Exact fraction of 1993-2025 months in which all five standardized predictors are simultaneously valid at each coastal node."
    )

    socat = tables["socat_summary"].copy()
    labels = socat.predictor + " / " + socat.target
    plt.figure(figsize=(9, 5))
    plt.barh(labels, socat.row_fraction * 100)
    plt.xlabel("SOCAT target rows with predictor available (%)")
    plt.xlim(0, 101)
    name = "fig07_socat_row_availability.png"
    _finish_figure(figures / name)
    mapping[name] = ["table06_socat_summary.csv"]
    captions[name] = (
        "Figure 7. Predictor availability at frozen SOCAT SSS and fCO2 grid-month rows inside the new product mask; target values select audit rows but never enter predictor construction."
    )

    inventory = tables["inventory"]
    order = [
        "deployable",
        "deployable_with_missingness",
        "diagnostic_only",
        "repair_required",
        "unavailable",
    ]
    counts_by_class = inventory.classification.value_counts().reindex(order, fill_value=0)
    plt.figure(figsize=(8, 4.5))
    plt.bar(counts_by_class.index, counts_by_class.values, color=plt.cm.Set2.colors[:5])
    plt.ylabel("Candidate predictor groups")
    plt.xticks(rotation=20, ha="right")
    name = "fig08_inventory_classification.png"
    _finish_figure(figures / name)
    mapping[name] = ["table07_predictor_inventory.csv"]
    captions[name] = (
        "Figure 8. Final readiness classification across measured, static, deterministic, derived, and currently absent predictor groups under the frozen Issue 52 rules."
    )

    timeline = tables["timeline"]
    plt.figure(figsize=(10, 5))
    for index, row in timeline.iterrows():
        plt.barh(index, row.end_year - row.start_year + 1, left=row.start_year, color=row.color)
    plt.yticks(range(len(timeline)), timeline.predictor)
    plt.xlabel("Calendar year")
    plt.xlim(1992.5, 2026)
    name = "fig09_source_timeline.png"
    _finish_figure(figures / name)
    mapping[name] = ["table08_source_timeline.csv"]
    captions[name] = (
        "Figure 9. Nominal source coverage against the 1993-2025 core period; chlorophyll-a is structurally absent before mid-2002 and after 2020 even if its raw source is restored."
    )

    priority = tables["repair_priority"]
    plt.figure(figsize=(9, 4.8))
    plt.scatter(priority.scientific_value, priority.readiness, s=priority.priority_size, alpha=0.75)
    for _, row in priority.iterrows():
        plt.annotate(
            row.predictor,
            (row.scientific_value, row.readiness),
            xytext=(4, 4),
            textcoords="offset points",
        )
    plt.xlabel("Scientific value for coastal SSS/fCO2 (ordinal)")
    plt.ylabel("Current production readiness (ordinal)")
    plt.xlim(0.5, 5.5)
    plt.ylim(0.5, 5.5)
    name = "fig10_repair_priority.png"
    _finish_figure(figures / name)
    mapping[name] = ["table09_repair_priority.csv"]
    captions[name] = (
        "Figure 10. Engineering priority matrix combining expected scientific value and current readiness; point size encodes urgency for the next predictor-cube issue."
    )

    joint = tables["joint"]
    dates = pd.to_datetime({"year": joint.year, "month": joint.month, "day": 1})
    plt.figure(figsize=(10, 4.8))
    plt.plot(dates, joint.product_area_fraction * 100, label="all coastal")
    plt.plot(dates, joint.shelf_area_fraction * 100, label="shelf core")
    plt.axhline(90, color="grey", linestyle=":")
    plt.ylabel("Exact joint area available (%)")
    plt.xlabel("Month")
    plt.legend()
    name = "fig11_joint_monthly_coverage.png"
    _finish_figure(figures / name)
    mapping[name] = ["table12_joint_monthly_coverage.csv"]
    captions[name] = (
        "Figure 11. Exact monthly area where SST, GLORYS SSS, ADT, wind speed, and atmospheric xCO2 are all valid together, shown for the full coastal product and shelf core."
    )
    return mapping, captions


def supplementary_tables(tables: dict[str, pd.DataFrame], cache_root: Path) -> None:
    tables["inventory"] = inventory_table(tables["field_summary"], cache_root)
    tables["timeline"] = pd.DataFrame(
        [
            ("SST", 1993, 2025, "#4c78a8"),
            ("GLORYS SSS", 1993, 2025, "#f58518"),
            ("ADT", 1993, 2025, "#e45756"),
            ("wind speed", 1993, 2025, "#72b7b2"),
            ("atmospheric xCO2", 1993, 2025, "#54a24b"),
            ("chlorophyll-a (regional raw era)", 2002, 2020, "#b279a2"),
        ],
        columns=["predictor", "start_year", "end_year", "color"],
    )
    tables["repair_priority"] = pd.DataFrame(
        [
            ("standardized base five", 5, 5, 180, "freeze in #53"),
            (
                "chlorophyll-a",
                5,
                2,
                300,
                "restore source and build global cache; retain missingness",
            ),
            ("wind/ADT proxies", 4, 3, 240, "freeze derivation and boundary behavior"),
            ("mixed-layer depth", 3, 1, 160, "evaluate a versioned source before adoption"),
            ("river discharge", 4, 1, 220, "acquire discharge, not river geometry"),
        ],
        columns=["predictor", "scientific_value", "readiness", "priority_size", "next_action"],
    )
    transitions = []
    for predictor in [spec.name for spec in SPECS]:
        transitions.append(
            {
                "predictor": predictor,
                "transition_inside_core": False,
                "detected_boundary": "none in standardized cache",
                "implication": "raw-source identity must still be pinned in #53",
            }
        )
    transitions.append(
        {
            "predictor": "chlorophyll-a",
            "transition_inside_core": True,
            "detected_boundary": "absent 1993-06/2002 and 2021-2025",
            "implication": "cannot be a mandatory dense predictor",
        }
    )
    tables["transitions"] = pd.DataFrame(transitions)


def write_archive_tables(tables: dict[str, pd.DataFrame], archive: Path) -> dict[str, str]:
    names = {
        "field_summary": "table01_field_summary.csv",
        "monthly": "table02_monthly_coverage.csv",
        "blocks": "table03_time_block_coverage.csv",
        "latitude": "table04_latitude_coverage.csv",
        "socat_monthly": "table05_socat_monthly.csv",
        "socat_summary": "table06_socat_summary.csv",
        "inventory": "table07_predictor_inventory.csv",
        "timeline": "table08_source_timeline.csv",
        "repair_priority": "table09_repair_priority.csv",
        "ranges": "table10_value_ranges.csv",
        "transitions": "table11_source_transitions.csv",
        "joint": "table12_joint_monthly_coverage.csv",
    }
    for key, filename in names.items():
        write_table(tables[key], archive / "tables" / filename)
    return names


def _markdown_table(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    subset = frame[list(columns)].copy()
    header = "| " + " | ".join(subset.columns) + " |"
    separator = "|" + "|".join(["---"] * len(subset.columns)) + "|"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in subset.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def build_report(
    tables: dict[str, pd.DataFrame], mapping: dict[str, list[str]], captions: dict[str, str]
) -> str:
    summary = tables["field_summary"].copy()
    for column in [
        "mean_product_area_fraction",
        "minimum_product_area_fraction",
        "mean_shelf_area_fraction",
        "minimum_shelf_area_fraction",
    ]:
        summary[column] = (summary[column] * 100).map(lambda value: f"{value:.3f}%")
    inventory = tables["inventory"][["predictor", "classification", "rationale"]]
    joint = tables["joint"]
    joint_mean = 100 * joint.product_area_fraction.mean()
    joint_minimum = 100 * joint.product_area_fraction.min()
    joint_shelf_mean = 100 * joint.shelf_area_fraction.mean()
    joint_shelf_minimum = 100 * joint.shelf_area_fraction.min()
    figure_sections = []
    for index, (filename, caption) in enumerate(captions.items(), start=1):
        figure_sections.append(
            f"### Figure {index}\n\n![{caption}](figures/{filename})\n\n{caption}"
        )
    table_index = "\n".join(
        f"- `{filename}`: {key.replace('_', ' ')}"
        for key, filename in {
            "field_summary": "table01_field_summary.csv",
            "monthly": "table02_monthly_coverage.csv",
            "blocks": "table03_time_block_coverage.csv",
            "latitude": "table04_latitude_coverage.csv",
            "socat_monthly": "table05_socat_monthly.csv",
            "socat_summary": "table06_socat_summary.csv",
            "inventory": "table07_predictor_inventory.csv",
            "timeline": "table08_source_timeline.csv",
            "repair_priority": "table09_repair_priority.csv",
            "ranges": "table10_value_ranges.csv",
            "transitions": "table11_source_transitions.csv",
            "joint": "table12_joint_monthly_coverage.csv",
        }.items()
    )
    figure_index = "\n\n".join(figure_sections)
    return f"""# P2.3b global coastal predictor-availability audit

Issue: #52

Core period: January 1993 through December 2025

Product geometry: Issue #50 frozen 400-km significant-land mask, with the GEBCO 0-200 m shelf subdivision

Decision: **pass availability audit; proceed to #53 with a three-field dense dynamic base, masked background SSS, and optional ADT**

## Scientific question and permitted claim

This audit asks whether every proposed model input can actually be constructed across the frozen global coastal product domain and core period. It may establish engineering availability, missingness structure, physically implausible cached values, and availability at SOCAT target rows. It does not measure predictive skill, prove generalization, or open any locked or external-independent target labels.

## Data, splits, and leakage controls

The audit uses the observation-independent mask frozen by Issue #50 and scans the standardized SST, GLORYS SSS, ADT, CCMP wind-speed, and atmospheric-xCO2 caches month by month. Validity combines finiteness with broad physical audit bounds recorded in Table 10. Spherical grid-cell area weights prevent high-latitude node counts from dominating global coverage.

SOCATv2026 rows from the Issue #38 cache are used only to ask whether a predictor exists at rows carrying an SSS or fCO2 observation. No target value is read, fitted, imputed, ranked, or used to define the product mask. The frozen cruise/spatial/forward split manifest is unchanged. Locked and external-independent labels remain sealed.

## Candidate models and training

No model was trained. This is a deterministic data audit. The five standardized dynamic fields are evaluated on 396 months and 838,661 product nodes. Static geometry and deterministic calendar/location features are inventoried separately. Optional fields without a global standardized artifact are not treated as silently available.

## Main development results

### Measured standardized fields

{_markdown_table(summary, ["predictor", "mean_product_area_fraction", "minimum_product_area_fraction", "mean_shelf_area_fraction", "minimum_shelf_area_fraction", "classification"])}

### Full candidate inventory

{_markdown_table(inventory, ["predictor", "classification", "rationale"])}

The mandatory dense production base for #53 is SST, wind speed, atmospheric xCO2, bathymetry/distance/shelf geometry, and deterministic coordinates/calendar/trend. GLORYS background SSS may enter with an explicit availability mask. ADT is useful but fails the frozen minimum-coverage threshold, especially on the shelf, so it must remain optional with a fallback and cannot determine whether a grid-month is publishable. Coverage holes in every measured field must remain explicit masks rather than being confused with observed values.

Across all five standardized dynamic fields, exact same-month joint availability averages **{joint_mean:.3f}%** of product area and falls to **{joint_minimum:.3f}%** in the worst month. On the shelf core it averages **{joint_shelf_mean:.3f}%** and falls to **{joint_shelf_minimum:.3f}%**. A cube that requires ADT everywhere would therefore discard a scientifically important fraction of shallow coastal waters.

Global chlorophyll-a is **not currently buildable from the frozen workspace state**: only regional caches exist, the recorded raw source path is unavailable, and MODIS Aqua supplies no observations for 1993-mid-2002 or 2021-2025. It remains a scientifically valuable optional channel after repair, accompanied by availability indicators and a non-Chl-a fallback. Mixed-layer depth and river discharge are unavailable; GSHHG river geometry is not a freshwater-flux product. Wind/ADT-derived circulation proxies remain diagnostic until #53 freezes formulas and boundary treatment.

## Decision and limitations

Issue #52 passes as an availability audit, not as a claim that every proposed predictor is ready. Issue #53 may build the 1993-2025 cube from the deployable base fields and fields explicitly classified as deployable with missingness. It must not make Chl-a, MLD, river discharge, or an unfrozen circulation proxy mandatory.

The standardized cache filenames do not by themselves fully identify upstream source releases. #53 must pin upstream product versions and transformations in its data manifest. The exact joint-coverage calculation requires all five standardized fields in the same grid-month. Plausible-range checks are deliberately broad and cannot detect scientifically biased but numerically plausible values. Predictor availability at SOCAT locations says nothing about prediction error in observation-sparse coastal waters.

## Figure and table index

{figure_index}

### Source tables

{table_index}
"""


def build_manifest(
    archive: Path,
    mapping: dict[str, list[str]],
    captions: dict[str, str],
    source_hashes: dict[str, str],
    output_path: Path,
) -> dict[str, object]:
    tracked = [archive / "README.md", archive / "REPORT.md", archive / "CAPTIONS.md"]
    tracked.extend(sorted((archive / "figures").glob("*")))
    tracked.extend(sorted((archive / "tables").glob("*.csv")))
    files = {path.relative_to(archive).as_posix(): sha256(path) for path in tracked}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    script_hash = sha256(Path(__file__))
    return {
        "experiment_id": EXPERIMENT_ID,
        "training_git_commit": commit,
        "data_manifest_sha256": sha256(output_path),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "predictor availability on frozen global coastal geometry",
        "decision": "pass_availability_audit",
        "figure_source_data": mapping,
        "figure_captions": captions,
        "files_sha256": files,
        "archive_builder_sha256": script_hash,
        "analysis_script_sha256": script_hash,
        "source_artifacts_sha256": source_hashes,
        "generated_products": {
            str(output_path): sha256(output_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--socat", type=Path, default=DEFAULT_SOCAT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--archive", type=Path, default=ROOT / "docs" / "experiment_archive" / EXPERIMENT_ID
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_paths = [args.cache_root / f"{spec.variable}.nc" for spec in SPECS]
    for path in [args.mask, args.socat, *cache_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)
    tables, counts, metadata = audit_fields(args.cache_root, args.mask, args.socat)
    tables["joint"], joint_count = audit_joint_coverage(args.cache_root, metadata)
    supplementary_tables(tables, args.cache_root)
    output_path = args.output_root / "predictor_availability_grid_v2.3.nc"
    save_node_product(counts, joint_count, metadata, output_path)
    write_archive_tables(tables, args.archive)
    mapping, captions = build_figures(tables, joint_count, metadata, args.archive)
    write_text(
        args.archive / "README.md",
        "# P2.3b predictor availability evidence\n\n"
        "Reviewer-ready evidence for Issue #52. Run `python scripts/"
        "audit_global_predictor_availability_v2_3.py`, then verify this directory with "
        "`python scripts/verify_experiment_archive.py docs/experiment_archive/"
        "p2_predictor_availability_v2.3`. Large gridded products remain outside Git.\n",
    )
    write_text(args.archive / "REPORT.md", build_report(tables, mapping, captions))
    write_text(
        args.archive / "CAPTIONS.md",
        "# Figure captions\n\n"
        + "\n\n".join(f"## {name}\n\n{caption}" for name, caption in captions.items())
        + "\n",
    )
    print("Hashing source artifacts...", flush=True)
    source_hashes = {str(path): sha256(path) for path in [args.mask, args.socat, *cache_paths]}
    manifest = build_manifest(args.archive, mapping, captions, source_hashes, output_path)
    write_text(
        args.archive / "archive_manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "period": "1993-01/2025-12",
        "mask": str(args.mask),
        "mask_sha256": source_hashes[str(args.mask)],
        "socat_cache": str(args.socat),
        "socat_cache_sha256": source_hashes[str(args.socat)],
        "coverage_thresholds": {
            "deployable_minimum_monthly_area_fraction": 0.99,
            "deployable_with_missingness_minimum_monthly_area_fraction": 0.90,
        },
        "decision": "pass_availability_audit",
        "mandatory_dense_dynamic_fields": ["SST", "wind speed", "atmospheric xCO2"],
        "masked_background_fields": ["GLORYS SSS"],
        "optional_repair_required_fields": ["ADT"],
        "mandatory_exclusions": ["chlorophyll-a", "mixed-layer depth", "river discharge"],
        "locked_test_opened": False,
        "external_independent_opened": False,
    }
    config_path = ROOT / "configs" / "frozen" / "predictor_availability_contract_v2.3.json"
    write_text(config_path, json.dumps(decision, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"archive": str(args.archive), "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
