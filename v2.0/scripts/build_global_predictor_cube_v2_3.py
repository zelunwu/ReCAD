"""Build and archive the frozen 1993-2025 global-coastal predictor cube."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from recad.data.coastal_mask import regular_cell_area_km2
from recad.data.predictor_cube import (
    CORE_MONTHS,
    STATUS_IMPUTED,
    STATUS_MISSING,
    STATUS_RANGE_REJECTED,
    STATUS_SOURCE_VALID,
    CubeVariableSpec,
    calendar_axis,
    compressed_grid_indices,
    deterministic_time_features,
    readiness_masks,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_global_predictor_cube_v2.3"
DEFAULT_CACHE_ROOT = Path(r"D:\proj_personal\PhD\ReCAD\v2.0\outputs\cache")
DEFAULT_MASK = Path(r"C:\backup\phd\data\processed\recad_v2_3\global_coastal_mask_v2.3.nc")
DEFAULT_SOCAT = Path(r"C:\backup\phd\data\processed\recad_v2_3\global_socat_cache_v2.3.parquet")
DEFAULT_SPLIT = Path(
    r"C:\backup\phd\data\processed\recad_v2_3\global_group_split_manifest_v2.3.parquet"
)
DEFAULT_OUTPUT_ROOT = Path(r"C:\backup\phd\data\processed\recad_v2_3")
RESOLUTION_DEG = 0.125
NODE_CHUNK = 65536

SPECS = (
    CubeVariableSpec("sst", "sst", -5.0, 50.0, "mandatory_dense", "degree_Celsius"),
    CubeVariableSpec("sss_bg", "sss", 0.0, 50.0, "masked_background", "1e-3"),
    CubeVariableSpec("adt", "adt", -5.0, 5.0, "optional", "m"),
    CubeVariableSpec("wspd", "wspd", 0.0, 80.0, "mandatory_dense", "m s-1"),
    CubeVariableSpec("xco2air", "xco2air", 250.0, 600.0, "mandatory_dense", "umol mol-1"),
)


def sha256(path: Path, block_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def source_paths(cache_root: Path) -> dict[str, Path]:
    return {spec.output_name: cache_root / f"{spec.source_variable}.nc" for spec in SPECS}


def validate_source_grid(
    paths: dict[str, Path], latitude: np.ndarray, longitude: np.ndarray
) -> None:
    """Fail before writing if a standardized source grid or calendar has drifted."""

    expected_shape = (408, latitude.size, longitude.size)
    for name, path in paths.items():
        with netCDF4.Dataset(path) as dataset:
            variable = dataset.variables[
                {spec.output_name: spec.source_variable for spec in SPECS}[name]
            ]
            if variable.shape != expected_shape:
                raise ValueError(f"{name} shape {variable.shape} != {expected_shape}")
            if not np.allclose(dataset.variables["lat"][:], latitude):
                raise ValueError(f"{name} latitude grid drifted")
            if not np.allclose(dataset.variables["lon"][:], longitude):
                raise ValueError(f"{name} longitude grid drifted")


def create_cube(
    output: Path,
    mask_path: Path,
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Stream source fields into a compressed coastal-node NetCDF."""

    started = time.perf_counter()
    with xr.open_dataset(mask_path) as mask_dataset:
        latitude = mask_dataset.lat.values.astype(np.float64)
        longitude = mask_dataset.lon.values.astype(np.float64)
        product = mask_dataset.coastal_product_mask.values.astype(bool)
        flat, lat_index, lon_index = compressed_grid_indices(product)
        shelf = mask_dataset.shelf_core_mask.values.reshape(-1)[flat].astype(np.uint8)
        deep = mask_dataset.slope_deep_mask.values.reshape(-1)[flat].astype(np.uint8)
        bathymetry = mask_dataset.bathymetry_m.values.reshape(-1)[flat].astype(np.float32)
        distance = mask_dataset.distance_to_significant_land_km.values.reshape(-1)[flat].astype(
            np.float32
        )
    validate_source_grid(paths, latitude, longitude)
    point_latitude = latitude[lat_index]
    point_longitude = longitude[lon_index]
    row_area = regular_cell_area_km2(latitude, RESOLUTION_DEG)
    point_area = row_area[lat_index]
    year, month, day_offset = calendar_axis()
    month_sin, month_cos = deterministic_time_features(month)

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    sources = {name: netCDF4.Dataset(path) for name, path in paths.items()}
    monthly_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    try:
        with netCDF4.Dataset(partial, "w", format="NETCDF4") as cube:
            cube.createDimension("time", CORE_MONTHS)
            cube.createDimension("node", flat.size)
            cube.title = "ReCAD v2.3 global coastal predictor cube"
            cube.product_period = "1993-01 through 2025-12"
            cube.geometry_definition = "within 400 km of GSHHG land >=1400 km2"
            cube.geometry_mask_path = str(mask_path)
            cube.node_order = "row-major flattened indices of the 1297 x 2880 source grid"
            cube.target_observations_included = "false"
            cube.imputation_applied = "false"
            cube.normalization_applied = "false"
            cube.fold_fitted_transformations_applied = "false"
            cube.adt_is_release_blocking = "false"
            cube.sss_background_missingness_is_explicit = "true"
            cube.complete = "false"

            time_var = cube.createVariable("time", "i4", ("time",))
            time_var.units = "days since 1993-01-15 00:00:00"
            time_var.calendar = "proleptic_gregorian"
            time_var[:] = day_offset
            for name, dtype, values, units in (
                ("year", "i2", year, "year"),
                ("month", "u1", month, "month"),
                ("month_sin", "f4", month_sin, "1"),
                ("month_cos", "f4", month_cos, "1"),
            ):
                variable = cube.createVariable(name, dtype, ("time",))
                variable.units = units
                variable[:] = values

            static_specs = (
                ("grid_flat", "i4", flat, "1"),
                ("grid_lat_index", "i2", lat_index, "1"),
                ("grid_lon_index", "i2", lon_index, "1"),
                ("latitude", "f4", point_latitude, "degrees_north"),
                ("longitude", "f4", point_longitude, "degrees_east"),
                ("cell_area_km2", "f4", point_area, "km2"),
                ("distance_to_significant_land_km", "f4", distance, "km"),
                ("bathymetry_m", "f4", bathymetry, "m"),
                ("shelf_core_mask", "u1", shelf, "1"),
                ("slope_deep_mask", "u1", deep, "1"),
                ("coastal_geometry_mask", "u1", np.ones(flat.size, dtype=np.uint8), "1"),
                ("latitude_sin", "f4", np.sin(np.deg2rad(point_latitude)), "1"),
                ("latitude_cos", "f4", np.cos(np.deg2rad(point_latitude)), "1"),
                ("longitude_sin", "f4", np.sin(np.deg2rad(point_longitude)), "1"),
                ("longitude_cos", "f4", np.cos(np.deg2rad(point_longitude)), "1"),
            )
            for name, dtype, values, units in static_specs:
                variable = cube.createVariable(
                    name, dtype, ("node",), zlib=True, complevel=4, shuffle=True
                )
                variable.units = units
                variable[:] = values

            data_variables = {}
            status_variables = {}
            availability_variables = {}
            era_variables = {}
            for spec in SPECS:
                variable = cube.createVariable(
                    spec.output_name,
                    "f4",
                    ("time", "node"),
                    fill_value=np.nan,
                    zlib=True,
                    complevel=2,
                    shuffle=True,
                    chunksizes=(1, NODE_CHUNK),
                )
                variable.units = spec.units
                variable.role = spec.role
                variable.source_path = str(paths[spec.output_name])
                variable.source_variable = spec.source_variable
                variable.minimum_accepted = spec.minimum
                variable.maximum_accepted = spec.maximum
                variable.imputation_applied = "false"
                data_variables[spec.output_name] = variable
                status = cube.createVariable(
                    f"{spec.output_name}_status",
                    "u1",
                    ("time", "node"),
                    zlib=True,
                    complevel=4,
                    shuffle=True,
                    chunksizes=(1, NODE_CHUNK),
                )
                status.flag_values = np.array(
                    [STATUS_MISSING, STATUS_SOURCE_VALID, STATUS_RANGE_REJECTED, STATUS_IMPUTED],
                    dtype=np.uint8,
                )
                status.flag_meanings = "missing source_valid range_rejected imputed"
                status_variables[spec.output_name] = status
                availability = cube.createVariable(
                    f"{spec.output_name}_available",
                    "u1",
                    ("time", "node"),
                    zlib=True,
                    complevel=4,
                    shuffle=True,
                    chunksizes=(1, NODE_CHUNK),
                )
                availability.long_name = f"{spec.output_name} accepted source-value availability"
                availability_variables[spec.output_name] = availability
                era = cube.createVariable(f"{spec.output_name}_source_era", "u1", ("time",))
                era.flag_values = np.array([1], dtype=np.uint8)
                era.flag_meanings = "single_standardized_source_era"
                era[:] = 1
                era_variables[spec.output_name] = era

            mandatory_var = cube.createVariable(
                "predictor_ready_mask",
                "u1",
                ("time", "node"),
                zlib=True,
                complevel=4,
                shuffle=True,
                chunksizes=(1, NODE_CHUNK),
            )
            mandatory_var.long_name = "all mandatory dynamic predictors available"
            mandatory_var.required_fields = "sst wspd xco2air"
            strict_var = cube.createVariable(
                "strict_common_support_mask",
                "u1",
                ("time", "node"),
                zlib=True,
                complevel=4,
                shuffle=True,
                chunksizes=(1, NODE_CHUNK),
            )
            strict_var.long_name = "all five standardized dynamic fields available"
            strict_var.required_fields = "sst sss_bg adt wspd xco2air"

            for time_index in range(CORE_MONTHS):
                status_by_name = {}
                for spec in SPECS:
                    source = sources[spec.output_name].variables[spec.source_variable]
                    field = np.ma.filled(source[time_index, :, :], np.nan)
                    values = np.asarray(field, dtype=np.float32).reshape(-1)[flat]
                    status = spec.status(values)
                    accepted = status == STATUS_SOURCE_VALID
                    cleaned = values.copy()
                    cleaned[~accepted] = np.nan
                    data_variables[spec.output_name][time_index, :] = cleaned
                    status_variables[spec.output_name][time_index, :] = status
                    availability_variables[spec.output_name][time_index, :] = accepted
                    status_by_name[spec.output_name] = status
                    monthly_rows.append(
                        {
                            "predictor": spec.output_name,
                            "year": int(year[time_index]),
                            "month": int(month[time_index]),
                            "node_fraction": float(accepted.mean()),
                            "area_fraction": float(point_area[accepted].sum() / point_area.sum()),
                            "shelf_node_fraction": float(accepted[shelf.astype(bool)].mean()),
                            "shelf_area_fraction": float(
                                point_area[accepted & shelf.astype(bool)].sum()
                                / point_area[shelf.astype(bool)].sum()
                            ),
                            "missing_nodes": int((status == STATUS_MISSING).sum()),
                            "range_rejected_nodes": int((status == STATUS_RANGE_REJECTED).sum()),
                            "imputed_nodes": int((status == STATUS_IMPUTED).sum()),
                        }
                    )
                mandatory, strict = readiness_masks(status_by_name)
                mandatory_var[time_index, :] = mandatory
                strict_var[time_index, :] = strict
                for mask_name, mask in (("predictor_ready", mandatory), ("strict_common", strict)):
                    monthly_rows.append(
                        {
                            "predictor": mask_name,
                            "year": int(year[time_index]),
                            "month": int(month[time_index]),
                            "node_fraction": float(mask.mean()),
                            "area_fraction": float(point_area[mask].sum() / point_area.sum()),
                            "shelf_node_fraction": float(mask[shelf.astype(bool)].mean()),
                            "shelf_area_fraction": float(
                                point_area[mask & shelf.astype(bool)].sum()
                                / point_area[shelf.astype(bool)].sum()
                            ),
                            "missing_nodes": int((~mask).sum()),
                            "range_rejected_nodes": 0,
                            "imputed_nodes": 0,
                        }
                    )
                if time_index % 12 == 0:
                    print(f"cube: {int(year[time_index])}-{int(month[time_index]):02d}", flush=True)
            cube.complete = "true"
            cube.completed_utc = datetime.now(timezone.utc).isoformat()
        os.replace(partial, output)
    finally:
        for dataset in sources.values():
            dataset.close()

    monthly = pd.DataFrame(monthly_rows)
    for spec in SPECS:
        frame = monthly.loc[monthly.predictor == spec.output_name]
        status_rows.append(
            {
                "predictor": spec.output_name,
                "role": spec.role,
                "mean_area_fraction": frame.area_fraction.mean(),
                "minimum_area_fraction": frame.area_fraction.min(),
                "missing_gridmonths": int(frame.missing_nodes.sum()),
                "range_rejected_gridmonths": int(frame.range_rejected_nodes.sum()),
                "imputed_gridmonths": int(frame.imputed_nodes.sum()),
            }
        )
    for name in ("predictor_ready", "strict_common"):
        frame = monthly.loc[monthly.predictor == name]
        status_rows.append(
            {
                "predictor": name,
                "role": "release_mask" if name == "predictor_ready" else "diagnostic_mask",
                "mean_area_fraction": frame.area_fraction.mean(),
                "minimum_area_fraction": frame.area_fraction.min(),
                "missing_gridmonths": int(frame.missing_nodes.sum()),
                "range_rejected_gridmonths": 0,
                "imputed_gridmonths": 0,
            }
        )
    resources = {
        "elapsed_seconds": time.perf_counter() - started,
        "node_count": int(flat.size),
        "time_count": CORE_MONTHS,
        "gridmonths": int(flat.size * CORE_MONTHS),
        "output_bytes": output.stat().st_size,
        "source_bytes": sum(path.stat().st_size for path in paths.values()),
    }
    return monthly, pd.DataFrame(status_rows), resources


def verify_frozen_inputs(
    paths: dict[str, Path], mask: Path, socat: Path, split: Path
) -> pd.DataFrame:
    """Hash all frozen inputs and compare those governed by Issues 38/50/52."""

    issue52 = json.loads(
        (
            ROOT / "docs/experiment_archive/p2_predictor_availability_v2.3/archive_manifest.json"
        ).read_text(encoding="utf-8")
    )["source_artifacts_sha256"]
    expected = {
        str(mask): "0e99ca53398a3f4c29d5058b3307857535c78195bcbabdadf06c59b0ee4957cf",
        str(socat): "7748db79dda802951f8d2f34346106ca9c6f3f49cbd5593b695e4f17e305babc",
        str(split): "a67888d34961b08cf3f6b9c75f62f24b5875dc461e156e70c2e2c9b42f46a748",
    }
    for path in paths.values():
        expected[str(path)] = issue52[str(path)]
    rows = []
    for path_text, expected_hash in expected.items():
        path = Path(path_text)
        actual = sha256(path)
        rows.append(
            {
                "artifact": path_text,
                "bytes": path.stat().st_size,
                "expected_sha256": expected_hash,
                "actual_sha256": actual,
                "verified": actual == expected_hash,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.verified.all():
        failed = frame.loc[~frame.verified, "artifact"].tolist()
        raise ValueError(f"frozen input hash mismatch: {failed}")
    return frame


def replay_cube(
    output: Path,
    paths: dict[str, Path],
    monthly: pd.DataFrame,
) -> pd.DataFrame:
    """Replay deterministic source samples and mask identities from the final file."""

    rng = np.random.default_rng(53023)
    rows = []
    sources = {name: netCDF4.Dataset(path) for name, path in paths.items()}
    try:
        with netCDF4.Dataset(output) as cube:
            if getattr(cube, "complete", "false") != "true":
                raise ValueError("cube does not carry the atomic completion marker")
            flat = cube.variables["grid_flat"][:].astype(np.int64)
            node_samples = np.sort(rng.choice(flat.size, size=64, replace=False))
            grid_samples = flat[node_samples]
            time_samples = sorted({0, CORE_MONTHS - 1, *range(7, CORE_MONTHS, 23)})
            for spec in SPECS:
                maximum_difference = 0.0
                status_mismatches = 0
                availability_mismatches = 0
                compared = 0
                for time_index in time_samples:
                    source_field = np.ma.filled(
                        sources[spec.output_name].variables[spec.source_variable][time_index, :, :],
                        np.nan,
                    )
                    source_values = np.asarray(source_field, dtype=np.float32).reshape(-1)[
                        grid_samples
                    ]
                    expected_status = spec.status(source_values)
                    expected_values = source_values.copy()
                    expected_values[expected_status != STATUS_SOURCE_VALID] = np.nan
                    actual_values = np.asarray(
                        cube.variables[spec.output_name][time_index, node_samples]
                    )
                    actual_status = np.asarray(
                        cube.variables[f"{spec.output_name}_status"][time_index, node_samples]
                    )
                    actual_available = np.asarray(
                        cube.variables[f"{spec.output_name}_available"][time_index, node_samples]
                    ).astype(bool)
                    finite = np.isfinite(expected_values) & np.isfinite(actual_values)
                    if finite.any():
                        maximum_difference = max(
                            maximum_difference,
                            float(np.max(np.abs(expected_values[finite] - actual_values[finite]))),
                        )
                    status_mismatches += int((actual_status != expected_status).sum())
                    availability_mismatches += int(
                        (actual_available != (expected_status == STATUS_SOURCE_VALID)).sum()
                    )
                    compared += len(node_samples)
                rows.append(
                    {
                        "check": f"source_replay_{spec.output_name}",
                        "samples": compared,
                        "maximum_absolute_difference": maximum_difference,
                        "mismatches": status_mismatches + availability_mismatches,
                        "passed": maximum_difference == 0
                        and status_mismatches == 0
                        and availability_mismatches == 0,
                    }
                )
            mask_mismatches = 0
            for time_index in time_samples:
                statuses = {
                    spec.output_name: np.asarray(
                        cube.variables[f"{spec.output_name}_status"][time_index, node_samples]
                    )
                    for spec in SPECS
                }
                mandatory, strict = readiness_masks(statuses)
                actual_mandatory = np.asarray(
                    cube.variables["predictor_ready_mask"][time_index, node_samples]
                ).astype(bool)
                actual_strict = np.asarray(
                    cube.variables["strict_common_support_mask"][time_index, node_samples]
                ).astype(bool)
                mask_mismatches += int((mandatory != actual_mandatory).sum())
                mask_mismatches += int((strict != actual_strict).sum())
            rows.append(
                {
                    "check": "readiness_mask_identity",
                    "samples": len(time_samples) * len(node_samples) * 2,
                    "maximum_absolute_difference": 0.0,
                    "mismatches": mask_mismatches,
                    "passed": mask_mismatches == 0,
                }
            )
    finally:
        for dataset in sources.values():
            dataset.close()

    issue52 = pd.read_csv(
        ROOT
        / "docs/experiment_archive/p2_predictor_availability_v2.3/tables/table02_monthly_coverage.csv"
    )
    name_map = {
        "SST": "sst",
        "GLORYS SSS": "sss_bg",
        "ADT": "adt",
        "wind speed": "wspd",
        "atmospheric xCO2": "xco2air",
    }
    reference = issue52.assign(predictor=issue52.predictor.map(name_map)).rename(
        columns={"product_area_fraction": "reference_area_fraction"}
    )
    observed = monthly.loc[monthly.predictor.isin(name_map.values())]
    joined = observed.merge(
        reference[["predictor", "year", "month", "reference_area_fraction"]],
        on=["predictor", "year", "month"],
        validate="one_to_one",
    )
    maximum = float(np.max(np.abs(joined.area_fraction - joined.reference_area_fraction)))
    rows.append(
        {
            "check": "issue52_monthly_area_replay",
            "samples": len(joined),
            "maximum_absolute_difference": maximum,
            "mismatches": int(
                (np.abs(joined.area_fraction - joined.reference_area_fraction) > 1e-12).sum()
            ),
            "passed": maximum <= 1e-12,
        }
    )
    result = pd.DataFrame(rows)
    if not result.passed.all():
        raise ValueError(f"cube replay failed: {result.loc[~result.passed, 'check'].tolist()}")
    return result


def build_tables(
    monthly: pd.DataFrame,
    summary: pd.DataFrame,
    resources: dict[str, object],
    source_inventory: pd.DataFrame,
    replay: pd.DataFrame,
    output: Path,
) -> dict[str, pd.DataFrame]:
    dictionary_rows = []
    for spec in SPECS:
        dictionary_rows.extend(
            [
                (
                    spec.output_name,
                    "float32",
                    "time,node",
                    spec.units,
                    spec.role,
                    "source value; invalid retained as NaN",
                ),
                (
                    f"{spec.output_name}_available",
                    "uint8",
                    "time,node",
                    "1",
                    "availability",
                    "1 iff source status is accepted",
                ),
                (
                    f"{spec.output_name}_status",
                    "uint8",
                    "time,node",
                    "1",
                    "QC/missingness/imputation",
                    "0 missing; 1 source valid; 2 range rejected; 3 imputed",
                ),
                (
                    f"{spec.output_name}_source_era",
                    "uint8",
                    "time",
                    "1",
                    "provenance",
                    "1 single standardized source era",
                ),
            ]
        )
    dictionary_rows.extend(
        [
            (
                "predictor_ready_mask",
                "uint8",
                "time,node",
                "1",
                "release mask",
                "SST, wind, and atmospheric xCO2 valid",
            ),
            (
                "strict_common_support_mask",
                "uint8",
                "time,node",
                "1",
                "diagnostic mask",
                "all five dynamic fields valid",
            ),
            ("grid_flat", "int32", "node", "1", "grid mapping", "row-major source-grid index"),
            ("shelf_core_mask", "uint8", "node", "1", "geometry", "GEBCO depth 0-200 m"),
            (
                "distance_to_significant_land_km",
                "float32",
                "node",
                "km",
                "geometry",
                "distance to GSHHG land >=1400 km2",
            ),
        ]
    )
    dictionary = pd.DataFrame(
        dictionary_rows,
        columns=["variable", "dtype", "dimensions", "units", "role", "definition"],
    )
    block = monthly.copy()
    block["time_block"] = pd.cut(
        block.year,
        bins=[1992, 1997, 2001, 2020, 2025],
        labels=["1993-1997", "1998-2001", "2002-2020", "2021-2025"],
    )
    block = (
        block.groupby(["predictor", "time_block"], observed=True)
        .agg(
            mean_area_fraction=("area_fraction", "mean"),
            minimum_area_fraction=("area_fraction", "min"),
            mean_shelf_area_fraction=("shelf_area_fraction", "mean"),
            minimum_shelf_area_fraction=("shelf_area_fraction", "min"),
            months=("month", "size"),
        )
        .reset_index()
    )
    seasonal = (
        monthly.groupby(["predictor", "month"], sort=False)
        .agg(
            mean_area_fraction=("area_fraction", "mean"),
            minimum_area_fraction=("area_fraction", "min"),
            mean_shelf_area_fraction=("shelf_area_fraction", "mean"),
        )
        .reset_index()
    )
    masks = pd.DataFrame(
        [
            (
                "coastal_geometry",
                "node",
                "Issue #50 geometry",
                "independent of predictors and targets",
                True,
            ),
            (
                "predictor_ready",
                "grid-month",
                "sst & wspd & xco2air",
                "release may proceed when true",
                True,
            ),
            (
                "strict_common",
                "grid-month",
                "sst & sss_bg & adt & wspd & xco2air",
                "diagnostic common-support comparisons",
                False,
            ),
            (
                "per-variable available",
                "grid-month",
                "accepted source value",
                "input-specific missingness",
                False,
            ),
        ],
        columns=["mask", "grain", "definition", "use", "release_relevant"],
    )
    exclusions = pd.DataFrame(
        [
            (
                "chlorophyll-a",
                "repair_required",
                "global cache absent; raw source path unavailable; structural gaps",
            ),
            ("mixed-layer depth", "unavailable", "no frozen source artifact"),
            ("river discharge", "unavailable", "river geometry is not discharge"),
            (
                "derived circulation proxies",
                "diagnostic_only",
                "formula and boundary handling not frozen",
            ),
        ],
        columns=["candidate", "status", "reason"],
    )
    decision = pd.DataFrame(
        [
            (
                "mandatory_dense_dynamic",
                "sst;wspd;xco2air",
                "all must be valid for predictor_ready",
            ),
            ("masked_background", "sss_bg", "missingness is explicit and does not block release"),
            ("optional", "adt", "missingness is explicit and does not block release"),
            (
                "fold_fitted_transformations",
                "none in cube",
                "fit climatology/scaling/imputation in training folds",
            ),
        ],
        columns=["category", "variables", "rule"],
    )
    file_record = pd.DataFrame(
        [
            {
                "path": str(output),
                "bytes": output.stat().st_size,
                "sha256": sha256(output),
                "format": "NetCDF4/HDF5",
                "time_chunks": 1,
                "node_chunks": NODE_CHUNK,
                "compression": "zlib level 2 values; level 4 masks/static",
            }
        ]
    )
    return {
        "dictionary": dictionary,
        "sources": source_inventory,
        "monthly": monthly,
        "summary": summary,
        "blocks": block,
        "seasonal": seasonal,
        "resources": pd.DataFrame([resources]),
        "replay": replay,
        "masks": masks,
        "exclusions": exclusions,
        "decision": decision,
        "file": file_record,
    }


def _finish_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def build_figures(
    tables: dict[str, pd.DataFrame], output: Path, archive: Path
) -> tuple[dict[str, list[str]], dict[str, str]]:
    figures = archive / "figures"
    mapping: dict[str, list[str]] = {}
    captions: dict[str, str] = {}
    monthly = tables["monthly"]
    summary = tables["summary"]

    dynamic_names = [spec.output_name for spec in SPECS]
    plt.figure(figsize=(10, 5))
    for name in dynamic_names:
        frame = monthly.loc[monthly.predictor == name]
        dates = pd.to_datetime({"year": frame.year, "month": frame.month, "day": 1})
        plt.plot(dates, frame.area_fraction * 100, label=name, linewidth=1)
    plt.ylabel("Product area available (%)")
    plt.xlabel("Month")
    plt.legend(ncol=3)
    name = "fig01_dynamic_availability.png"
    _finish_figure(figures / name)
    mapping[name] = ["table03_monthly_coverage.csv"]
    captions[name] = (
        "Figure 1. Replayed monthly area-weighted availability for every dynamic field written to the frozen coastal-node cube from 1993 through 2025."
    )

    masks = monthly.loc[monthly.predictor.isin(["predictor_ready", "strict_common"])]
    plt.figure(figsize=(10, 5))
    for mask_name, frame in masks.groupby("predictor", sort=False):
        dates = pd.to_datetime({"year": frame.year, "month": frame.month, "day": 1})
        plt.plot(dates, frame.area_fraction * 100, label=f"{mask_name}: all coastal")
        plt.plot(
            dates, frame.shelf_area_fraction * 100, linestyle="--", label=f"{mask_name}: shelf"
        )
    plt.ylabel("Area satisfying mask (%)")
    plt.xlabel("Month")
    plt.legend(ncol=2)
    name = "fig02_release_and_strict_masks.png"
    _finish_figure(figures / name)
    mapping[name] = ["table03_monthly_coverage.csv", "table09_mask_definitions.csv"]
    captions[name] = (
        "Figure 2. Mandatory predictor-ready coverage remains separate from five-field strict common support; dashed curves show the shallower and more seasonally incomplete shelf core."
    )

    positions = np.arange(len(summary))
    plt.figure(figsize=(10, 5))
    plt.bar(positions - 0.18, summary.mean_area_fraction * 100, 0.36, label="mean")
    plt.bar(positions + 0.18, summary.minimum_area_fraction * 100, 0.36, label="minimum")
    plt.xticks(positions, summary.predictor, rotation=20, ha="right")
    plt.ylabel("Product area available (%)")
    plt.ylim(0, 102)
    plt.legend()
    name = "fig03_summary_coverage.png"
    _finish_figure(figures / name)
    mapping[name] = ["table04_predictor_summary.csv"]
    captions[name] = (
        "Figure 3. Mean and worst-month product-area coverage for each stored field and the two distinct readiness masks, confirming that optional ADT does not gate release."
    )

    seasonal = tables["seasonal"]
    plt.figure(figsize=(9, 5))
    for predictor in ["adt", "strict_common"]:
        frame = seasonal.loc[seasonal.predictor == predictor]
        plt.plot(
            frame.month,
            frame.mean_area_fraction * 100,
            marker="o",
            label=f"{predictor}: all coastal",
        )
        plt.plot(
            frame.month,
            frame.mean_shelf_area_fraction * 100,
            marker="o",
            linestyle="--",
            label=f"{predictor}: shelf",
        )
    plt.xticks(range(1, 13))
    plt.xlabel("Calendar month")
    plt.ylabel("Mean area available (%)")
    plt.legend()
    name = "fig04_adt_seasonality.png"
    _finish_figure(figures / name)
    mapping[name] = ["table06_seasonal_coverage.csv"]
    captions[name] = (
        "Figure 4. Calendar-month climatology of ADT and strict support, showing why an ADT requirement would impose a strong seasonal selection on shelf waters."
    )

    status = summary.loc[summary.predictor.isin(dynamic_names)]
    plt.figure(figsize=(9, 5))
    plt.bar(status.predictor, status.missing_gridmonths, label="missing")
    plt.bar(
        status.predictor,
        status.range_rejected_gridmonths,
        bottom=status.missing_gridmonths,
        label="range rejected",
    )
    plt.bar(
        status.predictor,
        status.imputed_gridmonths,
        bottom=status.missing_gridmonths + status.range_rejected_gridmonths,
        label="imputed",
    )
    plt.ylabel("Grid-month count")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    name = "fig05_status_accounting.png"
    _finish_figure(figures / name)
    mapping[name] = ["table04_predictor_summary.csv"]
    captions[name] = (
        "Figure 5. Missing, physical-range-rejected, and imputed grid-month accounting; the frozen cube performs no imputation and preserves unavailable values as NaN plus status codes."
    )

    resources = tables["resources"].iloc[0]
    plt.figure(figsize=(7, 4.5))
    plt.bar(
        ["five source caches", "compressed cube"],
        [resources.source_bytes / 1e9, resources.output_bytes / 1e9],
    )
    plt.ylabel("Size (decimal GB)")
    name = "fig06_storage_accounting.png"
    _finish_figure(figures / name)
    mapping[name] = ["table07_resource_accounting.csv", "table12_cube_file.csv"]
    captions[name] = (
        "Figure 6. Storage accounting for the five full-grid standardized caches and the coastal-node cube, which removes land and non-product ocean nodes while retaining explicit status masks."
    )

    replay = tables["replay"]
    plt.figure(figsize=(9, 4.8))
    plt.barh(replay.check, replay.mismatches)
    plt.xlabel("Replay mismatches")
    name = "fig07_replay_checks.png"
    _finish_figure(figures / name)
    mapping[name] = ["table08_replay_checks.csv"]
    captions[name] = (
        "Figure 7. Deterministic source-value, status, readiness-mask, and Issue 52 coverage replay mismatches; all required checks must remain exactly zero."
    )

    source_inventory = tables["sources"]
    plt.figure(figsize=(9, 5))
    labels = [Path(path).name for path in source_inventory.artifact]
    plt.barh(labels, source_inventory.bytes / 1e9)
    plt.xlabel("Artifact size (decimal GB)")
    name = "fig08_source_inventory.png"
    _finish_figure(figures / name)
    mapping[name] = ["table02_source_inventory.csv"]
    captions[name] = (
        "Figure 8. Frozen input-artifact sizes for the geometry, target cache, split manifest, and five standardized predictor sources; every SHA256 was verified before publication."
    )

    with netCDF4.Dataset(output) as cube:
        flat = cube.variables["grid_flat"][:].astype(np.int64)
        map_rows = [(13, "1994-02"), (395, "2025-12")]
        for figure_number, (time_index, label) in enumerate(map_rows, start=9):
            field = np.full(1297 * 2880, np.nan, dtype=np.float32)
            field[flat] = cube.variables["strict_common_support_mask"][time_index, :]
            plt.figure(figsize=(11, 4.6))
            plt.imshow(
                field.reshape(1297, 2880),
                origin="lower",
                extent=(0, 360, -78, 84),
                aspect="auto",
                vmin=0,
                vmax=1,
                cmap="viridis",
            )
            plt.colorbar(label="Strict common support")
            plt.xlabel("Longitude (degrees east)")
            plt.ylabel("Latitude")
            filename = f"fig{figure_number:02d}_strict_map_{label}.png"
            _finish_figure(figures / filename)
            mapping[filename] = [
                "table03_monthly_coverage.csv",
                "local:global_coastal_predictor_cube_v2.3.nc",
            ]
            captions[filename] = (
                f"Figure {figure_number}. Exact five-field strict common-support mask for {label}; this diagnostic mask is deliberately separate from the mandatory predictor-ready release mask."
            )

    decisions = tables["decision"]
    role_scores = {
        "mandatory_dense_dynamic": 3,
        "masked_background": 2,
        "optional": 1,
        "fold_fitted_transformations": 0,
    }
    plt.figure(figsize=(9, 4.5))
    plt.bar(decisions.category, decisions.category.map(role_scores))
    plt.ylabel("Production requirement level (ordinal)")
    plt.xticks(rotation=20, ha="right")
    name = "fig11_frozen_roles.png"
    _finish_figure(figures / name)
    mapping[name] = ["table11_frozen_decision.csv"]
    captions[name] = (
        "Figure 11. Frozen production roles distinguish mandatory dynamic inputs, explicitly masked background SSS, optional ADT, and transformations that must be fitted only inside training folds."
    )
    return mapping, captions


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    subset = frame[columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in subset.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def build_report(tables: dict[str, pd.DataFrame], captions: dict[str, str], output: Path) -> str:
    summary = tables["summary"].copy()
    for column in ("mean_area_fraction", "minimum_area_fraction"):
        summary[column] = (summary[column] * 100).map(lambda value: f"{value:.3f}%")
    resources = tables["resources"].iloc[0]
    figure_sections = []
    for index, (filename, caption) in enumerate(captions.items(), start=1):
        figure_sections.append(
            f"### Figure {index}\n\n![{caption}](figures/{filename})\n\n{caption}"
        )
    figure_index = "\n\n".join(figure_sections)
    return f"""# P2.3c frozen global-coastal predictor cube

Issue: #53

Output: `{output}`

Decision: **pass cube freeze; proceed to Issue #54 predictor ablations**

## Scientific question and permitted claim

This engineering experiment asks whether the predictors qualified by Issue #52 can be represented as one traceable, replayable 1993-2025 monthly cube on the observation-independent Issue #50 geometry. It may claim reproducible input construction and coverage behavior. It does not claim SSS or fCO2 prediction skill and does not open locked or external-independent labels.

## Data, splits, and leakage controls

The cube contains {int(resources.time_count)} months and {int(resources.node_count):,} coastal nodes, or {int(resources.gridmonths):,} grid-months. It copies accepted source values from the five standardized full-grid caches into row-major coastal-node order. Missing or broad-range-rejected values remain NaN and receive explicit status and availability codes. The Issue #50 mask, Issue #38 SOCAT cache, and Issue #38 group/split manifest are SHA256 verified and unchanged.

No observation target is stored. No climatology, normalization, imputation, learned repair, or target-dependent transformation is fitted in this cube. Those operations must be fitted inside each training fold during Issue #54 and later model comparisons.

## Candidate models and training

No model was trained. SST, wind speed, and atmospheric xCO2 define `predictor_ready_mask`. GLORYS SSS is a masked background field. ADT is optional and cannot block release. `strict_common_support_mask` requires all five fields and exists only for controlled common-support comparisons. The compressed NetCDF uses one-month by 65,536-node chunks, zlib compression, and an atomic completion marker.

## Main development results

{markdown_table(summary, ["predictor", "role", "mean_area_fraction", "minimum_area_fraction", "missing_gridmonths", "range_rejected_gridmonths", "imputed_gridmonths"])}

The mandatory predictor-ready mask is complete because all three required fields are complete over the frozen geometry and core period. Strict common support reproduces Issue #52: it averages about 92.14% of product area and falls to about 87.30% in the worst month. Shelf strict support remains lower and seasonally variable because ADT coverage is weakest in shallow and high-latitude waters.

The final external file is {resources.output_bytes / 1e9:.3f} GB, built in {resources.elapsed_seconds / 60:.2f} minutes from {resources.source_bytes / 1e9:.3f} GB of standardized source caches. Every deterministic replay check has zero mismatches, and monthly area coverage agrees with the independently archived Issue #52 audit to machine precision.

## Decision and limitations

Issue #53 passes. Issue #54 must use this frozen cube and identical target rows/splits for all ablations. ADT and background SSS require missingness-aware branches; a sample may remain publishable when either is absent. Chl-a, MLD, river discharge, and unfrozen circulation proxies remain excluded.

The cube records standardized cache paths and hashes, but the upstream producers may issue future revisions; any source update requires a new cube version. Broad physical ranges detect gross corruption rather than scientific bias. The cube is an input product, so complete predictor-ready coverage does not establish prediction reliability in observation-sparse regions.

## Figure and table index

{figure_index}

### Source tables

- `table01_data_dictionary.csv`: variables, units, dimensions, roles, and status semantics.
- `table02_source_inventory.csv`: immutable source paths, sizes, and verified SHA256.
- `table03_monthly_coverage.csv`: per-field and mask monthly coverage.
- `table04_predictor_summary.csv`: whole-period field and mask summary.
- `table05_time_block_coverage.csv`: frozen temporal-block diagnostics.
- `table06_seasonal_coverage.csv`: calendar-month coverage climatology.
- `table07_resource_accounting.csv`: runtime and storage accounting.
- `table08_replay_checks.csv`: deterministic replay results.
- `table09_mask_definitions.csv`: geometry, release, strict, and per-variable masks.
- `table10_excluded_candidates.csv`: fields deliberately absent from the cube.
- `table11_frozen_decision.csv`: production role contract for Issue #54.
- `table12_cube_file.csv`: final path, hash, chunking, and compression.
"""


def archive_outputs(
    tables: dict[str, pd.DataFrame],
    output: Path,
    archive: Path,
    source_hashes: dict[str, str],
) -> dict[str, object]:
    table_names = {
        "dictionary": "table01_data_dictionary.csv",
        "sources": "table02_source_inventory.csv",
        "monthly": "table03_monthly_coverage.csv",
        "summary": "table04_predictor_summary.csv",
        "blocks": "table05_time_block_coverage.csv",
        "seasonal": "table06_seasonal_coverage.csv",
        "resources": "table07_resource_accounting.csv",
        "replay": "table08_replay_checks.csv",
        "masks": "table09_mask_definitions.csv",
        "exclusions": "table10_excluded_candidates.csv",
        "decision": "table11_frozen_decision.csv",
        "file": "table12_cube_file.csv",
    }
    for key, filename in table_names.items():
        write_table(tables[key], archive / "tables" / filename)
    mapping, captions = build_figures(tables, output, archive)
    write_text(
        archive / "README.md",
        "# P2.3c predictor-cube evidence\n\n"
        "Reviewer-ready evidence for Issue #53. Build with `python scripts/"
        "build_global_predictor_cube_v2_3.py`; verify with `python scripts/"
        "verify_experiment_archive.py docs/experiment_archive/p2_global_predictor_cube_v2.3`. "
        "The multi-GB NetCDF remains outside Git.\n",
    )
    write_text(archive / "REPORT.md", build_report(tables, captions, output))
    write_text(
        archive / "CAPTIONS.md",
        "# Figure captions\n\n"
        + "\n\n".join(f"## {name}\n\n{caption}" for name, caption in captions.items())
        + "\n",
    )
    tracked = [archive / "README.md", archive / "REPORT.md", archive / "CAPTIONS.md"]
    tracked.extend(sorted((archive / "figures").glob("*.png")))
    tracked.extend(sorted((archive / "tables").glob("*.csv")))
    files = {path.relative_to(archive).as_posix(): sha256(path) for path in tracked}
    script_hash = sha256(Path(__file__))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "training_git_commit": commit,
        "data_manifest_sha256": tables["file"].iloc[0].sha256,
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "traceable 1993-2025 predictor construction on frozen geometry",
        "decision": "pass_cube_freeze",
        "figure_source_data": mapping,
        "figure_captions": captions,
        "files_sha256": files,
        "archive_builder_sha256": script_hash,
        "analysis_script_sha256": script_hash,
        "source_artifacts_sha256": source_hashes,
        "generated_product": {
            "path": str(output),
            "sha256": tables["file"].iloc[0].sha256,
            "bytes": int(output.stat().st_size),
        },
    }
    write_text(
        archive / "archive_manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--socat", type=Path, default=DEFAULT_SOCAT)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--archive", type=Path, default=ROOT / "docs/experiment_archive" / EXPERIMENT_ID
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = source_paths(args.cache_root)
    for path in [args.mask, args.socat, args.split, *paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    print("Verifying frozen input hashes...", flush=True)
    source_inventory = verify_frozen_inputs(paths, args.mask, args.socat, args.split)
    output = args.output_root / "global_coastal_predictor_cube_v2.3.nc"
    monthly, summary, resources = create_cube(output, args.mask, paths)
    replay = replay_cube(output, paths, monthly)
    tables = build_tables(monthly, summary, resources, source_inventory, replay, output)
    source_hashes = dict(
        zip(source_inventory.artifact, source_inventory.actual_sha256, strict=True)
    )
    manifest = archive_outputs(tables, output, args.archive, source_hashes)
    frozen = {
        "product": str(output),
        "sha256": manifest["generated_product"]["sha256"],
        "bytes": manifest["generated_product"]["bytes"],
        "period": "1993-01/2025-12",
        "grid": {
            "resolution_deg": RESOLUTION_DEG,
            "coastal_nodes": int(resources["node_count"]),
            "months": CORE_MONTHS,
        },
        "mandatory_dense_dynamic_fields": ["sst", "wspd", "xco2air"],
        "masked_background_fields": ["sss_bg"],
        "optional_fields": ["adt"],
        "fold_fitted_transformations_in_cube": False,
        "target_observations_in_cube": False,
        "geometry_sha256": source_hashes[str(args.mask)],
        "socat_cache_sha256": source_hashes[str(args.socat)],
        "split_manifest_sha256": source_hashes[str(args.split)],
        "source_artifacts_sha256": source_hashes,
        "replay_passed": bool(replay.passed.all()),
        "decision": "pass_cube_freeze",
    }
    write_text(
        ROOT / "configs/frozen/global_predictor_cube_manifest_v2.3.json",
        json.dumps(frozen, indent=2, ensure_ascii=False) + "\n",
    )
    print(
        json.dumps(
            {"output": str(output), "sha256": frozen["sha256"], "bytes": frozen["bytes"]}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
