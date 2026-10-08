"""Global SOCAT coastal observation cache utilities.

The cache is indexed by cruise, coastal node, year, and month.  Cruise and
duplicate identities are assigned before target QC and before aggregation so
that every downstream split can keep an entire source group together.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

OBSERVATION_COLUMNS = (
    "Expocode",
    "version",
    "Source_DOI",
    "QC_Flag",
    "yr",
    "mon",
    "day",
    "hh",
    "mm",
    "ss",
    "longitude [dec.deg.E]",
    "latitude [dec.deg.N]",
    "sal",
    "fCO2rec [uatm]",
    "fCO2rec_flag",
)
BASIN_NAMES = {
    0: "Arctic",
    1: "Southern",
    2: "Atlantic",
    3: "Pacific",
    4: "Indian",
    5: "Mediterranean",
}


def normalize_group(value: object) -> str:
    """Return the source-independent cruise/duplicate key used by all splits."""
    if value is None or pd.isna(value):
        return "UNKNOWN"
    clean = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    return clean or "UNKNOWN"


def stable_fold(value: str, salt: str, folds: int = 5) -> int:
    """Map an identity to a deterministic fold without Python hash randomization."""
    digest = hashlib.sha256(f"{salt}|{value}".encode()).digest()[:8]
    return int(int.from_bytes(digest, "big") / 2**64 * folds) % folds


def nearest_indices(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Indices of nearest coordinates on a sorted one-dimensional grid."""
    indices = np.searchsorted(grid, values)
    indices = np.clip(indices, 1, len(grid) - 1)
    left = grid[indices - 1]
    move_left = np.abs(values - left) <= np.abs(values - grid[indices])
    return indices - move_left.astype(np.int64)


def coarse_basin(longitude_180: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    """Return neutral frozen basin IDs over the complete global grid."""
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


def observation_header(path: Path) -> int:
    """Locate the SOCAT observation header after its cruise metadata preamble."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            if line.startswith("Expocode\t") and "fCO2rec [uatm]" in line:
                return line_number
    raise RuntimeError(f"SOCAT observation header not found in {path}")


def platform_lookup(path: Path, stop_line: int | None = None) -> dict[str, str]:
    """Read Expocode-to-platform provenance from the SOCAT preamble."""
    stop = observation_header(path) if stop_line is None else stop_line
    lookup: dict[str, str] = {}
    header: list[str] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            if line_number >= stop:
                break
            fields = line.rstrip("\n\r").split("\t")
            if fields and fields[0] == "Expocode" and "Platform Name" in fields:
                header = fields
                continue
            if header is None or len(fields) < len(header):
                continue
            row = dict(zip(header, fields, strict=False))
            key = normalize_group(row.get("Expocode", ""))
            if key != "UNKNOWN":
                lookup[key] = row.get("Platform Name", "unknown") or "unknown"
    return lookup


@dataclass(frozen=True)
class CoastalGrid:
    """Frozen v2.2 coastal node mapping used by the v2.3 observation cache."""

    latitude: np.ndarray
    longitude: np.ndarray
    flat_to_node: np.ndarray
    node_latitude: np.ndarray
    node_longitude: np.ndarray
    lme_id: np.ndarray
    basin_id: np.ndarray
    regime_id: np.ndarray
    prepared_path: Path
    spatial_path: Path

    @classmethod
    def load(cls, prepared_path: Path, spatial_path: Path) -> CoastalGrid:
        with xr.open_dataset(prepared_path) as prepared:
            latitude = np.asarray(prepared["lat"].values, dtype=np.float64)
            longitude = np.asarray(prepared["lon"].values, dtype=np.float64)
            coastal = np.asarray(prepared["coastal_mask"].values, dtype=bool)
        with xr.open_dataset(spatial_path) as spatial:
            grid_flat = np.asarray(spatial["grid_flat"].values, dtype=np.int64)
            node_latitude = np.asarray(spatial["latitude"].values, dtype=np.float64)
            node_longitude = np.asarray(spatial["longitude"].values, dtype=np.float64)
            lme_id = np.asarray(spatial["lme_id"].values, dtype=np.int16)
            basin_id = np.asarray(spatial["basin_id"].values, dtype=np.int8)
            regime_id = np.asarray(spatial["regime_id"].values, dtype=np.int16)
        expected = np.flatnonzero(coastal.ravel())
        if not np.array_equal(grid_flat, expected):
            raise ValueError("v2.2 spatial support does not match the prepared coastal mask")
        flat_to_node = np.full(coastal.size, -1, dtype=np.int32)
        flat_to_node[grid_flat] = np.arange(len(grid_flat), dtype=np.int32)
        return cls(
            latitude,
            longitude,
            flat_to_node,
            node_latitude,
            node_longitude,
            lme_id,
            basin_id,
            regime_id,
            prepared_path,
            spatial_path,
        )

    @property
    def resolution_degrees(self) -> float:
        return float(np.median(np.diff(self.longitude)))

    def assign(self, longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
        """Return v2.2 legacy-mask nodes, or -1 outside that mask."""
        flat = self.assign_flat(longitude, latitude)
        return self.flat_to_node[flat]

    def assign_flat(self, longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
        """Return cells on the complete configured 1/8-degree global grid."""
        lat_index = nearest_indices(latitude, self.latitude)
        lon_index = nearest_indices(np.mod(longitude, 360.0), self.longitude)
        return lat_index * len(self.longitude) + lon_index


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _exact_sample_hash(frame: pd.DataFrame) -> np.ndarray:
    columns = [
        "group_key",
        "yr",
        "mon",
        "day",
        "hh",
        "mm",
        "ss",
        "longitude [dec.deg.E]",
        "latitude [dec.deg.N]",
        "sal",
        "fCO2rec [uatm]",
    ]
    return pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(np.uint64)


def build_global_socat_cache(
    source_path: Path,
    grid: CoastalGrid,
    *,
    chunksize: int = 750_000,
    start_year: int = 1993,
    core_end_year: int = 2025,
    data_end_year: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan SOCAT once and return cruise-node-month cache plus processing funnel."""
    header = observation_header(source_path)
    platforms = platform_lookup(source_path, header)
    pieces: list[pd.DataFrame] = []
    counts = {
        "source_observation_rows": 0,
        "valid_identity_space_time": 0,
        "within_1993_2026": 0,
        "after_exact_deduplication": 0,
        "at_least_one_qc_target": 0,
        "inside_complete_global_grid": 0,
        "inside_legacy_v2_2_product_mask": 0,
    }
    previous_hashes: set[int] = set()
    reader = pd.read_csv(
        source_path,
        sep="\t",
        skiprows=header,
        usecols=list(OBSERVATION_COLUMNS),
        dtype={"Expocode": str, "version": str, "Source_DOI": str, "QC_Flag": str},
        chunksize=chunksize,
        low_memory=False,
    )
    numeric = tuple(
        column
        for column in OBSERVATION_COLUMNS
        if column not in {"Expocode", "version", "Source_DOI", "QC_Flag"}
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        counts["source_observation_rows"] += len(chunk)
        _numeric(chunk, numeric)
        chunk["group_key"] = chunk["Expocode"].map(normalize_group)
        lon = chunk["longitude [dec.deg.E]"].to_numpy(float)
        lat = chunk["latitude [dec.deg.N]"].to_numpy(float)
        year = chunk["yr"].to_numpy(float)
        month = chunk["mon"].to_numpy(float)
        valid = (
            chunk["group_key"].ne("UNKNOWN").to_numpy()
            & np.isfinite(lon)
            & np.isfinite(lat)
            & (lat >= -90)
            & (lat <= 90)
            & np.isfinite(year)
            & np.isfinite(month)
            & (month >= 1)
            & (month <= 12)
        )
        chunk = chunk.loc[valid].copy()
        counts["valid_identity_space_time"] += len(chunk)
        period = chunk["yr"].between(start_year, data_end_year)
        chunk = chunk.loc[period].copy()
        counts["within_1993_2026"] += len(chunk)
        if chunk.empty:
            continue

        sample_hash = _exact_sample_hash(chunk)
        local_unique = ~pd.Series(sample_hash).duplicated().to_numpy()
        cross_unique = np.fromiter(
            (int(value) not in previous_hashes for value in sample_hash),
            dtype=bool,
            count=len(sample_hash),
        )
        keep = local_unique & cross_unique
        chunk = chunk.loc[keep].copy()
        sample_hash = sample_hash[keep]
        previous_hashes = {int(value) for value in sample_hash}
        counts["after_exact_deduplication"] += len(chunk)

        fco2 = chunk["fCO2rec [uatm]"].to_numpy(float)
        fco2_flag = chunk["fCO2rec_flag"].to_numpy(float)
        salinity = chunk["sal"].to_numpy(float)
        fco2_ok = np.isfinite(fco2) & (fco2 >= 1) & (fco2 <= 1000) & (fco2_flag <= 2)
        sss_ok = np.isfinite(salinity) & (salinity >= 0) & (salinity <= 50)
        any_target = fco2_ok | sss_ok
        chunk = chunk.loc[any_target].copy()
        fco2_ok = fco2_ok[any_target]
        sss_ok = sss_ok[any_target]
        counts["at_least_one_qc_target"] += len(chunk)
        if chunk.empty:
            continue

        inside_grid = (
            chunk["latitude [dec.deg.N]"]
            .between(float(grid.latitude.min()), float(grid.latitude.max()))
            .to_numpy()
        )
        chunk = chunk.loc[inside_grid].copy()
        fco2_ok = fco2_ok[inside_grid]
        sss_ok = sss_ok[inside_grid]
        counts["inside_complete_global_grid"] += len(chunk)
        if chunk.empty:
            continue

        grid_flat = grid.assign_flat(
            chunk["longitude [dec.deg.E]"].to_numpy(float),
            chunk["latitude [dec.deg.N]"].to_numpy(float),
        )
        frozen_node = grid.flat_to_node[grid_flat]
        chunk["grid_flat"] = grid_flat
        chunk["on_v2_2_product_mask"] = frozen_node >= 0
        counts["inside_legacy_v2_2_product_mask"] += int((frozen_node >= 0).sum())
        chunk["fco2_sum"] = np.where(fco2_ok, chunk["fCO2rec [uatm]"], 0.0)
        chunk["fco2_n"] = fco2_ok.astype(np.int32)
        chunk["sss_sum"] = np.where(sss_ok, chunk["sal"], 0.0)
        chunk["sss_n"] = sss_ok.astype(np.int32)
        chunk["sample_n"] = 1
        chunk["platform_name"] = chunk["group_key"].map(platforms).fillna("unknown")
        keys = [
            "group_key",
            "yr",
            "mon",
            "grid_flat",
            "on_v2_2_product_mask",
            "version",
            "Source_DOI",
            "QC_Flag",
            "platform_name",
        ]
        grouped = chunk.groupby(keys, as_index=False, dropna=False).agg(
            fco2_sum=("fco2_sum", "sum"),
            fco2_n=("fco2_n", "sum"),
            sss_sum=("sss_sum", "sum"),
            sss_n=("sss_n", "sum"),
            sample_n=("sample_n", "sum"),
        )
        pieces.append(grouped)
        print(
            f"SOCAT chunk {chunk_number}: scanned={counts['source_observation_rows']:,} "
            f"global_coastal_QC={counts['inside_complete_global_grid']:,}",
            flush=True,
        )

    if not pieces:
        raise RuntimeError("No SOCAT observations survived global coastal processing")
    cache = pd.concat(pieces, ignore_index=True)
    keys = ["group_key", "yr", "mon", "grid_flat"]
    provenance = cache.groupby(keys, as_index=False, dropna=False).agg(
        version=("version", "first"),
        source_doi=("Source_DOI", "first"),
        source_qc=("QC_Flag", "first"),
        platform_name=("platform_name", "first"),
        on_v2_2_product_mask=("on_v2_2_product_mask", "first"),
        provenance_variants=("Source_DOI", "nunique"),
        fco2_sum=("fco2_sum", "sum"),
        fco2_n=("fco2_n", "sum"),
        sss_sum=("sss_sum", "sum"),
        sss_n=("sss_n", "sum"),
        sample_n=("sample_n", "sum"),
    )
    flat = provenance["grid_flat"].to_numpy(int)
    lat_index = flat // len(grid.longitude)
    lon_index = flat % len(grid.longitude)
    frozen_node = grid.flat_to_node[flat]
    observed_flats = np.sort(provenance["grid_flat"].unique())
    observed_lookup = pd.Series(
        np.arange(len(observed_flats), dtype=np.int32), index=observed_flats
    )
    provenance["coastal_node"] = provenance["grid_flat"].map(observed_lookup).astype(np.int32)
    provenance["fco2"] = np.divide(
        provenance["fco2_sum"],
        provenance["fco2_n"],
        out=np.full(len(provenance), np.nan),
        where=provenance["fco2_n"].to_numpy() > 0,
    )
    provenance["salinity"] = np.divide(
        provenance["sss_sum"],
        provenance["sss_n"],
        out=np.full(len(provenance), np.nan),
        where=provenance["sss_n"].to_numpy() > 0,
    )
    provenance["latitude"] = grid.latitude[lat_index].astype(np.float32)
    provenance["longitude"] = grid.longitude[lon_index].astype(np.float32)
    lon180 = (provenance["longitude"].to_numpy(float) + 180) % 360 - 180
    safe_node = np.maximum(frozen_node, 0)
    provenance["lme_id"] = np.where(frozen_node >= 0, grid.lme_id[safe_node], -1)
    provenance["basin_id"] = coarse_basin(lon180, provenance["latitude"].to_numpy(float))
    provenance["regime_id"] = np.where(frozen_node >= 0, grid.regime_id[safe_node], -1)
    lat_block = np.floor((provenance["latitude"].to_numpy(float) + 90) / 5).astype(int)
    lon_block = np.floor((lon180 + 180) / 5).astype(int)
    provenance["region_key"] = np.where(
        provenance["lme_id"] >= 0,
        "LME-" + provenance["lme_id"].astype(str),
        "BASIN-"
        + provenance["basin_id"].astype(str)
        + "-GRID5-"
        + pd.Series(lat_block).astype(str)
        + ":"
        + pd.Series(lon_block).astype(str),
    )
    provenance["season"] = pd.cut(
        provenance["mon"],
        bins=[0, 2, 5, 8, 11, 12],
        labels=["DJF", "MAM", "JJA", "SON", "DJF"],
        ordered=False,
    ).astype(str)
    provenance["decade"] = (provenance["yr"] // 10 * 10).astype(int)
    provenance["record_status"] = np.where(
        provenance["yr"] <= core_end_year,
        "core_1993_2025",
        "provisional_2026",
    )
    provenance["source_release"] = "SOCATv2026_Coastal"
    provenance["target_provenance"] = "SOCAT in-situ sal; SOCAT recommended fCO2"
    diagnostic = "inside_legacy_v2_2_product_mask"
    pipeline = [name for name in counts if name != diagnostic]
    funnel = pd.DataFrame(
        {
            "stage": [*pipeline, "cruise_node_month_rows"],
            "rows": [*(counts[name] for name in pipeline), len(provenance)],
            "stage_kind": "ordered_filter_or_aggregation",
        }
    )
    funnel["retained_from_previous_fraction"] = funnel["rows"] / funnel["rows"].shift(1)
    funnel.loc[0, "retained_from_previous_fraction"] = 1.0
    funnel.loc[len(funnel)] = {
        "stage": diagnostic,
        "rows": counts[diagnostic],
        "stage_kind": "diagnostic_subset_not_filter",
        "retained_from_previous_fraction": np.nan,
    }
    return provenance, funnel


def build_group_manifest(cache: pd.DataFrame) -> pd.DataFrame:
    """Freeze cruise, spatial-block, LME-membership, and forward roles."""
    working = cache.copy()
    lon180 = (working["longitude"] + 180) % 360 - 180
    working["spatial_block"] = (
        np.floor((working["latitude"] + 90) / 5).astype(int).astype(str)
        + ":"
        + np.floor((lon180 + 180) / 5).astype(int).astype(str)
    )
    summary = working.groupby("group_key", as_index=False).agg(
        min_year=("yr", "min"),
        max_year=("yr", "max"),
        grid_month_rows=("coastal_node", "size"),
        fco2_rows=("fco2_n", lambda values: int((values > 0).sum())),
        sss_rows=("sss_n", lambda values: int((values > 0).sum())),
        platform_name=("platform_name", "first"),
        dominant_region=("region_key", lambda values: values.value_counts().index[0]),
        dominant_spatial_block=("spatial_block", lambda values: values.value_counts().index[0]),
        lme_memberships=(
            "lme_id",
            lambda values: ";".join(str(value) for value in sorted(set(values))),
        ),
    )
    summary["cruise_fold"] = (
        summary["group_key"]
        .map(lambda value: stable_fold(value, "recad-v2.3-global-cruise"))
        .astype(np.int8)
    )
    summary["spatial_fold"] = (
        summary["dominant_spatial_block"]
        .map(lambda value: stable_fold(value, "recad-v2.3-global-spatial"))
        .astype(np.int8)
    )
    summary["development_role"] = summary["group_key"].map(
        lambda value: (
            "grouped_development"
            if stable_fold(value, "recad-v2.3-global-development", 5) == 0
            else "train"
        )
    )
    summary["forward_role"] = np.select(
        [
            summary["max_year"] <= 2018,
            summary["max_year"] <= 2021,
            summary["max_year"] <= 2025,
        ],
        ["forward_train", "forward_development", "late_development"],
        default="provisional_2026",
    )
    summary["label_exposure"] = "development_accessible_in_SOCATv2026"
    return summary.sort_values("group_key").reset_index(drop=True)


def unit_xyz(longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    lon = np.deg2rad(longitude)
    lat = np.deg2rad(latitude)
    cosine = np.cos(lat)
    return np.column_stack((cosine * np.cos(lon), cosine * np.sin(lon), np.sin(lat)))


def build_node_support(cache: pd.DataFrame, grid: CoastalGrid) -> pd.DataFrame:
    """Quantify density and nearest training support for each observed global node."""
    del grid
    nodes = (
        cache[
            [
                "coastal_node",
                "grid_flat",
                "latitude",
                "longitude",
                "lme_id",
                "basin_id",
                "regime_id",
                "on_v2_2_product_mask",
            ]
        ]
        .drop_duplicates("coastal_node")
        .sort_values("coastal_node")
        .reset_index(drop=True)
    )
    for target, count_column in (("fco2", "fco2_n"), ("sss", "sss_n")):
        observed = cache.loc[cache[count_column] > 0]
        density = observed.groupby("coastal_node").agg(
            **{
                f"{target}_gridmonths": ("coastal_node", "size"),
                f"{target}_cruises": ("group_key", "nunique"),
            }
        )
        nodes = nodes.join(density, on="coastal_node")
        training = observed.loc[observed["training_eligible"]]
        support_nodes = training["coastal_node"].unique().astype(int)
        nearest = np.full(len(nodes), np.nan, dtype=np.float32)
        if len(support_nodes):
            by_node = nodes.set_index("coastal_node")
            tree = cKDTree(
                unit_xyz(
                    by_node.loc[support_nodes, "longitude"].to_numpy(),
                    by_node.loc[support_nodes, "latitude"].to_numpy(),
                )
            )
            chord, _ = tree.query(unit_xyz(nodes["longitude"], nodes["latitude"]), k=1)
            nearest = (6371.0088 * 2 * np.arcsin(np.clip(chord / 2, 0, 1))).astype(np.float32)
        nodes[f"nearest_{target}_support_km"] = nearest
    fill_columns = ["fco2_gridmonths", "fco2_cruises", "sss_gridmonths", "sss_cruises"]
    nodes[fill_columns] = nodes[fill_columns].fillna(0).astype(np.int32)
    return nodes
