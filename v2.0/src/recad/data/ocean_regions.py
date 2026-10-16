"""Auditable ocean-region provenance independent of product release status."""

from __future__ import annotations

import configparser
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import shape


@dataclass(frozen=True)
class OceanRegion:
    """One source polygon and its stable ReCAD reporting identifiers."""

    source_name: str
    canonical_name: str
    region_id: int
    macro_basin_id: int


@dataclass(frozen=True)
class OceanRegionContract:
    """Versioned source, geometry, and reporting definitions loaded from ``.in``."""

    contract_id: str
    schema_version: str
    source_title: str
    source_doi: str
    wfs_url: str
    raw_sha256: str
    simplification_tolerance_degrees: float
    simplified_sha256: str
    default_external_geometry: Path
    assignment_policy: str
    release_policy: str
    regions: tuple[OceanRegion, ...]
    macro_basins: dict[int, str]


@dataclass(frozen=True)
class OceanRegionAssignment:
    """Per-point region provenance with explicit ambiguity and missingness."""

    region_id: np.ndarray
    macro_basin_id: np.ndarray
    shared_boundary: np.ndarray
    unresolved: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required(parser: configparser.ConfigParser, section: str, key: str) -> str:
    if not parser.has_option(section, key):
        raise ValueError(f"missing [{section}] {key}")
    return parser.get(section, key)


def load_ocean_region_contract(path: str | Path) -> OceanRegionContract:
    """Load the frozen Global Oceans and Seas provenance contract."""

    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    with Path(path).open(encoding="utf-8") as handle:
        parser.read_file(handle)

    regions: list[OceanRegion] = []
    for section in parser.sections():
        if not section.startswith("region:"):
            continue
        source_name = section.removeprefix("region:")
        regions.append(
            OceanRegion(
                source_name=source_name,
                canonical_name=_required(parser, section, "canonical_name"),
                region_id=int(_required(parser, section, "region_id")),
                macro_basin_id=int(_required(parser, section, "macro_basin_id")),
            )
        )
    regions.sort(key=lambda region: region.region_id)
    if len(regions) != 10 or [region.region_id for region in regions] != list(range(10)):
        raise ValueError("ocean-region IDs must be the complete stable range 0..9")
    if len({region.source_name for region in regions}) != len(regions):
        raise ValueError("ocean-region source names must be unique")

    if not parser.has_section("macro_basins"):
        raise ValueError("missing [macro_basins] section")
    macro_basins = {int(identifier): name for identifier, name in parser.items("macro_basins")}
    referenced = {region.macro_basin_id for region in regions}
    if referenced != set(macro_basins):
        raise ValueError("macro-basin definitions must exactly cover referenced IDs")

    tolerance = float(_required(parser, "dataset", "simplification_tolerance_degrees"))
    if not 0 < tolerance <= 0.125:
        raise ValueError(
            "simplification tolerance must be positive and no larger than one grid cell"
        )
    return OceanRegionContract(
        contract_id=_required(parser, "dataset", "contract_id"),
        schema_version=_required(parser, "dataset", "schema_version"),
        source_title=_required(parser, "dataset", "source_title"),
        source_doi=_required(parser, "dataset", "source_doi"),
        wfs_url=_required(parser, "dataset", "wfs_url"),
        raw_sha256=_required(parser, "dataset", "raw_sha256"),
        simplification_tolerance_degrees=tolerance,
        simplified_sha256=_required(parser, "dataset", "simplified_sha256"),
        default_external_geometry=Path(_required(parser, "dataset", "default_external_geometry")),
        assignment_policy=_required(parser, "dataset", "assignment_policy"),
        release_policy=_required(parser, "dataset", "release_policy"),
        regions=tuple(regions),
        macro_basins=macro_basins,
    )


def load_region_geometries(
    path: str | Path,
    contract: OceanRegionContract,
    *,
    verify_hash: bool = True,
) -> tuple[object, ...]:
    """Load the ten source polygons in stable contract order."""

    source = Path(path)
    if verify_hash and sha256(source) != contract.simplified_sha256:
        raise ValueError("ocean-region geometry SHA256 does not match the frozen contract")
    payload = json.loads(source.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    by_name = {feature.get("properties", {}).get("name"): feature for feature in features}
    expected = {region.source_name for region in contract.regions}
    if set(by_name) != expected:
        missing = sorted(expected - set(by_name))
        extra = sorted(set(by_name) - expected)
        raise ValueError(f"ocean-region source names drifted; missing={missing}, extra={extra}")
    geometries = tuple(
        shape(by_name[region.source_name]["geometry"]) for region in contract.regions
    )
    if any(geometry.is_empty or not geometry.is_valid for geometry in geometries):
        raise ValueError("ocean-region source contains empty or invalid geometry")
    for geometry in geometries:
        shapely.prepare(geometry)
    return geometries


def assign_ocean_regions(
    longitude: np.ndarray,
    latitude: np.ndarray,
    geometries: tuple[object, ...],
    contract: OceanRegionContract,
) -> OceanRegionAssignment:
    """Assign points by polygon coverage without a longitude fallback."""

    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)
    if longitude.shape != latitude.shape:
        raise ValueError("longitude and latitude must have identical shapes")
    if len(geometries) != len(contract.regions):
        raise ValueError("geometry count does not match the region contract")
    if not np.all(np.isfinite(longitude) & np.isfinite(latitude)):
        raise ValueError("longitude and latitude must be finite")
    longitude_180 = (longitude + 180.0) % 360.0 - 180.0
    points = shapely.points(longitude_180.ravel(), latitude.ravel())
    hits = np.vstack(
        [np.asarray(shapely.covers(geometry, points), dtype=bool) for geometry in geometries]
    )
    hit_count = hits.sum(axis=0)
    first = hits.argmax(axis=0)
    unresolved = hit_count == 0
    shared_boundary = hit_count > 1
    region_id = np.full(points.shape, -1, dtype=np.int8)
    macro_basin_id = np.full(points.shape, -1, dtype=np.int8)
    for index, region in enumerate(contract.regions):
        selected = (first == index) & ~unresolved
        region_id[selected] = region.region_id
        macro_basin_id[selected] = region.macro_basin_id
    shape_out = longitude.shape
    return OceanRegionAssignment(
        region_id=region_id.reshape(shape_out),
        macro_basin_id=macro_basin_id.reshape(shape_out),
        shared_boundary=shared_boundary.reshape(shape_out),
        unresolved=unresolved.reshape(shape_out),
    )
