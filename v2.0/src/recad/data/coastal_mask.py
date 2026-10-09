"""Observation-independent coastal product-mask construction.

The product geometry is intentionally separate from observational support and
predictor availability.  Distances are supplied by a versioned external grid;
this module only applies the frozen, testable geometric contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class CoastalMaskSpec:
    """Frozen geometric definition for the ReCAD coastal product."""

    maximum_distance_km: float = 400.0
    significant_island_area_km2: float = 1400.0
    shelf_depth_m: float = 200.0

    def __post_init__(self) -> None:
        if self.maximum_distance_km <= 0:
            raise ValueError("maximum_distance_km must be positive")
        if self.significant_island_area_km2 < 0:
            raise ValueError("significant_island_area_km2 cannot be negative")
        if self.shelf_depth_m <= 0:
            raise ValueError("shelf_depth_m must be positive")


def regular_cell_area_km2(latitude: np.ndarray, resolution_deg: float) -> np.ndarray:
    """Return spherical cell areas for a regular longitude/latitude mesh.

    The returned array has one value per latitude row.  Latitude cells are
    clipped at the poles, so this also works for endpoint-centred grids.
    """

    latitude = np.asarray(latitude, dtype=np.float64)
    if latitude.ndim != 1:
        raise ValueError("latitude must be one-dimensional")
    if resolution_deg <= 0:
        raise ValueError("resolution_deg must be positive")
    half = np.deg2rad(resolution_deg / 2.0)
    south = np.maximum(np.deg2rad(latitude) - half, -np.pi / 2)
    north = np.minimum(np.deg2rad(latitude) + half, np.pi / 2)
    delta_lon = np.deg2rad(resolution_deg)
    return EARTH_RADIUS_KM**2 * delta_lon * (np.sin(north) - np.sin(south))


def build_coastal_mask_dataset(
    longitude: np.ndarray,
    latitude: np.ndarray,
    distance_all_land_km: np.ndarray,
    distance_significant_land_km: np.ndarray,
    *,
    bathymetry_m: np.ndarray | None = None,
    spec: CoastalMaskSpec | None = None,
) -> xr.Dataset:
    """Build product and shelf masks without consulting target observations.

    Signed distance fields must be negative on land and positive over ocean.
    The all-land field defines the physical ocean mask; the significant-land
    field defines the 400-km product extent.  This prevents excluded small
    islands from being misclassified as ocean.
    """

    spec = spec or CoastalMaskSpec()
    longitude = np.asarray(longitude, dtype=np.float64)
    latitude = np.asarray(latitude, dtype=np.float64)
    expected = (latitude.size, longitude.size)
    distance_all = np.asarray(distance_all_land_km, dtype=np.float64)
    distance_significant = np.asarray(distance_significant_land_km, dtype=np.float64)
    if longitude.ndim != 1 or latitude.ndim != 1:
        raise ValueError("longitude and latitude must be one-dimensional")
    if distance_all.shape != expected or distance_significant.shape != expected:
        raise ValueError(f"distance fields must have shape {expected}")

    ocean = np.isfinite(distance_all) & (distance_all > 0)
    coastal = (
        ocean
        & np.isfinite(distance_significant)
        & (distance_significant >= 0)
        & (distance_significant <= spec.maximum_distance_km)
    )

    data_vars: dict[str, tuple[tuple[str, str], np.ndarray, dict[str, object]]] = {
        "ocean_mask": (
            ("lat", "lon"),
            ocean.astype(np.int8),
            {"long_name": "ocean according to the all-land GSHHG distance sign"},
        ),
        "coastal_product_mask": (
            ("lat", "lon"),
            coastal.astype(np.int8),
            {
                "long_name": "observation-independent ReCAD coastal product geometry",
                "maximum_distance_km": spec.maximum_distance_km,
            },
        ),
        "distance_to_coast_km": (
            ("lat", "lon"),
            distance_all.astype(np.float32),
            {"long_name": "signed distance to any GSHHG coastline", "units": "km"},
        ),
        "distance_to_significant_land_km": (
            ("lat", "lon"),
            distance_significant.astype(np.float32),
            {
                "long_name": "signed distance to GSHHG land above the frozen area threshold",
                "units": "km",
                "minimum_island_area_km2": spec.significant_island_area_km2,
            },
        ),
    }

    if bathymetry_m is not None:
        bathymetry = np.asarray(bathymetry_m, dtype=np.float64)
        if bathymetry.shape != expected:
            raise ValueError(f"bathymetry_m must have shape {expected}")
        valid_bathymetry = np.isfinite(bathymetry)
        shelf = coastal & valid_bathymetry & (bathymetry <= 0) & (bathymetry >= -spec.shelf_depth_m)
        deep = coastal & valid_bathymetry & (bathymetry < -spec.shelf_depth_m)
        unclassified = coastal & ~(shelf | deep)
        data_vars.update(
            {
                "bathymetry_m": (
                    ("lat", "lon"),
                    bathymetry.astype(np.float32),
                    {"long_name": "seabed elevation relative to mean sea level", "units": "m"},
                ),
                "shelf_core_mask": (
                    ("lat", "lon"),
                    shelf.astype(np.int8),
                    {
                        "long_name": "coastal product cells on the bathymetric shelf",
                        "maximum_depth_m": spec.shelf_depth_m,
                    },
                ),
                "slope_deep_mask": (
                    ("lat", "lon"),
                    deep.astype(np.int8),
                    {
                        "long_name": "coastal product cells deeper than the shelf threshold",
                        "minimum_depth_m": spec.shelf_depth_m,
                    },
                ),
                "bathymetry_unclassified_mask": (
                    ("lat", "lon"),
                    unclassified.astype(np.int8),
                    {"long_name": "coastal cells lacking an ocean-consistent bathymetric class"},
                ),
            }
        )

    return xr.Dataset(
        data_vars=data_vars,
        coords={"lat": latitude.astype(np.float32), "lon": longitude.astype(np.float32)},
        attrs={
            "mask_independent_of_target_observations": "true",
            "maximum_distance_to_significant_land_km": spec.maximum_distance_km,
            "significant_island_area_km2": spec.significant_island_area_km2,
            "shelf_depth_m": spec.shelf_depth_m,
            "observation_support_included": "false",
            "predictor_availability_included": "false",
            "release_eligibility_included": "false",
        },
    )


def mask_area_km2(mask: np.ndarray, latitude: np.ndarray, resolution_deg: float) -> float:
    """Return the spherical area of a two-dimensional boolean mask."""

    mask = np.asarray(mask, dtype=bool)
    row_area = regular_cell_area_km2(latitude, resolution_deg)
    if mask.ndim != 2 or mask.shape[0] != row_area.size:
        raise ValueError("mask latitude dimension does not match latitude")
    return float(np.sum(mask * row_area[:, None]))
