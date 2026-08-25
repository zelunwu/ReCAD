"""Spatial reconstruction grid construction and patch layout.

The pipeline works on a regular lon/lat mesh over the configured domain.
Longitudes follow the ``[0, 360)`` convention (SOCAT gridded products and the
v1.1 pipeline both use this convention). The coastal ocean is selected by a
mask (see ``recad.data.specs`` / the coastal_mask config section); only masked
cells are part of the reconstruction.

For the transformer, the domain is tiled into square spatial patches (NN
cells per side, e.g. 16 cells at 1/8 deg = 2 deg x 2 deg). Each patch that
intersects the coastal mask becomes one token in the spatial encoder stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recad.config import ConfigError, GridConfig


@dataclass(frozen=True)
class DomainGrid:
    """A regular lon/lat mesh plus its patch layout."""

    lon: np.ndarray  # [n_lon], ascending, in [0, 360)
    lat: np.ndarray  # [n_lat], ascending
    patch_size: int  # cells per patch side

    @property
    def n_lon(self) -> int:
        return int(self.lon.size)

    @property
    def n_lat(self) -> int:
        return int(self.lat.size)

    @property
    def n_cells(self) -> int:
        return self.n_lat * self.n_lon

    @property
    def n_patches_lon(self) -> int:
        return int(np.ceil(self.n_lon / self.patch_size))

    @property
    def n_patches_lat(self) -> int:
        return int(np.ceil(self.n_lat / self.patch_size))

    @property
    def n_patches(self) -> int:
        return self.n_patches_lat * self.n_patches_lon

    def patch_id_map(self) -> np.ndarray:
        """Per-cell patch id, shape [n_lat, n_lon], int32 in [0, n_patches)."""
        plon = np.minimum(np.arange(self.n_lon) // self.patch_size, self.n_patches_lon - 1)
        plat = np.minimum(np.arange(self.n_lat) // self.patch_size, self.n_patches_lat - 1)
        return (plat[:, None] * self.n_patches_lon + plon[None, :]).astype(np.int32)

    def patch_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Center (lon, lat) of every patch, shape [n_patches]."""
        lon_centers = np.full(self.n_patches_lon, np.nan)
        lat_centers = np.full(self.n_patches_lat, np.nan)
        for i in range(self.n_patches_lon):
            cells = np.arange(i * self.patch_size, min((i + 1) * self.patch_size, self.n_lon))
            lon_centers[i] = self.lon[cells].mean()
        for j in range(self.n_patches_lat):
            cells = np.arange(j * self.patch_size, min((j + 1) * self.patch_size, self.n_lat))
            lat_centers[j] = self.lat[cells].mean()
        return (
            np.tile(lon_centers, self.n_patches_lat),
            np.repeat(lat_centers, self.n_patches_lon),
        )


def build_target_grid(cfg: GridConfig) -> DomainGrid:
    """Build the target mesh from a GridConfig."""
    if cfg.domain == "global":
        lon_min, lon_max, lat_min, lat_max = 0.0, 360.0, -78.0, 84.0
    elif cfg.domain == "naccom":
        lon_min, lon_max, lat_min, lat_max = 260.0, 320.0, 10.0, 65.0
    elif cfg.domain == "custom":
        lon_min, lon_max, lat_min, lat_max = cfg.lon_min, cfg.lon_max, cfg.lat_min, cfg.lat_max
    else:
        raise ConfigError(f"unknown domain preset: {cfg.domain}")

    n_lon = int(round((lon_max - lon_min) / cfg.resolution_deg))
    n_lat = int(round((lat_max - lat_min) / cfg.resolution_deg)) + 1
    # Longitude is half-open [min, max): the last node is max - res, so a
    # global 1/8-deg grid has exactly 2880 nodes and no duplicated wrap cell.
    lon = lon_min + np.arange(n_lon, dtype=np.float64) * cfg.resolution_deg
    lat = np.linspace(lat_min, lat_max, n_lat)  # latitude is closed
    lon = np.clip(lon, lon_min, lon_max)
    lat = np.clip(lat, lat_min, lat_max)
    return DomainGrid(lon=lon, lat=lat, patch_size=cfg.patch_size_cells)