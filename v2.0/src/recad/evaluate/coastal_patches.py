"""Leakage-safe patch extraction from the frozen Issue 53 predictor cube."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

DYNAMIC_VARIABLES = (
    "sst",
    "wspd",
    "xco2air",
    "sss_bg",
    "sss_bg_available",
    "adt",
    "adt_available",
)
STATIC_VARIABLES = (
    "distance_to_significant_land_km",
    "bathymetry_m",
    "shelf_core_mask",
    "slope_deep_mask",
    "latitude_sin",
    "latitude_cos",
    "longitude_sin",
    "longitude_cos",
)


class CoastalPatchReader:
    """Read sample-centred monthly patches without target-dependent neighbors."""

    def __init__(self, cube: str | Path, *, monthly_cache_size: int = 8) -> None:
        self.path = Path(cube)
        self.dataset = netCDF4.Dataset(self.path)
        self.monthly_cache_size = monthly_cache_size
        latitude_index = self.dataset.variables["grid_lat_index"][:].astype(np.int32)
        longitude_index = self.dataset.variables["grid_lon_index"][:].astype(np.int32)
        self.n_latitude = int(latitude_index.max()) + 1
        self.n_longitude = int(longitude_index.max()) + 1
        self.node_lookup = np.full((self.n_latitude, self.n_longitude), -1, dtype=np.int32)
        self.node_lookup[latitude_index, longitude_index] = np.arange(
            latitude_index.size, dtype=np.int32
        )
        self.node_latitude_index = latitude_index
        self.node_longitude_index = longitude_index
        self.static = np.column_stack(
            [self.dataset.variables[name][:] for name in STATIC_VARIABLES]
        ).astype(np.float32)
        self._monthly: OrderedDict[int, np.ndarray] = OrderedDict()

    @property
    def channels(self) -> tuple[str, ...]:
        return (*DYNAMIC_VARIABLES, *STATIC_VARIABLES, "valid_ocean")

    def close(self) -> None:
        self.dataset.close()

    def __enter__(self) -> CoastalPatchReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _month(self, time_index: int) -> np.ndarray:
        if time_index in self._monthly:
            self._monthly.move_to_end(time_index)
            return self._monthly[time_index]
        values = np.column_stack(
            [self.dataset.variables[name][time_index, :] for name in DYNAMIC_VARIABLES]
        ).astype(np.float32)
        self._monthly[time_index] = values
        while len(self._monthly) > self.monthly_cache_size:
            self._monthly.popitem(last=False)
        return values

    def neighborhood(self, nodes: np.ndarray, patch_size: int) -> np.ndarray:
        """Return node indices for square patches, with -1 over land."""

        nodes = np.asarray(nodes, dtype=np.int64)
        if patch_size < 2 or patch_size % 2:
            raise ValueError("patch_size must be an even integer >= 2")
        offsets = np.arange(patch_size, dtype=np.int32) - patch_size // 2
        latitude = self.node_latitude_index[nodes, None, None] + offsets[None, :, None]
        longitude = self.node_longitude_index[nodes, None, None] + offsets[None, None, :]
        valid_latitude = (latitude >= 0) & (latitude < self.n_latitude)
        latitude = np.clip(latitude, 0, self.n_latitude - 1)
        longitude = longitude % self.n_longitude
        neighborhood = self.node_lookup[latitude, longitude]
        return np.where(valid_latitude, neighborhood, -1)

    def read(
        self,
        frame: pd.DataFrame,
        *,
        sequence_months: int,
        patch_size: int,
    ) -> np.ndarray:
        """Materialize finite patches ending at each row's deployment month."""

        required = {"coastal_node", "time_index"}
        if not required.issubset(frame):
            raise ValueError(f"patch frame misses columns: {sorted(required - set(frame))}")
        node = frame.coastal_node.to_numpy(np.int64)
        ending_month = frame.time_index.to_numpy(np.int64)
        neighbors = self.neighborhood(node, patch_size)
        result = np.zeros(
            (
                len(frame),
                sequence_months,
                len(self.channels),
                patch_size,
                patch_size,
            ),
            dtype=np.float32,
        )
        valid_node = neighbors >= 0
        safe_node = np.where(valid_node, neighbors, 0)
        for lag in range(sequence_months):
            query_month = ending_month - (sequence_months - lag - 1)
            valid_month = query_month >= 0
            for time_index in np.unique(query_month[valid_month]):
                rows = np.flatnonzero(query_month == time_index)
                dynamic = self._month(int(time_index))[safe_node[rows]]
                static = self.static[safe_node[rows]]
                dynamic = np.where(np.isfinite(dynamic), dynamic, 0.0)
                static = np.where(np.isfinite(static), static, 0.0)
                combined = np.concatenate([dynamic, static, valid_node[rows, ..., None]], axis=-1)
                result[rows, lag] = combined.transpose(0, 3, 1, 2)
        return result
