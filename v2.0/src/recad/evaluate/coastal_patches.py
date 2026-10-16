"""Leakage-safe patch extraction from the frozen Issue 53 predictor cube."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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


class CoastalPatchMemmap:
    """Fast read-only access to a versioned dense-node patch cache."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        self.n_time = int(metadata["n_time"])
        self.n_node = int(metadata["n_node"])
        self.n_latitude = int(metadata["n_latitude"])
        self.n_longitude = int(metadata["n_longitude"])
        self.dynamic = np.memmap(
            self.root / "dynamic.float16.dat",
            dtype=np.float16,
            mode="r",
            shape=(self.n_time, self.n_node, len(DYNAMIC_VARIABLES)),
        )
        self.static = np.load(self.root / "static.float32.npy", mmap_mode="r")
        self.node_lookup = np.load(self.root / "node_lookup.int32.npy", mmap_mode="r")
        self.node_latitude_index = np.load(
            self.root / "node_latitude_index.int32.npy", mmap_mode="r"
        )
        self.node_longitude_index = np.load(
            self.root / "node_longitude_index.int32.npy", mmap_mode="r"
        )

    @property
    def channels(self) -> tuple[str, ...]:
        return (*DYNAMIC_VARIABLES, *STATIC_VARIABLES, "valid_ocean")

    def neighborhood(self, node: int, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
        if patch_size < 2 or patch_size % 2:
            raise ValueError("patch_size must be an even integer >= 2")
        offsets = np.arange(patch_size, dtype=np.int32) - patch_size // 2
        latitude = int(self.node_latitude_index[node]) + offsets[:, None]
        longitude = int(self.node_longitude_index[node]) + offsets[None, :]
        valid_latitude = (latitude >= 0) & (latitude < self.n_latitude)
        latitude = np.clip(latitude, 0, self.n_latitude - 1)
        longitude = longitude % self.n_longitude
        neighbors = np.asarray(self.node_lookup[latitude, longitude])
        valid = valid_latitude & (neighbors >= 0)
        return np.where(valid, neighbors, 0), valid

    def read_one(self, node: int, time_index: int, sequence: int, patch_size: int) -> np.ndarray:
        """Return one past-only finite sequence without reading target values."""

        neighbors, valid = self.neighborhood(node, patch_size)
        result = np.zeros((sequence, len(self.channels), patch_size, patch_size), dtype=np.float32)
        for lag in range(sequence):
            month = time_index - (sequence - lag - 1)
            if month < 0:
                continue
            dynamic = np.asarray(self.dynamic[month, neighbors], dtype=np.float32)
            static = np.asarray(self.static[neighbors], dtype=np.float32)
            combined = np.concatenate([dynamic, static, valid[..., None]], axis=-1)
            combined = np.where(np.isfinite(combined), combined, 0.0)
            result[lag] = combined.transpose(2, 0, 1)
        return result


class PatchFrameDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy observation dataset backed by the target-free coastal patch cache."""

    def __init__(
        self,
        cache: CoastalPatchMemmap,
        frame: pd.DataFrame,
        *,
        sequence: int,
        patch_size: int,
        scalar_features: tuple[str, ...] = (),
        channel_mean: np.ndarray | None = None,
        channel_scale: np.ndarray | None = None,
    ) -> None:
        self.cache = cache
        self.node = frame.coastal_node.to_numpy(np.int64)
        self.time_index = frame.time_index.to_numpy(np.int64)
        self.target = frame.truth.to_numpy(np.float32)
        self.scalar = (
            frame.loc[:, scalar_features].to_numpy(np.float32)
            if scalar_features
            else np.empty((len(frame), 0), dtype=np.float32)
        )
        self.sequence = sequence
        self.patch_size = patch_size
        channels = len(cache.channels) + self.scalar.shape[1]
        self.channel_mean = (
            np.zeros(channels, dtype=np.float32)
            if channel_mean is None
            else np.asarray(channel_mean, dtype=np.float32)
        )
        self.channel_scale = (
            np.ones(channels, dtype=np.float32)
            if channel_scale is None
            else np.asarray(channel_scale, dtype=np.float32)
        )
        if self.channel_mean.shape != (channels,) or self.channel_scale.shape != (channels,):
            raise ValueError("patch normalization vectors have the wrong channel count")

    def __len__(self) -> int:
        return len(self.node)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch = self.cache.read_one(
            int(self.node[index]),
            int(self.time_index[index]),
            self.sequence,
            self.patch_size,
        )
        if self.scalar.shape[1]:
            scalar = np.broadcast_to(
                self.scalar[index, None, :, None, None],
                (
                    self.sequence,
                    self.scalar.shape[1],
                    self.patch_size,
                    self.patch_size,
                ),
            )
            patch = np.concatenate([patch, scalar], axis=1)
        patch = (patch - self.channel_mean[None, :, None, None]) / self.channel_scale[
            None, :, None, None
        ]
        return torch.from_numpy(np.asarray(patch, dtype=np.float32)), torch.tensor(
            self.target[index]
        )


def patch_channel_statistics(
    dataset: PatchFrameDataset, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit channel normalization on a deterministic outer-training subset."""

    sums: np.ndarray | None = None
    squared: np.ndarray | None = None
    count = 0
    for index in np.asarray(indices, dtype=np.int64):
        patch, _ = dataset[int(index)]
        values = patch.numpy().astype(np.float64)
        current_sum = values.sum(axis=(0, 2, 3))
        current_squared = np.square(values).sum(axis=(0, 2, 3))
        sums = current_sum if sums is None else sums + current_sum
        squared = current_squared if squared is None else squared + current_squared
        count += values.shape[0] * values.shape[2] * values.shape[3]
    if sums is None or squared is None or count == 0:
        raise ValueError("normalization subset is empty")
    mean = sums / count
    variance = np.maximum(squared / count - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return mean.astype(np.float32), scale.astype(np.float32)
