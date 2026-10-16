"""Tests for target-independent coastal patch extraction."""

from __future__ import annotations

import json
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest

from recad.evaluate.coastal_patches import (
    DYNAMIC_VARIABLES,
    STATIC_VARIABLES,
    CoastalPatchMemmap,
    CoastalPatchReader,
    PatchFrameDataset,
    patch_channel_statistics,
)


def tiny_cube(path: Path) -> None:
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", 3)
        dataset.createDimension("node", 6)
        latitude = np.repeat(np.arange(2), 3)
        longitude = np.tile(np.arange(3), 2)
        for name, values in (
            ("grid_lat_index", latitude),
            ("grid_lon_index", longitude),
        ):
            variable = dataset.createVariable(name, "i4", ("node",))
            variable[:] = values
        for index, name in enumerate(STATIC_VARIABLES):
            variable = dataset.createVariable(name, "f4", ("node",))
            variable[:] = np.arange(6) + index
        for index, name in enumerate(DYNAMIC_VARIABLES):
            variable = dataset.createVariable(name, "f4", ("time", "node"))
            variable[:] = np.arange(18).reshape(3, 6) + 100 * index


@pytest.mark.l1
def test_patch_reader_preserves_time_and_wraps_longitude(tmp_path: Path) -> None:
    path = tmp_path / "cube.nc"
    tiny_cube(path)
    frame = pd.DataFrame({"coastal_node": [0, 5], "time_index": [2, 1]})
    with CoastalPatchReader(path, monthly_cache_size=2) as reader:
        patches = reader.read(frame, sequence_months=2, patch_size=2)
        assert patches.shape == (2, 2, 16, 2, 2)
        assert np.isfinite(patches).all()
        assert patches[0, 1, 0, 1, 1] == 12.0
        assert patches[0, 0, 0, 1, 1] == 6.0
        assert patches[0, 1, -1, 0, 0] == 0.0
        assert patches[0, 1, -1, 1, 0] == 1.0


@pytest.mark.l1
def test_patch_reader_rejects_odd_patch(tmp_path: Path) -> None:
    path = tmp_path / "cube.nc"
    tiny_cube(path)
    with CoastalPatchReader(path) as reader, pytest.raises(ValueError, match="even"):
        reader.neighborhood(np.array([0]), 3)


@pytest.mark.l1
def test_memmap_reader_is_past_only_and_target_free(tmp_path: Path) -> None:
    dynamic = np.memmap(
        tmp_path / "dynamic.float16.dat", dtype=np.float16, mode="w+", shape=(3, 6, 7)
    )
    dynamic[:] = np.arange(dynamic.size).reshape(dynamic.shape)
    dynamic.flush()
    np.save(tmp_path / "static.float32.npy", np.ones((6, 8), dtype=np.float32))
    lookup = np.arange(6, dtype=np.int32).reshape(2, 3)
    np.save(tmp_path / "node_lookup.int32.npy", lookup)
    np.save(tmp_path / "node_latitude_index.int32.npy", np.repeat(np.arange(2), 3))
    np.save(tmp_path / "node_longitude_index.int32.npy", np.tile(np.arange(3), 2))
    (tmp_path / "metadata.json").write_text(
        json.dumps({"n_time": 3, "n_node": 6, "n_latitude": 2, "n_longitude": 3}),
        encoding="utf-8",
    )
    cache = CoastalPatchMemmap(tmp_path)
    patch = cache.read_one(node=0, time_index=1, sequence=2, patch_size=2)
    assert patch.shape == (2, 16, 2, 2)
    assert patch[0, 0, 1, 1] == 0.0
    assert patch[1, 0, 1, 1] == 42.0
    assert np.isfinite(patch).all()

    frame = pd.DataFrame(
        {
            "coastal_node": [0, 1],
            "time_index": [1, 2],
            "truth": [10.0, 20.0],
            "year_centered": [0.5, 1.5],
        }
    )
    raw = PatchFrameDataset(
        cache, frame, sequence=2, patch_size=2, scalar_features=("year_centered",)
    )
    mean, scale = patch_channel_statistics(raw, np.array([0, 1]))
    normalized = PatchFrameDataset(
        cache,
        frame,
        sequence=2,
        patch_size=2,
        scalar_features=("year_centered",),
        channel_mean=mean,
        channel_scale=scale,
    )
    values, target = normalized[0]
    assert values.shape == (2, 17, 2, 2)
    assert float(target) == 10.0
    assert np.isfinite(values.numpy()).all()
