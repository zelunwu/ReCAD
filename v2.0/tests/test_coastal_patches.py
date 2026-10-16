"""Tests for target-independent coastal patch extraction."""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest

from recad.evaluate.coastal_patches import (
    DYNAMIC_VARIABLES,
    STATIC_VARIABLES,
    CoastalPatchReader,
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
