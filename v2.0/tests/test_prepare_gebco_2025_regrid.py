from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_gebco_2025_regrid", ROOT / "scripts" / "prepare_gebco_2025_regrid.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.l1
def test_official_tile_contract_is_complete_and_unique() -> None:
    assert len(MODULE.TILES) == 8
    assert len({tile.name for tile in MODULE.TILES}) == 8
    assert all(len(tile.md5) == 32 for tile in MODULE.TILES)


@pytest.mark.l2
def test_tile_coordinates_cover_every_target_node_once() -> None:
    pairs = []
    for tile in MODULE.TILES:
        longitude, latitude = MODULE.tile_coordinates(tile)
        lon2d, lat2d = np.meshgrid(longitude, latitude)
        pairs.append(np.column_stack([lon2d.ravel(), lat2d.ravel()]))
    coordinates = np.concatenate(pairs)

    assert coordinates.shape == (1297 * 2880, 2)
    assert np.unique(coordinates, axis=0).shape == coordinates.shape
    assert coordinates[:, 0].min() == -180.0
    assert coordinates[:, 0].max() == 179.875
    assert coordinates[:, 1].min() == -78.0
    assert coordinates[:, 1].max() == 84.0
