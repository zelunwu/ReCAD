"""Grid construction and patch layout tests."""

from __future__ import annotations

import numpy as np
import pytest

from recad.config import Config, ConfigError
from recad.data.grid import build_target_grid


def _grid(**grid_cfg):
    return build_target_grid(Config.from_dict({"grid": grid_cfg}).grid)


def test_global_1over8_grid():
    grid = _grid(domain="global", resolution_deg=0.125)
    assert grid.n_lon == 2880
    assert grid.n_lat == 1297
    assert grid.lon[0] == 0.0
    assert grid.lon[-1] == pytest.approx(359.875)
    assert grid.lat[0] == -78.0
    assert grid.lat[-1] == pytest.approx(84.0)


def test_half_open_lon_no_wrap_duplicate():
    grid = _grid(
        domain="custom", resolution_deg=0.25, lon_min=0.0, lon_max=10.0, lat_min=0.0, lat_max=5.0
    )
    assert grid.n_lon == 40
    assert grid.n_lat == 21
    assert grid.lon[0] == 0.0 and grid.lon[-1] == pytest.approx(9.75)


def test_patch_layout():
    grid = _grid(
        domain="custom",
        resolution_deg=0.25,
        lon_min=0.0,
        lon_max=10.0,
        lat_min=0.0,
        lat_max=5.0,
        patch_size_cells=2,
    )
    assert grid.n_patches_lon == 20 and grid.n_patches_lat == 11
    ids = grid.patch_id_map()
    assert ids.shape == (21, 40)
    assert ids.min() == 0 and ids.max() == grid.n_patches - 1
    assert len(np.unique(ids)) == grid.n_patches


def test_patch_centers():
    grid = _grid(
        domain="custom",
        resolution_deg=0.25,
        lon_min=0.0,
        lon_max=2.0,
        lat_min=0.0,
        lat_max=1.0,
        patch_size_cells=2,
    )
    lon_c, lat_c = grid.patch_centers()
    assert lon_c.shape == (grid.n_patches,)
    # first patch spans lon [0, 0.5) at 0.25-deg nodes -> cells {0.0, 0.25}
    assert lon_c[0] == pytest.approx(0.125)
    assert lat_c[0] == pytest.approx(0.125)


def test_unknown_domain_rejected():
    with pytest.raises(ConfigError):
        _grid(domain="mars")
