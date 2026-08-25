"""Shared fixtures for the ReCAD v2.0 test suite."""

from __future__ import annotations

import pytest
import torch

from recad.config import Config
from recad.data.features import PreparedData
from recad.data.grid import DomainGrid
from recad.data.split import make_split_masks


def _make_cfg(**overrides) -> Config:
    base: dict = {
        "grid": {
            "domain": "custom",
            "resolution_deg": 0.25,
            "lon_min": 0.0,
            "lon_max": 10.0,
            "lat_min": 0.0,
            "lat_max": 5.0,
            "patch_size_cells": 2,
        },
        "coastal_mask": {"method": "distance", "distance_km": 300.0},
        "data": {"root": "data/raw", "year_min": 1993, "year_max": 1996, "variable_name": "fco2"},
        "split": {
            "scheme": "random_80_20",
            "train_fraction": 0.8,
            "seed": 100,
            "test_holdout_years": [],
            "spatial_block_deg": 1.0,
            "temporal_block_months": 3,
        },
        "model": {
            "embed_dim": 16,
            "n_heads": 2,
            "spatial_layers": 1,
            "temporal_layers": 1,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "head": "gaussian",
            "temporal_window_months": 12,
        },
        "ensemble": {"n_members": 2, "seed_offset": 0, "bootstrap_fraction": 1.0},
        "train": {
            "n_epochs": 2,
            "batch_windows": 2,
            "lr": 1e-3,
            "use_amp": False,
            "device": "auto",
            "checkpoint_dir": "outputs/checkpoints",
            "early_stop_patience": 5,
        },
        "uncertainty": {"n_mc_draws": 3, "propagate_input": True, "combine": "rss"},
        "output": {"out_dir": "outputs", "product_name": "ReCAD-test"},
    }
    return Config.from_dict(base | overrides)


@pytest.fixture
def cfg_small():
    return _make_cfg()


@pytest.fixture
def prepared_small():
    """In-memory PreparedData on a 21x11 (lon 0-10, lat 0-5 at 0.5 deg) grid."""
    import numpy as np

    lon_vals = np.linspace(0.0, 10.0, 21)
    lat_vals = np.linspace(0.0, 5.0, 11)
    grid = DomainGrid(lon=lon_vals, lat=lat_vals, patch_size=2)

    rng = np.random.default_rng(7)
    n_year, n_lat, n_lon = 3, 11, 21
    shape4 = (n_year, 12, n_lat, n_lon)

    arrays = {}
    for name in ("sst", "sss", "adt", "pco2air", "wspd"):
        arrays[name] = rng.normal(size=shape4).astype(np.float32)
    arrays["sst"] += 20.0
    arrays["sss"] += 34.0
    arrays["pco2air"] += 400.0
    # fCO2 target driven by SST so the model has a signal to learn
    fco2 = 320.0 + 6.0 * (arrays["sst"] - 20.0) + rng.normal(0, 2, size=shape4)
    rng_mask = rng.random(shape4) > 0.15  # 85% coverage
    arrays["fco2"] = np.where(rng_mask, fco2, np.nan).astype(np.float32)

    coastal = np.zeros((n_lat, n_lon), dtype=bool)
    # a coastal band: rows 1..n_lat-2, with lon within 1 deg of the edges
    coast_lon = (lon_vals <= 1.0) | (lon_vals >= 9.0)
    coastal[1:-1, :] = True
    coastal = coastal & np.broadcast_to(coast_lon, (n_lat, n_lon))
    coastal[5, :] = False  # a little gap for interest

    prepared = PreparedData(
        grid=grid,
        coastal_mask=coastal,
        years=np.array([1993, 1994, 1995]),
        target="fco2",
        arrays=arrays,
    )
    return prepared


@pytest.fixture
def masks_small(prepared_small):
    cfg = _make_cfg()
    return make_split_masks(cfg.split, prepared_small)


@pytest.fixture
def model_cfg_small():
    return _make_cfg().model


@pytest.fixture(scope="session")
def torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
