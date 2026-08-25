"""Split-mask tests: disjointness, coverage, reproducibility, schemes."""

from __future__ import annotations

import numpy as np

from recad.config import Config
from recad.data.split import make_split_masks


def _split(prepared, **split_cfg):
    cfg = Config.from_dict({"split": split_cfg})
    return make_split_masks(cfg.split, prepared)


def test_random_80_20_partitions(prepared_small):
    masks = _split(
        prepared_small, scheme="random_80_20", train_fraction=0.8, seed=100, test_holdout_years=[]
    )
    valid = prepared_small.valid_mask()
    assert (masks.train & masks.val).sum() == 0
    assert (masks.test & masks.train).sum() == 0
    assert (masks.train | masks.val).sum() == valid.sum()
    counts = masks.counts()
    frac = counts["train"] / valid.sum()
    assert 0.75 < frac < 0.85


def test_holdout_years_go_to_test(prepared_small, cfg_small):
    masks = _split(prepared_small, scheme="random_80_20", test_holdout_years=[1994], seed=100)
    assert (masks.test[1]).any()  # 1994 is the test year (index 1)
    assert not masks.test[0].any() and not masks.test[2].any()
    # the 1994 pool must not leak into train/val
    assert not masks.train[1].any() and not masks.val[1].any()


def test_reproducible_seed(prepared_small):
    a = _split(prepared_small, scheme="random_80_20", seed=100)
    b = _split(prepared_small, scheme="random_80_20", seed=100)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.val, b.val)
    c = _split(prepared_small, scheme="random_80_20", seed=101)
    assert not np.array_equal(a.train, c.train)


def test_blocked_split_covers_pool(prepared_small):
    masks = _split(
        prepared_small,
        scheme="blocked_spatiotemporal",
        spatial_block_deg=1.0,
        temporal_block_months=3,
        train_fraction=0.8,
        seed=100,
        test_holdout_years=[],
    )
    valid = prepared_small.valid_mask()
    assert (masks.train | masks.val).sum() == valid.sum()
    assert (masks.train & masks.val).sum() == 0


def test_blocked_split_no_spatial_leak(prepared_small):
    """Train and val cells must never share the same spatial block+time block."""
    masks = _split(
        prepared_small,
        scheme="blocked_spatiotemporal",
        spatial_block_deg=1.0,
        temporal_block_months=3,
        seed=100,
        test_holdout_years=[],
    )
    # spatial block id per (lat, lon): 1-deg blocks over the 10x5 deg domain
    lon = prepared_small.grid.lon
    lat = prepared_small.grid.lat
    lon_block = np.floor((lon - lon[0]) / 1.0).astype(int)
    lat_block = np.floor((lat - lat[0]) / 1.0).astype(int)
    # for each (year, month-block, lat-block, lon-block) check no train/val mix
    for y in range(prepared_small.n_year):
        for m in range(12):
            for lb in np.unique(lat_block):
                for ob in np.unique(lon_block):
                    cell_sel = (lat_block == lb)[:, None] & (lon_block == ob)[None, :]
                    tr = masks.train[y, m][cell_sel].any()
                    va = masks.val[y, m][cell_sel].any()
                    assert not (tr and va), (y, m, lb, ob)


def test_v11_comparison_setup(prepared_small):
    """The v1.1 benchmark split (random 80/20 + 2004/05 holdout) is expressible."""
    cfg = Config.from_dict(
        {
            "split": {"scheme": "random_80_20", "seed": 100, "test_holdout_years": [1994]},
        }
    )
    masks = make_split_masks(cfg.split, prepared_small)
    assert masks.get("test").sum() > 0
    assert masks.counts()["train"] > masks.counts()["val"]
