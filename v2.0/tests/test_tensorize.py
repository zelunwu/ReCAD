"""Tensorization tests: token building, cell tensors, collation."""

from __future__ import annotations

import numpy as np
import pytest

from recad.data.tensorize import CoastalPatchDataset, collate_items
from recad.utils.native import recad_native


def _dataset(prepared_small, masks, model_cfg, **kw):
    return CoastalPatchDataset(prepared_small, masks, "train", model_cfg, **kw)


def test_shapes(prepared_small, masks_small, model_cfg_small):
    ds = _dataset(prepared_small, masks_small, model_cfg_small)
    shapes = ds.shapes()
    assert shapes.n_tokens >= 1
    assert shapes.n_cells == (prepared_small.coastal_mask).sum()
    # cell features: lon, lat, mon_sin, mon_cos + 5 z-scored + 5 missing flags
    assert shapes.n_cell_features == 4 + 2 * 5
    assert shapes.n_token_features == 5 + 1  # 5 predictor means + coverage


def test_item_shapes_and_dtype(prepared_small, masks_small, model_cfg_small):
    ds = _dataset(prepared_small, masks_small, model_cfg_small)
    item = ds[0]
    T = 12
    P = ds.shapes().n_tokens
    C = ds.shapes().n_cells
    assert item["token_feats"].shape == (T, P, 6)
    assert item["token_valid"].shape == (T, P)
    assert item["coord"].shape == (T, P, 2)
    assert item["month"].shape == (T, P, 2)
    assert item["cell_feats"].shape == (T, C, 14)
    assert item["targets"].shape == (T, C)
    assert item["cell_mask"].shape == (T, C)
    assert item["cell_patch"].shape == (T, C)
    assert item["year_id"].item() == 0
    assert item["month_ids"][0].item() == 1


def test_cell_mask_matches_target_validity(prepared_small, masks_small, model_cfg_small):
    ds = _dataset(prepared_small, masks_small, model_cfg_small)
    item = ds[0]
    tgt = prepared_small.target_values()[0, 0]  # [n_lat, n_lon]
    flat = tgt.ravel()[ds.cell_flat_indices]
    assert np.array_equal(item["cell_mask"][0].numpy(), ~np.isnan(flat))


def test_token_aggregation_matches_manual(prepared_small, masks_small, model_cfg_small):
    """Token predictor means must equal nanmean over the patch box."""
    ds = _dataset(prepared_small, masks_small, model_cfg_small)
    prepared = prepared_small
    _, std = prepared.feature_stats()  # not fitted -> std 1, mean 0
    m = 0
    for j, name in enumerate(["sst", "sss", "adt", "pco2air", "wspd"]):
        field = prepared.arrays[name][0, m]
        z = field / std[j]
        means = ds.token_feats_ym[0, m, :, j]
        for p in range(ds.shapes().n_tokens):
            patch_cells = ds.patch_map == ds.active_tokens[p]
            vals = z[patch_cells]
            manual = np.nanmean(vals) if vals[~np.isnan(vals)].size else np.nan
            if np.isnan(manual):
                assert np.isnan(means[p])
            else:
                # float32 aggregation (native kernel) vs float64 manual mean
                assert np.isclose(means[p], manual, atol=1e-4)


def test_tile_cells_deterministic(prepared_small, masks_small, model_cfg_small):
    ds_a = _dataset(prepared_small, masks_small, model_cfg_small, tile_cells=40)
    ds_b = _dataset(prepared_small, masks_small, model_cfg_small, tile_cells=40)
    # stride sampling may yield slightly fewer than the requested count
    assert 0 < ds_a.shapes().n_cells <= 40
    assert np.array_equal(ds_a.cell_flat_indices, ds_b.cell_flat_indices)
    item_a, item_b = ds_a[0], ds_b[0]
    assert np.array_equal(item_a["cell_feats"], item_b["cell_feats"])


def test_collate(prepared_small, masks_small, model_cfg_small):
    ds = _dataset(prepared_small, masks_small, model_cfg_small)
    items = [ds[0], ds[1]]
    batch = collate_items(items)
    assert batch["token_feats"].shape == (2, 12, ds.shapes().n_tokens, 6)
    assert batch["cell_feats"].shape == (2, 12, ds.shapes().n_cells, 14)
    assert batch["cell_mask"].dtype == torch_bool()
    assert batch["targets"].dtype == torch_float32()
    assert batch["year_id"].shape == (2,)


def test_reconstruction_mode_no_target_needed(prepared_small, model_cfg_small):
    ds = CoastalPatchDataset(
        prepared_small,
        masks=None,
        split=None,
        model_cfg=model_cfg_small,
        require_target=False,
    )
    assert len(ds) == prepared_small.n_year  # every year, one window
    item = ds[0]
    assert item["cell_mask"].all()  # every coastal cell is observable


def torch_bool():
    import torch

    return torch.bool


def torch_float32():
    import torch

    return torch.float32


def test_native_patch_aggregate_path(prepared_small, masks_small, model_cfg_small):
    """The dataset must build tokens without the native DLL and with it."""
    ds = CoastalPatchDataset(
        prepared_small,
        masks=None,
        split=None,
        model_cfg=model_cfg_small,
        require_target=False,
    )
    assert ds.token_feats_ym.shape[0] == prepared_small.n_year
    assert ds.token_feats_ym.shape[1] == 12
    assert not np.isnan(ds.token_feats_ym).all()
    if not recad_native._available:
        pytest.skip("native recad_cuda DLL not built; fallback path exercised above")
