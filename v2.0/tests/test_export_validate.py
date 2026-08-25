"""Export (CF product) and holdout-validation tests."""

from __future__ import annotations

import numpy as np
import torch

from recad.config import Config
from recad.data.split import make_split_masks
from recad.data.tensorize import CoastalPatchDataset
from recad.evaluate.validate import run_holdout_validation
from recad.model.ensemble import DeepEnsemble
from recad.model.factory import build_model
from recad.predict.export import product_dataset, subset_product, to_netcdf


def _cfg_with_holdout():
    return Config.from_dict(
        {
            "split": {
                "scheme": "random_80_20",
                "train_fraction": 0.8,
                "seed": 100,
                "test_holdout_years": [1994],
            },
            "model": {
                "embed_dim": 16,
                "n_heads": 2,
                "spatial_layers": 1,
                "temporal_layers": 1,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
            },
            "train": {"use_amp": False, "device": "cpu"},
        }
    )


def _tiny_ensemble(prepared, model_cfg):
    shapes = CoastalPatchDataset(
        prepared, masks=None, split=None, model_cfg=model_cfg, require_target=False
    ).shapes()
    members = [build_model(shapes, model_cfg, seed=i) for i in range(2)]
    return DeepEnsemble(members, torch.device("cpu"))


def test_product_dataset_and_netcdf(prepared_small, cfg_small, tmp_path):
    fields = {
        "fco2": np.full(prepared_small.shape4d, 400.0, dtype=np.float32),
        "pco2": np.full(prepared_small.shape4d, 405.0, dtype=np.float32),
        "fco2_err_total": np.full(prepared_small.shape4d, 8.0, dtype=np.float32),
    }
    ds = product_dataset(cfg_small, prepared_small.grid, prepared_small.years, fields)
    assert ds.attrs["Conventions"] == "CF-1.8"
    assert "config_sha256" in ds.attrs
    assert ds["fco2"].shape == prepared_small.shape4d

    path = to_netcdf(ds, tmp_path / "prod.nc")
    import xarray as xr

    again = xr.open_dataset(path)
    assert np.allclose(again["fco2"].values, 400.0)


def test_subset_product(prepared_small, cfg_small):
    fields = {"fco2": np.zeros(prepared_small.shape4d, dtype=np.float32)}
    ds = product_dataset(cfg_small, prepared_small.grid, prepared_small.years, fields)
    sub = subset_product(ds, {"lon": (2.0, 8.0), "lat": (1.0, 4.0)})
    assert sub.sizes["lon"] < ds.sizes["lon"]
    assert sub.sizes["lat"] < ds.sizes["lat"]


def test_holdout_validation_tables(prepared_small, model_cfg_small):
    cfg = _cfg_with_holdout()
    masks = make_split_masks(cfg.split, prepared_small)
    assert masks.test.sum() > 0  # 1994 held out
    ensemble = _tiny_ensemble(prepared_small, model_cfg_small)
    tables = run_holdout_validation(cfg, prepared_small, masks, ensemble, torch.device("cpu"))
    assert "test_years" in tables and "test_overall" in tables
    assert not tables["test_years"].empty
    assert {"year", "n", "r2", "rmse"}.issubset(tables["test_years"].columns)
    # the test-year table contains exactly the held-out year
    assert set(tables["test_years"]["year"]) == {1994}
