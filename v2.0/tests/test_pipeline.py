"""Full data-pipeline test on synthetic files (ingest -> prepared -> persist)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from recad.config import Config
from recad.data.pipeline import (
    build_prepared,
    ingest_variable,
    load_masks,
    load_prepared,
    prepare_split,
    save_masks,
    save_prepared,
)
from recad.testing.synthetic import make_synthetic_workdir


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("synth")
    config_path, cache, *_ = make_synthetic_workdir(workdir, n_years=2)
    return config_path, cache


def test_ingest_all_variables(synth):
    config_path, cache = synth
    cfg = Config.from_yaml(str(config_path))
    for name in ("sst", "sss", "adt", "wspd", "xco2air", "fco2", "mask"):
        out = ingest_variable(cfg, name, cache)
        assert Path(out).is_file()
    # fCO2 QC: no values < 1 or > 1000 survive
    import xarray as xr

    fco2 = xr.open_dataset(cache / "fco2.nc")["fco2"].values
    valid = fco2[~np.isnan(fco2)]
    assert (valid >= 1.0).all() and (valid <= 1000.0).all()


def test_build_and_persist_prepared(synth, tmp_path):
    config_path, cache = synth
    cfg = Config.from_yaml(str(config_path))
    prepared = build_prepared(cfg, cache)
    assert prepared.shape4d == (2, 12, 81, 160)
    assert prepared.coastal_mask.sum() > 0
    assert set(prepared.arrays) >= {"sst", "sss", "adt", "wspd", "xco2air", "fco2", "pco2air"}
    # pCO2air is a derived predictor
    assert not np.isnan(prepared.arrays["pco2air"]).all()

    masks = prepare_split(cfg, prepared)
    assert masks.counts()["train"] + masks.counts()["val"] == prepared.valid_mask().sum()

    p = tmp_path / "prepared.nc"
    m = tmp_path / "masks.nc"
    save_prepared(prepared, p)
    save_masks(masks, m, prepared.years, prepared.grid)
    reloaded = load_prepared(p)
    assert reloaded.shape4d == prepared.shape4d
    assert np.array_equal(reloaded.coastal_mask, prepared.coastal_mask)
    assert np.array_equal(reloaded.years, prepared.years)
    masks2 = load_masks(m)
    assert np.array_equal(masks2.train, masks.train)


def test_missing_template_raises(synth, tmp_path):
    config_path, cache = synth
    cfg = Config.from_yaml(str(config_path))

    bad = Config.from_dict(
        {
            "data": {**cfg.data.__dict__, "templates": {**cfg.data.templates, "sst": "missing.nc"}},
        }
    )
    with pytest.raises(FileNotFoundError):
        ingest_variable(bad, "sst", cache)
