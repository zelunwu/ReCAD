"""End-to-end pipeline smoke test: `recad e2e` on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from recad.cli import main


@pytest.mark.slow
def test_e2e_cli(tmp_path):
    workdir = tmp_path / "run"
    # needs the scripts import path only for the synthetic generator, which
    # now lives in the package; run the CLI directly
    rc = main(["e2e", "--workdir", str(workdir), "--n-years", "2"])
    assert rc == 0

    # ---- product file contents ----
    import xarray as xr

    product = workdir / "product.nc"
    assert product.is_file()
    ds = xr.open_dataset(product)
    for var in ("fco2", "pco2", "fco2_err_model", "fco2_err_aleatoric", "fco2_err_epistemic"):
        assert var in ds.data_vars, var
    assert ds["fco2"].shape == (2, 12, 81, 160)
    mean = ds["fco2"].values
    # a meaningful fraction of coastal cells reconstructed
    frac_valid = np.isfinite(mean).mean()
    assert frac_valid > 0.05, frac_valid
    # physical sanity: reconstructed fCO2 within a plausible range
    vals = mean[np.isfinite(mean)]
    assert vals.min() > 150 and vals.max() < 600

    # ---- checkpoints and histories ----
    assert (workdir / "outputs" / "checkpoints" / "member_0.pt").is_file()
    assert (workdir / "outputs" / "checkpoints" / "member_1.pt").is_file()
    for i in range(2):
        hist = workdir / "outputs" / f"history_member_{i}.json"
        assert hist.is_file()
    # training must have reduced loss (or at least recorded it)
    import json

    h0 = json.loads((workdir / "outputs" / "history_member_0.json").read_text())
    assert len(h0["epoch"]) >= 1 and np.isfinite(h0["train_loss"]).all()


@pytest.mark.slow
def test_e2e_training_improves_or_runs(tmp_path):
    """With 2 epochs the model must at least execute; loss stays finite."""
    workdir = tmp_path / "run2"
    rc = main(["e2e", "--workdir", str(workdir), "--n-years", "2"])
    assert rc == 0
    import json

    h = json.loads((workdir / "outputs" / "history_member_0.json").read_text())
    assert np.isfinite(h["val_rmse"]).all()
    assert (workdir / "product.nc").stat().st_size > 10_000
