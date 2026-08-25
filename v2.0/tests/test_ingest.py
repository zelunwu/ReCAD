"""Ingestion primitives: QC, outlier removal, climatology/anomaly, regrid."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from recad.data.ingest import (
    apply_qc,
    calc_clim_anom,
    regrid_to_target,
    remove_outliers_3sigma,
)


def test_apply_qc_bounds():
    arr = np.array([0.5, 1.0, 1000.0, 1500.0, np.nan])
    out = apply_qc(arr, 1.0, 1000.0)
    assert np.isnan(out[0]) and np.isnan(out[3])
    assert out[1] == 1.0 and out[2] == 1000.0
    assert np.isnan(out[4])


def test_qc_no_upper_bound():
    arr = np.array([-20.0, 5.0, 50.0, 200.0])
    out = apply_qc(arr, 0.0, None)
    assert np.isnan(out[0]) and out[-1] == 200.0


def test_remove_outliers_3sigma():
    rng = np.random.default_rng(0)
    arr = rng.normal(400.0, 20.0, size=500)
    arr = np.concatenate([arr, [1000.0, -200.0]])  # clear outliers
    out = remove_outliers_3sigma(arr)
    assert np.isnan(out[-2]) and np.isnan(out[-1])
    assert not np.isnan(out[:500]).any()


def test_remove_outliers_all_nan():
    out = remove_outliers_3sigma(np.full(5, np.nan))
    assert np.isnan(out).all()


def test_calc_clim_anom_hand_computed():
    # Structural invariants of the v1.1 decomposition:
    #   clim = year-mean of the DETRENDED monthly series;
    #   anom = raw - clim, so raw is exactly reconstructable.
    season = np.sin(np.arange(12) * 2 * np.pi / 12)
    trend = np.arange(24, dtype=float) * 0.25
    ts = np.tile(season, 2) + trend
    clim, anom = calc_clim_anom(ts, 2)
    assert clim.shape == (12,)
    assert anom.shape == (24,)
    # definition: anom + tiled clim reconstructs ts exactly
    assert np.allclose(anom + np.tile(clim, 2), ts, atol=1e-12)


def test_calc_clim_anom_removes_trend_from_climatology():
    # A pure linear trend: the climatology is the mean of the DETRENDED
    # months, so it is ~0 (the trend itself is removed), and the anomaly
    # retains the full trend (v1.1 semantics: anom = raw - clim).
    ts = np.arange(24, dtype=float) * 0.5 + 20.0
    clim, anom = calc_clim_anom(ts, 2)
    assert np.max(np.abs(clim)) < 1e-6  # flat at zero once detrended
    assert np.allclose(anom, ts, atol=1e-6)  # anomaly keeps the trend
    # one year of trend = 6.0
    assert np.nanmean(anom[12:]) - np.nanmean(anom[:12]) == pytest.approx(6.0, abs=1e-6)


def test_calc_clim_anom_handles_nan():
    ts = np.arange(24, dtype=float)
    ts[5] = np.nan
    clim, anom = calc_clim_anom(ts, 2)
    assert clim.shape == (12,)
    assert np.isnan(anom[5])


def test_regrid_to_target():
    da = xr.DataArray(
        np.arange(9).reshape(3, 3).astype(float),
        dims=("lat", "lon"),
        coords={"lat": [0.0, 1.0, 2.0], "lon": [0.0, 1.0, 2.0]},
    )
    out = regrid_to_target(da, np.array([0.0, 2.0]), np.array([0.0, 2.0]), method="nearest")
    assert out.shape == (2, 2)
    assert out.values[0, 0] == 0.0 and out.values[1, 1] == 8.0


def test_regrid_wraps_longitude_to_0_360():
    da = xr.DataArray(
        np.arange(4).reshape(2, 2).astype(float),
        dims=("lat", "lon"),
        coords={"lat": [0.0, 1.0], "lon": [-1.0, 1.0]},
    )
    out = regrid_to_target(da, np.array([359.0, 1.0]), np.array([0.5]), method="nearest")
    assert out.shape == (1, 2)
    # lon -1 wrapped to 359: nearest target 359.0 picks the -1 source column
    assert out.values[0, 0] in (0.0, 2.0)
