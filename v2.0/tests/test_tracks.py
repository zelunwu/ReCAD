"""Tests for recad.data.tracks (SOCAT scatter observations)."""

from __future__ import annotations

import numpy as np
import xarray as xr

from recad.data.features import PreparedData
from recad.data.grid import DomainGrid
from recad.data.tracks import load_tracks, prepare_point_table, qc_tracks, sample_predictors


def _make_tracks_nc(path, n=8):
    rng = np.random.default_rng(0)
    lon = rng.uniform(-90, -60, n)
    lat = rng.uniform(20, 50, n)
    month = np.arange(1, n + 1) % 12 + 1
    fco2 = rng.uniform(300, 420, n)
    if n > 2:
        fco2[2] = 1500.0  # out of range
    if n > 5:
        fco2[5] = np.nan  # missing
    qc = np.full(n, 2)
    if n > 4:
        qc[4] = 3  # flag=3 -> dropped by QC
    ds = xr.Dataset(
        {
            "time": (
                "row",
                np.array([f"1993-{m:02d}-15T12:00:00Z" for m in month], dtype="datetime64[ns]"),
            ),
            "latitude": ("row", lat),
            "longitude": ("row", lon),
            "fCO2_recommended": ("row", fco2),
            "WOCE_CO2_water": ("row", qc),
            "dataset_name": ("row", np.array([f"cruise_{i}" for i in range(n)])),
            "sal": ("row", rng.uniform(30, 36, n)),
            "temp": ("row", rng.uniform(15, 28, n)),
        }
    )
    ds.to_netcdf(path)
    return ds


def test_load_tracks(tmp_path):
    p = tmp_path / "tracks.nc"
    _make_tracks_nc(p)
    t = load_tracks(p)
    assert t.n == 8
    assert t.fco2.shape == (8,)
    assert np.all((t.lon >= 0) & (t.lon < 360))  # converted to 0-360
    assert t.year[0] == 1993
    assert t.month.min() >= 1 and t.month.max() <= 12


def test_qc_tracks_drops_bad(tmp_path):
    p = tmp_path / "tracks.nc"
    _make_tracks_nc(p)
    t = load_tracks(p)
    q = qc_tracks(t)
    # 2 rows removed: missing fCO2 (NaN) and flag=3; out-of-range fCO2 is
    # *masked* (row kept, value NaN) - mask semantics, as in the v1.1 grid QC
    assert q.n == 8 - 2
    assert np.isfinite(q.fco2).sum() == q.n - 1  # one row masked by range
    assert np.all(q.qc_flag <= 2)


def test_sample_predictors_values(tmp_path):
    p = tmp_path / "tracks.nc"
    _make_tracks_nc(p, n=3)
    t = load_tracks(p)
    # small prepared grid with known field: constant 5.0 everywhere
    lon = np.arange(260.0, 321.0, 0.25)
    lat = np.arange(10.0, 66.0, 0.25)
    prepared = PreparedData(
        grid=DomainGrid(lon, lat, patch_size=1),
        coastal_mask=np.ones((lat.size, lon.size), dtype=bool),
        years=np.array([1993, 1994]),
        target="fco2",
        arrays={
            "fco2": np.full((2, 12, lat.size, lon.size), 340.0, dtype=np.float32),
            "sst": np.full((2, 12, lat.size, lon.size), 5.0, dtype=np.float32),
            "sss": np.full((2, 12, lat.size, lon.size), 35.0, dtype=np.float32),
            "adt": np.full((2, 12, lat.size, lon.size), 0.1, dtype=np.float32),
            "pco2air": np.full((2, 12, lat.size, lon.size), 400.0, dtype=np.float32),
            "wspd": np.full((2, 12, lat.size, lon.size), 6.0, dtype=np.float32),
        },
    )
    pred = sample_predictors(t, prepared)
    assert pred.shape == (3, 5)
    assert np.allclose(pred[:, 0], 5.0)  # sst
    assert np.isfinite(pred).all()


def test_sample_predictors_year_guard(tmp_path):
    p = tmp_path / "tracks.nc"
    _make_tracks_nc(p, n=2)
    t = load_tracks(p)
    t.year = np.array([1993, 1999])  # 1999 outside the prepared years
    lon = np.arange(260.0, 321.0, 0.25)
    lat = np.arange(10.0, 66.0, 0.25)
    prepared = PreparedData(
        grid=DomainGrid(lon, lat, patch_size=1),
        coastal_mask=np.ones((lat.size, lon.size), dtype=bool),
        years=np.array([1993]),
        target="fco2",
        arrays={"sst": np.zeros((1, 12, lat.size, lon.size), dtype=np.float32)},
    )
    pred = sample_predictors(t, prepared)
    assert np.isnan(pred[1, 0]) and np.isfinite(pred[0, 0])


def test_prepare_point_table(tmp_path):
    p = tmp_path / "tracks.nc"
    _make_tracks_nc(p, n=3)
    t = load_tracks(p)
    lon = np.arange(260.0, 321.0, 0.25)
    lat = np.arange(10.0, 66.0, 0.25)
    prepared = PreparedData(
        grid=DomainGrid(lon, lat, patch_size=1),
        coastal_mask=np.ones((lat.size, lon.size), dtype=bool),
        years=np.array([1993]),
        target="fco2",
        arrays={"sst": np.ones((1, 12, lat.size, lon.size), dtype=np.float32)},
    )
    tab = prepare_point_table(t, prepared)
    assert tab.predictors is not None
    assert tab.predictors.shape == (3, 5)
    assert np.allclose(tab.predictors[:, 0], 1.0)
