"""Deterministic synthetic coastal-ocean data for pipeline tests / e2e.

The generator writes monolithic monthly NetCDF files (time, lat, lon) in the
same layout the pipeline expects from SOCAT-like gridded products, with
physically plausible values:

  * SST: seasonal cycle + latitude gradient + a weak warming trend;
  * SSS: seasonal cycle + coastal freshening noise;
  * ADT: seasonal cycle + noise;
  * wind: climatology + noise;
  * atmospheric xCO2: NOAA-like secular rise + seasonal cycle;
  * fCO2 target: driven by SST, seasonal phase, and the xCO2 trend, with
    synthetic SOCAT-like sparsity (deterministic 70% cover), so the
    reconstruction is learnable yet non-trivial.

All randomness is seeded: two runs with the same ``seed``/``n_years``
produce byte-identical files (industrial reproducibility).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from recad.utils.io import touch_dir

DEFAULT_SEED = 12345


def _coastal_mask(lon: np.ndarray, lat: np.ndarray, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Land/coast geometry: coastal = ocean cells within ~1.2 deg of land."""
    lon2, lat2 = np.meshgrid(lon, lat)

    def land(lo0, lo1, la0, la1):
        return (lon2 >= lo0) & (lon2 <= lo1) & (lat2 >= la0) & (lat2 <= la1)

    land_mask = land(7.0, 15.0, 12.0, 17.0) | land(0.0, 3.0, 13.0, 16.0)

    # coastal band: dilate the land mask by ~1.2 deg and take the ring
    dlon = abs(lon[1] - lon[0])
    dlat = abs(lat[1] - lat[0])
    ring_km = 1.2
    coast = np.zeros_like(land_mask, dtype=bool)
    nlon, nlat = lon.size, lat.size
    for i in range(nlat):
        for j in range(nlon):
            if land_mask[i, j]:
                continue
            dist = np.inf
            for ii in range(max(0, i - 3), min(nlat, i + 4)):
                for jj in range(max(0, j - 3), min(nlon, j + 4)):
                    if land_mask[ii, jj]:
                        d = np.hypot((ii - i) * dlat, (jj - j) * dlon)
                        dist = min(dist, d)
            if dist <= ring_km:
                coast[i, j] = True
    return coast


def _fields(n_years: int, lat: np.ndarray, lon: np.ndarray, seed: int = DEFAULT_SEED):
    """Return dict variable name -> [n_years, 12, n_lat, n_lon] float32."""
    rng = np.random.default_rng(seed)
    years = np.arange(1993, 1993 + n_years)
    n_lat, n_lon = lat.size, lon.size
    _, lat2 = np.meshgrid(lon, lat)

    month = np.arange(12)
    sst_clim = 26.0 - 0.5 * (lat2 - 10.0)  # warmer near equator
    seasonal = 3.0 * np.sin(2 * np.pi * (month + 3) / 12)  # N-Hemisphere winter trough
    trend = np.linspace(0.0, 0.3, n_years)  # 0.3 K warming across the record

    sst = np.zeros((n_years, 12, n_lat, n_lon), dtype=np.float32)
    sss = np.zeros_like(sst)
    adt = np.zeros_like(sst)
    wspd = np.zeros_like(sst)
    xco2 = np.zeros_like(sst)
    fco2 = np.zeros_like(sst)

    for y in range(n_years):
        sst[y] = (sst_clim + seasonal[None, :, None, None] + trend[y])[0, :, :, :]
        sss[y] = (
            34.5
            + 0.8 * np.cos(2 * np.pi * (month + 2) / 12)[:, None, None]
            + 0.15 * rng.standard_normal((12, n_lat, n_lon))
        )
        adt[y] = 0.25 * np.sin(2 * np.pi * (month + 1) / 12)[
            :, None, None
        ] + 0.05 * rng.standard_normal((12, n_lat, n_lon))
        wspd[y] = (
            6.0
            + 1.5 * np.sin(2 * np.pi * (month + 6) / 12)[:, None, None]
            + 0.8 * rng.standard_normal((12, n_lat, n_lon))
        )
        xco2[y] = (356.0 + 2.0 * (years[y] - 1993) + 4.0 * np.sin(2 * np.pi * (month + 5) / 12))[
            :, None, None
        ] + 0.2 * rng.standard_normal((12, n_lat, n_lon))
        # physically motivated target: pCO2-air + SST thermal drive + seasonal
        fco2[y] = (
            340.0
            + 9.0 * (sst[y] - 24.0)
            + 12.0 * np.sin(2 * np.pi * (month + 3) / 12)[:, None, None]
            + 1.8 * (years[y] - 1993)
            + 6.0 * rng.standard_normal((12, n_lat, n_lon))
        )

    # SOCAT-like sparsity: keep ~70% of coastal monthly samples deterministically
    keep = rng.random(fco2.shape) < 0.7
    fco2 = np.where(keep, fco2, np.nan)
    return {"sst": sst, "sss": sss, "adt": adt, "wspd": wspd, "xco2air": xco2, "fco2": fco2}


def make_synthetic_workdir(
    workdir: str | Path, n_years: int = 4, seed: int = DEFAULT_SEED
) -> tuple[Path, Path, Path, Path, Path]:
    """Materialise a synthetic raw-data + config workspace.

    Returns (config_path, cache_dir, prepared_path, masks_path, stats_path)
    with the raw data under ``workdir/data/raw`` and a ``test_smoke`` config.
    """
    workdir = Path(workdir)
    raw = touch_dir(workdir / "data" / "raw")
    configs = touch_dir(workdir / "configs")

    # small domain: 1/8 deg over [0, 20)E x [10, 20]N (lon half-open, lat closed,
    # matching the pipeline grid convention)
    res = 0.125
    lon = np.round(np.arange(0.0, 20.0, res), 6)
    lat = np.round(np.arange(10.0, 20.0 + res, res), 6)
    coast = _coastal_mask(lon, lat, seed)
    fields = _fields(n_years, lat, lon, seed)

    time_coords = np.array(
        [np.datetime64(f"{1993 + y:04d}-{m:02d}-15") for y in range(n_years) for m in range(1, 13)]
    )

    for name, values in fields.items():
        da = xr.DataArray(
            values.reshape(-1, lat.size, lon.size),
            dims=("time", "lat", "lon"),
            coords={"time": time_coords, "lat": lat, "lon": lon},
            attrs={"units": "synthetic"},
        )
        da.to_netcdf(raw / f"{name}.nc")
    # static coastal mask (int8 0/1)
    xr.DataArray(
        coast.astype(np.int8), dims=("lat", "lon"), coords={"lat": lat, "lon": lon}
    ).to_netcdf(raw / "mask.nc")

    config_path = configs / "synthetic.yaml"
    config_path.write_text(
        f"""# ReCAD v2.0 synthetic smoke-test configuration (generated).
grid:
  domain: custom
  resolution_deg: {res}
  lon_min: 0.0
  lon_max: 20.0
  lat_min: 10.0
  lat_max: 20.0
  patch_size_cells: 8

coastal_mask:
  method: file
  source_file: {raw.as_posix()}/mask.nc

data:
  root: {raw.as_posix()}
  year_min: 1993
  year_max: {1993 + n_years - 1}
  variable_name: fco2
  templates:
    sst: sst.nc
    sss: sss.nc
    adt: adt.nc
    wspd: wspd.nc
    xco2air: xco2air.nc
    fco2: fco2.nc
    mask: mask.nc

split:
  scheme: random_80_20
  train_fraction: 0.8
  seed: 100
  test_holdout_years: []

model:
  name: st_transformer
  embed_dim: 16
  n_heads: 2
  spatial_layers: 1
  temporal_layers: 1
  mlp_ratio: 2.0
  dropout: 0.0
  pos_encoding: sinusoidal
  head: gaussian
  temporal_window_months: 12

ensemble:
  n_members: 2
  seed_offset: 0
  bootstrap_fraction: 1.0

train:
  n_epochs: 8
  batch_windows: 2
  lr: 2.0e-3
  weight_decay: 1.0e-5
  warmup_epochs: 0
  scheduler: cosine
  grad_clip_norm: 1.0
  use_amp: false
  early_stop_patience: 2
  eval_every_epochs: 1
  device: auto
  checkpoint_dir: {workdir.as_posix()}/outputs/checkpoints

uncertainty:
  n_mc_draws: 3
  propagate_input: false
  combine: rss

output:
  out_dir: {workdir.as_posix()}/outputs
  product_name: ReCAD-synthetic
""",
        encoding="utf-8",
    )

    out = touch_dir(workdir / "outputs")
    return (
        config_path,
        touch_dir(out / "cache"),
        out / "prepared.nc",
        out / "masks.nc",
        out / "feature_stats.json",
    )
