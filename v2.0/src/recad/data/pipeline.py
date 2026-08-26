"""Pipeline glue: raw NetCDF/zarr files -> PreparedData.

The pipeline is deliberately split in two stages:

1. ``ingest`` - read each raw source (per the ``DataConfig.templates``),
   QC it, regrid it to the target mesh and cache it as a standardised
   monolithic NetCDF (proper dims/coords). This is where the heterogeneous
   source formats (SOCAT 0.25°, OISST daily, GLORYS12 1/12°, CMEMS, CCMP,
   NOAA xCO2) are normalised; see docs/data_sources.md for the full stack.
2. ``preprocess`` - combine the standardised files into a
   :class:`PreparedData` (target + predictors on the common grid), convert
   xCO2air to pCO2air with PyCO2SYS, apply the v1.1 QC rules, and persist it
   (with the split masks) for the training stage.

Both stages are deterministic and covered by unit tests on synthetic data
(``tests/test_pipeline.py``, ``scripts/make_synthetic_data.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from recad.config import Config
from recad.constants import QC_RANGES
from recad.data.features import PreparedData
from recad.data.grid import build_target_grid
from recad.data.ingest import apply_qc, regrid_to_target, remove_outliers_3sigma
from recad.data.split import SplitMasks, make_split_masks
from recad.utils.chem import fco2_to_pco2, xco2air_to_pco2air
from recad.utils.io import touch_dir
from recad.utils.logging import get_logger

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Stage 1: standardised monolithic cache per variable
# ---------------------------------------------------------------------------


def _monthly_time_coords(cfg: Config) -> np.ndarray:
    """Datetime64 mid-month stamps for every month in [year_min, year_max]."""
    stamps = []
    for year in range(cfg.data.year_min, cfg.data.year_max + 1):
        for month in range(1, 13):
            stamps.append(np.datetime64(f"{year:04d}-{month:02d}-15"))
    return np.array(stamps)


def ingest_variable(cfg: Config, name: str, cache_dir: str | Path) -> Path:
    """Read a raw variable, QC/regrid it, and cache the standardised file."""
    template = cfg.data.templates.get(name)
    if not template:
        raise KeyError(f"no template configured for variable '{name}'")
    cache_dir = touch_dir(cache_dir)

    grid = build_target_grid(cfg.grid)
    da = _read_monolithic_or_composite(cfg, name, template, grid)
    da = _qc_variable(da, name)
    da = da.rename(name)  # xarray interp can drop the name; pin it before write
    out_path = cache_dir / f"{name}.nc"
    da.to_netcdf(out_path, engine="netcdf4")
    _LOG.info("ingested '%s' -> %s", name, out_path)
    return out_path


def _read_monolithic_or_composite(cfg: Config, name: str, template: str, grid):
    """Open a monolithic monthly file or assemble {year}/{month} per-file data."""
    tpl = str(Path(cfg.data.root) / template) if not Path(template).is_absolute() else template
    path = Path(tpl)
    if not path.exists():
        raise FileNotFoundError(
            f"data source '{name}' not found at '{path}' - run "
            "scripts/make_synthetic_data.py first, or fix config data.templates"
        )

    time_coords = _monthly_time_coords(cfg)
    if "{year}" in tpl or "{month" in tpl:
        return _read_composite_files(cfg, name, tpl, grid, time_coords)

    ds = xr.open_dataset(path)
    var = name if name in ds.data_vars else (next(iter(ds.data_vars)) if ds.data_vars else name)
    da = ds[var]
    da = _normalise_dims(da, time_coords, cfg, grid)
    return da


def _read_composite_files(cfg, name, tpl, grid, time_coords) -> xr.DataArray:
    """Assemble a monthly series from per-year/per-month files (e.g. OISST)."""
    n_time = (cfg.data.year_max - cfg.data.year_min + 1) * 12
    lat = grid.lat
    lon = grid.lon
    out = np.full((n_time, lat.size, lon.size), np.nan, dtype=np.float32)
    var_candidates = ("sst", "sss", "adt", "sla", "wspd", "chla", "fco2", "xco2air")
    t = 0
    for year in range(cfg.data.year_min, cfg.data.year_max + 1):
        for month in range(1, 13):
            file = Path(tpl.format(year=year, month=month))
            if not file.exists():
                raise FileNotFoundError(f"missing composite file: {file}")
            ds = xr.open_dataset(file)
            var = (
                name
                if name in ds.data_vars
                else next(
                    (v for v in var_candidates if v in ds.data_vars),
                    next(iter(ds.data_vars)),
                )
            )
            plane = ds[var]
            # assume 2-D lat/lon data per file (or squeeze leading day dim)
            plane = plane.squeeze()
            plane = regrid_to_target(plane, lon, lat, method="linear")
            out[t] = plane.values
            t += 1
    return xr.DataArray(
        out, dims=("time", "lat", "lon"), coords={"time": time_coords, "lat": lat, "lon": lon}
    )


def _normalise_dims(da: xr.DataArray, time_coords, cfg, grid) -> xr.DataArray:
    """Bring a DataArray to (time, lat, lon) on the target grid.

    2-D inputs (static fields such as the coastal mask) keep their (lat, lon)
    shape and receive the first timestamp only as a scalar annotation.
    """
    da = da.squeeze()
    if "lat" not in da.dims or "lon" not in da.dims:
        # try common aliases
        da = da.rename(
            {
                k: v
                for k, v in (
                    ("latitude", "lat"),
                    ("longitude", "lon"),
                    ("ylat", "lat"),
                    ("xlon", "lon"),
                )
                if k in da.dims
            }
        )
    if "time" not in da.dims and da.ndim == 3:
        da = da.rename({da.dims[0]: "time"})
    if cfg.data.longitude_shift:
        da = da.assign_coords(lon=da.lon + cfg.data.longitude_shift)
    da = regrid_to_target(da, grid.lon, grid.lat, method="linear")
    if "time" in da.dims:
        if da.sizes["time"] != time_coords.size:
            raise ValueError(f"time axis length {da.sizes['time']} != expected {time_coords.size}")
        da = da.assign_coords(time=time_coords)
    elif time_coords.size:
        da = da.assign_coords(time=time_coords[0])  # static field annotation
    return da


def _qc_variable(da: xr.DataArray, name: str) -> xr.DataArray:
    """Apply the v1.1 QC ranges plus the 3-sigma outlier rule for fCO2."""
    vmin, vmax = QC_RANGES.get(name, (None, None))
    values = apply_qc(da.values, vmin, vmax)
    if name in ("fco2", "pco2"):
        values = remove_outliers_3sigma(values)
    return da.copy(data=values)


# ---------------------------------------------------------------------------
# Stage 2: PreparedData assembly
# ---------------------------------------------------------------------------


def build_prepared(cfg: Config, cache_dir: str | Path) -> PreparedData:
    """Combine the standardised variables into model-ready fields."""
    cache_dir = Path(cache_dir)
    grid = build_target_grid(cfg.grid)
    years = np.array(
        [cfg.data.year_min + y for y in range((cfg.data.year_max - cfg.data.year_min) + 1)]
    )

    n_year = years.size
    shape4 = (n_year, 12, grid.n_lat, grid.n_lon)

    arrays: dict[str, np.ndarray] = {}
    for name in ("sst", "sss", "adt", "wspd", "xco2air", "fco2"):
        path = cache_dir / f"{name}.nc"
        if not path.exists():
            raise FileNotFoundError(
                f"standardised variable '{name}' missing ({path}) - run the 'ingest' step first"
            )
        da = xr.open_dataset(path)[name].load()
        arrays[name] = da.values.reshape(shape4).astype(np.float32)

    # xCO2air -> pCO2air at in-situ SST/SSS (v1.1 cell 8); then v1.1 QC
    pco2air = xco2air_to_pco2air(arrays["xco2air"], arrays["sst"], arrays["sss"]).astype(np.float32)
    vmin_p, vmax_p = QC_RANGES.get("pco2air", (None, None))
    pco2air = apply_qc(pco2air, vmin_p, vmax_p)
    arrays["pco2air"] = pco2air

    # Optional derived target: pCO2 from SOCAT fCO2 (v1.1 conversion)
    if cfg.data.variable_name == "pco2":
        arrays["pco2_target"] = fco2_to_pco2(arrays["fco2"], arrays["sst"]).astype(np.float32)
        target_name = "pco2_target"
    else:
        target_name = "fco2"

    # Coastal mask from a cached mask file, else fallback: any target cell
    mask_path = cache_dir / "mask.nc"
    if mask_path.exists():
        coastal = xr.open_dataset(mask_path)["mask"].values.astype(bool)
    else:
        _LOG.warning("no coastal mask cached; using cells with any observed fCO2")
        coastal = (~np.isnan(arrays["fco2"])).any(axis=(0, 1))

    prepared = PreparedData(
        grid=grid,
        coastal_mask=coastal,
        years=years,
        target=target_name,
        arrays=arrays,
    )
    _LOG.info(
        "prepared data: %d years, %d coastal cells, target='%s'",
        n_year,
        int(coastal.sum()),
        target_name,
    )
    return prepared


# ---------------------------------------------------------------------------
# Persistence of PreparedData
# ---------------------------------------------------------------------------


def save_prepared(prepared: PreparedData, path: str | Path) -> Path:
    """Persist PreparedData to a single NetCDF file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data_vars = {
        "coastal_mask": (("lat", "lon"), prepared.coastal_mask.astype(np.int8)),
    }
    for name, arr in prepared.arrays.items():
        data_vars[name] = (("year", "month", "lat", "lon"), arr)
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "year": ("year", prepared.years),
            "month": ("month", np.arange(1, 13)),
            "lat": ("lat", prepared.grid.lat),
            "lon": ("lon", prepared.grid.lon),
        },
        attrs={"target": prepared.target, "patch_size": int(prepared.grid.patch_size)},
    )
    ds.to_netcdf(p, engine="netcdf4")
    return p


def load_prepared(path: str | Path) -> PreparedData:
    """Reload a persisted PreparedData."""
    from recad.data.grid import DomainGrid

    ds = xr.open_dataset(path)
    grid = DomainGrid(lon=ds.lon.values, lat=ds.lat.values, patch_size=int(ds.attrs["patch_size"]))
    arrays = {
        name: ds[name].values.astype(np.float32)
        for name in ds.data_vars
        if name != "coastal_mask" and ds[name].ndim == 4
    }
    return PreparedData(
        grid=grid,
        coastal_mask=ds["coastal_mask"].values.astype(bool),
        years=ds.year.values.astype(np.int64),
        target=str(ds.attrs.get("target", "fco2")),
        arrays=arrays,
    )


def prepare_split(cfg: Config, prepared: PreparedData) -> SplitMasks:
    """Compute the split masks and fit the normalisation on training only."""
    masks = make_split_masks(cfg.split, prepared)
    prepared.fit_feature_stats(masks.train)
    _LOG.info("split counts: %s", masks.counts())
    return masks


def save_masks(masks: SplitMasks, path: str | Path, years: np.ndarray, grid) -> Path:
    """Persist the split masks as int8 NetCDF (train/val/test)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data_vars = {
        name: (("year", "month", "lat", "lon"), masks.get(name).astype(np.int8))
        for name in masks.names
    }
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "year": ("year", years),
            "month": ("month", np.arange(1, 13)),
            "lat": ("lat", grid.lat),
            "lon": ("lon", grid.lon),
        },
    )
    ds.to_netcdf(p, engine="netcdf4")
    return p


def load_masks(path: str | Path) -> SplitMasks:
    """Reload persisted split masks."""
    ds = xr.open_dataset(path)
    return SplitMasks(
        train=ds["train"].values.astype(bool),
        val=ds["val"].values.astype(bool),
        test=ds["test"].values.astype(bool),
    )
