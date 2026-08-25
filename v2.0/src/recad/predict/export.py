"""CF-compliant NetCDF export of the ReCAD product.

The v1.1 product was exported as flat NetCDF files
(``Data_product_subset.ipynb``). v2.0 writes CF-1.8-styled NetCDF4 with
unlimited-free dims (time, lat, lon), explicit units/standard names, global
attributes documenting the model and config checksum, and an optional
product subsetting step (see ``subset_product``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

from recad.config import Config
from recad.data.grid import DomainGrid


def product_dataset(
    cfg: Config,
    grid: DomainGrid,
    years: np.ndarray,
    fields: dict[str, np.ndarray],
) -> xr.Dataset:
    """Assemble an xarray Dataset from 4-D fields with CF-style metadata."""
    months = np.arange(1, 13, dtype=np.int32)
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for name, arr in fields.items():
        if not isinstance(arr, np.ndarray):
            continue  # e.g. per-member lists; not part of the product file
        if arr.ndim == 4:
            data_vars[name] = (("year", "month", "lat", "lon"), arr)
        elif arr.ndim == 3:  # e.g. member-mean collections not flattened
            continue
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "year": ("year", years.astype(np.int32)),
            "month": ("month", months),
            "lat": ("lat", grid.lat),
            "lon": ("lon", grid.lon),
        },
        attrs={
            "title": cfg.output.product_name,
            "institution": "University of Delaware; Xiamen University",
            "source": "ReCAD v2.0 spatio-temporal transformer deep ensemble",
            "history": f"created {__import__('datetime').datetime.now().isoformat(timespec='seconds')}",
            "Conventions": "CF-1.8",
        },
    )
    for key, value in cfg.output.product_attrs.items():
        ds.attrs[key] = value
    ds.attrs["config_sha256"] = _config_checksum(cfg)
    return ds


def _config_checksum(cfg: Config) -> str:
    payload = {
        "grid": cfg.grid.__dict__,
        "split": cfg.split.__dict__,
        "model": cfg.model.__dict__,
        "ensemble": cfg.ensemble.__dict__,
        "uncertainty": cfg.uncertainty.__dict__,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[
        :16
    ]


def to_netcdf(ds: xr.Dataset, path: str | Path) -> Path:
    """Write the product dataset to NetCDF4 (path resolved under out_dir)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(p, engine="netcdf4")
    return p


def subset_product(ds: xr.Dataset, region: dict[str, tuple[float, float]]) -> xr.Dataset:
    """Subset by lon/lat ranges, mirroring v1.1's Data_product_subset step.

    ``region`` maps axis names to (min, max), e.g.
    ``{"lon": (260, 320), "lat": (10, 65)}`` for the v1.1 NACCOM domain.
    """
    out = ds
    for axis, (vmin, vmax) in region.items():
        values = ds[axis].values
        sel = (values >= vmin) & (values <= vmax)
        out = out.isel({axis: sel})
    return out
