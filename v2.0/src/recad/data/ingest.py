"""Data ingestion primitives: QC, climatology/anomaly, regridding.

These mirror the v1.1 preprocessing steps (``Data_readalldata.ipynb``) with
two hardening changes documented in ``docs/migration_v1_to_v2.md``:
  * NaN-aware linear detrending for the climatology/anomaly split
    (``scipy.signal.detrend`` propagates NaNs; the v2.0 version fits the
    trend only on valid samples);
  * explicit, unit-tested QC ranges (``recad.constants.QC_RANGES``).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import xarray as xr

from recad.constants import N_STD_OUTLIER


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------


def apply_qc(
    values: np.ndarray, vmin: float | None, vmax: float | None
) -> np.ndarray:
    """Set values outside [vmin, vmax] to NaN (bounds inclusive keep)."""
    out = np.asarray(values, dtype=np.float64).copy()
    if vmin is not None:
        out[out < vmin] = np.nan
    if vmax is not None:
        out[out > vmax] = np.nan
    return out


def remove_outliers_3sigma(values: np.ndarray) -> np.ndarray:
    """Set values more than 3 population std devs from the mean to NaN.

    Mirrors the v1.1 rule applied to SOCAT pCO2
    (``Data_readalldata.ipynb`` cell 3). ``N_STD_OUTLIER`` is configurable in
    ``recad.constants``.
    """
    out = np.asarray(values, dtype=np.float64).copy()
    valid = out[~np.isnan(out)]
    if valid.size == 0:
        return out
    mean = np.nanmean(valid)
    std = np.nanstd(valid)
    out[(out < mean - N_STD_OUTLIER * std) | (out > mean + N_STD_OUTLIER * std)] = np.nan
    return out


# ---------------------------------------------------------------------------
# Climatology / anomaly decomposition (v1.1 calc_clim_anom)
# ---------------------------------------------------------------------------


def _nan_aware_linear_detrend(series: np.ndarray) -> np.ndarray:
    """Linear detrend that ignores NaN entries (v2.0 hardening)."""
    series = np.asarray(series, dtype=np.float64)
    n = series.size
    idx = np.arange(n, dtype=np.float64)
    valid = ~np.isnan(series)
    if valid.sum() < 2:
        return series.copy()
    coef = np.polyfit(idx[valid], series[valid], deg=1)
    return series - np.polyval(coef, idx)


def calc_clim_anom(ts: np.ndarray, n_years: int) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a monthly series into (climatology, anomaly).

    Mirrors the v1.1 ``calc_clim_anom``:
      1. reshape into ``[n_years, 12]``;
      2. linear detrend the flattened series (NaN-aware in v2.0);
      3. climatology = year-mean of the detrended monthly values;
      4. anomaly   = raw monthly values minus the climatology.

    Args:
        ts: flat monthly series of length ``n_years * 12``.
        n_years: number of years spanned by ``ts``.

    Returns:
        (clim, anom), both shaped like a 12-month year and the original
        shape respectively (matching v1.1 semantics).
    """
    ts = np.asarray(ts, dtype=np.float64)
    if ts.ndim == 0:
        ts = ts[np.newaxis]
    expected = n_years * 12
    # For multi-dimensional input, operate on the leading time axis.
    leading = ts.reshape(expected, -1)
    n_tails = leading.shape[1]
    clim_flat = np.full((12, n_tails), np.nan)
    anom_flat = np.full_like(leading, np.nan)
    for k in range(n_tails):
        series = leading[:, k]
        detrended = _nan_aware_linear_detrend(series).reshape(n_years, 12)
        clim = np.nanmean(detrended, axis=0)
        anom = series.reshape(n_years, 12) - clim
        clim_flat[:, k] = clim
        anom_flat[:, k] = anom.ravel()
    clim = clim_flat.reshape((12,) + ts.shape[1:])
    anom = anom_flat.reshape(ts.shape)
    return clim, anom


# ---------------------------------------------------------------------------
# Regridding helpers
# ---------------------------------------------------------------------------


def regrid_to_target(
    da: xr.DataArray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    *,
    method: str = "linear",
) -> xr.DataArray:
    """Interpolate a DataArray onto the target lon/lat mesh.

    Longitudes of ``da`` are first wrapped into ``[0, 360)`` so the target
    convention is honoured regardless of the source convention. Dimension
    names are preserved (an explicit rename would leak into the result).
    """
    if "lon" not in da.dims or "lat" not in da.dims:
        raise ValueError("source DataArray must have lon/lat dimensions")
    lon_src = ((da.lon.values % 360.0) + 360.0) % 360.0
    da = da.assign_coords(lon=lon_src).sortby("lon")
    if method == "nearest":
        return da.interp(lon=target_lon, lat=target_lat, method="nearest")
    if method == "linear":
        return da.interp(lon=target_lon, lat=target_lat, method="linear")
    raise ValueError(f"unknown regrid method: {method}")


def flatten_time(ds: xr.Dataset, year_dim: str, month_dim: str) -> xr.Dataset:
    """Fully materialize a (year, month, lat, lon) dataset to NumPy arrays."""
    return ds.transpose(year_dim, month_dim, "lat", "lon")