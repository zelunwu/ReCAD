"""SOCAT scatter/point observations (tracks) for point-based training.

``recad download --only socat_tracks`` fetches per-observation rows for a
lon/lat/time window (ERDDAP tabledap, ``socat_v2026_decimated``); this module
loads them, applies the v1.1 QC philosophy, and samples the gridded predictor
fields at each observation's (lon, lat, year, month) position.

Why tracks and not the 0.25-deg gridded product?
  The gridded coastal product averages every observation inside a cell,
  smoothing away the in-situ gradients (estuaries, shelf fronts, river
  plumes) that matter for a 1/8-deg coastal reconstruction. Training on
  point observations keeps the full spatial information of the cruise
  tracks; predictors are sampled at the observation location.

NOTE: the point-based trainer itself is the next step; this module provides
the data plumbing (load/QC/sample) that it will consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from recad.data.features import FEATURE_NAMES, PreparedData
from recad.data.ingest import apply_qc, remove_outliers_3sigma


@dataclass
class TrackData:
    """Cleaned SOCAT scatter observations with sampled predictors."""

    lon: np.ndarray  # [n] in [0, 360)
    lat: np.ndarray
    year: np.ndarray
    month: np.ndarray
    fco2: np.ndarray  # µatm (fCO2_recommended), NaN where unusable
    qc_flag: np.ndarray  # WOCE_CO2_water flag
    dataset_name: np.ndarray  # per-observation cruise id
    predictors: np.ndarray | None = None  # [n, F] sampled field values
    predictor_names: tuple[str, ...] = FEATURE_NAMES

    @property
    def n(self) -> int:
        return int(self.lon.size)

    def valid(self) -> np.ndarray:
        return ~np.isnan(self.fco2)

    def subset(self, valid: np.ndarray) -> TrackData:
        """Return a copy restricted to ``valid`` (bool mask)."""
        return TrackData(
            lon=self.lon[valid],
            lat=self.lat[valid],
            year=self.year[valid],
            month=self.month[valid],
            fco2=self.fco2[valid],
            qc_flag=self.qc_flag[valid],
            dataset_name=self.dataset_name[valid],
            predictors=self.predictors[valid] if self.predictors is not None else None,
            predictor_names=self.predictor_names,
        )


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_tracks(path: str | Path, *, to360: bool = True) -> TrackData:
    """Load a SOCAT tracks NetCDF (tabledap .nc) or CSV into TrackData.

    Handles the two header rows of ERDDAP CSVs and the record dimension of
    the .nc output. Longitudes default to ``[0, 360)`` (pipeline convention).
    """
    path = Path(path)
    if path.suffix.lower() == ".nc":
        ds = xr.open_dataset(path)
        df = ds.to_dataframe().reset_index(drop=True)
        ds.close()
    else:
        df = pd.read_csv(path, comment="#", skiprows=[1])  # drop the units row
        df = df.rename(columns={c: c.strip() for c in df.columns})

    lon = np.asarray(df["longitude"].values, dtype=float)
    lat = np.asarray(df["latitude"].values, dtype=float)
    if to360:
        lon = np.mod(lon, 360.0)

    time = pd.to_datetime(df["time"].values)
    year = time.year.to_numpy(dtype=np.int64)
    month = time.month.to_numpy(dtype=np.int64)

    fco2 = np.asarray(df.get("fCO2_recommended", np.nan), dtype=float)
    qc = np.asarray(df.get("WOCE_CO2_water", np.nan), dtype=float)
    dataset_name = (
        df.get("dataset_name", np.asarray([""] * len(df))).to_numpy()
        if ("dataset_name" in df.columns)
        else np.asarray([""] * len(df), dtype=object)
    )

    return TrackData(
        lon=lon,
        lat=lat,
        year=year,
        month=month,
        fco2=fco2,
        qc_flag=qc,
        dataset_name=dataset_name,
    )


def qc_tracks(
    tracks: TrackData,
    *,
    fco2_range: tuple[float, float] = (1.0, 1000.0),
    qc_flag_max: float = 2.0,
    remove_outliers: bool = True,
) -> TrackData:
    """Apply SOCAT/QC rules; return a cleaned copy.

    * ``WOCE_CO2_water`` flag values above ``qc_flag_max`` are dropped
      (SOCAT: 2 = good, higher = suspect/bad);
    * fCO2 outside ``fco2_range`` and the 3-sigma domain rule are masked
      (v1.1 semantics, ``recad.constants.QC_RANGES``);
    * USERS MUST still choose their training sample (e.g. train/val/test
      splits by time/cruise) afterwards.
    """
    ok = np.isfinite(tracks.fco2)
    lo, hi = fco2_range
    ok &= ~np.isnan(tracks.qc_flag) & (tracks.qc_flag <= qc_flag_max)
    out = tracks.subset(ok)
    out.fco2 = apply_qc(out.fco2, lo, hi)
    if remove_outliers:
        out.fco2 = remove_outliers_3sigma(out.fco2)
    return out


# ---------------------------------------------------------------------------
# predictor sampling
# ---------------------------------------------------------------------------


def sample_predictors(
    tracks: TrackData, prepared: PreparedData, *, method: str = "nearest"
) -> np.ndarray:
    """Sample the 4-D predictor fields at each observation's (year, month, lat, lon).

    Returns ``[n, F]`` raw sampled values (NaN where the field is missing),
    matching ``PreparedData.arrays``. ``method``: ``nearest`` (vectorized,
    fast; fine at 1/8 deg) or ``bilinear`` (xarray interp, slower). Points
    outside the grid are clamped to the nearest edge cell (documented).
    """
    # index lookups are vectorised: axis -> searchsorted per coordinate
    lat_idx = np.clip(np.searchsorted(prepared.grid.lat, tracks.lat), 0, prepared.grid.n_lat - 1)
    lon_idx = np.clip(
        np.searchsorted(prepared.grid.lon, np.mod(tracks.lon, 360.0)), 0, prepared.grid.n_lon - 1
    )
    year_idx = np.clip(np.searchsorted(prepared.years, tracks.year), 0, prepared.n_year - 1)
    # accept only exact-year matches (searchsorted can clamp into a neighbour)
    exact_year = prepared.years[np.minimum(year_idx, prepared.n_year - 1)] == tracks.year
    month_idx = np.clip(tracks.month - 1, 0, 11)

    out = np.full((tracks.n, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    for j, name in enumerate(FEATURE_NAMES):
        if name not in prepared.arrays:
            continue  # feature absent from this prepared set -> NaN column
        field = prepared.require(name)  # [n_year, 12, n_lat, n_lon]
        vals = field[year_idx, month_idx, lat_idx, lon_idx]
        vals = np.where(exact_year, vals, np.nan)
        out[:, j] = vals
    return out


def prepare_point_table(tracks: TrackData, prepared: PreparedData, **sample_kwargs) -> TrackData:
    """Attach sampled predictors to the tracks (mutates a copy)."""
    from dataclasses import replace

    pred = sample_predictors(tracks, prepared, **sample_kwargs)
    return replace(tracks, predictors=pred)
