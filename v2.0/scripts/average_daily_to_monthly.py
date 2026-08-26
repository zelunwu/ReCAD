#!/usr/bin/env python
"""Average daily NetCDF files (e.g. GLORYS daily subsets) into one monthly file.

The CMEMS Toolbox subsets GLORYS12 daily means, but ``recad ingest`` expects
a monthly time axis (one record per month of [year_min, year_max]). This
helper converts a directory/glob of daily files into a single monthly-mean
NetCDF that the pipeline accepts (xarray ``resample(time="MS")``, so
multi-year inputs keep their per-year records).

Usage (run from v2.0/):
    python scripts/average_daily_to_monthly.py \
        --glob "data/raw/sss/glorys12_so_daily_*.nc" \
        --out data/raw/sss/glorys12_so_monthly.nc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xarray as xr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", required=True, help="glob pattern of daily files")
    parser.add_argument("--out", required=True, help="output monthly NetCDF")
    args = parser.parse_args()

    files = sorted(Path().glob(args.glob))
    if len(files) < 2:
        print(f"no/matching files too few for {args.glob} (found {len(files)})")
        return 1

    datasets = [xr.open_dataset(f, decode_timedelta=True) for f in files]
    merged = xr.concat(datasets, dim="time").sortby("time")
    with xr.set_options(keep_attrs=True):
        monthly = merged.resample(time="MS", skipna=True).mean()

    out_attrs = dict(monthly.attrs)
    out_attrs["source"] = "daily files averaged to monthly"
    out_attrs["n_input_files"] = str(len(files))
    monthly.attrs = out_attrs

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    monthly.to_netcdf(args.out)
    n_month = monthly.sizes.get("time", 0)
    print(f"wrote {args.out}: {n_month} months from {len(files)} daily files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
