"""Audit CODAP-NA v2026 surface carbonate observations.

The default, publication-grade counts require WOCE-style QC flag 2 (good).
Flags 3 and 6 are reported separately and can be used only in sensitivity tests.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


REGIONS = {
    "north_america_all": (-180.0, -45.0, 0.0, 75.0),
    "us_east": (-85.0, -60.0, 20.0, 55.0),
    "us_west": (-130.0, -110.0, 24.0, 50.0),
    "alaska": (-180.0, -130.0, 45.0, 75.0),
    "gulf_of_mexico": (-100.0, -80.0, 18.0, 32.0),
}


def number(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > -900 else None


def is_surface(row: dict[str, str], max_depth: float) -> bool:
    depth = number(row["Depth_meter"])
    if depth is not None:
        return depth <= max_depth
    pressure = number(row["CTDPRES_dbar"])
    return pressure is not None and pressure <= max_depth


def in_box(lon: float, lat: float, box: tuple[float, float, float, float]) -> bool:
    lon_min, lon_max, lat_min, lat_max = box
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


def new_stats() -> dict:
    return {
        "rows": 0,
        "cruises": set(),
        "ta_good_rows": 0,
        "ta_good_cruises": set(),
        "dic_good_rows": 0,
        "dic_good_cruises": set(),
        "both_good_rows": 0,
        "both_good_cruises": set(),
        "salinity_rows": 0,
        "salinity_cruises": set(),
    }


def finalize(stats: dict) -> dict:
    return {
        key: (len(value) if isinstance(value, set) else value)
        for key, value in stats.items()
    }


def audit(csv_gz: Path, max_depth: float) -> dict:
    regions = {name: new_stats() for name in REGIONS}
    all_flags = {"TALK_flag": Counter(), "DIC_flag": Counter()}
    years: list[int] = []
    profiles: set[str] = set()

    with gzip.open(csv_gz, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for field in all_flags:
                all_flags[field][row[field]] += 1
            if not is_surface(row, max_depth):
                continue

            lon = number(row["Longitude"])
            lat = number(row["Latitude"])
            if lon is None or lat is None:
                continue
            lon = lon - 360.0 if lon > 180.0 else lon
            cruise = row["EXPOCODE"].strip()
            year = number(row["Year_UTC"])
            if year is not None:
                years.append(int(year))
            profiles.add((cruise, row["Profile_number"].strip()))

            ta_good = number(row["TALK_umol_kg"]) is not None and row["TALK_flag"] == "2"
            dic_good = number(row["DIC_umol_kg"]) is not None and row["DIC_flag"] == "2"
            salinity = number(row["recommended_Salinity_PSS78"]) is not None

            for name, box in REGIONS.items():
                if not in_box(lon, lat, box):
                    continue
                stat = regions[name]
                stat["rows"] += 1
                stat["cruises"].add(cruise)
                if ta_good:
                    stat["ta_good_rows"] += 1
                    stat["ta_good_cruises"].add(cruise)
                if dic_good:
                    stat["dic_good_rows"] += 1
                    stat["dic_good_cruises"].add(cruise)
                if ta_good and dic_good:
                    stat["both_good_rows"] += 1
                    stat["both_good_cruises"].add(cruise)
                if salinity:
                    stat["salinity_rows"] += 1
                    stat["salinity_cruises"].add(cruise)

    return {
        "source": str(csv_gz),
        "surface_definition": f"Depth_meter <= {max_depth:g} m; fall back to CTDPRES_dbar <= {max_depth:g}",
        "primary_carbon_qc": "flag == 2 (good)",
        "sensitivity_only_flags": [3, 6],
        "surface_year_min": min(years),
        "surface_year_max": max(years),
        "surface_profile_ids": len(profiles),
        "all_depth_flag_counts": {
            field: dict(sorted(counts.items())) for field, counts in all_flags.items()
        },
        "regions": {name: finalize(stat) for name, stat in regions.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-depth", type=float, default=5.0)
    args = parser.parse_args()
    result = audit(args.csv, args.max_depth)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
