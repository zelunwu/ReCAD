"""Build an auditable CODAP-NA + GLODAP surface carbonate point product.

The output is an observation-indexed NetCDF, not a monthly gridded product.
Source rows are never discarded during cross-product de-duplication. Matching
rows receive the same ``duplicate_group`` and exactly one row is marked
``is_primary=1`` so downstream training does not count a bottle twice.

Only code belongs in git. Write the resulting NetCDF and JSON audit outside
the repository, for example under ``C:/backup/phd/data/processed/carbon``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

CODAP_COLUMNS = [
    "EXPOCODE",
    "Cruise_ID",
    "Observation_type",
    "Accession",
    "Station_ID",
    "Cast_number",
    "Profile_number",
    "Year_UTC",
    "Month_UTC",
    "Day_UTC",
    "Time_UTC",
    "Latitude",
    "Longitude",
    "Niskin_ID",
    "Sample_ID",
    "CTDPRES_dbar",
    "Depth_meter",
    "CTDTEMP_ITS90_deg_C",
    "recommended_Salinity_PSS78",
    "recommended_Salinity_flag",
    "recommended_Oxygen_umol_kg",
    "recommended_Oxygen_flag",
    "recommended_Nitrate_and_Nitrite_umol_kg",
    "recommended_Nitrate_and_Nitrite_flag",
    "Phosphate_umol_kg",
    "Phosphate_flag",
    "Silicate_umol_kg",
    "Silicate_flag",
    "DIC_umol_kg",
    "DIC_flag",
    "TALK_umol_kg",
    "TALK_flag",
    "pH_TS_measured",
    "pH_flag",
    "pH_TS_insitu_measured",
    "pH_TS_insitu_calculated",
    "fCO2_insitu_measured_uatm",
    "fCO2_insitu_calculated_uatm",
    "fCO2_flag",
]

GLODAP_COLUMNS = [
    "G2expocode",
    "G2cruise",
    "G2station",
    "G2cast",
    "G2year",
    "G2month",
    "G2day",
    "G2hour",
    "G2minute",
    "G2latitude",
    "G2longitude",
    "G2bottle",
    "G2pressure",
    "G2depth",
    "G2temperature",
    "G2salinity",
    "G2salinityf",
    "G2oxygen",
    "G2oxygenf",
    "G2nitrate",
    "G2nitratef",
    "G2phosphate",
    "G2phosphatef",
    "G2silicate",
    "G2silicatef",
    "G2tco2",
    "G2tco2f",
    "G2talk",
    "G2talkf",
    "G2fco2",
    "G2fco2f",
    "G2phtsinsitutp",
    "G2phtsinsitutpf",
    "G2doi",
]

NUMERIC_COLUMNS = [
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "latitude",
    "longitude",
    "depth",
    "pressure",
    "temperature",
    "salinity",
    "oxygen",
    "nitrate",
    "phosphate",
    "silicate",
    "dic",
    "ta",
    "ph",
    "fco2",
    "salinity_qc",
    "oxygen_qc",
    "nitrate_qc",
    "phosphate_qc",
    "silicate_qc",
    "dic_qc",
    "ta_qc",
    "ph_qc",
    "fco2_qc",
    "source_row",
    "parameter_method_ta",
    "parameter_method_dic",
    "parameter_method_ph",
    "parameter_method_fco2",
]

STRING_COLUMNS = [
    "obs_id",
    "source",
    "accession",
    "doi",
    "expocode",
    "cruise_id",
    "station_id",
    "cast_number",
    "bottle_id",
    "profile_number",
    "observation_type",
]


def numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.mask(values <= -900)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_string(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).replace({"nan": "", "<NA>": ""})


def normalize_longitude(values: pd.Series) -> pd.Series:
    result = numeric(values)
    return ((result + 180.0) % 360.0) - 180.0


def normalized_expocode(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def north_america(latitude: pd.Series, longitude: pd.Series) -> pd.Series:
    return latitude.between(0.0, 75.0) & longitude.between(-180.0, -45.0)


def effective_depth(depth: pd.Series, pressure: pd.Series) -> pd.Series:
    return depth.where(depth.notna(), pressure)


def empty_standard_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=STRING_COLUMNS + NUMERIC_COLUMNS)


def load_codap(path: Path, max_depth: float) -> pd.DataFrame:
    raw = pd.read_csv(path, usecols=CODAP_COLUMNS, low_memory=False)
    depth = effective_depth(numeric(raw["Depth_meter"]), numeric(raw["CTDPRES_dbar"]))
    lat = numeric(raw["Latitude"])
    lon = normalize_longitude(raw["Longitude"])
    ta_qc = numeric(raw["TALK_flag"])
    dic_qc = numeric(raw["DIC_flag"])
    ta = numeric(raw["TALK_umol_kg"])
    dic = numeric(raw["DIC_umol_kg"])
    good_ta = ta.notna() & ta_qc.eq(2)
    good_dic = dic.notna() & dic_qc.eq(2)
    keep = depth.le(max_depth) & north_america(lat, lon) & (good_ta | good_dic)
    raw = raw.loc[keep].copy()

    measured_ph = numeric(raw["pH_TS_insitu_measured"])
    measured_ph = measured_ph.where(measured_ph.notna(), numeric(raw["pH_TS_measured"]))
    calculated_ph = numeric(raw["pH_TS_insitu_calculated"])
    ph = measured_ph.where(measured_ph.notna(), calculated_ph)
    ph_method = np.select([measured_ph.notna(), calculated_ph.notna()], [1, 3], default=0)

    measured_fco2 = numeric(raw["fCO2_insitu_measured_uatm"])
    calculated_fco2 = numeric(raw["fCO2_insitu_calculated_uatm"])
    fco2 = measured_fco2.where(measured_fco2.notna(), calculated_fco2)
    fco2_method = np.select([measured_fco2.notna(), calculated_fco2.notna()], [1, 3], default=0)

    index = raw.index.to_numpy(np.int64)
    out = pd.DataFrame(
        {
            "obs_id": [f"codap:{i}" for i in index],
            "source": "CODAP-NA_v2026",
            "source_row": index,
            "accession": clean_string(raw["Accession"]),
            "doi": "10.25921/h2ff-9d66",
            "expocode": clean_string(raw["EXPOCODE"]),
            "cruise_id": clean_string(raw["Cruise_ID"]),
            "station_id": clean_string(raw["Station_ID"]),
            "cast_number": clean_string(raw["Cast_number"]),
            "bottle_id": clean_string(raw["Niskin_ID"]),
            "profile_number": clean_string(raw["Profile_number"]),
            "observation_type": clean_string(raw["Observation_type"]),
            "year": numeric(raw["Year_UTC"]),
            "month": numeric(raw["Month_UTC"]),
            "day": numeric(raw["Day_UTC"]),
            "hour": numeric(raw["Time_UTC"]),  # decimal UTC hour in CODAP
            "minute": np.nan,
            "latitude": numeric(raw["Latitude"]),
            "longitude": normalize_longitude(raw["Longitude"]),
            "depth": effective_depth(numeric(raw["Depth_meter"]), numeric(raw["CTDPRES_dbar"])),
            "pressure": numeric(raw["CTDPRES_dbar"]),
            "temperature": numeric(raw["CTDTEMP_ITS90_deg_C"]),
            "salinity": numeric(raw["recommended_Salinity_PSS78"]),
            "salinity_qc": numeric(raw["recommended_Salinity_flag"]),
            "oxygen": numeric(raw["recommended_Oxygen_umol_kg"]),
            "oxygen_qc": numeric(raw["recommended_Oxygen_flag"]),
            "nitrate": numeric(raw["recommended_Nitrate_and_Nitrite_umol_kg"]),
            "nitrate_qc": numeric(raw["recommended_Nitrate_and_Nitrite_flag"]),
            "phosphate": numeric(raw["Phosphate_umol_kg"]),
            "phosphate_qc": numeric(raw["Phosphate_flag"]),
            "silicate": numeric(raw["Silicate_umol_kg"]),
            "silicate_qc": numeric(raw["Silicate_flag"]),
            "dic": numeric(raw["DIC_umol_kg"]).where(numeric(raw["DIC_flag"]).eq(2)),
            "dic_qc": numeric(raw["DIC_flag"]),
            "ta": numeric(raw["TALK_umol_kg"]).where(numeric(raw["TALK_flag"]).eq(2)),
            "ta_qc": numeric(raw["TALK_flag"]),
            "ph": ph,
            "ph_qc": numeric(raw["pH_flag"]),
            "fco2": fco2,
            "fco2_qc": numeric(raw["fCO2_flag"]),
            "parameter_method_ta": np.where(
                numeric(raw["TALK_umol_kg"]).notna() & numeric(raw["TALK_flag"]).eq(2), 1, 0
            ),
            "parameter_method_dic": np.where(
                numeric(raw["DIC_umol_kg"]).notna() & numeric(raw["DIC_flag"]).eq(2), 1, 0
            ),
            "parameter_method_ph": ph_method,
            "parameter_method_fco2": fco2_method,
        }
    )
    return out.reset_index(drop=True)


def transform_glodap(raw: pd.DataFrame, source_rows: np.ndarray, max_depth: float) -> pd.DataFrame:
    depth = effective_depth(numeric(raw["G2depth"]), numeric(raw["G2pressure"]))
    lat = numeric(raw["G2latitude"])
    lon = normalize_longitude(raw["G2longitude"])
    ta_qc = numeric(raw["G2talkf"])
    dic_qc = numeric(raw["G2tco2f"])
    ta = numeric(raw["G2talk"])
    dic = numeric(raw["G2tco2"])
    good_ta = ta.notna() & ta_qc.eq(2)
    good_dic = dic.notna() & dic_qc.eq(2)
    keep = depth.le(max_depth) & north_america(lat, lon) & (good_ta | good_dic)
    raw = raw.loc[keep].copy()
    source_rows = source_rows[keep.to_numpy()]

    out = pd.DataFrame(
        {
            "obs_id": [f"glodap:{i}" for i in source_rows],
            "source": "GLODAPv2.2023",
            "source_row": source_rows,
            "accession": "",
            "doi": clean_string(raw["G2doi"]),
            "expocode": clean_string(raw["G2expocode"]),
            "cruise_id": clean_string(raw["G2cruise"]),
            "station_id": clean_string(raw["G2station"]),
            "cast_number": clean_string(raw["G2cast"]),
            "bottle_id": clean_string(raw["G2bottle"]),
            "profile_number": "",
            "observation_type": "bottle",
            "year": numeric(raw["G2year"]),
            "month": numeric(raw["G2month"]),
            "day": numeric(raw["G2day"]),
            "hour": numeric(raw["G2hour"]),
            "minute": numeric(raw["G2minute"]),
            "latitude": numeric(raw["G2latitude"]),
            "longitude": normalize_longitude(raw["G2longitude"]),
            "depth": effective_depth(numeric(raw["G2depth"]), numeric(raw["G2pressure"])),
            "pressure": numeric(raw["G2pressure"]),
            "temperature": numeric(raw["G2temperature"]),
            "salinity": numeric(raw["G2salinity"]),
            "salinity_qc": numeric(raw["G2salinityf"]),
            "oxygen": numeric(raw["G2oxygen"]),
            "oxygen_qc": numeric(raw["G2oxygenf"]),
            "nitrate": numeric(raw["G2nitrate"]),
            "nitrate_qc": numeric(raw["G2nitratef"]),
            "phosphate": numeric(raw["G2phosphate"]),
            "phosphate_qc": numeric(raw["G2phosphatef"]),
            "silicate": numeric(raw["G2silicate"]),
            "silicate_qc": numeric(raw["G2silicatef"]),
            "dic": numeric(raw["G2tco2"]).where(numeric(raw["G2tco2f"]).eq(2)),
            "dic_qc": numeric(raw["G2tco2f"]),
            "ta": numeric(raw["G2talk"]).where(numeric(raw["G2talkf"]).eq(2)),
            "ta_qc": numeric(raw["G2talkf"]),
            "ph": numeric(raw["G2phtsinsitutp"]).where(numeric(raw["G2phtsinsitutpf"]).eq(2)),
            "ph_qc": numeric(raw["G2phtsinsitutpf"]),
            "fco2": numeric(raw["G2fco2"]).where(numeric(raw["G2fco2f"]).eq(2)),
            "fco2_qc": numeric(raw["G2fco2f"]),
            # GLODAP values are measured data with secondary-QC adjustments.
            "parameter_method_ta": np.where(
                numeric(raw["G2talk"]).notna() & numeric(raw["G2talkf"]).eq(2), 2, 0
            ),
            "parameter_method_dic": np.where(
                numeric(raw["G2tco2"]).notna() & numeric(raw["G2tco2f"]).eq(2), 2, 0
            ),
            "parameter_method_ph": np.where(
                numeric(raw["G2phtsinsitutp"]).notna() & numeric(raw["G2phtsinsitutpf"]).eq(2), 2, 0
            ),
            "parameter_method_fco2": np.where(
                numeric(raw["G2fco2"]).notna() & numeric(raw["G2fco2f"]).eq(2), 2, 0
            ),
        }
    )
    return out.reset_index(drop=True)


def load_glodap(path: Path, max_depth: float, chunksize: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    offset = 0
    for raw in pd.read_csv(
        path,
        usecols=GLODAP_COLUMNS,
        low_memory=False,
        chunksize=chunksize,
        on_bad_lines="skip",
    ):
        rows = np.arange(offset, offset + len(raw), dtype=np.int64)
        frames.append(transform_glodap(raw, rows, max_depth))
        offset += len(raw)
        print(f"  scanned GLODAP rows: {offset:,}", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else empty_standard_frame()


def date_key(row: pd.Series) -> tuple[int, int, int] | None:
    values = (row["year"], row["month"], row["day"])
    if any(pd.isna(v) for v in values):
        return None
    return tuple(int(v) for v in values)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(a)))


def carbon_difference(a: pd.Series, b: pd.Series) -> float:
    differences = []
    for field in ("ta", "dic"):
        if pd.notna(a[field]) and pd.notna(b[field]):
            differences.append(abs(float(a[field]) - float(b[field])))
    return max(differences) if differences else math.inf


def candidate_pairs(codap: pd.DataFrame, glodap: pd.DataFrame) -> list[tuple[float, int, int, int]]:
    """Return (score, codap row, glodap row, reason code) candidates.

    reason=1: same normalized EXPOCODE + date + close position/depth.
    reason=2: different/missing EXPOCODE but exact date, very close sample and
              TA/DIC agree within 5 umol kg-1. The latter is deliberately strict.
    """
    by_expo_date: dict[tuple[str, tuple[int, int, int]], list[int]] = defaultdict(list)
    by_date: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for gi, row in glodap.iterrows():
        date = date_key(row)
        if date is None:
            continue
        by_date[date].append(gi)
        expo = normalized_expocode(row["expocode"])
        if expo:
            by_expo_date[(expo, date)].append(gi)

    candidates: list[tuple[float, int, int, int]] = []
    for ci, row in codap.iterrows():
        date = date_key(row)
        if date is None:
            continue
        expo = normalized_expocode(row["expocode"])
        same = set(by_expo_date.get((expo, date), ())) if expo else set()
        for gi in same:
            other = glodap.loc[gi]
            geo = haversine_km(row.latitude, row.longitude, other.latitude, other.longitude)
            dz = abs(float(row.depth) - float(other.depth))
            if geo <= 2.0 and dz <= 2.0:
                chemistry = carbon_difference(row, other)
                score = geo + dz / 2.0 + min(chemistry, 50.0) / 50.0
                candidates.append((score, ci, gi, 1))

        # Catch only extremely strong aliases. Same-EXPOCODE candidates were
        # already added above and are not duplicated here.
        for gi in by_date.get(date, ()):
            if gi in same:
                continue
            other = glodap.loc[gi]
            geo = haversine_km(row.latitude, row.longitude, other.latitude, other.longitude)
            dz = abs(float(row.depth) - float(other.depth))
            chemistry = carbon_difference(row, other)
            if geo <= 0.25 and dz <= 0.5 and chemistry <= 5.0:
                score = 100.0 + geo + dz + chemistry / 5.0
                candidates.append((score, ci, gi, 2))
    return candidates


def assign_duplicates(codap: pd.DataFrame, glodap: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    pairs = candidate_pairs(codap, glodap)
    used_c: set[int] = set()
    used_g: set[int] = set()
    selected: list[tuple[int, int, int, float]] = []
    for score, ci, gi, reason in sorted(pairs):
        if ci in used_c or gi in used_g:
            continue
        used_c.add(ci)
        used_g.add(gi)
        selected.append((ci, gi, reason, score))

    combined = pd.concat([codap, glodap], ignore_index=True)
    combined["duplicate_group"] = np.int32(0)
    combined["duplicate_reason"] = np.int8(0)
    combined["is_primary"] = np.int8(1)
    offset = len(codap)
    for group, (ci, gi, reason, _score) in enumerate(selected, start=1):
        combined.loc[ci, ["duplicate_group", "duplicate_reason"]] = (group, reason)
        combined.loc[offset + gi, ["duplicate_group", "duplicate_reason", "is_primary"]] = (
            group,
            reason,
            0,
        )

    diffs = []
    for ci, gi, reason, score in selected:
        c, g = codap.loc[ci], glodap.loc[gi]
        diffs.append(
            {
                "reason": reason,
                "score": score,
                "ta_abs_diff": abs(c.ta - g.ta) if pd.notna(c.ta) and pd.notna(g.ta) else None,
                "dic_abs_diff": abs(c.dic - g.dic) if pd.notna(c.dic) and pd.notna(g.dic) else None,
            }
        )
    diff = pd.DataFrame(diffs)
    audit = {
        "matched_pairs": len(selected),
        "same_expocode_pairs": sum(x[2] == 1 for x in selected),
        "strong_alias_pairs": sum(x[2] == 2 for x in selected),
        "ta_abs_diff_median": float(diff.ta_abs_diff.median()) if len(diff) else None,
        "ta_abs_diff_p95": float(diff.ta_abs_diff.quantile(0.95)) if len(diff) else None,
        "dic_abs_diff_median": float(diff.dic_abs_diff.median()) if len(diff) else None,
        "dic_abs_diff_p95": float(diff.dic_abs_diff.quantile(0.95)) if len(diff) else None,
    }
    return combined, audit


def build_time(frame: pd.DataFrame) -> pd.Series:
    hour = numeric(frame["hour"]).fillna(0.0)
    minute = numeric(frame["minute"]).fillna(0.0)
    whole_hour = np.floor(hour).astype(int)
    whole_minute = (minute + (hour - whole_hour) * 60.0).round().astype(int)
    whole_hour = whole_hour + whole_minute // 60
    whole_minute = whole_minute % 60
    parts = pd.DataFrame(
        {
            "year": numeric(frame.year).astype("Int64"),
            "month": numeric(frame.month).astype("Int64"),
            "day": numeric(frame.day).astype("Int64"),
            "hour": whole_hour,
            "minute": whole_minute,
        }
    )
    return pd.to_datetime(parts, errors="coerce", utc=True).dt.tz_localize(None)


def to_dataset(frame: pd.DataFrame, attrs: dict) -> xr.Dataset:
    frame = frame.copy()
    frame["time"] = build_time(frame)
    data_vars: dict[str, tuple[str, np.ndarray]] = {}
    for field in STRING_COLUMNS:
        data_vars[field] = ("obs", clean_string(frame[field]).to_numpy(dtype=str))
    for field in NUMERIC_COLUMNS:
        values = numeric(frame[field])
        if field in {
            "year",
            "month",
            "day",
            "source_row",
            "salinity_qc",
            "oxygen_qc",
            "nitrate_qc",
            "phosphate_qc",
            "silicate_qc",
            "dic_qc",
            "ta_qc",
            "ph_qc",
            "fco2_qc",
            "parameter_method_ta",
            "parameter_method_dic",
            "parameter_method_ph",
            "parameter_method_fco2",
        }:
            data_vars[field] = ("obs", values.fillna(-1).to_numpy(np.int32))
        else:
            data_vars[field] = ("obs", values.to_numpy(np.float32))
    data_vars["time"] = ("obs", frame["time"].to_numpy(dtype="datetime64[ns]"))
    data_vars["duplicate_group"] = ("obs", frame.duplicate_group.to_numpy(np.int32))
    data_vars["duplicate_reason"] = ("obs", frame.duplicate_reason.to_numpy(np.int8))
    data_vars["is_primary"] = ("obs", frame.is_primary.to_numpy(np.int8))

    ds = xr.Dataset(data_vars, coords={"obs": np.arange(len(frame), dtype=np.int64)}, attrs=attrs)
    units = {
        "latitude": "degrees_north",
        "longitude": "degrees_east",
        "depth": "m",
        "pressure": "dbar",
        "temperature": "degree_Celsius",
        "salinity": "1e-3",
        "oxygen": "umol kg-1",
        "nitrate": "umol kg-1",
        "phosphate": "umol kg-1",
        "silicate": "umol kg-1",
        "dic": "umol kg-1",
        "ta": "umol kg-1",
        "fco2": "uatm",
        "ph": "1",
    }
    for name, unit in units.items():
        ds[name].attrs["units"] = unit
    for field in (
        "parameter_method_ta",
        "parameter_method_dic",
        "parameter_method_ph",
        "parameter_method_fco2",
    ):
        ds[field].attrs.update(
            flag_values=np.array([0, 1, 2, 3], dtype=np.int32),
            flag_meanings="unknown measured measured_secondary_qc_adjusted co2sys_derived",
        )
    ds["duplicate_reason"].attrs.update(
        flag_values=np.array([0, 1, 2], dtype=np.int8),
        flag_meanings="unique same_expocode_spatiotemporal strong_spatiotemporal_alias",
    )
    ds["is_primary"].attrs.update(
        flag_values=np.array([0, 1], dtype=np.int8), flag_meanings="duplicate_non_primary primary"
    )
    return ds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codap", required=True, type=Path)
    parser.add_argument("--glodap", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    print("hashing source files...", flush=True)
    codap_sha256 = sha256(args.codap)
    glodap_sha256 = sha256(args.glodap)

    print("reading CODAP...", flush=True)
    codap = load_codap(args.codap, args.max_depth)
    print(f"  CODAP retained: {len(codap):,}", flush=True)
    print("reading GLODAP...", flush=True)
    glodap = load_glodap(args.glodap, args.max_depth, args.chunksize)
    print(f"  GLODAP retained: {len(glodap):,}", flush=True)
    print("matching cross-product duplicate observations...", flush=True)
    combined, duplicate_audit = assign_duplicates(codap, glodap)

    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    attrs = {
        "Conventions": "CF-1.10",
        "featureType": "point",
        "title": "CODAP-NA v2026 and GLODAPv2.2023 North American surface carbonate observations",
        "history": f"created {created} by build_codap_glodap_surface_nc.py",
        "surface_definition": f"depth <= {args.max_depth:g} m; pressure used when depth is missing",
        "spatial_subset": "0 to 75 degrees_north, 180 degrees_west to 45 degrees_west",
        "carbon_qc_policy": "retain rows with TA flag=2 or DIC flag=2; non-good TA/DIC values masked",
        "deduplication_policy": "retain both source rows; CODAP is primary for matched cross-product duplicates",
        "source_codap": str(args.codap),
        "source_codap_sha256": codap_sha256,
        "source_glodap": str(args.glodap),
        "source_glodap_sha256": glodap_sha256,
    }
    ds = to_dataset(combined, attrs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        name: {"zlib": True, "complevel": 5, "shuffle": True}
        for name, variable in ds.data_vars.items()
        if variable.dtype.kind not in "OUSM"
    }
    ds.to_netcdf(args.out, engine="h5netcdf", encoding=encoding)

    primary = combined.is_primary.eq(1)
    audit = {
        "created_utc": created,
        "output": str(args.out),
        "source_codap_sha256": codap_sha256,
        "source_glodap_sha256": glodap_sha256,
        "surface_max_depth_m": args.max_depth,
        "codap_retained_rows": len(codap),
        "glodap_retained_rows": len(glodap),
        "combined_rows": len(combined),
        "primary_rows": int(primary.sum()),
        "primary_good_ta_rows": int((primary & combined.ta.notna()).sum()),
        "primary_good_dic_rows": int((primary & combined.dic.notna()).sum()),
        "primary_good_ta_dic_rows": int(
            (primary & combined.ta.notna() & combined.dic.notna()).sum()
        ),
        "codap_cruises": int(codap.expocode.nunique()),
        "glodap_cruises": int(glodap.expocode.nunique()),
        "primary_cruises": int(combined.loc[primary, "expocode"].nunique()),
        "year_min": int(combined.year.min()),
        "year_max": int(combined.year.max()),
        "duplicates": duplicate_audit,
    }
    audit_path = args.audit or args.out.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    # Round-trip checks catch invalid strings, dimensions and compression now,
    # rather than during the much more expensive training-cache build.
    with xr.open_dataset(args.out, engine="h5netcdf") as check:
        assert check.sizes["obs"] == len(combined)
        assert int(check.is_primary.sum()) == audit["primary_rows"]
        assert check.attrs["carbon_qc_policy"] == attrs["carbon_qc_policy"]
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
