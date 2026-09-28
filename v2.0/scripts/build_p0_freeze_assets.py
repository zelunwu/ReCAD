"""Build the frozen v2.1 P0 spatial, split, support, and compact-cache assets.

Large generated artifacts are written outside Git.  The repository receives
only small manifests that contain their paths, SHA256 values, and protocols.
The formal cutoff is 2024-12-31; incomplete 2026 fields are never sampled.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, save_npz
from shapely import contains_xy
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path(r"C:\backup\phd\data\processed\recad_v2_1")
DEFAULT_CARBON = Path(r"C:\backup\phd\data\processed\carbon\codap_glodap_na_surface5m_v1.nc")
DEFAULT_SOCAT = Path(r"C:\backup\phd\data\raw\socat\SOCATv2026_Coastal.tsv")
LME_URL = ("https://maps.edc.uri.edu/ArcGIS/rest/services/LME/LMEWebMap/"
           "MapServer/2/query?where=1%3D1&outFields=*&returnGeometry=true&f=geojson")
FEATURES = ("sst", "sss", "adt", "wspd", "pco2air")
FORMAL_END_YEAR = 2024


def sha256(path: Path, block: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(block):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def stable_fraction(value: str, salt: str) -> float:
    b = hashlib.sha256(f"{salt}|{value}".encode()).digest()[:8]
    return int.from_bytes(b, "big") / 2**64


def group_key(value: object) -> str:
    clean = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    return clean or "UNKNOWN"


def split_for_group(key: str) -> str:
    u = stable_fraction(key, "recad-v2.1-cruise-split")
    return "locked_test" if u < 0.15 else ("development" if u < 0.30 else "train")


def cv_for_group(key: str) -> int:
    return int(stable_fraction(key, "recad-v2.1-group-cv") * 5) % 5


def coarse_basin(lon180: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Reproducible coarse evaluation basins; LMEs remain the coastal ecology label."""
    out = np.full(lat.shape, 4, np.int8)  # Indian
    out[lat >= 66] = 0                    # Arctic
    out[lat <= -50] = 1                   # Southern
    mid = (lat > -50) & (lat < 66)
    out[mid & (lon180 >= -70) & (lon180 < 20)] = 2       # Atlantic
    out[mid & ((lon180 < -70) | (lon180 >= 145))] = 3    # Pacific
    out[mid & (lon180 >= 20) & (lon180 < 145)] = 4       # Indian
    out[(lat >= 30) & (lat <= 47) & (lon180 >= -6) & (lon180 <= 42)] = 5
    return out


def nearest_indices(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    ix = np.searchsorted(grid, values)
    ix = np.clip(ix, 1, len(grid) - 1)
    left = grid[ix - 1]
    ix -= (np.abs(values - left) <= np.abs(values - grid[ix])).astype(np.int64)
    return ix


def unit_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    lon = np.deg2rad(lon_deg); lat = np.deg2rad(lat_deg); c = np.cos(lat)
    return np.column_stack((c * np.cos(lon), c * np.sin(lon), np.sin(lat)))


def environmental_climatology(ds: netCDF4.Dataset, years_ix: np.ndarray,
                               flats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(flats); sums = np.zeros((n, len(FEATURES)), np.float64)
    sums2 = np.zeros_like(sums); counts = np.zeros_like(sums, np.int32)
    fcount = np.zeros(n, np.int32)
    nlon = len(ds.dimensions["lon"])
    li, oi = flats // nlon, flats % nlon
    # Read latitude strips, then select coastal cells.  This avoids slow
    # netCDF fancy point indexing while keeping peak memory under ~100 MB.
    for yi in years_ix:
        for mi in range(12):
            for j, name in enumerate(FEATURES):
                a = np.asarray(ds.variables[name][yi, mi], np.float32).reshape(-1)[flats]
                ok = np.isfinite(a)
                sums[ok, j] += a[ok]; sums2[ok, j] += a[ok] ** 2; counts[ok, j] += 1
            f = np.asarray(ds.variables["fco2"][yi, mi], np.float32).reshape(-1)[flats]
            fcount += np.isfinite(f)
        print(f"  climatology year {int(ds.variables['year'][yi])}", flush=True)
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    var = np.divide(sums2, counts, out=np.full_like(sums2, np.nan), where=counts > 0) - means**2
    stds = np.sqrt(np.maximum(var, 0))
    return np.column_stack((means, stds)).astype(np.float32), fcount


def label_lmes(lme_path: Path, lon180: np.ndarray, lat: np.ndarray):
    raw = json.loads(lme_path.read_text(encoding="utf-8"))
    labels = np.full(len(lat), -1, np.int16); names = {}
    for feat in raw["features"]:
        p = feat["properties"]
        lid = int(p["LMEs_66.LME_NUMBER"]); names[str(lid)] = p["LMEs_66.LME_NAME"]
        hit = contains_xy(shape(feat["geometry"]), lon180, lat)
        labels[(labels < 0) & hit] = lid
    return labels, names


def build_graph(mask: np.ndarray, flats: np.ndarray, out: Path) -> dict:
    nlat, nlon = mask.shape
    node = np.full(mask.size, -1, np.int32); node[flats] = np.arange(len(flats), dtype=np.int32)
    rows, cols = [], []
    li, oi = flats // nlon, flats % nlon
    for dl, do in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nl = li + dl; no = (oi + do) % nlon
        valid = (nl >= 0) & (nl < nlat)
        dst = np.full(len(flats), -1, np.int32)
        dst[valid] = node[nl[valid] * nlon + no[valid]]
        keep = dst >= 0
        rows.append(np.flatnonzero(keep)); cols.append(dst[keep])
    row = np.concatenate(rows); col = np.concatenate(cols)
    graph = csr_matrix((np.ones(len(row), np.uint8), (row, col)), shape=(len(flats), len(flats)))
    graph.sum_duplicates(); graph.sort_indices(); save_npz(out, graph, compressed=True)
    # Every edge was constructed from cardinal adjacent water cells.
    edge_ok = bool(np.all(mask.reshape(-1)[flats[graph.indices]]))
    return {"nodes": int(graph.shape[0]), "directed_edges": int(graph.nnz),
            "cardinal_water_edges_only": edge_ok, "longitude_wrap": True}


def read_carbon(path: Path) -> pd.DataFrame:
    keep = ["obs_id", "source", "expocode", "cruise_id", "year", "month", "time",
            "latitude", "longitude", "temperature", "salinity", "dic", "ta", "ph", "fco2",
            "salinity_qc", "dic_qc", "ta_qc", "ph_qc", "fco2_qc", "parameter_method_ta",
            "parameter_method_dic", "parameter_method_ph", "parameter_method_fco2",
            "duplicate_group", "is_primary"]
    with xr.open_dataset(path) as d:
        frame = d[keep].to_dataframe().reset_index(drop=True)
    frame = frame[(frame.is_primary == 1) & (frame.year >= 1993) &
                  (frame.year <= FORMAL_END_YEAR)].copy()
    raw_group = frame.expocode.where(frame.expocode.astype(str).str.len() > 2, frame.cruise_id)
    frame["group_key"] = raw_group.map(group_key)
    return frame


def socat_header(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if line.startswith("Expocode\t") and "fCO2rec [uatm]" in line:
                return i
    raise RuntimeError("SOCAT observation header not found")


def build_socat_na(path: Path, lat_grid: np.ndarray, lon_grid: np.ndarray,
                   coastal: np.ndarray) -> pd.DataFrame:
    cols = ("Expocode", "yr", "mon", "longitude [dec.deg.E]", "latitude [dec.deg.N]",
            "fCO2rec [uatm]", "fCO2rec_flag", "sal")
    pieces = []; scanned = kept = 0
    reader = pd.read_csv(path, sep="\t", skiprows=socat_header(path), usecols=list(cols),
                         dtype={"Expocode": str}, chunksize=750_000, low_memory=False)
    for chunk in reader:
        scanned += len(chunk)
        for c in cols[1:]: chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        lon180 = ((chunk[cols[3]].to_numpy(float) + 180) % 360) - 180
        lat = chunk[cols[4]].to_numpy(float); yr = chunk.yr.to_numpy(float)
        flag = chunk[cols[6]].to_numpy(float); f = chunk[cols[5]].to_numpy(float)
        ok = (np.isfinite(f) & (f >= 1) & (f <= 1000) & (flag <= 2) &
              (yr >= 1993) & (yr <= FORMAL_END_YEAR) & (lat >= 0) & (lat <= 75) &
              (lon180 >= -180) & (lon180 <= -45))
        sub = chunk.loc[ok].copy()
        if sub.empty: continue
        x = np.mod(sub[cols[3]].to_numpy(float), 360); y = sub[cols[4]].to_numpy(float)
        li = nearest_indices(y, lat_grid); oi = nearest_indices(x, lon_grid)
        water = coastal[li, oi]
        sub = sub.loc[water].copy(); li = li[water]; oi = oi[water]
        if sub.empty: continue
        sub["group_key"] = sub.Expocode.map(group_key); sub["li"] = li; sub["oi"] = oi
        sub["f_sum"] = sub[cols[5]]; sub["f_n"] = 1
        sal = sub.sal.to_numpy(float); good_sal = np.isfinite(sal) & (sal >= 0) & (sal <= 50)
        sub["sal_sum"] = np.where(good_sal, sal, 0.0); sub["sal_n"] = good_sal.astype(np.int32)
        g = sub.groupby(["group_key", "yr", "mon", "li", "oi"], as_index=False).agg(
            f_sum=("f_sum", "sum"), f_n=("f_n", "sum"),
            sal_sum=("sal_sum", "sum"), sal_n=("sal_n", "sum"))
        pieces.append(g); kept += len(sub)
        if scanned % 3_000_000 < 750_000:
            print(f"  SOCAT scanned={scanned:,} NA-kept={kept:,}", flush=True)
    allg = pd.concat(pieces, ignore_index=True)
    out = allg.groupby(["group_key", "yr", "mon", "li", "oi"], as_index=False).sum()
    out["fco2"] = out.f_sum / out.f_n
    out["salinity"] = np.where(out.sal_n > 0, out.sal_sum / out.sal_n, np.nan)
    return out


def attach_grid(frame: pd.DataFrame, ds: netCDF4.Dataset, spatial: dict[str, np.ndarray],
                lat_grid: np.ndarray, lon_grid: np.ndarray, is_socat: bool) -> pd.DataFrame:
    if is_socat:
        li = frame.li.to_numpy(int); oi = frame.oi.to_numpy(int)
        years = frame.yr.to_numpy(int); months = frame.mon.to_numpy(int)
        frame["latitude"] = lat_grid[li]; frame["longitude"] = lon_grid[oi]
    else:
        li = nearest_indices(frame.latitude.to_numpy(float), lat_grid)
        oi = nearest_indices(np.mod(frame.longitude.to_numpy(float), 360), lon_grid)
        years = frame.year.to_numpy(int); months = frame.month.to_numpy(int)
        frame["li"] = li; frame["oi"] = oi
    yi = years - int(ds.variables["year"][0]); mi = months - 1
    valid = (yi >= 0) & (yi < len(ds.dimensions["year"])) & (mi >= 0) & (mi < 12)
    frame = frame.loc[valid].copy(); li = li[valid]; oi = oi[valid]; yi = yi[valid]; mi = mi[valid]
    flat = li * len(lon_grid) + oi; frame["grid_flat"] = flat
    lookup = spatial["flat_to_node"]; node = lookup[flat]; frame["coastal_node"] = node
    water = node >= 0
    frame = frame.loc[water].copy(); node = node[water]; yi = yi[water]; mi = mi[water]
    li = frame.li.to_numpy(int); oi = frame.oi.to_numpy(int)
    frame["patch_id"] = (li // 32) * int(np.ceil(len(lon_grid) / 32)) + (oi // 32)

    def sample_points(name: str) -> np.ndarray:
        result = np.full(len(frame), np.nan, np.float32)
        key = yi * 12 + mi
        for value in np.unique(key):
            take = np.flatnonzero(key == value)
            slab = np.asarray(ds.variables[name][int(value // 12), int(value % 12)], np.float32)
            result[take] = slab[li[take], oi[take]]
        return result

    for name in FEATURES + ("xco2air",):
        frame[name] = sample_points(name)
    frame["lme_id"] = spatial["lme_id"][node]
    frame["basin_id"] = spatial["basin_id"][node]
    frame["regime_id"] = spatial["regime_id"][node]
    frame["nearest_carbon_km"] = spatial["nearest_carbon_km"][node]
    frame["split"] = frame.group_key.map(split_for_group)
    frame["cv_fold"] = frame.group_key.map(cv_for_group).astype(np.int8)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, default=ROOT / "outputs/prepared_global_p32.nc")
    ap.add_argument("--carbon", type=Path, default=DEFAULT_CARBON)
    ap.add_argument("--socat", type=Path, default=DEFAULT_SOCAT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--regimes", type=int, default=12)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat()

    lme = args.out / "lme_66.geojson"
    if not lme.exists(): urllib.request.urlretrieve(LME_URL, lme)
    carbon = read_carbon(args.carbon)
    # Source- and time-separated carbonate validation: CODAP cruises first
    # observed in 2022--2024 that have no matched GLODAP duplicate.  Freeze the
    # whole cruise before any formal model comparison; a matching SOCAT cruise
    # is sealed as well.
    external_groups = set(carbon.loc[
        carbon.source.astype(str).str.startswith("CODAP") &
        (carbon.duplicate_group == 0) & (carbon.year >= 2022), "group_key"])

    with netCDF4.Dataset(args.prepared) as ds:
        lat_grid = np.asarray(ds.variables["lat"][:]); lon_grid = np.asarray(ds.variables["lon"][:])
        mask = np.asarray(ds.variables["coastal_mask"][:], bool); flats = np.flatnonzero(mask.ravel())
        li, oi = flats // len(lon_grid), flats % len(lon_grid)
        node_lon = lon_grid[oi]; node_lat = lat_grid[li]; lon180 = ((node_lon + 180) % 360) - 180
        years = np.asarray(ds.variables["year"][:], int)
        year_ix = np.flatnonzero((years >= 1993) & (years <= FORMAL_END_YEAR))
        env, socat_month_count = environmental_climatology(ds, year_ix, flats)

        # Target-free regimes: location plus mean/variability of predictors only.
        feat = np.column_stack((node_lat / 90, np.sin(np.deg2rad(node_lon)),
                                np.cos(np.deg2rad(node_lon)), env)).astype(np.float64)
        for j in range(feat.shape[1]):
            med = np.nanmedian(feat[:, j]); feat[~np.isfinite(feat[:, j]), j] = med
            sd = feat[:, j].std(); feat[:, j] = (feat[:, j] - feat[:, j].mean()) / (sd or 1)
        _, regime = kmeans2(feat, args.regimes, iter=60, minit="++", seed=20260928)
        lme_id, lme_names = label_lmes(lme, lon180, node_lat)
        basin_id = coarse_basin(lon180, node_lat)

        cli = nearest_indices(carbon.latitude.to_numpy(float), lat_grid)
        coi = nearest_indices(np.mod(carbon.longitude.to_numpy(float), 360), lon_grid)
        cflat = cli * len(lon_grid) + coi
        flat_to_node = np.full(mask.size, -1, np.int32); flat_to_node[flats] = np.arange(len(flats))
        cnode = flat_to_node[cflat]; valid_c = cnode >= 0
        obs_count = np.bincount(cnode[valid_c], minlength=len(flats)).astype(np.int32)
        pairs = pd.DataFrame({"node": cnode[valid_c], "group": carbon.group_key.to_numpy()[valid_c]}).drop_duplicates()
        cruise_count = np.bincount(pairs.node, minlength=len(flats)).astype(np.int32)
        tree = cKDTree(unit_xyz(node_lon[cnode[valid_c]], node_lat[cnode[valid_c]]))
        chord, _ = tree.query(unit_xyz(node_lon, node_lat), k=1)
        nearest_km = (6371.0088 * 2 * np.arcsin(np.clip(chord / 2, 0, 1))).astype(np.float32)
        spatial = {"flat_to_node": flat_to_node, "lme_id": lme_id, "basin_id": basin_id,
                   "regime_id": regime.astype(np.int16), "nearest_carbon_km": nearest_km}

        graph_path = args.out / "coastal_graph_v2.1.npz"
        graph_audit = build_graph(mask, flats, graph_path)
        spatial_path = args.out / "spatial_support_v2.1.nc"
        sds = xr.Dataset(
            {"grid_flat": ("node", flats.astype(np.int32)), "lat_index": ("node", li.astype(np.int16)),
             "lon_index": ("node", oi.astype(np.int16)), "latitude": ("node", node_lat.astype(np.float32)),
             "longitude": ("node", node_lon.astype(np.float32)), "lme_id": ("node", lme_id),
             "basin_id": ("node", basin_id), "regime_id": ("node", regime.astype(np.int16)),
             "carbon_observation_count": ("node", obs_count), "carbon_cruise_count": ("node", cruise_count),
             "carbon_nearest_support_km": ("node", nearest_km),
             "socat_observed_gridmonth_count": ("node", socat_month_count)},
            attrs={"generated_utc": generated, "formal_period": "1993-2024",
                   "regime_inputs": "lat, lon sin/cos, predictor means/stds: " + ",".join(FEATURES),
                   "regime_excludes_targets": "fco2,TA,DIC and all observation labels",
                   "lme_names_json": json.dumps(lme_names, ensure_ascii=False),
                   "basin_names": "0 Arctic;1 Southern;2 Atlantic;3 Pacific;4 Indian;5 Mediterranean"})
        enc = {v: {"zlib": True, "complevel": 4} for v in sds.data_vars}
        sds.to_netcdf(spatial_path, engine="netcdf4", encoding=enc)

        print("Scanning SOCAT and building North-American compact caches", flush=True)
        socat = build_socat_na(args.socat, lat_grid, lon_grid, mask)
        carbon_cache = attach_grid(carbon, ds, spatial, lat_grid, lon_grid, False)
        socat_cache = attach_grid(socat, ds, spatial, lat_grid, lon_grid, True)

    carbon_cache.loc[carbon_cache.group_key.isin(external_groups), "split"] = "external_independent"
    socat_cache.loc[socat_cache.group_key.isin(external_groups), "split"] = "external_independent"

    # Keep explicit label provenance and masks. Calculated fCO2 (method 3) is
    # retained for audits but excluded from independent fCO2 supervision.
    carbon_cache["label_sss_ok"] = np.isfinite(carbon_cache.salinity) & (carbon_cache.salinity_qc == 2)
    carbon_cache["label_ta_ok"] = np.isfinite(carbon_cache.ta) & (carbon_cache.ta_qc == 2)
    carbon_cache["label_dic_ok"] = np.isfinite(carbon_cache.dic) & (carbon_cache.dic_qc == 2)
    carbon_cache["label_fco2_measured_ok"] = (np.isfinite(carbon_cache.fco2) &
        (carbon_cache.fco2_qc == 2) & carbon_cache.parameter_method_fco2.isin([1, 2]))
    carbon_path = args.out / "na_carbon_cache_v2.1.parquet"
    socat_path = args.out / "na_socat_cache_v2.1.parquet"
    carbon_cache.to_parquet(carbon_path, index=False, compression="zstd")
    socat_cache.to_parquet(socat_path, index=False, compression="zstd")

    groups = pd.concat([
        carbon_cache[["group_key", "split", "cv_fold"]].assign(dataset="carbon"),
        socat_cache[["group_key", "split", "cv_fold"]].assign(dataset="socat")
    ]).drop_duplicates().sort_values(["group_key", "dataset"])
    group_year = pd.concat([
        carbon_cache[["group_key", "year"]],
        socat_cache[["group_key", "yr"]].rename(columns={"yr": "year"})
    ]).groupby("group_key").year.max()
    groups["max_year"] = groups.group_key.map(group_year)
    groups["forward_split"] = np.where(groups.group_key.isin(external_groups), "external_independent",
        np.where(groups.max_year <= 2018, "train",
        np.where(groups.max_year <= 2021, "development", "locked_test")))
    split_path = args.out / "split_manifest_v2.1.parquet"
    groups.to_parquet(split_path, index=False, compression="zstd")

    leakage = {
        "primary_carbon_rows": int(len(carbon_cache)),
        "target_counts": {k: int(carbon_cache[k].sum()) for k in
            ("label_sss_ok", "label_fco2_measured_ok", "label_ta_ok", "label_dic_ok")},
        "coobserved": {
            "ta_dic": int((carbon_cache.label_ta_ok & carbon_cache.label_dic_ok).sum()),
            "all_four_measured": int((carbon_cache.label_sss_ok & carbon_cache.label_fco2_measured_ok &
                                      carbon_cache.label_ta_ok & carbon_cache.label_dic_ok).sum())},
        "fco2_method_counts": {str(k): int(v) for k, v in
            carbon_cache.parameter_method_fco2.value_counts(dropna=False).items()},
        "leakage_rule": "parameter_method_fco2=3 is derived and forbidden as an independent fCO2 label",
        "predictor_regimes_target_free": True,
    }
    split_audit = {
        "rows": int(len(groups)), "unique_groups": int(groups.group_key.nunique()),
        "split_group_counts": groups.drop_duplicates("group_key").split.value_counts().to_dict(),
        "cv_group_counts": {str(k): int(v) for k, v in groups.drop_duplicates("group_key").cv_fold.value_counts().items()},
        "group_cross_split_violations": int((groups.groupby("group_key").split.nunique() > 1).sum()),
        "group_cross_cv_violations": int((groups.groupby("group_key").cv_fold.nunique() > 1).sum()),
        "forward_group_cross_split_violations": int((groups.groupby("group_key").forward_split.nunique() > 1).sum()),
        "whole_region_protocol": "hold out one lme_id at a time using sample lme_id in compact caches",
    }
    if any(split_audit[k] for k in ("group_cross_split_violations", "group_cross_cv_violations",
                                    "forward_group_cross_split_violations")):
        raise RuntimeError(f"split leakage detected: {split_audit}")

    coverage_path = args.out / "coverage_independence_audit_v2.1.json"
    atomic_json(coverage_path, {"generated_utc": generated, "formal_cutoff": "2024-12-31",
        "excluded": "2025-2026 (2026 incomplete; conservative common cutoff)",
        "spatial": {"coastal_nodes": int(len(flats)), "lme_assigned": int((lme_id >= 0).sum()),
                    "regimes": int(args.regimes), "graph": graph_audit},
        "splits": split_audit, "leakage": leakage,
        "cache_rows": {"socat_cruise_gridmonth": int(len(socat_cache)), "carbon_primary": int(len(carbon_cache))}})

    external_manifest = {
        "schema_version": "2.1", "frozen_utc": generated,
        "carbonate_external": {
            "source": "CODAP-NA v2026 cruises from 2022-2024 unmatched to GLODAPv2.2023",
            "selection_uses_labels": False, "grouping": "whole normalized Expocode/cruise",
            "group_count": len(external_groups), "group_keys": sorted(external_groups),
            "isolation": "split=external_independent in both compact caches; forbidden for training, tuning, checkpoint selection and ablation decisions",
            "deduplication": "duplicate_group=0 only; matched GLODAP rows excluded from this external set"},
        "sss_fco2_external": {
            "source": "prospective SOCAT release after v2026",
            "eligibility": "new Expocode/cruise groups absent from frozen SOCATv2026 SHA256, observation time after 2024-12-31",
            "labels_available_at_freeze": False,
            "isolation": "do not download or inspect labels until P3 model and analysis plan are frozen"},
        "access_rule": "external labels may be opened once for P3 final evaluation; never for model selection"}
    external_path = ROOT / "configs/frozen/external_validation_manifest_v2.1.json"
    atomic_json(external_path, external_manifest)

    artifacts = [spatial_path, graph_path, split_path, carbon_path, socat_path, coverage_path]
    hashes = {str(p): sha256(p) for p in artifacts}
    print("Hashing formal inputs (prepared file is large)", flush=True)
    inputs = {str(p): sha256(p) for p in (args.prepared, args.carbon, args.socat, lme)}
    manifest = {
        "schema_version": "2.1", "frozen_utc": generated, "formal_time_range": "1993-01 through 2024-12",
        "exclusion": {"2025": "excluded for conservative complete common window",
                      "2026": "excluded because source year is incomplete at freeze time"},
        "inputs_sha256": inputs, "artifacts_sha256": hashes,
        "qc": {"SOCAT": "fCO2 1..1000 uatm and fCO2rec_flag<=2; salinity 0..50",
               "carbon": "is_primary=1; TA/DIC/SSS require QC=2; measured fCO2 requires QC=2 and method in {1,2}"},
        "target_provenance": {"SSS": "SOCAT in-situ sal", "fCO2": "SOCAT fCO2rec",
            "TA": "CODAP/GLODAP observed or adjusted with method retained", "DIC": "CODAP/GLODAP observed or adjusted with method retained",
            "forbidden": "calculated carbon-product fCO2 method=3 is audit-only"},
        "split_protocol": {"group": "normalized Expocode/cruise key", "assignment": "SHA256 salted deterministic",
            "fractions": {"train": 0.70, "development": 0.15, "locked_test": 0.15},
            "grouped_cv": "5-fold salted SHA256", "whole_region": "leave-one-LME-out",
            "forward_chain": "cruise max year: <=2018 train, 2019-2021 development, 2022-2024 locked_test"},
        "independent_validation": {"path": "configs/frozen/external_validation_manifest_v2.1.json",
                                   "sha256": sha256(external_path)},
        "audit": str(coverage_path), "split_manifest": str(split_path),
    }
    manifest_path = ROOT / "configs/frozen/data_manifest_v2.1.json"
    atomic_json(manifest_path, manifest)
    print(f"DONE manifest={manifest_path} manifest_sha256={sha256(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
