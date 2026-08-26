"""Data downloader for ReCAD v2.0 (``recad download``).

Downloads the *latest* recommended data stack (docs/data_sources.md) into
``data/raw/<source>/`` — a directory that is permanently git-ignored
(``v2.0/.gitignore`` → ``data/``). Nothing scientific is ever committed; the
manifest + a generated ``MANIFEST.md`` record what was fetched, when, from
which URL, and with which checksum, so the pipeline is reproducible.

Source types
------------
* open, direct download (implements now): NOAA GML atmospheric CO2 (xCO2),
  SOCAT coastal gridded fCO2 via PMEL ERDDAP (with live lon/lat/time subset
  by indices), GSHHG coastline via Zenodo;
* requires registration/credentials (implemented as documented commands +
  env-var tokens; the script refuses to run without them):
  CMEMS GLORYS12 (SSS) and SEALEVEL SSH, CCMP winds (NASA Earthdata),
  GEBCO 2025 bathymetry, OISST v2.1 (NCEI/cloud distribution).

Usage (also exposed as the ``recad download`` CLI):
    python -m recad.data.download --list
    python -m recad.data.download --only xco2air
    python -m recad.data.download --only socat --region -100,-40,10,65 --years 1993,2021
    python -m recad.data.download --doc            # print per-source instructions
"""

from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recad.utils.io import touch_dir
from recad.utils.logging import get_logger

_LOG = get_logger(__name__)

DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw"
"""data/raw under the repository root - git-ignored, see v2.0/.gitignore."""

# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DownloadSpec:
    """One data source in the manifest."""

    name: str  # logical variable name (matches DataConfig.templates keys where relevant)
    title: str
    kind: str  # target | sst | sss | adt | wind | atmosphere | mask | bathymetry
    dest_dir: str  # relative to data/raw
    note: str = ""
    requires_auth: str | None = None  # human instruction if credentials needed
    unzip: bool = False


SOURCES: dict[str, DownloadSpec] = {
    "xco2air": DownloadSpec(
        name="xco2air",
        title="NOAA GML Marine Boundary Layer xCO2 (monthly, global zones)",
        kind="atmosphere",
        dest_dir="xco2air/noaa_gml_mbl",
        note="Converted to pCO2air at in-situ SST/SSS with PyCO2SYS (utils/chem.py). "
        "The MBL reference page (https://gml.noaa.gov/ccgg/mbl/) serves the zonal-mean "
        "surface files v1.1 used (Global_90S90N_Surface.txt); co2_mm_gl.txt is the "
        "global-mean monthly product served on the standard trend page.",
    ),
    "socat": DownloadSpec(
        name="socat",
        title="SOCAT quarter-degree coastal monthly gridded fCO2 (PMEL ERDDAP)",
        kind="target",
        dest_dir="socat",
        note="Latest gridded coastal product (v2026 on ERDDAP; v2025 database documented). "
        "Subsets are built live by index from the .dds metadata and coordinate arrays, "
        "so any lon/lat/time window fits in one NetCDF.",
    ),
    "socat_tracks": DownloadSpec(
        name="socat_tracks",
        title="SOCAT scatter observations (per-cruise tracks, decimated/full)",
        kind="target_scatter",
        dest_dir="socat_tracks",
        note="original scatter observations (per-cruise tracks) at full spatial "
        "precision of the in-situ positions - the 0.25-deg gridded product "
        "averages them away.  Windows are subset via ERDDAP tabledap "
        "(socat_v2026_decimated / socat_v2026_fulldata); columns include "
        "fCO2_recommended and the WOCE_CO2_water QC flag.",
    ),
    "gshhg": DownloadSpec(
        name="gshhg",
        title="GSHHG coastline database v2.3.7 (Zenodo)",
        kind="mask",
        dest_dir="gshhg",
        note="Base for the reproducible coastal mask (coastal_mask.method: distance). "
        "Resolved through the Zenodo API so the file URL does not hard-code.",
        unzip=True,
    ),
    "sst": DownloadSpec(
        name="sst",
        title="OISST v2.1 sea-surface temperature (0.25 deg, daily, no login)",
        kind="sst",
        dest_dir="sst",
        note="NCEI open archive (no account): monthly dirs access/avhrr/YYYYMM/ with "
        "daily files oisst-avhrr-v02r01.YYYYMMDD.nc (~1.7 MB each, sst+err+ice). "
        "recad ingest averages daily -> monthly on the target grid.",
    ),
    "sss": DownloadSpec(
        name="sss",
        title="GLORYS12v1 reanalysis surface salinity (CMEMS GLOBAL_MULTIYEAR_PHY_001_030)",
        kind="sss",
        dest_dir="sss",
        note="1/12 deg global reanalysis 1993-present; CMEMS requires a free account.",
        requires_auth="CMEMS registration: https://data.marine.copernicus.eu ; "
        "download via motu-client with CMEMS_USERNAME/CMEMS_PASSWORD env vars - "
        "see docs/data_download.md",
    ),
    "adt": DownloadSpec(
        name="adt",
        title="CMEMS SEALEVEL_GLO_PHY_L4_MY_008_047 (absolute dynamic topography)",
        kind="adt",
        dest_dir="adt",
        note="0.25 deg daily delayed-time sea level; CMEMS account required.",
        requires_auth="CMEMS registration: https://data.marine.copernicus.eu ; "
        "use the product sub-setting service or motu-client - see docs/data_download.md",
    ),
    "wspd": DownloadSpec(
        name="wspd",
        title="CCMP v3.1 cross-calibrated 10-m winds (NASA/RSS)",
        kind="wind",
        dest_dir="wspd",
        note="6-hourly 0.25 deg; NASA Earthdata login + token required.",
        requires_auth="NASA Earthdata (urs.earthdata.nasa.gov) login; set "
        "EARTHDATA_TOKEN env var before running - see docs/data_download.md",
    ),
    "bathymetry": DownloadSpec(
        name="bathymetry",
        title="GEBCO 2025 global bathymetry grid (15 arc-sec)",
        kind="bathymetry",
        dest_dir="bathymetry",
        note="Optional auxiliary covariate; registration-gated website.",
        requires_auth="GEBCO registration: https://www.gebco.net/data_and_products/"
        "gridded_bathymetry_data/ ; after agreeing to the terms, download the "
        "netCDF grid manually into data/raw/bathymetry/ - see docs/data_download.md",
    ),
}


# ---------------------------------------------------------------------------
# generic file download (resumable)
# ---------------------------------------------------------------------------


def download_file(
    url: str,
    dest: Path,
    *,
    resume: bool = True,
    retries: int = 3,
    timeout: int = 120,
    show_progress: bool = True,
) -> Path:
    """Download ``url`` to ``dest`` with HTTP Range resume and retries.

    ``file://`` URLs are supported (unit tests, local mirrors). Returns dest.
    """
    touch_dir(dest.parent)
    headers: dict[str, str] = {"User-Agent": "recad-v2.0/2.0 (scientific download)"}
    offset = 0
    if resume and dest.is_file():
        offset = dest.stat().st_size
        if offset:
            headers["Range"] = f"bytes={offset}-"

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                content_range = resp.headers.get("Content-Range")
                # Only append when the server actually honoured the Range
                # request (206 + Content-Range). A 200 (or a server that
                # ignores Range, e.g. file://) means the full payload:
                # truncate and rewrite from scratch.
                append = offset > 0 and status == 206 and content_range is not None
                mode = "ab" if append else "wb"
                if not append and offset:
                    offset = 0
                    headers.pop("Range", None)
                total = None
                if content_range and "/" in content_range:
                    total = int(content_range.rsplit("/", 1)[1])
                with open(dest, mode) as fh:
                    _pipe(resp, fh, total if show_progress else None, offset)
            return dest
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and dest.is_file():
                # server has no further bytes: our partial file is already
                # complete (Range from EOF) - treat as success.
                _LOG.info("download complete (server reported 416 at EOF): %s", url)
                return dest
            last_err = exc
            _LOG.warning("download attempt %d failed for %s: %s", attempt + 1, url, exc)
            time.sleep(2 * (attempt + 1))
            if dest.is_file():
                offset = dest.stat().st_size
                headers["Range"] = f"bytes={offset}-"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_err = exc
            _LOG.warning("download attempt %d failed for %s: %s", attempt + 1, url, exc)
            time.sleep(2 * (attempt + 1))
            if dest.is_file():
                offset = dest.stat().st_size
                headers["Range"] = f"bytes={offset}-"
    raise RuntimeError(f"failed to download {url}: {last_err}")


def _pipe(resp, fh, total: int | None, offset: int) -> None:
    import tqdm  # optional; degrades to silent copy

    chunk = 1024 * 1024
    if total and offset:
        total = max(total - offset, 0)
    if total:
        with tqdm.tqdm(total=total, unit="B", unit_scale=True, desc=fh.name) as bar:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                fh.write(block)
                bar.update(len(block))
    else:
        shutil.copyfileobj(resp, fh, chunk)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_checksum(path: Path) -> Path:
    """Sidecar ``<name>.sha256`` next to a downloaded file."""
    root = path.parent
    dest = root / f"{path.name}.sha256"
    dest.write_text(f"{sha256_of(path)}  {path.name}\n", encoding="utf-8")
    return dest


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract a zip or tar(.gz) archive into ``dest_dir`` (path-safe)."""
    touch_dir(dest_dir)
    name = archive_path.name.lower()
    n_entries = 0
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            n_entries = len(zf.namelist())
            zf.extractall(dest_dir)
    elif name.endswith((".tar.gz", ".tgz", ".tar")):
        import tarfile

        with tarfile.open(archive_path, mode="r:*") as tf:
            members = tf.getmembers()
            n_entries = len(members)
            tf.extractall(dest_dir, members=members, filter="data")
    else:  # pragma: no cover - manifest guards this
        raise ValueError(f"unsupported archive type: {archive_path.name}")
    _LOG.info("extracted %s -> %s (%d entries)", archive_path.name, dest_dir, n_entries)


# ---------------------------------------------------------------------------
# ERDDAP subsetting (SOCAT)
# ---------------------------------------------------------------------------


def _fetch_text(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "recad-v2.0/2.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _dds_dimensions(dds: str) -> dict[str, int]:
    """Parse an ERDDAP .dds response into {dim: size}."""
    dims: dict[str, int] = {}
    for line in dds.splitlines():
        line = line.strip()
        if not line.startswith("Float64") and not line.startswith("double"):
            continue
        # e.g. Float64 time[time = 372];
        if "[" not in line:
            continue
        content = line[line.index("[") + 1 : line.rindex("]")]
        dim_name = content.split("=")[0].strip()
        size = int(content.split("=")[1].strip().rstrip(";"))
        dim_name = dim_name.split("_")[0] if "_" in dim_name else dim_name
        dims.setdefault(dim_name, size)
    return dims


def _ident_lonlat(sizes: dict[str, int]) -> tuple[str, str, str]:
    """Find axis names for longitude, latitude, time (name-agnostic)."""
    lower = {k.lower(): k for k in sizes}
    lon = lower.get("longitude") or lower.get("lon") or lower.get("xlon")
    lat = lower.get("latitude") or lower.get("lat") or lower.get("ylat")
    time = lower.get("time")
    if not (lon and lat and time):
        raise ValueError(f"cannot identify lon/lat/time axes in {sorted(sizes)}")
    return lon, lat, time


def _axis_values(url_base: str, dim: str, n: int) -> np.ndarray:
    """Fetch a full coordinate axis from ERDDAP (.csv, small).

    ERDDAP coordinate CSVs start with two header rows (variable name and
    units, e.g. ``time`` / ``UTC``), hence ``skip_header=2``. Some instances
    emit a trailing/blank row beyond the declared dimension, which is
    truncated.
    """
    csv = _fetch_text(f"{url_base}.csv?{dim}")
    table = np.genfromtxt(io.StringIO(csv), delimiter=",", dtype=float, skip_header=2)
    axis = table[:, 1] if table.ndim == 2 else table
    axis = np.asarray(axis, dtype=float).ravel()
    axis = axis[~np.isnan(axis)]
    if axis.size < n:
        raise RuntimeError(f"axis '{dim}' returned {axis.size} values, expected {n}")
    return axis[:n]


def _index_range(axis: np.ndarray, lo: float, hi: float) -> tuple[int, int]:
    idx = np.where((axis >= lo) & (axis <= hi))[0]
    if idx.size == 0:
        raise ValueError(f"no axis values in [{lo}, {hi}]")
    return int(idx.min()), int(idx.max())


def _month_index_range(times: np.ndarray, y0: int, y1: int) -> tuple[int, int]:
    years = np.asarray([str(t)[:4] for t in times], dtype=int)
    sel = np.where((years >= y0) & (years <= y1))[0]
    if sel.size == 0:
        raise ValueError(f"no months in [{y0}, {y1}]")
    return int(sel.min()), int(sel.max())


def build_socat_subset_url(
    base: str,
    variables: tuple[str, ...],
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    year_range: tuple[int, int],
) -> str:
    """Build an ERDDAP griddap .nc subset URL for the SOCAT coastal product.

    Axis sizes come from the .dds; coordinate values and monthly times are
    fetched as small .csv and clamped to index ranges locally, so any
    lon/lat/time window maps to exact index constraints (works across
    ``xlon/ylat`` and ``longitude/latitude`` axis conventions).
    """
    dds = _fetch_text(f"{base}.dds")
    sizes = _dds_dimensions(dds)
    if not sizes:
        raise RuntimeError(f"could not parse .dds for {base}")
    lon_ax, lat_ax, time_ax = _ident_lonlat(sizes)

    lon0, lon1 = lon_range
    lat0, lat1 = lat_range
    ystart, yend = int(year_range[0]), int(year_range[1])

    lon_vals = _axis_values(base, lon_ax, sizes[lon_ax])
    lat_vals = _axis_values(base, lat_ax, sizes[lat_ax])
    time_csv = _fetch_text(f"{base}.csv?{time_ax}")
    table = np.genfromtxt(io.StringIO(time_csv), delimiter=",", dtype=str, skip_header=2)
    times = table if table.ndim == 1 else table[:, 0]

    t0, t1 = _month_index_range(times, ystart, yend)
    # translate the requested window into the dataset's native lon convention
    if lon_vals.min() >= 0:  # 0-360 axis -> shift negative longitudes by +360
        lon0, lon1 = (lon0 + 360) % 360, (lon1 + 360) % 360
    xi, xf = _index_range(lon_vals, lon0, lon1)
    yi, yf = _index_range(lat_vals, lat0, lat1)

    var_spec = ",".join(f"{v}[{t0}:1:{t1}][{yi}:1:{yf}][{xi}:1:{xf}]" for v in variables)
    return f"{base}.nc?{var_spec}"


# ---------------------------------------------------------------------------
# per-source downloaders
# ---------------------------------------------------------------------------


def download_xco2air(root: Path) -> Path:
    dest_dir = touch_dir(root / SOURCES["xco2air"].dest_dir)
    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_gl.txt"
    dest = dest_dir / "co2_mm_gl.txt"
    download_file(url, dest)
    write_checksum(dest)
    return dest


OISST_BASE = (
    "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/"
    "v2.1/access/avhrr/"
)
"""NCEI open archive root for OISST v2.1 daily AVHRR files (no login)."""


def _list_month_files(month_url: str) -> list[str]:
    """List daily .nc files in one OISST YYYYMM/ directory."""
    import re

    txt = _fetch_text(month_url)
    files = sorted(set(re.findall(r'href="([^"]+\.nc)"', txt)))
    return files


def _download_oisst_month(dest_dir: Path, year: int, month: str) -> int:
    """Download one OISST YYYYMM/ directory; return the file count."""
    month_url = f"{OISST_BASE}{year:04d}{month}/"
    files = _list_month_files(month_url)
    for fname in files:
        dest = dest_dir / fname
        download_file(month_url + fname, dest, timeout=300, show_progress=False)
    _LOG.info("OISST: %04d-%s -> %d files", year, month, len(files))
    return len(files)


def download_sst(
    root: Path,
    year_range: tuple[int, int],
    *,
    months: tuple[str, ...] | None = None,
    workers: int = 4,
) -> Path:
    """Download OISST v2.1 daily files for the given years (open archive).

    ``months`` selects 2-digit month names (e.g. ("01", "02")); None = all 12.
    ``workers`` parallel HTTP streams within/across month directories (this
    archive tolerates concurrent connections; the bottleneck is usually the
    link). Files keep their original names under ``data/raw/sst/``; ingest
    averages them to the monthly target field. ~1.7 MB/day -> ~620 MB/year.
    """
    dest_dir = touch_dir(root / SOURCES["sst"].dest_dir)
    y0, y1 = int(year_range[0]), int(year_range[1])
    month_list = months or tuple(f"{m:02d}" for m in range(1, 13))
    tasks = [(y, mm) for y in range(y0, y1 + 1) for mm in month_list]
    total = 0
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_download_oisst_month, dest_dir, y, mm): (y, mm) for y, mm in tasks}
        for future in futures:
            y, mm = futures[future]
            try:
                total += future.result()
            except Exception as exc:
                errors.append(f"{y:04d}-{mm}: {exc}")
                _LOG.error("OISST %04d-%s failed: %s", y, mm, exc)
    if errors:
        _LOG.warning(
            "OISST: %d/%d month dirs had failures -> rerun to resume", len(errors), len(tasks)
        )
    _LOG.info("OISST: %d daily files for %d-%d -> %s", total, y0, y1, dest_dir)
    return dest_dir


def download_socat_tracks(root: Path, region, years, *, kind: str = "decimated") -> Path:
    """SOCAT scatter observations via ERDDAP tabledap (window subset).

    ``kind`` selects ``decimated`` (1/minute, the modelling standard) or
    ``fulldata`` (all observations; much larger). The returned NetCDF keeps
    per-observation rows with ``fCO2_recommended`` and the WOCE QC flag.
    """
    if kind not in ("decimated", "fulldata"):
        raise ValueError(f"unknown tracks kind '{kind}'")
    dest_dir = touch_dir(root / SOURCES["socat_tracks"].dest_dir)
    version = "v2026"
    dataset = f"socat_{version}_{kind}"
    base = f"https://data.pmel.noaa.gov/socat/erddap/tabledap/{dataset}"
    lon0, lon1, lat0, lat1 = region
    y0, y1 = int(years[0]), int(years[1])
    cols = ",".join(
        (
            "time",
            "latitude",
            "longitude",
            "fCO2_recommended",
            "WOCE_CO2_water",
            "dataset_name",
            "sal",
            "temp",
        )
    )
    url = (
        f"{base}.nc?{cols}"
        f"&longitude>={lon0}&longitude<={lon1}"
        f"&latitude>={lat0}&latitude<={lat1}"
        f"&year>={y0}&year<={y1}"
    )
    fname = f"SOCAT_{version}_{kind}_lon{lon0}_{lon1}_lat{lat0}_{lat1}_{y0}-{y1}.nc"
    dest = dest_dir / fname
    download_file(url, dest, timeout=600)
    write_checksum(dest)
    return dest


def download_gshhg(root: Path) -> Path:
    import json as _json

    dest_dir = touch_dir(root / SOURCES["gshhg"].dest_dir)
    record = "7007502"
    api = f"https://zenodo.org/api/records/{record}"
    meta = _json.loads(_fetch_text(api))
    files = meta.get("files", [])
    if not files:
        raise RuntimeError(f"Zenodo record {record} exposes no files")
    # prefer the shapefile archive when present, else the first file
    pick = next(
        (f for f in files if "shp" in f["key"] or "tar" in f["key"] or "zip" in f["key"]), files[0]
    )
    url = pick["links"]["self"]
    dest = dest_dir / pick["key"]
    download_file(url, dest)
    write_checksum(dest)
    extract_archive(dest, dest_dir)
    return dest


def download_socat(root: Path, region, years) -> Path:
    dest_dir = touch_dir(root / SOURCES["socat"].dest_dir)
    base = (
        "https://data.pmel.noaa.gov/socat/erddap/griddap/SOCAT_v2026_qrtrdeg_gridded_coast_monthly"
    )
    # region translated: NACCOM lon [-100, -40] in the dataset's 0-360 axis
    lon_range = region[:2]
    lat_range = region[2:]
    url = build_socat_subset_url(
        base,
        ("coast_fco2_ave_weighted", "coast_sst_ave_weighted", "coast_salinity_ave_weighted"),
        (lon_range[0], lon_range[1]),
        (lat_range[0], lat_range[1]),
        (years[0], years[1]),
    )
    fname = f"SOCAT_v2026_coast_monthly_lon{lon_range[0]}_{lon_range[1]}_lat{lat_range[0]}_{lat_range[1]}_{years[0]}-{years[1]}.nc"
    dest = dest_dir / fname
    download_file(url, dest)
    write_checksum(dest)
    return dest


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------


def download_source(
    name: str,
    root: Path = DATA_ROOT,
    *,
    region=(-100.0, -40.0, 10.0, 65.0),
    years=(1993, 2021),
    kind: str = "decimated",
    months: tuple[str, ...] | None = None,
    workers: int = 4,
) -> Path:
    """Download one manifest source into ``root``; raise on auth-gated ones."""
    if name not in SOURCES:
        raise ValueError(f"unknown source '{name}'; known: {sorted(SOURCES)}")
    spec = SOURCES[name]
    if spec.requires_auth:
        raise RuntimeError(f"'{name}' requires external access: {spec.requires_auth}")
    if name == "xco2air":
        return download_xco2air(root)
    if name == "gshhg":
        return download_gshhg(root)
    if name == "socat":
        return download_socat(root, region, years)
    if name == "socat_tracks":
        return download_socat_tracks(root, region, years, kind=kind)
    if name == "sst":
        return download_sst(root, years, months=months, workers=workers)
    raise ValueError(f"no downloader implemented for '{name}'")


def write_manifest_log(root: Path = DATA_ROOT) -> Path:
    """Record what is on disk in data/raw/MANIFEST.md."""
    lines = ["# ReCAD v2.0 data/raw manifest (auto-generated)", ""]
    for name, spec in SOURCES.items():
        target = root / spec.dest_dir
        files = sorted(p.name for p in target.rglob("*") if p.is_file()) if target.exists() else []
        lines.append(f"## {name} - {spec.title}")
        lines.append(f"- status: {'present' if files else 'not downloaded'}")
        lines.append(f"- files: {', '.join(files) if files else 'none'}")
        if spec.requires_auth:
            lines.append(f"- access: {spec.requires_auth}")
        lines.append("")
    manifest = root.parent / "MANIFEST.md"
    manifest.write_text("\n".join(lines), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recad-download", description=__doc__)
    parser.add_argument("--list", action="store_true", help="print the manifest")
    parser.add_argument("--doc", action="store_true", help="print per-source instructions")
    parser.add_argument(
        "--only",
        default=None,
        help="download one source (xco2air|socat|socat_tracks|gshhg|sst|sss|adt|wspd|bathymetry)",
    )
    parser.add_argument(
        "--kind",
        default="decimated",
        choices=["decimated", "fulldata"],
        help="SOCAT tracks kind (decimated=1/minute standard; fulldata=all obs)",
    )
    parser.add_argument(
        "--region",
        default="-100,-40,10,65",
        nargs="?",
        help="lon0,lon1,lat0,lat1 (SOCAT subset); use = form with negatives, "
        "e.g. --region=-100,-40,10,65",
    )
    parser.add_argument(
        "--years",
        default="1993,2021",
        nargs="?",
        help="year0,year1 (SOCAT subset); e.g. --years=1993,2021",
    )
    parser.add_argument(
        "--months",
        default=None,
        help="OISST months to fetch, comma-separated (e.g. 01,02); default: all 12",
    )
    parser.add_argument(
        "--sst-workers",
        type=int,
        default=4,
        help="parallel OISST month-download threads (default 4)",
    )
    parser.add_argument("--dry-run", action="store_true", help="resolve URLs/plans only")
    args = parser.parse_args(argv)

    if args.list:
        for name, spec in SOURCES.items():
            auth = f"  [auth: {spec.requires_auth[:40]}...]" if spec.requires_auth else ""
            _LOG.info("%-12s %s%s", name, spec.title, auth)
        return 0

    if args.doc:
        from recad.data.download import __doc__ as _d

        print(_d)
        return 0

    try:
        region = tuple(float(x) for x in args.region.split(","))
        years = tuple(int(x) for x in args.years.split(","))
        months = tuple(args.months.split(",")) if args.months else None
    except ValueError:
        parser.error("--region/--years must be four/two comma-separated numbers")

    if args.dry_run:
        _LOG.info("dry run: downloading to %s", DATA_ROOT)
        for name in SOURCES:
            _LOG.info("  %-12s -> data/raw/%s", name, SOURCES[name].dest_dir)
        return 0

    if args.only:
        try:
            path = download_source(
                args.only,
                region=region,
                years=years,
                kind=args.kind,
                months=months,
                workers=args.sst_workers,
            )
            _LOG.info("downloaded %s -> %s", args.only, path)
        except (RuntimeError, ValueError) as exc:
            _LOG.error("%s", exc)
            return 1
    else:
        # open sources by default (large/opt-in ones skipped - use --only)
        for name in SOURCES:
            if SOURCES[name].requires_auth:
                _LOG.warning(
                    "skipping '%s' (external access required): %s",
                    name,
                    SOURCES[name].requires_auth,
                )
                continue
            if name in ("socat_tracks", "sst"):
                _LOG.warning(
                    "skipping '%s' (large opt-in; use --only %s with --years/--months)",
                    name,
                    name,
                )
                continue
            try:
                download_source(name, region=region, years=years)
            except Exception as exc:
                _LOG.error("failed '%s': %s", name, exc)
    write_manifest_log()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
