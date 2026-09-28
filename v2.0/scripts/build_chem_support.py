"""Build the carbonate-chemistry observation-support field (chem_support.nc).

Rationale (see docs/backup/ta_dic_constraint_review.md):
    Only 1.4% of the 0.125-deg global coastal cells hold any TA/DIC anchor
    (3.3-3.5% for the US coasts). Applying a uniform CO2SYS closure penalty
    over the whole domain therefore imposes a *pseudo-constraint* on 98.6% of
    cells where (TA, DIC) is unidentifiable from observations. This script
    quantifies the support so the chemistry loss can be weighted by it.

Output (one file per domain in outputs/chem/):
    chem_support_<tag>.nc with variables on the (lat, lon) grid:
        n_anchors   int16   number of GLODAP TA/DIC anchors in the cell
        dist_km     float32 great-circle distance to the nearest anchor (km)
        support     float32 chemistry-loss weight in [0, 1] (see policy below)

Weight policy (three tiers):
    support = 1.0   if the cell itself holds >=1 anchor
    support = w_near(dist_km)  for cells within NEAR_KM of an anchor
              w_near = 0.1 * (1 - dist/NEAR_KM)      (linear decay 0.1 -> 0)
    support = 0.0   beyond NEAR_KM
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recad.data.grid import build_target_grid  # noqa: E402
from recad.config import Config  # noqa: E402

ANCHORS = Path(r"C:\backup\phd\data\raw\ocads\glodapv22023\glodap_surface5m_split.csv")
NEAR_KM = 200.0  # neighbourhood radius for the decay tier


def hav_km(lon1, lat1, lon2, lat2):
    """Great-circle distance (km) between arrays of lon/lat (degrees)."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def nearest_anchor_distance(grid, alon, alat, chunk: int = 200_000):
    """Distance (km) from every grid cell to the nearest anchor.

    Uses a 3-D Cartesian KD-tree on the unit sphere: exact great-circle
    nearest neighbour in O(n log n), and it never materialises the
    (cells x anchors) broadcast (which would need ~64 GB here).
    """
    from scipy.spatial import cKDTree

    def to_xyz(lon, lat):
        lo, la = np.radians(lon), np.radians(lat)
        return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])

    tree = cKDTree(to_xyz(alon, alat))
    lon_g, lat_g = np.meshgrid(grid.lon, grid.lat)
    flat = to_xyz(lon_g.ravel(), lat_g.ravel())
    dist = np.empty(flat.shape[0], dtype=np.float64)
    for i0 in range(0, flat.shape[0], chunk):
        i1 = min(i0 + chunk, flat.shape[0])
        d, _ = tree.query(flat[i0:i1], k=1)
        dist[i0:i1] = d
    # chord length -> great-circle distance
    dist = 2.0 * 6371.0 * np.arcsin(np.clip(dist / 2.0, 0, 1))
    return dist.reshape(grid.n_lat, grid.n_lon).astype(np.float32)


def build(tag: str, bounds: tuple[float, float, float, float] | None, out_dir: Path) -> Path:
    cfg = Config.from_yaml(str(ROOT / "configs" / "global_1over8.yaml"))
    if bounds is not None:
        lon0, lon1, lat0, lat1 = bounds
        cfg = cfg  # grid config untouched; we crop afterwards
    grid = build_target_grid(cfg.grid)

    df = pd.read_csv(ANCHORS)
    alon = df["lon"].to_numpy(float) % 360.0
    alat = df["lat"].to_numpy(float)
    keep = np.isfinite(alon) & np.isfinite(alat)
    if bounds is not None:
        lon0, lon1, lat0, lat1 = bounds
        keep &= (alon >= lon0) & (alon < lon1) & (alat >= lat0) & (alat <= lat1)
    alon, alat = alon[keep], alat[keep]
    print(f"[{tag}] {len(alon)} anchors used")

    # cell counts
    li = np.clip(np.searchsorted(grid.lat, alat), 0, grid.n_lat - 1)
    oi = np.clip(np.searchsorted(grid.lon, alon, side="right") - 1, 0, grid.n_lon - 1)
    n_anchors = np.zeros((grid.n_lat, grid.n_lon), dtype=np.int16)
    np.add.at(n_anchors, (li, oi), 1)

    dist = nearest_anchor_distance(grid, alon, alat)
    support = np.zeros_like(dist, dtype=np.float32)
    on_anchor = n_anchors > 0
    support[on_anchor] = 1.0
    near = (~on_anchor) & (dist < NEAR_KM)
    support[near] = 0.1 * (1.0 - dist[near] / NEAR_KM)

    ds = xr.Dataset(
        {
            "n_anchors": (("lat", "lon"), n_anchors),
            "dist_km": (("lat", "lon"), dist),
            "support": (("lat", "lon"), support),
        },
        coords={"lat": grid.lat, "lon": grid.lon},
        attrs={
            "source": str(ANCHORS),
            "near_km": NEAR_KM,
            "policy": "1.0 on anchor cells; 0.1*(1-d/near_km) within near_km; 0 beyond",
            "n_anchors_total": int(len(alon)),
        },
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"chem_support_{tag}.nc"
    ds.to_netcdf(path)
    n_on = int(on_anchor.sum())
    n_near = int(near.sum())
    n_zero = int((support == 0).sum())
    print(f"[{tag}] support: on-anchor {n_on} | near {n_near} | zero {n_zero} | saved {path}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "chem"))
    args = ap.parse_args()
    out = Path(args.out)
    build("global", None, out)
    build("us_east", (275.0, 300.0, 20.0, 55.0), out)
    build("us_west", (225.0, 250.0, 24.0, 50.0), out)
    build("naccom", (260.0, 320.0, 10.0, 65.0), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())