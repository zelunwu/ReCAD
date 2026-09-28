"""Observation-support lookup for the spatially-weighted CO2SYS closure loss.

The chemistry constraint is only identifiable where TA/DIC observations exist
(1.4% of global coastal cells).  This module loads the pre-computed support
field (scripts/build_chem_support.py -> outputs/chem/chem_support_<tag>.nc)
and returns a weight per sample position, so the training loss can down-weight
(and zero out) the closure penalty where it would be a pseudo-constraint.

Weight policy (from the support file):
    1.0   the cell holds >= 1 GLODAP anchor
    0.1*(1 - d/200km)   within 200 km of an anchor (linear decay)
    0.0   beyond 200 km
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

DEFAULT_SUPPORT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "chem"


@dataclass
class ChemSupport:
    """Support-weight field on the 0.125-degree grid of one domain."""

    lon: np.ndarray  # [n_lon]
    lat: np.ndarray  # [n_lat]
    support: np.ndarray  # [n_lat, n_lon] float32 in [0, 1]
    n_anchors: np.ndarray  # [n_lat, n_lon] int16
    lime: float = 0.0  # floor applied to positive weights (0 = hard zero)

    @classmethod
    def load(cls, tag: str = "global", support_dir: Path | str | None = None) -> "ChemSupport":
        d = Path(support_dir) if support_dir else DEFAULT_SUPPORT_DIR
        path = d / f"chem_support_{tag}.nc"
        if not path.exists():
            raise FileNotFoundError(f"chem support field missing: {path}")
        with Dataset(str(path)) as ds:
            lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
            lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
            sup = np.asarray(ds.variables["support"][:], dtype=np.float32)
            nanc = np.asarray(ds.variables["n_anchors"][:], dtype=np.int16)
        return cls(lon=lon, lat=lat, support=sup, n_anchors=nanc)

    # ------------------------------------------------------------------
    def query(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Nearest-cell support weight for arrays of lon (deg E) / lat (deg N)."""
        lon = np.mod(np.asarray(lon, dtype=np.float64), 360.0)
        lat = np.asarray(lat, dtype=np.float64)
        li = np.clip(np.searchsorted(self.lat, lat), 0, self.lat.size - 1)
        oi = np.clip(np.searchsorted(self.lon, lon, side="right") - 1, 0, self.lon.size - 1)
        w = self.support[li, oi].astype(np.float32)
        if self.lime > 0:
            w = np.where(w > 0, np.maximum(w, self.lime), 0.0).astype(np.float32)
        return w

    # ------------------------------------------------------------------
    def stats(self) -> dict[str, float]:
        s = self.support
        return {
            "frac_on_anchor": float(np.mean(self.n_anchors > 0)),
            "frac_positive": float(np.mean(s > 0)),
            "frac_zero": float(np.mean(s == 0)),
            "mean_support": float(s.mean()),
        }


def masked_weighted_mse(
    pred: np.ndarray, target: np.ndarray, weight: np.ndarray, *, scale: float = 1.0
) -> tuple[float, int]:
    """Sum(w * ((pred-target)/scale)^2) / max(sum(w), eps) with NaN masking."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(t) & (w > 0)
    if not ok.any():
        return 0.0, 0
    r = (p[ok] - t[ok]) / scale
    wv = w[ok]
    return float(np.sum(wv * r * r) / np.sum(wv)), int(ok.sum())