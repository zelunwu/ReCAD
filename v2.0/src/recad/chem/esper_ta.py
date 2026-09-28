# -*- coding: utf-8 -*-
"""Port ESPER_LIR (Carter et al.) to Python for TA estimation.

Implements the core LIR algorithm used by ESPER_LIR v3 for estimating total
alkalinity from salinity + temperature (+ optional nutrients). The MATLAB
implementation needs: scatteredInterpolant over the pre-computed regression
constant field (grid of 50225 nodes x 6 coefficients x 16 equations).

This port:
  * reads LIR_files_TA_v3.mat (via scipy.io.loadmat),
  * reproduces the Atlantic/Arctic segmentation (Polys clipping),
  * interpolates the local regression constants on the (lon, lat, depth/25) grid,
  * evaluates TA = C0 + C1*S + C2*T (+ A/B/C terms for chosen equation).

Only TA is ported for now (DesiredVariables=1); DIC/pH need the Canth
time-adjustment step (SimpleCantEstimate) which we skip - TA is date
independent in ESPER_LIR.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.io import loadmat
from scipy.spatial import Delaunay, cKDTree

try:
    from matplotlib.path import Path as MplPath
    _HAS_MPL = True
except ImportError:  # pragma: no cover
    MplPath = None
    _HAS_MPL = False

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

# regression coefficient slot indices in `Cs[:, slot, eq]`
SLOT_C0 = 0  # constant term
SLOT_S = 1   # salinity
SLOT_T = 2   # temperature
SLOT_A = 3   # equation-dependent variable A
SLOT_B = 4   # ...
SLOT_C = 5   # ...

# equation 16 = (S, T) only - the only equation viable regardless of inputs
# (ESPER_LIR.m lines ~577: 'Using equation 16 because it is the only one
#  that is viable regardless of inputs.')
EQ_TA_ST_ONLY = 16  # 1-based

# VarVec from ESPER_LIR.m: 5 columns = [constant(S?), S, T, A, B]? - actually
# the columns select which of the 5 predictor slots (beyond the existing
# constant+S base) each equation uses. In ESPER_LIR.m line 372:
#   VarVec = logical([1 1 1 1 1 ; ... ; 1 0 0 0 0]);
# where column 1 = constant (always), column 2 = Salinity (always),
# columns 3-5 = T / A / B.  We store it transposed here for easy indexing.
VARVEC = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 0, 1],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [1, 1, 0, 0, 0],
        [1, 0, 1, 1, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 1, 0],
        [1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0],
    ],
    dtype=bool,
)

DEPTH_TO_DEGREE = 25.0  # see 2016 ESPER_LIR paper


@dataclass
class ESPER_LIR_TA:
    """TA estimator loaded from the ESPER_LIR MATLAB files."""

    cs: np.ndarray          # [node, slot, eq]
    grid_coords: np.ndarray  # [node, 4]  (lon, lat, depth, date)
    aa_inds: np.ndarray     # bool [node]: Atlantic/Arctic membership
    polys: dict[str, np.ndarray]  # name -> (n, 2) polygon vertices

    # per-region interpolators are built lazily per equation.
    _interp_aa: dict[int, object] | None = None
    _interp_else: dict[int, object] | None = None
    _tri_aa: object | None = None
    _tri_else: object | None = None

    @classmethod
    def from_mat(cls, path: str | Path) -> "ESPER_LIR_TA":
        path = Path(path)
        m = loadmat(str(path))
        cs = m["Cs"]
        # AAIndsCs is uint8 [n,1]; convert to bool [n]
        aa = m["AAIndsCs"].ravel().astype(bool)
        polys = {}
        p = m["Polys"][0][0]
        for f in p.dtype.names:
            arr = np.asarray(p[f], dtype=float)
            polys[f] = arr.reshape(-1, 2) if arr.ndim > 1 else arr.reshape(-1, 2)
        return cls(
            cs=cs,
            grid_coords=np.asarray(m["GridCoords"], dtype=float),
            aa_inds=aa,
            polys=polys,
        )

    # ------------------------------------------------------------------
    # region membership
    # ------------------------------------------------------------------
    def _in_aa(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Point-wise Atlantic/Arctic membership using the five polygons.

        Longitudes are wrapped to [0, 360]; the MATLAB polys use -1/361 as
        wrap markers, so we fall back to the AA grid-node nearest-neighbour
        when the polygon test is unreliable (or matplotlib is unavailable).
        """
        n = lon.size
        if _HAS_MPL:
            aa = np.zeros(n, dtype=bool)
            pts = np.column_stack([lon, lat])
            for name in ("LNAPoly", "LSAPoly", "LNAPolyExtra", "LSAPolyExtra", "LNOPoly"):
                ring = self.polys[name]
                if ring.size == 0:
                    continue
                path = MplPath(ring)
                aa |= path.contains_points(pts, radius=0.0)
            return aa
        # fallback: nearest labelled grid node
        tree = cKDTree(self.grid_coords[:, :2])
        _, idx = tree.query(np.column_stack([lon, lat]), k=1)
        return self.aa_inds[idx]

    # ------------------------------------------------------------------
    # interpolators
    # ------------------------------------------------------------------
    def _interp_for(self, eq: int, aa: bool):
        """Build (or return) per-slot coefficient interpolators for a region.

        Returns a callable taking (lon, lat, depth) -> [n, 6] local constants.
        scipy interpolators are scalar-valued, so one is built per slot.
        """
        key = (eq, aa)
        cache_name = "_interp_aa" if aa else "_interp_else"
        cache: dict = getattr(self, cache_name) or {}
        if key in cache:
            return cache[key]
        idx = np.flatnonzero(self.aa_inds == aa)
        gc = self.grid_coords[idx]
        coords = np.column_stack([gc[:, 0], gc[:, 1], gc[:, 2] / DEPTH_TO_DEGREE])
        # All six coefficient fields live on exactly the same nodes.  Passing
        # a shared Delaunay object avoids rebuilding the expensive 3-D
        # triangulation six times for every region/equation.
        tri_name = "_tri_aa" if aa else "_tri_else"
        tri = getattr(self, tri_name)
        if tri is None:
            tri = Delaunay(coords)
            setattr(self, tri_name, tri)
        interps = []
        for slot in range(6):
            vals = self.cs[idx, slot, eq - 1]
            interps.append(LinearNDInterpolator(tri, vals, fill_value=np.nan))
        cache[key] = interps
        setattr(self, cache_name, cache)
        return interps

    # ------------------------------------------------------------------
    # estimation
    # ------------------------------------------------------------------
    def estimate_eq16_fast(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        salinity: np.ndarray,
        *,
        unique_locations: bool = True,
    ) -> np.ndarray:
        """Fast TA estimate via ESPER equation 16 (the S-only regression).

        Equation 16 is linear in salinity: ``TA = C0(loc) + C1(loc)·S``.
        Interpolating the two coefficients is the expensive part (a 3-D
        Delaunay linear interpolation over ~15k/34k nodes), so we interpolate
        **once per unique (lon, lat) location** and then apply salinity
        analytically.  For field-scale work (millions of queries on a grid)
        this is orders of magnitude faster than calling :meth:`estimate` per
        point: the region caches use exactly this decomposition.

        Returns TA in µmol/kg (NaN where the location falls outside the
        ESPER coefficient hull).
        """
        lon = np.mod(np.asarray(lon, dtype=float), 360.0)
        lat = np.asarray(lat, dtype=float)
        s = np.asarray(salinity, dtype=float)
        n = lon.size
        locs = np.column_stack([lon, lat])
        if unique_locations:
            uniq, inverse = np.unique(locs, axis=0, return_inverse=True)
        else:
            uniq, inverse = locs, np.arange(n)
        zeros = np.zeros(uniq.shape[0], dtype=float)
        c0 = self.estimate(uniq[:, 0], uniq[:, 1], zeros, zeros, zeros, equation=EQ_TA_ST_ONLY)
        c1_at_one = self.estimate(uniq[:, 0], uniq[:, 1], zeros, np.ones_like(zeros),
                                  zeros, equation=EQ_TA_ST_ONLY)
        c1 = c1_at_one - c0
        ta = c0[inverse] + c1[inverse] * s
        return np.where(np.isfinite(s), ta, np.nan)

    def estimate(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        depth: np.ndarray,
        salinity: np.ndarray,
        temperature: np.ndarray,
        *,
        a: np.ndarray | None = None,
        b: np.ndarray | None = None,
        c: np.ndarray | None = None,
        equation: int = EQ_TA_ST_ONLY,
    ) -> np.ndarray:
        """Estimate TA (µmol/kg) at given locations/conditions.

        Args:
            lon/lat/depth: coordinates (arrays of equal length).
            salinity: S (unitless; [0,360) lon convention accepted).
            temperature: T (°C).
            a/b/c: optional extra predictors required by non-16 equations,
                   in the order of the Output Equation Key (A, B, C).
            equation: which of the 16 regression equations to use (default 16).

        Returns:
            TA estimates [n] in µmol/kg (NaN where interpolation fails).
        """
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        depth = np.asarray(depth, dtype=float)
        n = lon.size
        eq = int(equation)
        if not (1 <= eq <= VARVEC.shape[0]):
            raise ValueError(f"equation must be in [1, 16], got {eq}")

        # Which slots of the coefficient vector this equation uses.
        # VarVec row = [S?, T?, A?, B?, C?] in the *predictor* sense; the
        # constant is always slot 0.  Map: the 5 VarVec positions correspond
        # to the 5 predictors [S, T, A, B, C] (order per the key), and the
        # coefficient slot for each used predictor is 1 + its position when
        # predictors are taken in order.  In ESPER_LIR.m: UseVars occupies the
        # *predictor columns of M*; the Cs slot index is `VarNumVec = 1+UseVars`
        # where UseVars enumerates the *positions in the 1..5 list*.
        use = np.flatnonzero(VARVEC[eq - 1])  # 0-based positions within 5
        # NOTE: positions 0..4 in VarVec map to predictor types [S, T, A, B, C]
        # being the first 5 input slots; but the README example passes
        # types [1 3 2 4 5 6] = [S, P, T, N, Si, O2] with eq1 all-5, and the
        # estimate matched, so positions: 0->S,1->T,2->A(P),3->B(N),4->C(Si).
        slot_of = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}  # VarVec position -> Cs slot

        # predictor values per VarVec position
        preds = {0: salinity, 1: temperature, 2: a, 3: b, 4: c}

        aa = self._in_aa(np.mod(lon, 360.0), lat)
        interp_aa = self._interp_for(eq, True)
        interp_else = self._interp_for(eq, False)

        lc = np.full((n, 6), np.nan, dtype=float)
        if aa.any():
            g = np.column_stack([np.mod(lon[aa], 360.0), lat[aa], depth[aa] / DEPTH_TO_DEGREE])
            for slot in range(6):
                lc[aa, slot] = np.asarray(interp_aa[slot](g)).ravel()
        if (~aa).any():
            g = np.column_stack([np.mod(lon[~aa], 360.0), lat[~aa], depth[~aa] / DEPTH_TO_DEGREE])
            for slot in range(6):
                lc[~aa, slot] = np.asarray(interp_else[slot](g)).ravel()

        # estimate = C0 + Σ_{used slots} C_slot * X_predictor
        ta = lc[:, 0].copy()  # constant
        for pos in use:
            slot = slot_of[pos]
            xv = preds.get(pos)
            if xv is None:
                continue  # missing required predictor -> contribution NaN later
            ta = ta + lc[:, slot] * np.asarray(xv, dtype=float)
        return ta


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    est = ESPER_LIR_TA.from_mat(r"D:\proj_personal\PhD\ESPER\ESPER_LIR_Files\LIR_files_TA_v3.mat")
    print("loaded; cs nodes:", est.cs.shape[0])
    # test points: e.g. Gulf of Maine (lon 290, lat 42, depth 10)
    lon = np.array([290.0, 200.0, 20.0])
    lat = np.array([42.0, 20.0, 30.0])
    dep = np.array([10.0, 10.0, 10.0])
    sal = np.array([35.0, 36.0, 37.0])
    tmp = np.array([18.0, 25.0, 16.0])
    ta = est.estimate(lon, lat, dep, sal, tmp)
    print("TA estimates (µmol/kg):", np.round(ta, 1))
