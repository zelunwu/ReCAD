"""Regional masks of the v1.1 NA-Atlantic product (faithful port).

``Figures_Maintext.ipynb``/``Figure_Uncertainties.ipynb`` define the NACCOM
sub-regions as hard-coded lon/lat boxes (in -180\\u2026180 convention) plus the
GoMx/CbS diagonal cut-off. This module ports them to functions so regional
statistics and figures are reproducible. Input longitudes may be in either
convention (converted internally).
"""

from __future__ import annotations

import numpy as np

from recad.viz.geo import lon_to_180


def _ref(vmin, vmax):  # strict v1.1 open-closed semantics
    return (vmin, vmax)


def region_masks(lon: np.ndarray, lat: np.ndarray) -> dict[str, np.ndarray]:
    """Return {name: bool[n_lat, n_lon]} masks for every v1.1 region.

    Pure geometry (no data dependence), exactly as the v1.1 hard-coded
    definitions. The GoMx diagonal boundaries reproduce the v1.1 linear cuts.
    """
    lon2, lat2 = np.meshgrid(lon_to_180(lon), lat)
    m: dict[str, np.ndarray] = {}

    def box(lo0, lo1, la0, la1):
        return (lon2 > lo0) & (lon2 <= lo1) & (lat2 >= la0) & (lat2 <= la1)

    def box_closed(lo0, lo1, la0, la1):
        return (lon2 >= lo0) & (lon2 <= lo1) & (lat2 >= la0) & (lat2 <= la1)

    m["GoMeSS"] = box(-70, -60, 41.5, 46)
    m["GoMe"] = box(-71, -65.5, 30, 46)
    m["SS"] = box(-65.5, -60, 37, 46)
    m["GStL"] = box(-69, -45, 41.5, 51)
    m["GStL"] = m["GStL"] & ~m["GoMeSS"]
    m["SAB"] = box(-82, -70, 27, 35.5)
    m["MAB"] = box(-82, -70, 35.5, 41.5)

    # GoMx: box minus three linear cuts and the NW corner (v1.1 exact)
    gm = box(-100, -80, 17, 31)
    a1, b1 = _slope_intercept((-83, 31), (-81, 25.5))
    gm &= (a1 * lon2 + b1) >= lat2
    a2, b2 = _slope_intercept((-87.0, 21), (-80.5, 25))
    gm &= (a2 * lon2 + b2) <= lat2
    gm &= ~((lon2 <= -89.75) & (lat2 >= 29.5))
    m["GoMx"] = gm

    m["CbS"] = box(-90, -55, 15, 27) & ~m["GoMx"] & ~m["SAB"]
    m["Atlantic"] = m["GoMx"] | m["SAB"] | m["MAB"] | m["GoMe"] | m["SS"] | m["GStL"]
    m["Atlantic_north"] = m["GoMe"] | m["SS"] | m["GStL"]
    m["Atlantic_south"] = m["GoMx"] | m["SAB"] | m["MAB"]

    # Analysis boxes used in the trend/seasonality figures
    m["LAS"] = box_closed(-94, -87, 28, 31)
    m["WFS"] = box_closed(-84, -81, 24, 29)
    m["MAB_box"] = box_closed(-76, -71, 36, 41.5)
    m["North_box"] = box_closed(-65, -50, 43, 50)
    m["S_GStL_box"] = box_closed(-64, -58, 45, 48)
    return m


def _slope_intercept(p1, p2) -> tuple[float, float]:
    (x1, y1), (x2, y2) = p1, p2
    a = (y2 - y1) / (x2 - x1)
    return a, y2 - a * x2


# Regional display names in v1.1 figure order
V11_REGION_NAMES: tuple[str, ...] = (
    "GStL&GB",
    "SS",
    "GoMe",
    "MAB",
    "SAB",
    "GoMx",
    "NAACOM",
)


def v11_region_names() -> tuple[str, ...]:
    """Display names for the v1.1 NA-Atlantic regions (figure order)."""
    return V11_REGION_NAMES


def v11_region_keys() -> list[str]:
    """Mask keys in figure order (Atlantic = the whole product domain)."""
    return ["GStL", "SS", "GoMe", "MAB", "SAB", "GoMx", "Atlantic"]
