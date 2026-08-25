"""Matplotlib style presets mirroring the published v1.1 figures.

The v1.1 notebooks (``Figures_Maintext.ipynb``) set ``plt.rcParams.update(
{"font.size": N})`` per figure and used ``plt.tight_layout()`` everywhere;
this module centralises that formatting so every v2.0 figure looks like the
published paper rather than a fresh matplotlib default.
"""

from __future__ import annotations

import matplotlib as mpl

_NAME = "recad.v11"

_LIGHT_STYLE = {
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 600,  # v1.1 saved jpg at dpi=600
    "axes.grid": False,
}

_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_SEASON_NAMES = ["Spring", "Summer", "Fall", "Winter"]
# month indices (0-based) for each season, v1.1 convention
_SEASON_MONTHS = [[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 0, 1]]


def use_v11_style() -> None:
    """Apply the v1.1 figure style (idempotent)."""
    mpl.rcParams.update(_LIGHT_STYLE)
    # NOTE: no figure.autolayout - AxesGrid manages its own layout and the
    # figure functions call tight_layout / bbox_inches="tight" explicitly.
