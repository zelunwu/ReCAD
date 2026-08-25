"""Plotting tools for ReCAD v2.0 (``recad.viz``).

This package turns the ad-hoc plotting code of the v1/v1.1 notebooks
(``Figures_Maintext.ipynb``, ``Figure_Uncertainties.ipynb``,
``Figures_Dissertation_Chapter3.ipynb``, ``u_inputs_Monte_Carlo.mlx``) into
reusable, tested tools. See ``docs/plotting.md`` for the one-to-one mapping
between every v1.1 figure/function and its v2.0 counterpart.

Conventions
-----------
- 4-D fields are ``[n_year, 12, n_lat, n_lon]`` with NaN = missing.
- Longitudes default to ``[0, 360)``; the geo helpers convert as needed.
- ``statsmodels`` (HAC-OLS, the exact v1.1 recipe) is preferred when
  installed; pure-NumPy fallbacks are used otherwise (documented deviations).
- ``cartopy`` is optional for maps: without it the tools render PlateCarree
  projections as plain axes (no coastlines/land), so the same code runs in CI.
"""

from recad.viz.geo import field_map, panel_maps, setup_geo_axes
from recad.viz.regions import region_masks, v11_region_names
from recad.viz.scatter import bin_counts, density_scatter, metrics_text
from recad.viz.style import use_v11_style
from recad.viz.trends import calc_trend_sutton, trend_map
from recad.viz.uncertainty import contribution_histograms, uncertainty_panels

__all__ = [
    "bin_counts",
    "calc_trend_sutton",
    "contribution_histograms",
    "density_scatter",
    "field_map",
    "metrics_text",
    "panel_maps",
    "region_masks",
    "setup_geo_axes",
    "trend_map",
    "uncertainty_panels",
    "use_v11_style",
    "v11_region_names",
]
