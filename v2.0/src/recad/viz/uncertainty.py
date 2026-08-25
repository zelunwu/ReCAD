"""Uncertainty figures (v1.1 ``Figure_Uncertainties.ipynb`` / Fig B1).

Reproduces the two v1.1 uncertainty figure families:

* component maps: mean maps of each input-error contribution (u_SSS, u_SST,
  u_SSH, u_pCO2air) and their RSS total, in one panel grid sharing a colorbar
  (Figure_B1, Figure_Uncertainties);
* contribution histograms: the 2x3 histograms of per-grid-cell standard
  deviations that ``u_inputs_Monte_Carlo.mlx`` printed for each contributor.
"""

from __future__ import annotations

import numpy as np

from recad.viz.geo import panel_maps


def uncertainty_panels(
    fields: dict[str, np.ndarray],
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    vmax: float = 15.0,
    cmap: str = "RdYlBu_r",
    n_levels: int = 15,
    label: str = "fCO$_2$ (\\u788catm)",
    nrows_ncols: tuple[int, int] = (2, 3),
    figsize=(15.0, 7.0),
) -> tuple[object, object]:
    """Component uncertainty maps sharing one colorbar (v1.1 Fig B1 style).

    ``fields`` maps a key (e.g. "u_SSS") to a 2-D mean-of-time uncertainty
    field; the color scale is common (default 0-15 \\u788catm, as in the paper).
    """
    use_v11_style = __import__("recad.viz.style", fromlist=["use_v11_style"]).use_v11_style
    use_v11_style()
    display = {key: key.replace("_", "$_{") + "}$" for key in fields}
    fig, pc = panel_maps(
        fields,
        lon,
        lat,
        nrows_ncols=nrows_ncols,
        vmin=0.0,
        vmax=vmax,
        cmap=cmap,
        n_levels=n_levels,
        label=label,
        titles=display,
        figsize=figsize,
    )
    return fig, pc


def contribution_histograms(
    std_fields: dict[str, np.ndarray],
    *,
    nrows_ncols: tuple[int, int] = (2, 3),
    bins: int = 100,
    xlim: tuple[float, float] = (-10.0, 100.0),
    figsize=(12.0, 8.0),
    fill: bool = True,
) -> tuple[object, dict[str, object]]:
    """Histograms of per-cell uncertainty stds (v1.1 MC hist panels).

    ``std_fields`` maps contributor -> 1-D array of per-cell std values
    (spatially resolved). Each panel shows the histogram and the mean \\u5364 std
    in the title.
    """
    from matplotlib import pyplot as plt

    use_v11_style = __import__("recad.viz.style", fromlist=["use_v11_style"]).use_v11_style
    use_v11_style()
    fig, axes = plt.subplots(*nrows_ncols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).ravel()
    artists: dict[str, object] = {}
    for ax, (key, values) in zip(axes_flat, std_fields.items(), strict=False):
        values = np.asarray(values, dtype=float)
        vals = values[np.isfinite(values)]
        ax.hist(vals, bins=bins, color="steelblue", alpha=0.8, fill=fill)
        if vals.size:
            ax.set_title(
                "%s: mean = %.2f \\u00b1 %.2f \\u00b5atm" % (key, np.mean(vals), np.std(vals))
            )
        ax.set_xlim(xlim)
        artists[key] = ax
    fig.tight_layout()
    return fig, artists
