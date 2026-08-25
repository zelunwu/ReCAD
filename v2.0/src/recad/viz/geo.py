"""Geo-map infrastructure (v1.1 ``Figures_Maintext.ipynb`` style).

The v1.1 notebooks built every map with cartopy ``PlateCarree`` + AxesGrid:
coastlines, grey land, rivers, a lon/lat grid with degree formatters, hard
``xlim/ylim`` and region boundary polylines with labels. This module
reproduces that recipe as functions.

Dependency policy: ``cartopy`` is optional. When it is missing, the same
functions render onto plain matplotlib axes (no coastlines/land/rivers, no
map projection), which keeps the tools fully testable in CI. ``cartopy_ok()``
reports which path is active.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

try:  # optional renderer
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.geoaxes import GeoAxes
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

    _CARTOPY_OK = True
except ImportError:  # pragma: no cover - environment dependent
    _CARTOPY_OK = False

try:
    from mpl_toolkits.axes_grid1 import AxesGrid

    _AXESGRID_OK = True
except ImportError:  # pragma: no cover
    _AXESGRID_OK = False


def cartopy_ok() -> bool:
    """True when the full cartopy renderer is available."""
    return _CARTOPY_OK


def _cartopy_version() -> tuple[int, int]:
    import cartopy

    return tuple(int(x) for x in cartopy.__version__.split(".")[:2])  # type: ignore[return-value]


def lon_to_180(lon: np.ndarray) -> np.ndarray:
    """Map longitudes into [-180, 180) (v1.1 regional coordinates)."""
    lon = np.mod(np.asarray(lon, dtype=float), 360.0)
    return np.where(lon >= 180.0, lon - 360.0, lon)


def projection_crs():
    """The equatorial PlateCarree projection used by all v1.1 maps."""
    return ccrs.PlateCarree()


def setup_geo_axes(
    ax,
    lon_range: tuple[float, float] = (-98.0, -45.0),
    lat_range: tuple[float, float] = (17.0, 52.0),
    *,
    xticks: Iterable[float] | None = None,
    yticks: Iterable[float] | None = None,
    grid: bool = True,
) -> None:
    """Configure one map axes exactly like the v1.1 notebooks.

    Applies coastlines/land/rivers (when cartopy is present), a lon/lat grid
    with degree formatters, and the hard xlim/ylim used in the paper.
    """
    if _CARTOPY_OK:
        ax.coastlines(alpha=0.4)
        ax.add_feature(cfeature.LAND, facecolor="grey", alpha=0.3)
        ax.add_feature(cfeature.RIVERS)
        lon_fmt = LongitudeFormatter()
        lat_fmt = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_fmt)
        ax.yaxis.set_major_formatter(lat_fmt)
        ticks_lon = xticks if xticks is not None else np.arange(-180, 181, 10)
        ticks_lat = yticks if yticks is not None else np.arange(20, 66, 10)
        ax.set_xticks(ticks_lon, crs=projection_crs())
        ax.set_yticks(ticks_lat, crs=projection_crs())
        ax.grid("on", alpha=0.4)
    elif grid:  # pragma: no cover - exercised in plain-axes mode too
        ax.grid("on", alpha=0.4)
    ax.set_xlim(lon_range)
    ax.set_ylim(lat_range)


class _AxesFactory:
    """Axes/AxesGrid factories selecting cartopy GeoAxes or plain Axes."""

    @staticmethod
    def single(fig, subplot_spec=111):
        if _CARTOPY_OK:
            return fig.add_subplot(subplot_spec, projection=projection_crs())
        return fig.add_subplot(subplot_spec)

    @staticmethod
    def grid(fig, nrows_ncols, *, pad=0.7, cbar_mode="single", cbar_location="right"):
        if not _AXESGRID_OK:  # pragma: no cover - axe_grid1 ships with mpl
            raise RuntimeError("mpl_toolkits.axes_grid1.AxesGrid unavailable")
        axes_class = _AxesFactory._geoaxes_class() if _CARTOPY_OK else None
        kwargs = {
            "nrows_ncols": nrows_ncols,
            "axes_pad": pad,
            "cbar_location": cbar_location,
            "cbar_mode": cbar_mode,
            "cbar_pad": 0.2 if cbar_mode != "none" else None,
            "cbar_size": "3%",
            # v1.1 used label_mode='' (legacy); modern mpl accepts 'L' (left
            # column labels only), which is the same intended layout.
            "label_mode": "L",
        }
        if axes_class is not None and cbar_mode == "each":
            # AxesGrid support for per-axis colorbars needs label_mode='' either way
            pass
        if axes_class is not None:
            kwargs["axes_class"] = axes_class
        return AxesGrid(fig, 111, **kwargs)

    @staticmethod
    def _geoaxes_class():
        # cartopy >= 0.25 renamed the axes_class keyword from `map_projection`
        # to `projection`; older versions only accept `map_projection`.
        kwarg = "projection" if cartopy_ok() and _cartopy_version() >= (0, 25) else "map_projection"
        return (GeoAxes, {kwarg: projection_crs()})


def _axes_kwargs(ax) -> dict:
    """transform= keyword for pcolor on geo axes (empty on plain axes)."""
    if _CARTOPY_OK:
        return {"transform": projection_crs()}
    return {}


def field_map(
    ax,
    field2d: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap="jet",
    label: str | None = None,
    n_levels: int | None = None,
    extend: str = "both",
    mask2d: np.ndarray | None = None,
    title: str | None = None,
) -> object:
    """Draw one 2-D field with pcolor on the axes (v1.1 style).

    ``field2d`` may carry longitudes in [0, 360) or [-180, 180); the 180
    conversion is applied automatically for the plot. Returns the pcolor
    mappable so the caller can attach a colorbar.
    """
    if mask2d is not None:
        field2d = np.where(mask2d, field2d, np.nan)
    if n_levels is not None:
        cmap = __import__("matplotlib").pyplot.get_cmap(cmap, n_levels)
    pc = ax.pcolormesh(
        lon_to_180(lon),
        lat,
        field2d,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        **_axes_kwargs(ax),
    )
    if title:
        ax.set_title(title)
    colorbar_kwargs = {"label": label} if label else {}
    if _CARTOPY_OK and colorbar_kwargs:
        from matplotlib import pyplot as plt

        plt.colorbar(pc, ax=ax, extend=extend, **colorbar_kwargs)
    return pc


def panel_maps(
    fields: dict[str, np.ndarray],
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    nrows_ncols: tuple[int, int] = (2, 2),
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "jet",
    label: str = "fCO2 (\\u788catm)",
    titles: dict[str, str] | None = None,
    mask2d: np.ndarray | None = None,
    n_levels: int | None = None,
    figsize=(14.0, 8.0),
) -> tuple[object, list]:
    """Multiple 2-D maps sharing one colorbar (AxesGrid, v1.1 style).

    ``fields`` maps a panel key to a 2-D array; panel order follows dict
    insertion. Returns ``(fig, mappable)``.
    """
    from matplotlib import pyplot as plt

    use_v11_style = __import__("recad.viz.style", fromlist=["use_v11_style"]).use_v11_style
    use_v11_style()
    fig = plt.figure(figsize=figsize)
    grid = _AxesFactory.grid(fig, nrows_ncols)
    axes = list(grid)
    pc = None
    for ax, (key, field) in zip(axes, fields.items(), strict=False):
        _setup_like_v11(ax)
        pc = field_map(
            ax,
            field,
            lon,
            lat,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            n_levels=n_levels,
            mask2d=mask2d,
        )
        title = (titles or {}).get(key, str(key))
        ax.set_title(title)
    if _CARTOPY_OK and pc is not None:
        grid.cbar_axes[0].colorbar(pc, extend="both", label=label)
    return fig, pc


def _setup_like_v11(ax) -> None:
    """Default v1.1 map look: NA-Atlantic window + region boundaries."""
    setup_geo_axes(ax)
    add_region_boundaries(ax, color="tab:orange", width=1.5)


def add_region_boundaries(ax, *, color: str = "tab:orange", width: float = 1.5) -> None:
    """Draw the v1.1 regional boundary polylines and labels (NA Atlantic).

    Region abbreviations (v1.1 figure annotations): GStL&GB, SS, GoMe, MAB,
    SAB, GoMx.
    """
    segments = (
        ((-80.5, -87), (25, 21)),  # GoMx & CbS
        ((-80.5, -77), (27, 27)),  # CbS & SAB
        ((-76, -72), (35.5, 35.5)),  # SAB & MAB
        ((-70, -70), (41.5, 38.1)),  # MAB & SS
        ((-65.6, -65.6), (40, 45)),  # SS & GoMe
        ((-60, -60), (42, 45.5)),  # SS & GStL
    )
    for xs, ys in segments:
        ax.plot(xs, ys, c=color, linewidth=width)
    labels = {
        (-57, 42): "GStL&GB",
        (-62.5, 39.5): "SS",
        (-71, 45): "GoMe",
        (-76, 42): "MAB",
        (-76, 30): "SAB",
        (-92, 31): "GoMx",
    }
    for (x, y), text in labels.items():
        ax.text(x, y, text, c=color, fontdict={"size": 13})
