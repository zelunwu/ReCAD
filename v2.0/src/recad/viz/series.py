"""Seasonal-cycle and time-series figures (v1.1 Figure_06/A1/A2 style).

Two families of the v1.1 figures are reproduced:

* monthly climatology comparisons (``errorbar(month, mean \\u5364 std)``) for one
  or several series on the same axes - e.g. SOCAT vs product, including the
  "Difference (Product - SOCAT)" annotation box (Figure_06, Figure_A1);
* deseasonalized time series with the Sutton trend line and the
  ``trend \\u5364 err (N, p)`` annotation (Figure_08) or R\\u864f/RMSE boxes (Figure_A2).
"""

from __future__ import annotations

import numpy as np

from recad.viz.trends import TrendResult, calc_trend_sutton

_MONTHS = np.arange(1, 13)


def monthly_climatology(
    *,
    series: np.ndarray,
    label: str,
    ax,
    color="k",
    marker="o",
    offset: float = 0.0,
    linewidth: float = 2.0,
    markersize: float = 6.0,
    capsize: float = 3.0,
) -> object:
    """Draw one monthly climatology as errorbar(mean \\u5364 std across years).

    ``series`` must be 2-D ``[n_year, 12]`` (a spatial mean reshaped to
    years-by-months); NaN years are dropped.
    """
    mean = np.nanmean(series, axis=0)
    std = np.nanstd(series, axis=0)
    lns = ax.errorbar(
        _MONTHS + offset,
        mean,
        yerr=std,
        c=color,
        fmt=marker,
        ls="-",
        markerfacecolor="white",
        capsize=capsize,
        ms=markersize,
        label=label,
        linewidth=linewidth,
    )
    return lns, mean, std


def climatology_comparison(
    ax,
    series: dict[str, tuple[np.ndarray, str, str]],
    *,
    title: str | None = None,
    ylabel: str = "fCO$_2$ (\\u788catm)",
    diff_text_xy: tuple[float, float] | None = (2.0, 420.0),
    diff_legend_labels: tuple[str, str] | None = None,
) -> None:
    """Several monthly climatologies on one axes with a diff annotation box.

    ``series`` maps ``(label, color) -> 2-D [n_year, 12]``; the first two
    entries are compared in an annotation box, mirroring Figure_06's
    "Difference (Product - SOCAT)" text.
    """
    handles = []
    keys = list(series.items())
    for i, (meta, data) in enumerate(keys):
        label, color = meta
        lns, mean, _std = monthly_climatology(
            series=data,
            label=label,
            ax=ax,
            color=color,
            offset=0.0,
        )
        handles.append(lns)
        if diff_text_xy and i == 1:
            ref_mean = np.nanmean(np.nanmean(keys[0][1], axis=0))
            this_mean = np.nanmean(mean)
            diff = this_mean - ref_mean
            ax.text(
                *diff_text_xy,
                "Difference (%s - %s) = \n %+.2f \\u00b1 %.2f \\u00b5atm"
                % (label, keys[0][0], diff, np.nanstd(mean - np.nanmean(keys[0][1], axis=0))),
                ha="left",
                fontdict={"size": 12},
            )
    ax.set_xticks(_MONTHS)
    ax.set_xlim([0.5, 12.5])
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.5)
    if title:
        ax.set_title(title)
    if diff_legend_labels is None:
        ax.legend(handles, [k[0] for k in keys], loc="lower right")
    else:
        ax.legend(handles, list(diff_legend_labels), loc="lower right")


def deseasonalized_timeseries(
    ax,
    t: np.ndarray,
    y: np.ndarray,
    *,
    label: str = "",
    color: str = "tab:blue",
    marker: str = "o",
    alpha: float = 0.3,
    trend: bool = True,
    window: float = 5.0,
    year_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> TrendResult:
    """Deseasonalized monthly series with the Sutton trend fit (Figure_08).

    Draws scatter points of the deseasonalized series and the fitted trend
    line, and annotates ``trend \\u5364 err (N = n, p < 0.0001 | p = x)`` in the
    series color. Returns the trend result.
    """
    res = calc_trend_sutton(t, y, window)
    ax.scatter(
        t,
        res.ts_monthly_deseason,
        s=20,
        marker=marker,
        color=color,
        alpha=alpha,
        facecolors=color,
        label=label,
    )
    if trend and np.isfinite(res.slope):
        tt = np.linspace(t.min(), t.max(), 200)
        ax.plot(
            tt,
            res.slope * tt + (res.ts_monthly_deseason.mean() - res.slope * t.mean()),
            linewidth=3,
            color=color,
        )
        if np.isfinite(res.p_value):
            if res.p_value < 0.0001:
                txt = "trend = %+.2f \\u00b1 %.2f \\u00b5atm yr$^{-1}$ (N = %d, p < 0.0001)" % (
                    res.slope,
                    res.slope_err,
                    res.n_valid,
                )
            else:
                txt = "trend = %+.2f \\u00b1 %.2f \\u00b5atm yr$^{-1}$ (N = %d, p = %.4f)" % (
                    res.slope,
                    res.slope_err,
                    res.n_valid,
                    res.p_value,
                )
        else:
            txt = "trend = %+.2f \\u00b5atm yr$^{-1}$ (N = %d)" % (res.slope, res.n_valid)
        ax.text(t.min() + 2, _text_y(ax, y_range), txt, color=color)
    if year_range:
        ax.set_xlim(year_range)
    if y_range:
        ax.set_ylim(y_range)
    ax.set_ylabel("pCO$_2$ (\\u788catm)")
    ax.grid("on", alpha=0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return res


def _text_y(ax, y_range):
    if y_range is None:
        lo = ax.get_ylim()[0]
        return lo + (ax.get_ylim()[1] - lo) * 0.15
    return y_range[0] + (y_range[1] - y_range[0]) * 0.15


def region_time_series(field4d: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
    """Spatially averaged monthly series for a (boolean) region mask.

    Returns ``[n_year, 12]`` nan-mean over the masked cells, preserving the
    v1.1 convention (mean over cells, then the caller reshapes by year).
    """
    field = np.asarray(field4d, dtype=float)
    masked = np.where(mask2d[None, None, :, :], field, np.nan)
    return np.nanmean(np.nanmean(masked, axis=3), axis=2)  # [n_year, 12]
