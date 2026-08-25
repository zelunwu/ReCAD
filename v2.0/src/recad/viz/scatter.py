"""Density-scatter tools (v1.1 ``Figure_03`` style).

``bin_counts`` is the exact v1.1 2-D histogram; ``density_scatter`` draws
predicted-vs-observed counts as a pcolor with a 1:1 line and the OLS/RMSE/
MAE/MBE/N statistics box that decorated every v1.1 accuracy figure.
"""

from __future__ import annotations

import numpy as np


def bin_counts(x, y, xlim, ylim, step):
    """2-D bin densities (exact v1.1 implementation).

    Returns ``(dens[n_y, n_x], x_new, y_new)`` with NaN where no samples
    fall; the caller draws ``ax.pcolor(x_new, y_new, dens.T)``.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    valid = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[valid], y[valid]
    x_new = np.arange(xlim[0], xlim[1] + step / 2, step)
    y_new = np.arange(ylim[0], ylim[1] + step / 2, step)
    dens = np.full((len(y_new), len(x_new)), np.nan)
    for idxx in range(len(x_new)):
        for idxy in range(len(y_new)):
            n = np.sum(
                (x >= x_new[idxx] - step / 2)
                & (x < x_new[idxx] + step / 2)
                & (y >= y_new[idxy] - step / 2)
                & (y < y_new[idxy] + step / 2)
            )
            if n > 0:
                dens[idxx, idxy] = n
    return dens, x_new, y_new


# ---------------------------------------------------------------------------
# metrics helpers
# ---------------------------------------------------------------------------


def calculate_metrics(y_true, y_pred):
    """R\\u864f and RMSE with v1.1 (notebook) semantics.

    R\\u864f = 1 - SS_res/SS_tot (the notebook ``calculate_metrics``), distinct
    from the product-validation corr\\u864f used in train/metrics.py.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return np.nan, np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return float(r2), rmse


def _ols_fit(x, y):
    """OLS slope/intercept/R\\u864f (NumPy lstsq; statsmodels-equivalent)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    valid = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return np.nan, np.nan, np.nan
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2, _ = calculate_metrics(y, coef[0] + coef[1] * x)
    return float(coef[1]), float(coef[0]), float(r2)


def metrics_text(y_pred, y_obs) -> str:
    """The v1.1 statistics box: R\\u864f / RMSE / MAE / MBE / N."""
    y_pred = np.asarray(y_pred, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    valid = ~np.isnan(y_pred) & ~np.isnan(y_obs)
    x = y_pred[valid]
    y = y_obs[valid]
    n = x.size
    if n == 0:
        return "N = 0"
    _, _, r2 = _ols_fit(x, y)
    rmse_i = float(np.sqrt(np.mean((x - y) ** 2)))
    mae_i = float(np.mean(np.abs(x - y)))
    mbias = float(np.mean(x - y))
    return (
        "R$^2$ = %.2f\nRMSE = %.1f \\u00b5atm\nMAE = %.2f \\u00b5atm\nMBE = %.2f \\u00b5atm\nN = %d"
        % (r2, rmse_i, mae_i, mbias, n)
    )


def density_scatter(
    ax,
    y_pred,
    y_obs,
    *,
    bin_step: float = 2.0,
    xlim=(200.0, 550.0),
    ylim=(200.0, 550.0),
    nmax: int = 10,
    title: str | None = None,
    label_x: str = "${p}$CO$_{2,est}$ (\\u788catm)",
    label_y: str = "${p}$CO$_{2,obs}$ (\\u788catm)",
    text_xy=(210.0, 465.0),
    **pcolor_kwargs,
):
    """2-D density scatter of est vs obs with 1:1 line and metrics box.

    Reproduces v1.1 Figure_03 / Figure_Model_Training panels. Returns the
    pcolor mappable (for an AxesGrid colorbar).
    """
    dens, x_new, y_new = bin_counts(y_pred, y_obs, xlim, ylim, bin_step)
    ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], "k", linewidth=1, alpha=0.5)
    pc = ax.pcolormesh(
        x_new,
        y_new,
        dens.T,
        vmin=0,
        vmax=nmax,
        cmap=__import__("matplotlib").pyplot.get_cmap("RdYlBu_r", nmax),
        **pcolor_kwargs,
    )
    ax.grid("minor", alpha=0.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.text(*text_xy, metrics_text(y_pred, y_obs), fontdict={"size": 12})
    if title:
        ax.set_title(title)
    return pc
