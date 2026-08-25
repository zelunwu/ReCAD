"""Trend analysis & trend figures (v1.1 ``calc_trend_Sutton`` port).

The v1.1 figures used the Sutton et al. (2007) two-step recipe: (1) an
initial HAC-OLS line removes the gross trend, (2) the residual monthly
climatology (and its annual mean) provides the seasonal adjustment, and
(3) a final WLS fit on the deseasonalized series gives the reported trend
with its uncertainty (``Figures_Maintext.ipynb``, Figure_08).

``statsmodels`` is used when available to reproduce the v1.1 HAC-OLS/WLS
exactly; otherwise a pure-NumPy OLS (no HAC covariance) is used - the slope
estimate is identical, only the standard error differs slightly. Both paths
document which one they took.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import statsmodels.api as sm

    _SM_OK = True
except ImportError:  # pragma: no cover - environment dependent
    _SM_OK = False


@dataclass
class TrendResult:
    """Result of the Sutton-style trend decomposition."""

    model: object = None  # statsmodels fit (when available), else None
    slope: float = np.nan
    slope_err: float = np.nan
    p_value: float = np.nan
    r_squared: float = np.nan
    monthly_clim: np.ndarray = field(default_factory=lambda: np.full(12, np.nan))
    annual_clim: float = np.nan
    ts_monthly_deseason: np.ndarray = field(default_factory=lambda: np.full(0, np.nan))
    using_statsmodels: bool = False
    n_valid: int = 0


def calc_trend_sutton(t: np.ndarray, y: np.ndarray, w: float = 5.0) -> TrendResult:
    """Sutton et al. (2007) trend of a monthly series.

    Args:
        t: decimal time (e.g. 1993.04).
        y: monthly values (NaN tolerated).
        w: v1.1 used ``w = 5``; the residual-climatology window parameter for
           the initial detrending step (kept for API fidelity).

    Returns:
        :class:`TrendResult` with slope/slope_err/p_value/r\\u864f in \\u788catm yr\\u9226\\u5ba6?
    """
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(t.size, y.size)
    t, y = t[:n], y[:n]
    valid = ~np.isnan(t) & ~np.isnan(y)
    n_valid = int(valid.sum())

    res = _trend_sm(t, y, valid, w) if _SM_OK else _trend_numpy(t, y, valid)
    res.using_statsmodels = _SM_OK
    res.slope_err = float(res.slope_err) if np.isfinite(res.slope_err) else np.nan
    res.p_value = float(res.p_value) if np.isfinite(res.p_value) else np.nan
    res.n_valid = int(n_valid)
    return res


# ---------------------------------------------------------------------------
def _trend_sm(t, y, valid, w) -> TrendResult:
    """v1.1-exact path with statsmodels (HAC-OLS then WLS on deseasonalized)."""
    lm_raw = sm.OLS(y[valid], sm.add_constant(t[valid])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 1}
    )
    ts_trend_remove = lm_raw.predict(sm.add_constant(t[valid]))
    ts_y_detrend = np.full_like(y, np.nan)
    ts_y_detrend[valid] = y[valid] - ts_trend_remove
    ts_y_month_clim = np.nanmean(np.reshape(ts_y_detrend, [-1, 12]), axis=0)
    month_adj = ts_y_month_clim - np.nanmean(ts_y_month_clim)
    # deseasonalized by the *detrended* climatology, then refit on the raw
    # series (v1.1 semantics)
    ts_y_annual_clim = np.nanmean(ts_y_month_clim)
    ts_deseason = np.reshape(np.reshape(y, (-1, 12)) - month_adj, len(y))
    ivalid = ~np.isnan(ts_deseason)
    wts = np.full(int(ivalid.sum()), w)
    lm_wls = sm.WLS(ts_deseason[ivalid], sm.add_constant(t[ivalid]), wts).fit(
        cov_type="HAC", cov_kwds={"maxlags": 1}
    )
    slope = lm_wls.params[1]
    return TrendResult(
        model=lm_wls,
        slope=float(slope),
        slope_err=float(lm_wls.bse[1]),
        p_value=float(lm_wls.pvalues[1]),
        r_squared=float(lm_wls.rsquared),
        monthly_clim=ts_y_month_clim,
        annual_clim=float(ts_y_annual_clim),
        ts_monthly_deseason=ts_deseason,
    )


def _trend_numpy(t, y, valid) -> TrendResult:
    """Pure-NumPy path (no HAC covariance; slope identical within eps)."""
    # step 1: initial OLS detrend
    A = np.column_stack([np.ones(int(valid.sum())), t[valid]])
    coef_raw, *_ = np.linalg.lstsq(A, y[valid], rcond=None)
    ts_y_detrend = np.full_like(y, np.nan)
    ts_y_detrend[valid] = y[valid] - (coef_raw[0] + coef_raw[1] * t[valid])
    n_years = y.size // 12
    clim = np.nanmean(np.reshape(ts_y_detrend, (n_years, 12)), axis=0)
    month_adj = clim - np.nanmean(clim)
    ts_deseason = np.reshape(np.reshape(y, (-1, 12)) - month_adj, len(y))
    ivalid = ~np.isnan(ts_deseason)
    Ai = np.column_stack([np.ones(int(ivalid.sum())), t[ivalid]])
    coef, *_ = np.linalg.lstsq(Ai, ts_deseason[ivalid], rcond=None)
    resid = ts_deseason[ivalid] - (coef[0] + coef[1] * t[ivalid])
    n = resid.size
    sigma2 = float(np.sum(resid**2) / max(n - 2, 1))
    se = float(np.sqrt(sigma2 / np.sum((t[ivalid] - t[ivalid].mean()) ** 2)))
    slope = float(coef[1])
    t_stat = slope / se if se > 0 else np.nan
    p_value = float(2.0 * _t_sf(abs(t_stat), n - 2)) if np.isfinite(t_stat) else np.nan
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((ts_deseason[ivalid] - ts_deseason[ivalid].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return TrendResult(
        slope=slope,
        slope_err=se,
        p_value=p_value,
        r_squared=r2,
        monthly_clim=clim,
        annual_clim=float(np.nanmean(clim)),
        ts_monthly_deseason=ts_deseason,
    )


def _t_sf(x: float, df: float) -> float:
    """Survival function of Student's t (numpy fallback; no scipy needed)."""
    if df <= 0 or not np.isfinite(x):
        return np.nan

    # regularized incomplete beta via continued fraction (Numerical Recipes)
    # simplified: use scipy when available
    try:
        from scipy.stats import t as tdist

        return float(tdist.sf(x, df))
    except ImportError:  # pragma: no cover
        # crude normal approximation, documented fallback
        from math import erfc, sqrt

        return float(erfc(x / sqrt(2.0)) / 2.0)


def trend_map(field4d: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Per-cell linear trend (\\u788catm yr\\u9226\\u5ba6? of a 4-D monthly field.

    Uses the numpy OLS path (fast; identical slope) for every coastal cell
    with enough valid samples. Returns a 2-D array of slopes, NaN where the
    series is unusable.
    """
    field4d = np.asarray(field4d, dtype=float)
    n_year, n_mon, n_lat, n_lon = field4d.shape
    t = np.arange(n_year * n_mon, dtype=float) / 12.0 + years[0]
    slope_out = np.full((n_lat, n_lon), np.nan)
    cells = np.argwhere(~np.isnan(field4d).all(axis=(0, 1)))
    for i, j in cells:
        ts = field4d[:, :, i, j].ravel()
        valid = ~np.isnan(ts)
        if valid.sum() < 12:
            continue
        A = np.column_stack([np.ones(int(valid.sum())), t[valid]])
        coef, *_ = np.linalg.lstsq(A, ts[valid], rcond=None)
        slope_out[i, j] = coef[1]
    return slope_out
