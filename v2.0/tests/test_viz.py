"""Tests for the recad.viz plotting tools (Agg backend, no GUI)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import pytest

from recad.viz.regions import region_masks
from recad.viz.scatter import bin_counts, metrics_text
from recad.viz.style import use_v11_style
from recad.viz.trends import calc_trend_sutton, trend_map

# ---------------------------------------------------------------------------
# bin_counts / metrics_text
# ---------------------------------------------------------------------------


def test_bin_counts_discrete():
    x = np.array([100.0, 100.0, 101.0, 103.0])
    y = np.array([100.0, 101.0, 102.0, 103.0])
    dens, x_new, y_new = bin_counts(x, y, [99.0, 105.0], [99.0, 105.0], 2.0)
    # 1-D diagonal: each bin has at most 1-2 samples
    assert dens.shape == (len(y_new), len(x_new))
    assert np.nansum(dens) == 4


def test_bin_counts_hand_computed():
    x = np.full(3, 50.0)
    y = np.full(3, 60.0)
    dens, x_new, y_new = bin_counts(x, y, [40.0, 60.0], [50.0, 70.0], 2.0)
    # center bin (49-51, 59-61) contains all 3 samples
    ix = np.argmin(np.abs(x_new - 50.0))
    iy = np.argmin(np.abs(y_new - 60.0))
    assert dens[ix, iy] == 3
    assert np.nansum(~np.isnan(dens)) == 1


def test_metrics_text_content():
    yobs = np.linspace(300, 350, 40)
    yest = yobs + 2.0
    txt = metrics_text(yest, yobs)
    assert "R$^2$" in txt and "N = 40" in txt and "RMSE" in txt


# ---------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------


def test_region_masks_shapes_and_content():
    lon = np.arange(260.0, 321.0, 0.25)  # -100..-40 in 0-360
    lat = np.arange(10.0, 66.0, 0.25)
    masks = region_masks(lon, lat)
    assert masks["GoMx"].sum() > 0
    assert masks["Atlantic"].sum() > 0
    # union of the six sub-regions equals Atlantic
    union = (
        masks["GoMx"] | masks["SAB"] | masks["MAB"] | masks["GoMe"] | masks["SS"] | masks["GStL"]
    )
    assert np.array_equal(union, masks["Atlantic"])
    # GoMeSS is carved out of GStL (v1.1 semantics)
    assert not (masks["GStL"] & masks["GoMeSS"]).any()


def test_region_masks_accept_0_360_longitude():
    lon = np.arange(0.0, 360.0, 1.0)
    lat = np.arange(-80.0, 85.0, 1.0)
    masks = region_masks(lon, lat)
    assert masks["GoMx"].sum() > 0
    assert masks["Atlantic"].sum() > 0


# ---------------------------------------------------------------------------
# trend analysis
# ---------------------------------------------------------------------------


def test_calc_trend_sutton_linear():
    rng = np.random.default_rng(0)
    t = np.arange(1993.0, 2003.0, 1 / 12)
    y = 1.8 * (t - 1993) + 350.0 + 8.0 * np.sin(2 * np.pi * (np.arange(t.size) % 12) / 12)
    y = y + rng.normal(0, 1, y.size)
    res = calc_trend_sutton(t, y, w=5.0)
    assert abs(res.slope - 1.8) < 0.05
    assert res.slope_err > 0
    assert res.ts_monthly_deseason.shape == t.shape
    assert res.monthly_clim.shape == (12,)


def test_calc_trend_sutton_with_nan():
    t = np.arange(1993.0, 1996.0, 1 / 12)
    y = np.full(t.size, 360.0)
    y[3] = np.nan
    res = calc_trend_sutton(t, y, w=5.0)
    assert np.isfinite(res.slope)
    assert res.n_valid == t.size - 1


def test_trend_map_shape_and_signal():
    _ = np.random.default_rng(1)
    years = np.arange(1993, 1996)
    # a true monthly linear ramp: every point lies on a line -> slope exact
    field = np.zeros((3, 12, 8, 10))
    trend_slope = 2.0
    for y in range(3):
        field[y] = (
            trend_slope * (years[y] - 1993)
            + trend_slope * (np.arange(12)[:, None, None] / 12.0)
            + 340.0
        )
    slope = trend_map(field, years)
    assert slope.shape == (8, 10)
    valid = slope[np.isfinite(slope)]
    assert np.allclose(valid, trend_slope, atol=1e-6)


# ---------------------------------------------------------------------------
# rendering smoke tests
# ---------------------------------------------------------------------------


def test_figure_smoke(tmp_path):
    import matplotlib.pyplot as plt

    use_v11_style()
    lon = np.arange(260.0, 321.0, 1.0)
    lat = np.arange(10.0, 66.0, 1.0)
    rng = np.random.default_rng(2)
    field = rng.normal(360.0, 20.0, (lat.size, lon.size))
    masks = region_masks(lon, lat)
    field = np.where(masks["Atlantic"], field, np.nan)

    fig = plt.figure(figsize=(7, 4.5))
    from recad.viz.geo import _AxesFactory, field_map, setup_geo_axes

    ax = _AxesFactory.single(fig)
    setup_geo_axes(ax)
    pc = field_map(ax, field, lon, lat, label="fCO$_2$ (碌atm)")
    assert pc is not None
    fig.savefig(tmp_path / "map_smoke.png", dpi=80)
    plt.close(fig)
    assert (tmp_path / "map_smoke.png").stat().st_size > 5_000


def test_density_scatter_renders(tmp_path):
    import matplotlib.pyplot as plt

    from recad.viz.scatter import density_scatter

    rng = np.random.default_rng(3)
    obs = rng.normal(360.0, 20.0, 500)
    est = obs + rng.normal(0, 5.0, 500)
    fig, ax = plt.subplots(figsize=(5, 5))
    pc = density_scatter(ax, est, obs, bin_step=2.0, xlim=(280, 440), ylim=(280, 440))
    fig.savefig(tmp_path / "scatter_smoke.png", dpi=80)
    plt.close(fig)
    assert pc is not None and (tmp_path / "scatter_smoke.png").stat().st_size > 5_000


def test_climatology_comparison_renders(tmp_path):
    import matplotlib.pyplot as plt

    from recad.viz.series import climatology_comparison

    obs = 350.0 + np.tile(np.sin(np.arange(12) / 12 * 2 * np.pi) * 10, (5, 1))
    est = obs + 3.0
    fig, ax = plt.subplots(figsize=(7, 4))
    climatology_comparison(
        ax,
        {("SOCAT", "k"): obs, ("Product", "tab:red"): est},
        title="monthly climatology",
    )
    fig.savefig(tmp_path / "clim_smoke.png", dpi=80)
    plt.close(fig)
    assert (tmp_path / "clim_smoke.png").stat().st_size > 5_000


def test_trend_series_renders(tmp_path):
    import matplotlib.pyplot as plt

    from recad.viz.series import deseasonalized_timeseries

    t = np.arange(1993.0, 2003.0, 1 / 12)
    y = 1.8 * (t - 1993) + 350 + 8 * np.sin(2 * np.pi * (np.arange(t.size) % 12) / 12)
    fig, ax = plt.subplots(figsize=(7, 4))
    res = deseasonalized_timeseries(ax, t, y, label="pCO2", color="tab:blue")
    assert np.isfinite(res.slope)
    fig.savefig(tmp_path / "trend_smoke.png", dpi=80)
    plt.close(fig)
    assert (tmp_path / "trend_smoke.png").stat().st_size > 5_000


def test_uncertainty_histograms_renders(tmp_path):
    import matplotlib.pyplot as plt

    from recad.viz.uncertainty import contribution_histograms

    rng = np.random.default_rng(5)
    fields = {
        k: rng.gamma(2.0, 1.0, 2000) for k in ("u_SSS", "u_SST", "u_SSH", "u_pCO2air", "u_inputs")
    }
    fig, _ = contribution_histograms(fields)
    fig.savefig(tmp_path / "hist_smoke.png", dpi=80)
    plt.close(fig)
    assert (tmp_path / "hist_smoke.png").stat().st_size > 5_000


def test_panel_maps_renders(tmp_path):
    import matplotlib.pyplot as plt

    from recad.viz.geo import panel_maps

    lon = np.arange(260.0, 321.0, 1.0)
    lat = np.arange(10.0, 66.0, 1.0)
    rng = np.random.default_rng(6)
    fields = {
        "a": rng.normal(360, 20, (lat.size, lon.size)),
        "b": rng.normal(360, 20, (lat.size, lon.size)),
    }
    fig, pc = panel_maps(fields, lon, lat, nrows_ncols=(1, 2), label="fCO2")
    fig.savefig(tmp_path / "panels_smoke.png", dpi=80)
    plt.close(fig)
    assert pc is not None and (tmp_path / "panels_smoke.png").stat().st_size > 5_000


# ---------------------------------------------------------------------------
# full report render (uses the e2e synthetic workspace)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_render_figure_set_end_to_end(tmp_path):
    import matplotlib.pyplot as plt

    from recad.cli import main

    workdir = tmp_path / "run"
    assert main(["e2e", "--workdir", str(workdir), "--n-years", "2"]) == 0

    from recad.data.pipeline import load_masks, load_prepared
    from recad.viz.report import render_figure_set

    prepared = load_prepared(workdir / "prepared.nc")
    masks = load_masks(workdir / "masks.nc")
    out = tmp_path / "figs"
    outputs = render_figure_set(
        str(workdir / "product.nc"),
        prepared,
        masks,
        str(out),
        fmts=("png",),
    )
    assert "product_mean_map" in outputs
    for key in (
        "seasonal_maps",
        "density_scatter_ttv",
        "trend_map",
        "uncertainty_panels",
        "uncertainty_histograms",
        "model_summary.csv",
    ):
        assert key in outputs, key
        assert Path(outputs[key]).is_file(), key
    plt.close("all")
