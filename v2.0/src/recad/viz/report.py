"""One-call rendering of the v1.1-style figure set (`recad plot`).

Given a v2.0 product (NetCDF from ``recad predict``/``recad uncertainty``),
the prepared data (SOCAT target + grid) and the split masks, this module
regenerates the whole family of v1.1 figures and summary tables:

    Figure / table                    v1.1 counterpart
    -------------------------------   ----------------------------------
    01_product_mean_map               Figure_07 (ReCAD panel)
    02_seasonal_maps                  Figure_02 (seasonal climatology)
    03_density_scatter_train_val_test Figure_03 (Train/Validation/Test)
    04_regional_scatter               Figure_Model_Training (region rows)
    05_climatology_comparison_ns      Figure_06 (North/South climatology)
    06_trend_map                      Figure_08 (per-cell trend)
    07_regional_trend_series          Figure_08 (box deseasonalized series)
    08_uncertainty_panels             Figure_B1 / Figure_Uncertainties
    09_uncertainty_histograms         u_inputs_Monte_Carlo histograms
    model_summary.csv                 model_summary_chapter3.xlsx
    uncertainty_table.csv             uncertainty_table.xlsx

Everything defaults to the region definitions of the v1.1 NA-Atlantic domain
(``recad.viz.regions``); for other domains the region-specific panels fall
back to the whole domain.
"""

from __future__ import annotations

import contextlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from recad.data.features import PreparedData
from recad.data.grid import DomainGrid
from recad.data.split import SplitMasks
from recad.train.metrics import metrics_summary
from recad.utils.io import touch_dir
from recad.utils.logging import get_logger
from recad.viz.trends import trend_map

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def _product_generator(product_path: str) -> xr.Dataset:
    return xr.open_dataset(product_path)


def _region_masks_for(grid: DomainGrid) -> dict[str, np.ndarray] | None:
    """v1.1 NA-Atlantic region masks when the grid overlaps that domain."""
    from recad.viz.regions import region_masks

    try:
        masks = region_masks(grid.lon, grid.lat)
    except Exception:  # pragma: no cover - geometry should not fail
        return None
    # require a meaningful overlap with the NA-Atlantic window
    lon180 = ((grid.lon % 360) + 180) % 360 - 180
    if not ((lon180 >= -100).any() and (lon180 <= -40).any()):
        return None
    return masks


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _save(fig, out_dir: Path, name: str, fmts: tuple[str, ...]) -> None:
    for fmt in fmts:
        fig.savefig(
            out_dir / f"{name}.{fmt}", dpi=150 if fmt == "png" else None, bbox_inches="tight"
        )
    from matplotlib import pyplot as plt

    plt.close(fig)


@contextlib.contextmanager
def _quiet_nanmean():
    """Silence the benign 'Mean of empty slice' RuntimeWarning: spatially
    sparse fields legitimately have all-NaN time slices."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def _nanmean2(field4d: np.ndarray) -> np.ndarray:
    with _quiet_nanmean():
        return np.nanmean(np.nanmean(field4d, axis=1), axis=0)


def _mean2d(field4d: np.ndarray) -> np.ndarray:
    return _nanmean2(field4d)


def _seasonal_mean(field4d: np.ndarray, months: list[int]) -> np.ndarray:
    with _quiet_nanmean():
        return np.nanmean(field4d[:, months, :, :], axis=(0, 1))


def fig_product_mean_map(
    product: xr.Dataset,
    lon: np.ndarray,
    lat: np.ndarray,
    out_dir: Path,
    fmts: tuple[str, ...],
    var: str = "fco2",
) -> Path:
    from recad.viz.geo import field_map, setup_geo_axes

    fields = {var: _mean2d(product[var].values)}
    from matplotlib import pyplot as plt

    from recad.viz.geo import _AxesFactory

    use_v11_style = __import__("recad.viz.style", fromlist=["use_v11_style"]).use_v11_style
    use_v11_style()
    fig = plt.figure(figsize=(7, 4.5))
    ax = _AxesFactory.single(fig)
    setup_geo_axes(ax)
    field_map(
        ax,
        fields[var],
        lon,
        lat,
        vmin=float(np.nanpercentile(fields[var], 2)),
        vmax=float(np.nanpercentile(fields[var], 98)),
        label="fCO$_2$ (\\u788catm)",
        title="ReCAD mean fCO$_2$",
    )
    _save(fig, out_dir, "01_product_mean_map", fmts)
    return out_dir / "01_product_mean_map.png"


def fig_seasonal_maps(
    field4d: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    out_dir: Path,
    fmts: tuple[str, ...],
) -> Path:
    from recad.viz.geo import panel_maps
    from recad.viz.style import _SEASON_MONTHS, _SEASON_NAMES

    fields = {
        name: _seasonal_mean(field4d, months)
        for name, months in zip(_SEASON_NAMES, _SEASON_MONTHS, strict=False)
    }
    fig, _ = panel_maps(
        fields,
        lon,
        lat,
        nrows_ncols=(2, 2),
        vmin=float(np.nanpercentile(_mean2d(field4d), 2)),
        vmax=float(np.nanpercentile(_mean2d(field4d), 98)),
        label="fCO$_2$ (\\u788catm)",
        titles={k: k for k in fields},
    )
    _save(fig, out_dir, "02_seasonal_maps", fmts)
    return out_dir / "02_seasonal_maps.png"


def fig_density_scatter_ttv(
    est4d: np.ndarray,
    obs4d: np.ndarray,
    masks: SplitMasks,
    out_dir: Path,
    fmts: tuple[str, ...],
    var_label: str = "fCO$_2$",
) -> Path:
    from matplotlib import pyplot as plt

    from recad.viz.scatter import density_scatter

    use_v11_style = __import__("recad.viz.style", fromlist=["use_v11_style"]).use_v11_style
    use_v11_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    panels = (
        ("train", "Model Training"),
        ("val", "Model Validation"),
        ("test", "Independent Test"),
    )
    for ax, (key, title) in zip(axes, panels, strict=False):
        sel = masks.get(key)
        x = est4d[sel]
        y = obs4d[sel]
        valid = ~np.isnan(x) & ~np.isnan(y)
        density_scatter(
            ax,
            x[valid],
            y[valid],
            bin_step=2.0,
            title=title,
            label_x=f"{var_label}$_{{est}}$ (\\u788catm)",
            label_y=f"{var_label}$_{{obs}}$ (\\u788catm)",
        )
    fig.tight_layout()
    _save(fig, out_dir, "03_density_scatter_train_val_test", fmts)
    return out_dir / "03_density_scatter_train_val_test.png"


def fig_regional_scatter(
    est4d: np.ndarray,
    obs4d: np.ndarray,
    masks: SplitMasks,
    masks4d: np.ndarray,
    out_dir: Path,
    fmts: tuple[str, ...],
) -> Path:
    from matplotlib import pyplot as plt

    from recad.viz.scatter import density_scatter

    use_v11_style = __import__("recad.viz.style", fromlist=["use_v11_style"]).use_v11_style
    use_v11_style()
    keys = ["GStL", "SS", "GoMe", "MAB", "SAB", "GoMx"]
    names = ["GStL & GB", "SS", "GoME", "MAB", "SAB", "GoMX"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    train = masks.get("train")
    if masks4d and "Atlantic" in masks4d:
        domain = np.broadcast_to(masks4d["Atlantic"], est4d.shape)
    elif masks4d:
        domain = None  # use per-region masks below
    else:
        domain = np.ones(est4d.shape, dtype=bool)
    for ax, key, name in zip(axes.ravel(), keys, names, strict=False):
        if domain is not None:
            sel = train & domain
        else:
            sel = train & np.broadcast_to(masks4d[key], est4d.shape)
        x, y = est4d[sel], obs4d[sel]
        valid = ~np.isnan(x) & ~np.isnan(y)
        density_scatter(
            ax, x[valid], y[valid], bin_step=2.0, title=name, xlim=(200, 500), ylim=(200, 500)
        )
    fig.tight_layout()
    _save(fig, out_dir, "04_regional_scatter", fmts)
    return out_dir / "04_regional_scatter.png"


def fig_climatology_comparison(
    est4d: np.ndarray,
    obs4d: np.ndarray,
    masks4d: np.ndarray,
    out_dir: Path,
    fmts: tuple[str, ...],
) -> Path:
    from matplotlib import pyplot as plt

    from recad.viz.series import climatology_comparison, region_time_series

    use_v11_style = __import__("recad.viz.style", fromlist=["use_v11_style"]).use_v11_style
    use_v11_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    domains = [("Atlantic_south", "Southern areas"), ("Atlantic_north", "Northern areas")]
    for ax, (reg_key, title) in zip(axes, domains, strict=False):
        if reg_key not in masks4d:
            ax.axis("off")
            continue
        _ = np.broadcast_to(masks4d[reg_key], est4d.shape)
        obs_ts = region_time_series(obs4d, masks4d[reg_key])
        est_ts = region_time_series(np.where(~np.isnan(obs4d), est4d, np.nan), masks4d[reg_key])
        climatology_comparison(
            ax,
            {
                ("SOCAT (with missing data)", "k"): obs_ts,
                ("Product (gap-filled)", "tab:red"): est_ts,
            },
            title=title,
        )
    fig.tight_layout()
    _save(fig, out_dir, "05_climatology_comparison_ns", fmts)
    return out_dir / "05_climatology_comparison_ns.png"


def fig_trend_map(
    est4d: np.ndarray,
    years: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    out_dir: Path,
    fmts: tuple[str, ...],
) -> Path:
    from recad.viz.geo import field_map, setup_geo_axes

    use_v11_style = __import__("recad.viz.style", fromlist=["use_v11_style"]).use_v11_style
    use_v11_style()
    slope = trend_map(est4d, years)
    from matplotlib import pyplot as plt

    from recad.viz.geo import _AxesFactory

    fig = plt.figure(figsize=(7, 4.5))
    ax = _AxesFactory.single(fig)
    setup_geo_axes(ax)
    field_map(
        ax,
        slope,
        lon,
        lat,
        cmap="RdBu_r",
        vmin=float(np.nanpercentile(slope, 2)),
        vmax=float(np.nanpercentile(slope, 98)),
        label="fCO$_2$ trend (\\u788catm yr$^{-1}$)",
        title="fCO$_2$ linear trend",
    )
    _save(fig, out_dir, "06_trend_map", fmts)
    return out_dir / "06_trend_map.png"


def fig_uncertainty_panels(
    product: xr.Dataset,
    lon: np.ndarray,
    lat: np.ndarray,
    out_dir: Path,
    fmts: tuple[str, ...],
) -> Path:

    candidates = (
        "fco2_err_aleatoric",
        "fco2_err_epistemic",
        "fco2_err_input",
        "fco2_err_model",
        "fco2_err_total",
    )
    fields = {k: _mean2d(product[k].values) for k in candidates if k in product.data_vars}
    if not fields:
        _LOG.warning("no uncertainty variables in product; skipping fig 08/09")
        return out_dir / "08_uncertainty_panels.png"
    vmax = float(min(np.nanpercentile(_mean2d(product[next(iter(fields))].values), 98), 30.0))
    from recad.viz.geo import panel_maps
    from recad.viz.style import use_v11_style

    use_v11_style()
    fig, _ = panel_maps(
        fields,
        lon,
        lat,
        nrows_ncols=(2, 3),
        vmin=0.0,
        vmax=vmax,
        label="fCO$_2$ (\\u788catm)",
        figsize=(15, 7),
    )
    _save(fig, out_dir, "08_uncertainty_panels", fmts)
    return out_dir / "08_uncertainty_panels.png"


def fig_uncertainty_histograms(product: xr.Dataset, out_dir: Path, fmts: tuple[str, ...]) -> Path:
    from recad.viz.uncertainty import contribution_histograms

    candidates = (
        "fco2_err_aleatoric",
        "fco2_err_epistemic",
        "fco2_err_input",
        "fco2_err_model",
        "fco2_err_total",
    )
    std_fields = {k: product[k].values.ravel() for k in candidates if k in product.data_vars}
    if not std_fields:
        return out_dir / "09_uncertainty_histograms.png"
    fig, _ = contribution_histograms(std_fields)
    _save(fig, out_dir, "09_uncertainty_histograms", fmts)
    return out_dir / "09_uncertainty_histograms.png"


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def model_summary_table(
    est4d: np.ndarray,
    obs4d: np.ndarray,
    masks: SplitMasks,
    masks4d: dict[str, np.ndarray] | None,
) -> pd.DataFrame:
    """Region x Type(Train/Validation/Test/All) R\\u864f/RMSE/MAE/MBE table."""
    regions = [
        ("GStL", "GStL & GB"),
        ("SS", "SS"),
        ("GoMe", "GoME"),
        ("MAB", "MAB"),
        ("SAB", "SAB"),
        ("GoMx", "GoMX"),
        ("Atlantic", "NAACOM"),
    ]
    rows: list[dict] = []
    for reg_key, reg_name in regions:
        if masks4d is not None and reg_key in masks4d:
            reg = np.broadcast_to(masks4d[reg_key], est4d.shape)
        elif reg_key == "Atlantic":
            reg = np.ones(est4d.shape, dtype=bool)
        else:
            continue
        for type_name in ("train", "val", "test", "all"):
            sel = reg if type_name == "all" else reg & masks.get(type_name)
            x, y = est4d[sel], obs4d[sel]
            valid = ~np.isnan(x) & ~np.isnan(y)
            m = (
                metrics_summary(y[valid], x[valid])
                if valid.any()
                else {
                    "r2": np.nan,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "bias": np.nan,
                    "n": 0,
                }
            )
            rows.append(
                {
                    "Region": reg_name,
                    "Type": type_name.title(),
                    "R2": m["r2"],
                    "RMSE": m["rmse"],
                    "MAE": m["mae"],
                    "MBE": m["bias"],
                    "N": m["n"],
                }
            )
    return pd.DataFrame(rows)


def uncertainty_table(product: xr.Dataset, out_dir: Path) -> pd.DataFrame:
    """Region-level combined uncertainty table (v1.1 uncertainty_table.xlsx)."""
    from recad.viz.regions import region_masks

    lon = product.lon.values
    lat = product.lat.values
    masks = region_masks(lon, lat)
    regions = [
        ("GStL", "GStL&GB"),
        ("SS", "SS"),
        ("GoMe", "GoMe"),
        ("MAB", "MAB"),
        ("SAB", "SAB"),
        ("GoMx", "GoMx"),
        ("Atlantic", "NAACOM"),
    ]
    cols = [
        k
        for k in (
            "fco2_err_aleatoric",
            "fco2_err_epistemic",
            "fco2_err_input",
            "fco2_err_model",
            "fco2_err_total",
        )
        if k in product.data_vars
    ]
    rows = []
    field0 = product[cols[0]].values if cols else None
    for reg_key, reg_name in regions:
        if reg_key not in masks or field0 is None:
            continue
        sel = np.broadcast_to(masks[reg_key], field0.shape)
        if not sel.any():
            continue  # domain does not overlap this v1.1 region
        row = {"Region": reg_name}
        for col in cols:
            field = product[col].values
            row[col.replace("fco2_err_", "u_")] = np.nanmean(field[sel])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# top-level
# ---------------------------------------------------------------------------


def render_figure_set(
    product_path: str,
    prepared: PreparedData,
    masks: SplitMasks,
    out_dir: str,
    *,
    fmts: tuple[str, ...] = ("png",),
    target_var: str = "fco2",
) -> dict[str, str]:
    """Regenerate the v1.1 figure/table set from a product + data + masks.

    Returns a mapping of logical name -> output path.
    """
    out = touch_dir(out_dir)
    ds = xr.open_dataset(product_path)
    lon, lat = ds.lon.values, ds.lat.values
    years = ds.year.values.astype(int)
    est = ds[target_var].values
    obs = prepared.target_values()
    masks4d = _region_masks_for(prepared.grid)

    outputs: dict[str, str] = {}
    if obs.shape != est.shape:
        # product may cover only part of the domain; align by index
        raise ValueError("product and prepared fields must share the 4-D shape")
    outputs["product_mean_map"] = str(fig_product_mean_map(ds, lon, lat, out, fmts))
    outputs["seasonal_maps"] = str(fig_seasonal_maps(est, lon, lat, out, fmts))
    outputs["density_scatter_ttv"] = str(fig_density_scatter_ttv(est, obs, masks, out, fmts))
    outputs["regional_scatter"] = str(
        fig_regional_scatter(est, obs, masks, masks4d or {}, out, fmts)
    )
    outputs["climatology_comparison"] = str(
        fig_climatology_comparison(est, obs, masks4d or {}, out, fmts)
    )
    outputs["trend_map"] = str(fig_trend_map(est, years, lon, lat, out, fmts))
    outputs["uncertainty_panels"] = str(fig_uncertainty_panels(ds, lon, lat, out, fmts))
    outputs["uncertainty_histograms"] = str(fig_uncertainty_histograms(ds, out, fmts))

    summary = model_summary_table(est, obs, masks, masks4d)
    summary.to_csv(out / "model_summary.csv", index=False)
    outputs["model_summary.csv"] = str(out / "model_summary.csv")
    try:
        unc = uncertainty_table(ds, out)
        unc.to_csv(out / "uncertainty_table.csv", index=False)
        outputs["uncertainty_table.csv"] = str(out / "uncertainty_table.csv")
    except Exception as exc:  # pragma: no cover - table is best-effort
        _LOG.warning("uncertainty_table skipped: %s", exc)
    ds.close()
    _LOG.info("figure set rendered to %s (%d outputs)", out, len(outputs))
    return outputs
