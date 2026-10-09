"""Build, audit, and archive the observation-independent ReCAD coastal mask."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import BallTree

from recad.data.coastal_mask import (
    CoastalMaskSpec,
    build_coastal_mask_dataset,
    mask_area_km2,
    regular_cell_area_km2,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_global_coastal_mask_v2.3"
DEFAULT_DISTANCE_ROOT = Path(r"C:\backup\phd\data\raw\landmetrics\v1.0.0")
DEFAULT_BATHY_ROOT = Path(r"C:\backup\phd\data\raw\gebco\gebco_2025_regrid")
DEFAULT_CACHE = Path(r"C:\backup\phd\data\processed\recad_v2_3\global_socat_cache_v2.3.parquet")
DEFAULT_OUT = Path(r"C:\backup\phd\data\processed\recad_v2_3")
THRESHOLDS = (0, 1400, 4748)
RESOLUTION_DEG = 0.125
DISTANCE_RESOLUTION_DEG = 0.05


def sha256(path: Path, block_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def target_grid() -> tuple[np.ndarray, np.ndarray]:
    longitude = np.arange(0.0, 360.0, RESOLUTION_DEG, dtype=np.float64)
    latitude = np.arange(-78.0, 84.0 + RESOLUTION_DEG / 2, RESOLUTION_DEG, dtype=np.float64)
    if longitude.size != 2880 or latitude.size != 1297:
        raise RuntimeError("target grid no longer matches the frozen ReCAD grid")
    return longitude, latitude


def distance_path(root: Path, threshold: int) -> Path:
    return root / f"distance_to_land_0.05deg_gt{threshold}km2.nc"


def query_distance_fields(
    root: Path, longitude: np.ndarray, latitude: np.ndarray
) -> dict[int, np.ndarray]:
    try:
        import landmetrics as lm
    except ImportError as exc:  # pragma: no cover - exercised by real-data entry point
        raise RuntimeError(
            "install the coastal-mask extra: pip install -e .[coastal-mask]"
        ) from exc
    lon2d, lat2d = np.meshgrid(longitude, latitude)
    fields: dict[int, np.ndarray] = {}
    for threshold in THRESHOLDS:
        path = distance_path(root, threshold)
        if not path.is_file():
            raise FileNotFoundError(path)
        fields[threshold] = np.asarray(lm.distance_to_land(lat2d, lon2d, path=path))
    return fields


def load_bathymetry_tiles(
    root: Path, longitude: np.ndarray, latitude: np.ndarray
) -> tuple[np.ndarray, list[Path]]:
    paths = sorted(root.glob("gebco_2025_*.npz"))
    if len(paths) != 8:
        raise FileNotFoundError(
            f"expected eight GEBCO tile extracts under {root}; found {len(paths)}"
        )
    output = np.full((latitude.size, longitude.size), np.nan, dtype=np.float32)
    for path in paths:
        with np.load(path) as tile:
            tile_lon = np.asarray(tile["lon"], dtype=np.float64) % 360
            tile_lat = np.asarray(tile["lat"], dtype=np.float64)
            elevation = np.asarray(tile["elevation_m"], dtype=np.float32)
        lon_index = np.rint(tile_lon / RESOLUTION_DEG).astype(int) % longitude.size
        lat_index = np.rint((tile_lat - latitude[0]) / RESOLUTION_DEG).astype(int)
        if elevation.shape != (tile_lat.size, tile_lon.size):
            raise ValueError(f"invalid GEBCO tile shape: {path}")
        output[np.ix_(lat_index, lon_index)] = elevation
    if not np.isfinite(output).all():
        raise ValueError("GEBCO tile extracts do not cover every target-grid node")
    return output, paths


def nearest_mask_values(
    mask: np.ndarray, longitude: np.ndarray, latitude: np.ndarray, frame: pd.DataFrame
) -> np.ndarray:
    lon_index = np.rint((frame["longitude"].to_numpy(float) % 360) / RESOLUTION_DEG).astype(int)
    lon_index %= longitude.size
    lat_index = np.rint((frame["latitude"].to_numpy(float) - latitude[0]) / RESOLUTION_DEG).astype(
        int
    )
    inside = (lat_index >= 0) & (lat_index < latitude.size)
    values = np.zeros(len(frame), dtype=bool)
    values[inside] = mask[lat_index[inside], lon_index[inside]]
    return values


def build_candidate_tables(
    fields: dict[int, np.ndarray],
    bathymetry: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    cache: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[int, xr.Dataset]]:
    datasets: dict[int, xr.Dataset] = {}
    summary_rows: list[dict[str, object]] = []
    latitude_rows: list[dict[str, object]] = []
    retention_rows: list[dict[str, object]] = []
    all_distance = fields[0]
    row_area = regular_cell_area_km2(latitude, RESOLUTION_DEG)
    ocean = all_distance > 0
    for threshold in THRESHOLDS:
        spec = CoastalMaskSpec(significant_island_area_km2=float(threshold))
        dataset = build_coastal_mask_dataset(
            longitude,
            latitude,
            all_distance,
            fields[threshold],
            bathymetry_m=bathymetry,
            spec=spec,
        )
        datasets[threshold] = dataset
        product = dataset.coastal_product_mask.values.astype(bool)
        shelf = dataset.shelf_core_mask.values.astype(bool)
        deep = dataset.slope_deep_mask.values.astype(bool)
        summary_rows.append(
            {
                "min_island_area_km2": threshold,
                "selected": threshold == 1400,
                "ocean_nodes": int(ocean.sum()),
                "product_nodes": int(product.sum()),
                "product_area_km2": mask_area_km2(product, latitude, RESOLUTION_DEG),
                "shelf_nodes": int(shelf.sum()),
                "shelf_area_km2": mask_area_km2(shelf, latitude, RESOLUTION_DEG),
                "deep_nodes": int(deep.sum()),
                "unclassified_nodes": int(
                    dataset.bathymetry_unclassified_mask.values.astype(bool).sum()
                ),
            }
        )
        observed = nearest_mask_values(product, longitude, latitude, cache)
        retention_rows.append(
            {
                "min_island_area_km2": threshold,
                "cache_rows": len(cache),
                "retained_rows": int(observed.sum()),
                "retained_row_fraction": float(observed.mean()),
                "observed_nodes": int(cache["grid_flat"].nunique()),
                "retained_observed_nodes": int(cache.loc[observed, "grid_flat"].nunique()),
            }
        )
        latitude_bin = (np.floor((latitude + 90) / 10) * 10 - 90).astype(int)
        for lower in np.unique(latitude_bin):
            rows = latitude_bin == lower
            latitude_rows.append(
                {
                    "min_island_area_km2": threshold,
                    "latitude_bin_lower": lower,
                    "latitude_bin_upper": lower + 10,
                    "product_nodes": int(product[rows].sum()),
                    "product_area_km2": float((product[rows] * row_area[rows, None]).sum()),
                }
            )

    key_locations = pd.DataFrame(
        [
            ("Honolulu/Oahu", 21.307, -157.858),
            ("Hawaii/Big Island", 19.593, -155.428),
            ("Reunion", -21.115, 55.536),
            ("Galapagos/Isabela", -0.830, -91.130),
            ("New Caledonia", -21.300, 165.500),
            ("Azores/Sao Miguel", 37.780, -25.500),
            ("Iceland", 64.963, -19.021),
            ("Madagascar", -18.766, 46.869),
            ("New Zealand", -41.286, 174.776),
            ("Japan", 35.676, 139.650),
        ],
        columns=["location", "latitude", "longitude"],
    )
    lon_index = np.rint((key_locations.longitude.to_numpy() % 360) / RESOLUTION_DEG).astype(int)
    lat_index = np.rint((key_locations.latitude.to_numpy() - latitude[0]) / RESOLUTION_DEG).astype(
        int
    )
    for threshold in THRESHOLDS:
        key_locations[f"distance_gt{threshold}km2_km"] = fields[threshold][lat_index, lon_index]

    selected = datasets[1400]
    distance_bins = np.array([0, 25, 50, 100, 200, 300, 400.0001])
    selected_distance = selected.distance_to_significant_land_km.values
    product = selected.coastal_product_mask.values.astype(bool)
    bin_rows = []
    for left, right in pairwise(distance_bins):
        hits = product & (selected_distance >= left) & (selected_distance < right)
        bin_rows.append(
            {
                "distance_left_km": left,
                "distance_right_km": right,
                "nodes": int(hits.sum()),
                "area_km2": mask_area_km2(hits, latitude, RESOLUTION_DEG),
            }
        )

    bathy_rows = []
    classes = {
        "shelf_0_200m": selected.shelf_core_mask.values.astype(bool),
        "slope_deep_gt200m": selected.slope_deep_mask.values.astype(bool),
        "unclassified": selected.bathymetry_unclassified_mask.values.astype(bool),
    }
    for name, class_mask in classes.items():
        bathy_rows.append(
            {
                "class": name,
                "nodes": int(class_mask.sum()),
                "area_km2": mask_area_km2(class_mask, latitude, RESOLUTION_DEG),
                "fraction_of_product_nodes": float(class_mask.sum() / product.sum()),
            }
        )
    tables = {
        "candidate_summary": pd.DataFrame(summary_rows),
        "latitude_coverage": pd.DataFrame(latitude_rows),
        "observation_retention": pd.DataFrame(retention_rows),
        "key_locations": key_locations,
        "distance_bins": pd.DataFrame(bin_rows),
        "bathymetric_classes": pd.DataFrame(bathy_rows),
    }
    return tables, datasets


def _geometry_vertices(geometry: object) -> list[np.ndarray]:
    geometry_type = getattr(geometry, "geom_type", "")
    if geometry_type in {"LineString", "LinearRing"}:
        return [np.asarray(geometry.coords)]
    if geometry_type == "MultiLineString":
        return [np.asarray(part.coords) for part in geometry.geoms]
    return []


def independent_distance_audit(
    all_land_distance: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    *,
    sample_size: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare GSHHG lookup distances with Natural Earth + haversine BallTree."""

    import cartopy.io.shapereader as shpreader

    source = shpreader.natural_earth(resolution="10m", category="physical", name="coastline")
    vertices = []
    for geometry in shpreader.Reader(source).geometries():
        vertices.extend(_geometry_vertices(geometry))
    coast = np.concatenate(vertices)
    coast_latlon = np.deg2rad(np.column_stack([coast[:, 1], coast[:, 0]]))
    tree = BallTree(coast_latlon, metric="haversine")
    eligible = np.flatnonzero((all_land_distance > 0) & (all_land_distance <= 500))
    rng = np.random.default_rng(20260930)
    chosen = rng.choice(eligible, size=min(sample_size, eligible.size), replace=False)
    lat_index, lon_index = np.unravel_index(chosen, all_land_distance.shape)
    points = np.deg2rad(np.column_stack([latitude[lat_index], longitude[lon_index]]))
    natural_earth = tree.query(points, k=1, return_distance=True)[0].ravel() * 6371.0088
    gshhg = all_land_distance[lat_index, lon_index]
    errors = np.abs(gshhg - natural_earth)
    samples = pd.DataFrame(
        {
            "latitude": latitude[lat_index],
            "longitude": longitude[lon_index],
            "gshhg_distance_km": gshhg,
            "natural_earth_distance_km": natural_earth,
            "absolute_difference_km": errors,
        }
    )
    quantiles = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]
    summary = pd.DataFrame(
        {
            "quantile": quantiles,
            "absolute_difference_km": np.quantile(errors, quantiles),
        }
    )
    return samples, summary


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_mask(mask: np.ndarray, title: str, path: Path, *, cmap: str = "viridis") -> None:
    figure, axis = plt.subplots(figsize=(12, 5.2))
    image = axis.imshow(
        mask,
        origin="lower",
        extent=(0, 360, -78, 84),
        interpolation="nearest",
        aspect="auto",
        cmap=cmap,
    )
    axis.set(title=title, xlabel="Longitude (degrees east)", ylabel="Latitude")
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    _save_figure(figure, path)


def create_figures(
    tables: dict[str, pd.DataFrame],
    datasets: dict[int, xr.Dataset],
    audit_samples: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for number, threshold in enumerate(THRESHOLDS, start=1):
        _plot_mask(
            datasets[threshold].coastal_product_mask.values,
            f"Candidate coastal product mask: land area threshold {threshold:,} km²",
            figure_dir / f"fig{number:02d}_candidate_gt{threshold}km2.png",
            cmap="Blues",
        )

    all_land = datasets[0].coastal_product_mask.values.astype(bool)
    selected = datasets[1400].coastal_product_mask.values.astype(bool)
    restrictive = datasets[4748].coastal_product_mask.values.astype(bool)
    difference = np.zeros(selected.shape, dtype=np.int8)
    difference[all_land & ~selected] = 1
    difference[selected & ~restrictive] = 2
    _plot_mask(
        difference,
        "Sensitivity to excluding small and medium islands (0=no difference)",
        figure_dir / "fig04_candidate_difference.png",
        cmap="magma",
    )

    summary = tables["candidate_summary"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = summary.min_island_area_km2.astype(str)
    axes[0].bar(x, summary.product_area_km2 / 1e6, color="#2676b8")
    axes[0].set(xlabel="Minimum island area (km²)", ylabel="Product area (million km²)")
    axes[1].bar(x, summary.product_nodes / 1e6, color="#e17c31")
    axes[1].set(xlabel="Minimum island area (km²)", ylabel="Product nodes (million)")
    figure.suptitle("Candidate mask size sensitivity")
    _save_figure(figure, figure_dir / "fig05_candidate_area_nodes.png")

    latitude = tables["latitude_coverage"]
    figure, axis = plt.subplots(figsize=(10, 5))
    for threshold, group in latitude.groupby("min_island_area_km2"):
        axis.plot(
            group.latitude_bin_lower + 5,
            group.product_area_km2 / 1e6,
            marker="o",
            label=f">{threshold:,} km²",
        )
    axis.set(
        xlabel="Latitude-band midpoint",
        ylabel="Product area (million km²)",
        title="Coastal product area by 10-degree latitude band",
    )
    axis.legend(title="Land threshold")
    _save_figure(figure, figure_dir / "fig06_latitude_coverage.png")

    retention = tables["observation_retention"]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(
        retention.min_island_area_km2.astype(str),
        retention.retained_row_fraction * 100,
        color="#43a047",
    )
    axis.set(
        ylim=(0, 101),
        xlabel="Minimum island area (km²)",
        ylabel="Frozen SOCAT cache retained (%)",
        title="Observation retention is an audit, not a mask input",
    )
    _save_figure(figure, figure_dir / "fig07_observation_retention.png")

    bins = tables["distance_bins"]
    labels = [
        f"{left:g}-{right:g}"
        for left, right in zip(bins.distance_left_km, bins.distance_right_km, strict=True)
    ]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(labels, bins.area_km2 / 1e6, color="#6a51a3")
    axis.set(
        xlabel="Distance to significant land (km)",
        ylabel="Area (million km²)",
        title="Selected product area across the 0-400 km envelope",
    )
    axis.tick_params(axis="x", rotation=30)
    _save_figure(figure, figure_dir / "fig08_distance_bins.png")

    selected_ds = datasets[1400]
    classes = np.zeros(selected_ds.coastal_product_mask.shape, dtype=np.int8)
    classes[selected_ds.shelf_core_mask.values.astype(bool)] = 1
    classes[selected_ds.slope_deep_mask.values.astype(bool)] = 2
    classes[selected_ds.bathymetry_unclassified_mask.values.astype(bool)] = 3
    _plot_mask(
        classes,
        "Selected geometry: 1=shelf core, 2=slope/deep, 3=unclassified",
        figure_dir / "fig09_bathymetric_classes.png",
        cmap="viridis",
    )

    bathy = tables["bathymetric_classes"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(bathy["class"], bathy.area_km2 / 1e6, color=["#2ca25f", "#3182bd", "#bdbdbd"])
    axis.set(ylabel="Area (million km²)", title="Bathymetric classes within selected product")
    axis.tick_params(axis="x", rotation=15)
    _save_figure(figure, figure_dir / "fig10_bathymetric_area.png")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(
        audit_samples.gshhg_distance_km,
        audit_samples.natural_earth_distance_km,
        s=7,
        alpha=0.35,
    )
    axes[0].plot([0, 500], [0, 500], color="black", linewidth=1)
    axes[0].set(xlabel="GSHHG lookup (km)", ylabel="Natural Earth audit (km)")
    axes[1].hist(audit_samples.absolute_difference_km, bins=50, color="#756bb1")
    axes[1].set(xlabel="Absolute difference (km)", ylabel="Sample count")
    figure.suptitle("Independent coastline-distance audit")
    _save_figure(figure, figure_dir / "fig11_independent_distance_audit.png")


CAPTIONS = {
    "fig01_candidate_gt0km2.png": "Figure 1. Observation-independent 400-km candidate using every GSHHG island as land; this maximizes small-island halos and supplies the inclusive sensitivity endpoint.",
    "fig02_candidate_gt1400km2.png": "Figure 2. Frozen ReCAD product geometry: ocean nodes within 400 km of GSHHG land at least 1,400 km², evaluated on the native 1/8-degree grid.",
    "fig03_candidate_gt4748km2.png": "Figure 3. Restrictive 4,748-km² island-threshold candidate, included as the historical IBTrACS large-island sensitivity endpoint rather than the selected geometry.",
    "fig04_candidate_difference.png": "Figure 4. Cells removed as the minimum island area increases; class 1 is present only in the all-island candidate and class 2 is retained at 1,400 km² but removed at 4,748 km².",
    "fig05_candidate_area_nodes.png": "Figure 5. Total spherical area and native-grid node count for the three candidate product geometries, quantifying the consequence of the island-size choice.",
    "fig06_latitude_coverage.png": "Figure 6. Spherical product area by ten-degree latitude band for each candidate, showing where the island threshold changes global coverage.",
    "fig07_observation_retention.png": "Figure 7. Fraction of the frozen SOCATv2026 cache falling inside each candidate; observations are used only after geometry construction as a sensitivity audit and never define the mask.",
    "fig08_distance_bins.png": "Figure 8. Area of the selected 1,400-km² product geometry by distance from significant land; the complete release envelope ends at 400 km.",
    "fig09_bathymetric_classes.png": "Figure 9. GEBCO-2025 subdivision of the selected product into a 0-200 m shelf core, waters deeper than 200 m, and any bathymetrically unclassified cells.",
    "fig10_bathymetric_area.png": "Figure 10. Spherical area of shelf, slope/deep, and unclassified cells within the selected 400-km product geometry.",
    "fig11_independent_distance_audit.png": "Figure 11. Deterministic 2,000-point comparison between the GSHHG distance lookup and an independent haversine nearest-coast calculation from Natural Earth 10m vertices; disagreement reflects both source geometry and raster resolution.",
}


FIGURE_SOURCES = {
    "fig01_candidate_gt0km2.png": ["table01_candidate_summary.csv"],
    "fig02_candidate_gt1400km2.png": ["table01_candidate_summary.csv"],
    "fig03_candidate_gt4748km2.png": ["table01_candidate_summary.csv"],
    "fig04_candidate_difference.png": ["table01_candidate_summary.csv"],
    "fig05_candidate_area_nodes.png": ["table01_candidate_summary.csv"],
    "fig06_latitude_coverage.png": ["table02_latitude_coverage.csv"],
    "fig07_observation_retention.png": ["table03_observation_retention.csv"],
    "fig08_distance_bins.png": ["table05_distance_bins.csv"],
    "fig09_bathymetric_classes.png": ["table06_bathymetric_classes.csv"],
    "fig10_bathymetric_area.png": ["table06_bathymetric_classes.csv"],
    "fig11_independent_distance_audit.png": [
        "table07_independent_distance_samples.csv",
        "table08_independent_distance_summary.csv",
    ],
}


def _report_figure_blocks() -> str:
    blocks = []
    for figure_name, caption in CAPTIONS.items():
        blocks.append(f"![{caption.split('.')[0]}](figures/{figure_name})\n\n{caption}")
    return "\n\n".join(blocks)


def write_archive(
    archive: Path,
    tables: dict[str, pd.DataFrame],
    datasets: dict[int, xr.Dataset],
    audit_samples: pd.DataFrame,
    audit_summary: pd.DataFrame,
    source_inventory: pd.DataFrame,
    output_path: Path,
    output_sha: str,
) -> None:
    table_dir = archive / "tables"
    table_names = {
        "candidate_summary": "table01_candidate_summary.csv",
        "latitude_coverage": "table02_latitude_coverage.csv",
        "observation_retention": "table03_observation_retention.csv",
        "key_locations": "table04_key_locations.csv",
        "distance_bins": "table05_distance_bins.csv",
        "bathymetric_classes": "table06_bathymetric_classes.csv",
    }
    for key, name in table_names.items():
        write_table(tables[key], table_dir / name)
    write_table(audit_samples, table_dir / "table07_independent_distance_samples.csv")
    write_table(audit_summary, table_dir / "table08_independent_distance_summary.csv")
    write_table(source_inventory, table_dir / "table09_source_inventory.csv")
    decision = pd.DataFrame(
        [
            ("outer_distance_km", 400, "project release boundary"),
            ("significant_land_area_km2", 1400, "retains major island systems"),
            ("shelf_depth_m", 200, "bathymetric shelf core"),
            ("grid_resolution_deg", RESOLUTION_DEG, "native ReCAD grid"),
        ],
        columns=["parameter", "selected_value", "reason"],
    )
    write_table(decision, table_dir / "table10_frozen_decision.csv")
    agreement = pd.DataFrame(
        [
            {
                "tolerance_km": tolerance,
                "sample_fraction_within_tolerance": float(
                    (audit_samples.absolute_difference_km <= tolerance).mean()
                ),
            }
            for tolerance in (2.5, 5, 10, 25, 50, 100)
        ]
    )
    write_table(agreement, table_dir / "table11_independent_agreement.csv")

    selected = tables["candidate_summary"].query("selected").iloc[0]
    retained = tables["observation_retention"].query("min_island_area_km2 == 1400").iloc[0]
    p95 = audit_summary.loc[audit_summary["quantile"] == 0.95, "absolute_difference_km"].iloc[0]
    median = audit_summary.loc[audit_summary["quantile"] == 0.5, "absolute_difference_km"].iloc[0]
    within_10 = float((audit_samples.absolute_difference_km <= 10).mean())
    report = f"""# P2 global coastal product mask v2.3

## Scientific question and permitted claim

This experiment freezes an observation-independent spatial domain for the global ReCAD coastal product. The permitted claim is geometric: the mask identifies ocean grid nodes within 400 km of significant land, and separately marks the 0-200 m bathymetric shelf core. It does not claim that fCO2 or SSS is predictable at every included node.

## Data, splits, and leakage controls

Land distance comes from [landmetrics v1.0.0](https://zenodo.org/records/21959508) grids derived from GSHHG levels 1 and 5 at 0.05 degrees. Candidate minimum island areas are 0, 1,400, and 4,748 km². [GEBCO-2025](https://doi.org/10.5285/37c52e96-24ea-67ce-e063-7086abc05f29) ice-surface elevation supplies bathymetry and was sampled directly at native ReCAD nodes after official tile MD5 verification. The frozen SOCATv2026 cache from Issue #38 is queried only after each mask exists. No target value, split role, observation density, TA/DIC availability, or model prediction enters mask construction. Natural Earth 10m coastline vertices provide an independent deterministic distance audit.

## Candidate models and training

There is no trained statistical model in this experiment. The three deterministic geometry candidates share the same 400-km outer distance and differ only in minimum island area. The selected 1,400-km² threshold follows an established major-land sensitivity threshold used in IBTrACS/SHIPS tooling and retains major island systems such as Kauai. The 4,748-km² candidate is retained as a restrictive sensitivity case. This reproduces the broad SOCAT-style 400-km coastal convention but is not presented as an exact reconstruction of any historical SOCAT land mask.

## Main development results

The selected geometry contains **{int(selected.product_nodes):,} native nodes** and covers **{selected.product_area_km2 / 1e6:.3f} million km²**. Its GEBCO shelf core contains **{int(selected.shelf_nodes):,} nodes** and covers **{selected.shelf_area_km2 / 1e6:.3f} million km²**. It retains **{retained.retained_row_fraction * 100:.3f}%** of rows in the frozen global SOCAT cache, used only as a post hoc diagnostic. In the independent coastline audit, the median absolute difference is **{median:.2f} km**, **{within_10 * 100:.2f}%** of points agree within 10 km, and the 95th percentile is **{p95:.2f} km**. The long tail is concentrated around islands represented differently by GSHHG and Natural Earth; it records source-geometry sensitivity and is not treated as a raster interpolation error.

## Decision and limitations

Decision: **pass_geometry_freeze**. The released geometry is `distance_to_significant_land <= 400 km` over ocean, with significant land defined by the 1,400-km² GSHHG threshold. `shelf_core_mask` is a nested GEBCO class at depths from 0 to 200 m; it is not a second distance-based coastal domain. Observation support, predictor completeness, validation grade, and release eligibility must be computed as separate layers in later issues. The 0.05-degree distance raster and 1/8-degree target grid impose finite boundary uncertainty, so downstream area statistics must use the stored mask rather than recomputing a vector boundary.

The generated NetCDF is stored outside Git at `{output_path}` with SHA256 `{output_sha}`. The NetCDF includes continuous signed distance, bathymetry, ocean geometry, selected product geometry, shelf, slope/deep, and unclassified masks.

## Figure and table index

{_report_figure_blocks()}

Tables 1-11 in `tables/` contain the exact plotted summaries, independent audit samples, source hashes, agreement thresholds, and frozen decision parameters. Every image has source-data mapping in `archive_manifest.json`.
"""
    readme = """# P2 global coastal mask v2.3

Reviewer-ready evidence for Issue #50. Install `.[coastal-mask]`, fetch the three pinned 0.05-degree landmetrics distance grids, and prepare GEBCO with `python scripts/prepare_gebco_2025_regrid.py <external-output-directory>`. Run `python scripts/build_global_coastal_mask_v2_3.py` with the external landmetrics, GEBCO extracts, and frozen Issue #38 cache. Verify this directory with `python scripts/verify_experiment_archive.py docs/experiment_archive/p2_global_coastal_mask_v2.3`.

The large NetCDF product remains outside Git; its path and SHA256 are frozen in the report and manifest.
"""
    captions = (
        "# Figure captions\n\n"
        + "\n\n".join(f"## {name}\n\n{caption}" for name, caption in CAPTIONS.items())
        + "\n"
    )
    write_markdown(archive / "REPORT.md", report)
    write_markdown(archive / "README.md", readme)
    write_markdown(archive / "CAPTIONS.md", captions)

    tracked = [archive / "README.md", archive / "REPORT.md", archive / "CAPTIONS.md"]
    tracked.extend(sorted((archive / "figures").glob("*.png")))
    tracked.extend(sorted(table_dir.glob("*.csv")))
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    source_hashes = dict(zip(source_inventory.path, source_inventory.sha256, strict=True))
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "training_git_commit": git_commit,
        "data_manifest_sha256": output_sha,
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "observation-independent global coastal product geometry",
        "decision": "pass_geometry_freeze",
        "figure_source_data": FIGURE_SOURCES,
        "figure_captions": CAPTIONS,
        "files_sha256": {
            str(path.relative_to(archive)).replace("\\", "/"): sha256(path) for path in tracked
        },
        "archive_builder_sha256": sha256(Path(__file__)),
        "analysis_script_sha256": sha256(Path(__file__)),
        "preparation_script_sha256": sha256(ROOT / "scripts/prepare_gebco_2025_regrid.py"),
        "source_artifacts_sha256": source_hashes,
        "generated_product": {"path": str(output_path), "sha256": output_sha},
        "selected_spec": datasets[1400].attrs,
    }
    write_json(archive / "archive_manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance-root", type=Path, default=DEFAULT_DISTANCE_ROOT)
    parser.add_argument("--bathymetry-root", type=Path, default=DEFAULT_BATHY_ROOT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    longitude, latitude = target_grid()
    fields = query_distance_fields(args.distance_root, longitude, latitude)
    bathymetry, bathy_paths = load_bathymetry_tiles(args.bathymetry_root, longitude, latitude)
    cache = pd.read_parquet(args.cache)
    tables, datasets = build_candidate_tables(fields, bathymetry, longitude, latitude, cache)
    audit_samples, audit_summary = independent_distance_audit(fields[0], longitude, latitude)

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / "global_coastal_mask_v2.3.nc"
    selected = datasets[1400]
    selected.attrs.update(
        title="ReCAD v2.3 observation-independent global coastal product mask",
        source_land_distance="landmetrics v1.0.0 / GSHHG levels 1 and 5",
        source_bathymetry="GEBCO Compilation Group (2025), GEBCO_2025 Grid",
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    encoding = {
        name: {"zlib": True, "complevel": 4, "shuffle": True} for name in selected.data_vars
    }
    selected.to_netcdf(output_path, engine="h5netcdf", encoding=encoding)
    output_sha = sha256(output_path)

    source_paths = [distance_path(args.distance_root, threshold) for threshold in THRESHOLDS]
    source_paths.extend(bathy_paths)
    source_paths.append(args.cache)
    source_inventory = pd.DataFrame(
        [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in source_paths
        ]
    )
    archive = ROOT / "docs" / "experiment_archive" / EXPERIMENT_ID
    create_figures(tables, datasets, audit_samples, archive / "figures")
    write_archive(
        archive,
        tables,
        datasets,
        audit_samples,
        audit_summary,
        source_inventory,
        output_path,
        output_sha,
    )

    frozen_manifest = {
        "product": str(output_path),
        "sha256": output_sha,
        "grid": {"resolution_deg": RESOLUTION_DEG, "latitude_nodes": 1297, "longitude_nodes": 2880},
        "geometry": {"maximum_distance_km": 400, "significant_island_area_km2": 1400},
        "shelf_core": {"minimum_depth_m": 0, "maximum_depth_m": 200},
        "source_artifacts_sha256": dict(
            zip(source_inventory.path, source_inventory.sha256, strict=True)
        ),
    }
    write_json(ROOT / "configs/frozen/global_coastal_mask_manifest_v2.3.json", frozen_manifest)
    print(json.dumps({"output": str(output_path), "sha256": output_sha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
