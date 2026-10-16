"""Build the Issue 81 ocean-region provenance layer and reviewer archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

from recad.data.ocean_regions import (
    OceanRegionAssignment,
    assign_ocean_regions,
    load_ocean_region_contract,
    load_region_geometries,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "frozen" / "ocean_region_provenance_v2.5.in"
DEFAULT_PRODUCT = Path(
    r"C:\backup\phd\data\processed\recad_v2_4\recad_v2_release_v2.4\products"
    r"\ReCAD_v2.4_fCO2_1993_2025.nc"
)
DEFAULT_OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_5\p2_ocean_region_provenance_v2.5")
ARCHIVE = ROOT / "docs" / "experiment_archive" / "p2_ocean_region_provenance_v2.5"

REGION_COLORS = [
    "#6baed6",
    "#756bb1",
    "#2171b5",
    "#9ecae1",
    "#cb181d",
    "#fb6a4a",
    "#31a354",
    "#fdae6b",
    "#9ecae1",
    "#dd1c77",
]
LEGACY_NAMES = {
    0: "Arctic",
    1: "Southern",
    2: "Atlantic",
    3: "Pacific",
    4: "Indian",
    5: "Mediterranean",
}
FIXTURES = [
    ("Japan east coast", 140.0, 35.0, 4),
    ("Japan west coast", 135.0, 37.0, 4),
    ("Western North America", -125.0, 40.0, 4),
    ("Central America Pacific", -90.0, 10.0, 4),
    ("Central America Caribbean", -80.0, 15.0, 2),
    ("South China Sea", 115.0, 10.0, 9),
    ("Mediterranean", 15.0, 40.0, 7),
    ("Baltic", 20.0, 58.0, 8),
    ("Indian", 72.0, 19.0, 6),
    ("Arctic", 0.0, 80.0, 0),
    ("Southern", 0.0, -65.0, 1),
]


def write_utf8_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def sha256_tracked_text(path: Path) -> str:
    """Hash the LF form stored by Git, independent of checkout line endings."""
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def write_table(frame: pd.DataFrame, path: Path) -> None:
    write_utf8_lf(path, frame.to_csv(index=False, lineterminator="\n"))


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def load_nodes(product: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with netCDF4.Dataset(product) as dataset:
        return (
            np.asarray(dataset.variables["longitude"][:], dtype=np.float64),
            np.asarray(dataset.variables["latitude"][:], dtype=np.float64),
            np.asarray(dataset.variables["basin_id"][:], dtype=np.int8),
        )


def write_product(
    path: Path,
    longitude: np.ndarray,
    latitude: np.ndarray,
    legacy: np.ndarray,
    assignment: OceanRegionAssignment,
    config_hash: str,
    geometry_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # NetCDF-3 avoids HDF5 object timestamps, making the external evidence
    # artifact byte-reproducible for identical inputs.
    with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as dataset:
        dataset.createDimension("node", len(longitude))
        for name, values, dtype, long_name in (
            ("longitude", longitude, "f8", "longitude in degrees east"),
            ("latitude", latitude, "f8", "latitude in degrees north"),
            ("legacy_coarse_basin_id", legacy, "i1", "legacy v2.4 longitude-rule basin"),
            ("ocean_region_id", assignment.region_id, "i1", "Global Oceans and Seas region"),
            ("macro_basin_id", assignment.macro_basin_id, "i1", "reporting-only macro basin"),
            (
                "shared_boundary",
                assignment.shared_boundary.astype("i1"),
                "i1",
                "point covers multiple shared-boundary polygons",
            ),
            (
                "unresolved",
                assignment.unresolved.astype("i1"),
                "i1",
                "point not covered by the source geometry",
            ),
        ):
            variable = dataset.createVariable(name, dtype, ("node",))
            variable[:] = values
            variable.long_name = long_name
        dataset.contract_id = "ocean_region_provenance_v2.5"
        dataset.source = "Flanders Marine Institute Global Oceans and Seas v1"
        dataset.source_doi = "https://doi.org/10.14284/542"
        dataset.config_sha256 = config_hash
        dataset.geometry_sha256 = geometry_hash
        dataset.release_policy = "provenance and reporting only; never a direct quality switch"


def node_frame(
    longitude: np.ndarray,
    latitude: np.ndarray,
    legacy: np.ndarray,
    assignment: OceanRegionAssignment,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "longitude": (longitude + 180.0) % 360.0 - 180.0,
            "latitude": latitude,
            "legacy_basin_id": legacy,
            "ocean_region_id": assignment.region_id,
            "macro_basin_id": assignment.macro_basin_id,
            "shared_boundary": assignment.shared_boundary,
            "unresolved": assignment.unresolved,
        }
    )


def build_tables(frame: pd.DataFrame, contract, geometries) -> dict[str, pd.DataFrame]:
    names = {region.region_id: region.canonical_name for region in contract.regions}
    region_counts = (
        frame.groupby("ocean_region_id", dropna=False).size().rename("nodes").reset_index()
    )
    region_counts["region"] = region_counts.ocean_region_id.map(names).fillna("Unresolved")
    region_counts["fraction"] = region_counts.nodes / len(frame)

    confusion = (
        frame.groupby(["legacy_basin_id", "macro_basin_id"], dropna=False)
        .size()
        .rename("nodes")
        .reset_index()
    )
    confusion["legacy_basin"] = confusion.legacy_basin_id.map(LEGACY_NAMES)
    confusion["new_macro_basin"] = confusion.macro_basin_id.map(contract.macro_basins).fillna(
        "Unresolved"
    )

    lon = np.asarray([fixture[1] for fixture in FIXTURES])
    lat = np.asarray([fixture[2] for fixture in FIXTURES])
    fixture_assignment = assign_ocean_regions(lon, lat, geometries, contract)
    fixtures = pd.DataFrame(
        {
            "fixture": [fixture[0] for fixture in FIXTURES],
            "longitude": lon,
            "latitude": lat,
            "expected_region_id": [fixture[3] for fixture in FIXTURES],
            "assigned_region_id": fixture_assignment.region_id,
        }
    )
    fixtures["assigned_region"] = fixtures.assigned_region_id.map(names)
    fixtures["passed"] = fixtures.expected_region_id.eq(fixtures.assigned_region_id)
    if not fixtures.passed.all():
        raise RuntimeError("one or more frozen geographic fixtures failed")

    block = frame.assign(
        block_lat=np.floor((frame.latitude + 90.0) / 5.0).astype(int),
        block_lon=np.floor((frame.longitude + 180.0) / 5.0).astype(int),
    )
    unresolved = (
        block.loc[block.unresolved]
        .groupby(["block_lat", "block_lon"])
        .size()
        .rename("nodes")
        .reset_index()
        .sort_values("nodes", ascending=False)
    )
    boundaries = (
        block.loc[block.shared_boundary]
        .groupby(["block_lat", "block_lon"])
        .size()
        .rename("nodes")
        .reset_index()
        .sort_values("nodes", ascending=False)
    )
    macro = frame.groupby("macro_basin_id").size().rename("nodes").reset_index()
    macro["macro_basin"] = macro.macro_basin_id.map(contract.macro_basins).fillna("Unresolved")
    macro["fraction"] = macro.nodes / len(frame)

    changed = frame.assign(classification_changed=frame.macro_basin_id.ne(frame.legacy_basin_id))
    changed = (
        changed.groupby("ocean_region_id")
        .agg(nodes=("ocean_region_id", "size"), changed=("classification_changed", "sum"))
        .reset_index()
    )
    changed["region"] = changed.ocean_region_id.map(names).fillna("Unresolved")
    changed["changed_fraction"] = changed.changed / changed.nodes

    source = pd.DataFrame(
        [
            ("source_title", contract.source_title),
            ("source_doi", contract.source_doi),
            ("source_sha256", contract.raw_sha256),
            ("simplified_sha256", contract.simplified_sha256),
            ("simplification_tolerance_degrees", contract.simplification_tolerance_degrees),
            ("assignment_policy", contract.assignment_policy),
            ("release_policy", contract.release_policy),
        ],
        columns=["field", "value"],
    )
    return {
        "table01_region_node_counts.csv": region_counts,
        "table02_legacy_new_confusion.csv": confusion,
        "table03_boundary_fixtures.csv": fixtures,
        "table04_unresolved_blocks.csv": unresolved,
        "table05_shared_boundary_blocks.csv": boundaries,
        "table06_macro_basin_counts.csv": macro,
        "table07_changed_fraction_by_region.csv": changed,
        "table08_source_and_policy.csv": source,
    }


def grid_values(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latitudes = np.sort(frame.latitude.unique())
    longitudes = np.sort(frame.longitude.unique())
    grid = np.full((len(latitudes), len(longitudes)), np.nan)
    lat_index = np.searchsorted(latitudes, frame.latitude.to_numpy())
    lon_index = np.searchsorted(longitudes, frame.longitude.to_numpy())
    values = frame[column].to_numpy(float)
    values[values < 0] = np.nan
    grid[lat_index, lon_index] = values
    return longitudes, latitudes, grid


def base_map(axis, extent=None) -> None:
    axis.add_feature(cfeature.LAND, facecolor="#e5e5e5", zorder=2)
    axis.coastlines(linewidth=0.35, zorder=3)
    grid = axis.gridlines(draw_labels=True, linewidth=0.25, color="gray", alpha=0.6)
    grid.top_labels = False
    grid.right_labels = False
    if extent:
        axis.set_extent(extent, crs=ccrs.PlateCarree())


def save_map(frame: pd.DataFrame, column: str, title: str, path: Path, colors, labels) -> None:
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), len(colors))
    figure, axis = plt.subplots(figsize=(13.5, 6.8), subplot_kw={"projection": ccrs.Robinson()})
    axis.set_global()
    selected = frame.loc[frame[column].ge(0)]
    image = axis.scatter(
        selected.longitude,
        selected.latitude,
        c=selected[column],
        s=0.12,
        linewidths=0,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        rasterized=True,
    )
    base_map(axis)
    axis.set_title(title, weight="bold")
    colorbar = figure.colorbar(image, ax=axis, orientation="horizontal", pad=0.07, shrink=0.78)
    colorbar.set_ticks(range(len(labels)), labels=labels, rotation=18, ha="right")
    colorbar.ax.tick_params(labelsize=8)
    # Cartopy can collapse the projected GeoAxes when ``bbox_inches='tight'``
    # is combined with a horizontal colorbar.  Keep the fixed canvas so the
    # scientific map, rather than only its colorbar, is archived.
    figure.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.18)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_figures(frame: pd.DataFrame, tables: dict[str, pd.DataFrame], contract) -> None:
    figures = ARCHIVE / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    region_labels = [region.canonical_name for region in contract.regions]
    save_map(
        frame,
        "ocean_region_id",
        "Global Oceans and Seas v1 provenance on ReCAD coastal nodes",
        figures / "fig01_corrected_region_map.png",
        REGION_COLORS,
        region_labels,
    )
    save_map(
        frame,
        "legacy_basin_id",
        "Legacy v2.4 longitude-rule basin labels",
        figures / "fig02_legacy_basin_map.png",
        ["#6baed6", "#756bb1", "#2171b5", "#cb181d", "#31a354", "#fdae6b"],
        list(LEGACY_NAMES.values()),
    )
    changed = frame.assign(changed=frame.macro_basin_id.ne(frame.legacy_basin_id).astype(int))
    save_map(
        changed,
        "changed",
        "Nodes whose provenance changes under polygon assignment",
        figures / "fig03_changed_nodes_map.png",
        ["#d9d9d9", "#7a0177"],
        ["unchanged", "changed"],
    )

    japan = frame.loc[frame.longitude.between(125, 150) & frame.latitude.between(25, 50)].copy()
    longitude, latitude, grid = grid_values(japan, "ocean_region_id")
    figure, axis = plt.subplots(figsize=(8.5, 7), subplot_kw={"projection": ccrs.PlateCarree()})
    axis.pcolormesh(
        longitude,
        latitude,
        grid,
        cmap=ListedColormap(REGION_COLORS),
        norm=BoundaryNorm(np.arange(-0.5, 10.5), 10),
        transform=ccrs.PlateCarree(),
    )
    base_map(axis, [125, 150, 25, 50])
    axis.set_title("Japan-area nodes are North Pacific under GOaS v1", weight="bold")
    figure.savefig(figures / "fig04_japan_corrected_map.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    confusion = tables["table02_legacy_new_confusion.csv"].pivot_table(
        index="new_macro_basin", columns="legacy_basin", values="nodes", fill_value=0
    )
    figure, axis = plt.subplots(figsize=(9, 6.8))
    image = axis.imshow(np.log10(confusion.to_numpy() + 1), cmap="Blues", aspect="auto")
    axis.set_xticks(range(len(confusion.columns)), confusion.columns, rotation=30, ha="right")
    axis.set_yticks(range(len(confusion.index)), confusion.index)
    for row in range(len(confusion.index)):
        for column in range(len(confusion.columns)):
            value = int(confusion.iloc[row, column])
            if value:
                axis.text(column, row, f"{value:,}", ha="center", va="center", fontsize=7)
    axis.set_title("Legacy versus polygon-based provenance (log color, raw labels)", weight="bold")
    figure.colorbar(image, ax=axis, label="log10(nodes + 1)")
    figure.savefig(figures / "fig05_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    counts = tables["table01_region_node_counts.csv"].query("ocean_region_id >= 0")
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(counts.region, counts.nodes, color=REGION_COLORS)
    axis.invert_yaxis()
    axis.set_xlabel("Coastal nodes")
    axis.set_title("Product-domain nodes by GOaS v1 region", weight="bold")
    figure.savefig(figures / "fig06_region_node_counts.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    for column, title, filename, color in (
        (
            "unresolved",
            "Nodes outside the GOaS source coverage",
            "fig07_unresolved_nodes.png",
            "#d7301f",
        ),
        (
            "shared_boundary",
            "Nodes exactly on a shared GOaS boundary",
            "fig08_shared_boundary_nodes.png",
            "#54278f",
        ),
    ):
        selected = frame.loc[frame[column]]
        figure, axis = plt.subplots(figsize=(13.5, 6.8), subplot_kw={"projection": ccrs.Robinson()})
        axis.scatter(
            selected.longitude,
            selected.latitude,
            s=2,
            c=color,
            transform=ccrs.PlateCarree(),
            rasterized=True,
        )
        base_map(axis)
        axis.set_global()
        axis.set_title(title, weight="bold")
        figure.savefig(figures / filename, dpi=180, bbox_inches="tight")
        plt.close(figure)


def archive_text(
    frame: pd.DataFrame, tables: dict[str, pd.DataFrame], contract
) -> tuple[str, dict[str, str]]:
    unresolved = int(frame.unresolved.sum())
    boundaries = int(frame.shared_boundary.sum())
    changed = int(frame.macro_basin_id.ne(frame.legacy_basin_id).sum())
    japan = frame.loc[
        frame.longitude.between(125, 150)
        & frame.latitude.between(25, 50)
        & frame.ocean_region_id.eq(4)
    ]
    japan_legacy_indian = int(japan.legacy_basin_id.eq(4).sum())
    captions = {
        "fig01_corrected_region_map.png": "Figure 1. Polygon-based Global Oceans and Seas v1 provenance for every ReCAD v2.4 coastal node. The ten source regions are mutually auditable reporting strata and do not directly determine reconstruction quality or release status.",
        "fig02_legacy_basin_map.png": "Figure 2. Legacy v2.4 basin labels produced by latitude and longitude thresholds. The 145 degree east boundary cuts through the western North Pacific and assigns part of Japan and adjacent seas to the Indian label.",
        "fig03_changed_nodes_map.png": "Figure 3. Nodes whose reporting provenance changes when the coarse threshold rule is replaced by GOaS polygons. Purple denotes a changed macro label or a newly explicit Baltic or archipelagic region; gray denotes agreement.",
        "fig04_japan_corrected_map.png": "Figure 4. Japan-area product nodes assigned from the GOaS North Pacific polygon. Both the Pacific-facing and Sea of Japan margins receive North Pacific provenance, removing the legacy Indian-label bypass.",
        "fig05_confusion_matrix.png": "Figure 5. Node-count cross-tabulation between legacy coarse basins and new reporting macro basins. Cell labels are raw counts and color is logarithmic, exposing substantial off-diagonal assignments hidden by the old longitude boundaries.",
        "fig06_region_node_counts.png": "Figure 6. Number of ReCAD coastal nodes in each GOaS source region. These counts describe geographic product support only and are not weights, quality grades, or target-observation counts.",
        "fig07_unresolved_nodes.png": "Figure 7. Product nodes not covered by the frozen GOaS polygon geometry. They remain explicitly unresolved rather than receiving a fallback basin from longitude, and must be investigated before provenance is written into a release product.",
        "fig08_shared_boundary_nodes.png": "Figure 8. Product nodes covered by more than one GOaS polygon because they lie exactly on a shared boundary. The stable lower region ID is retained for serialization while the boundary flag preserves the ambiguity for audits.",
    }
    blocks = []
    for filename, caption in captions.items():
        blocks.append(f"![{caption.split('. ', 1)[1]}](figures/{filename})\n\n{caption}")
    report = f"""# Ocean-region provenance repair

## Scientific question and permitted claim

Can the coarse longitude-rule basin field be replaced by a versioned polygon source so reporting provenance is geographically auditable and cannot create a release-status bypass? The permitted claim is limited to geographic provenance. Region identity does not establish reconstruction accuracy.

## Data, splits, and leakage controls

The analysis uses the immutable spatial nodes from the ReCAD v2.4 fCO2 release and no target labels. Region polygons come from {contract.source_title}, DOI {contract.source_doi}. The raw WFS and topology-preserving 0.005-degree simplification are pinned by SHA256. This label-free operation cannot tune models, uncertainty thresholds, or the Issue 5 audit.

## Candidate models and training

No statistical model is trained. The legacy candidate is the v2.4 latitude/longitude threshold. The replacement uses polygon coverage, records exact shared-boundary hits, and leaves uncovered nodes unresolved. No nearest-region or basin-name release fallback is permitted.

## Main development results

The product contains {len(frame):,} coastal nodes. Polygon assignment leaves {unresolved:,} unresolved ({unresolved / len(frame):.3%}) and marks {boundaries:,} shared-boundary nodes ({boundaries / len(frame):.3%}). The new macro provenance differs from the legacy label at {changed:,} nodes ({changed / len(frame):.1%}). Within 125-150 E and 25-50 N, {japan_legacy_indian:,} nodes assigned to the GOaS North Pacific region carried the legacy Indian basin label.

{chr(10).join([block + chr(10) for block in blocks])}

## Decision and limitations

Decision: `adopt_goas_v1_for_provenance_only; prohibit_basin_release_switches`. The Japan bypass is repaired by the source polygon rather than a hand-written exception. Baltic and South China/Eastern Archipelagic Seas remain explicit regions instead of being forced into a neighboring ocean. Unresolved and shared-boundary nodes are preserved as auditable states. This phase does not replace the v2.4 release, estimate physical error, or assign A-D/U evidence.

The source is a static hydrographic partition, not a dynamic water-mass classification. It is suitable for provenance and stratified reporting but cannot define model applicability. The {unresolved:,} uncovered nodes require a later source/coastline reconciliation; they cannot silently inherit a basin.

## Figure and table index

- Figures 1-8 correspond to source tables 1-8 in order.
- Table 1: node counts by GOaS region.
- Table 2: legacy/new confusion counts.
- Table 3: frozen geographic regression fixtures.
- Tables 4-5: unresolved and shared-boundary 5-degree blocks.
- Table 6: macro-basin counts.
- Table 7: changed fraction by source region.
- Table 8: source, hashes, and release policy.
"""
    return report, captions


def build_archive(
    frame: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    contract,
    config: Path,
    geometry: Path,
    product: Path,
    output_product: Path,
) -> None:
    table_dir = ARCHIVE / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in tables.items():
        write_table(table, table_dir / filename)
    build_figures(frame, tables, contract)
    report, captions = archive_text(frame, tables, contract)
    write_utf8_lf(ARCHIVE / "REPORT.md", report)
    caption_text = "# Figure captions\n\n" + "\n\n".join(
        f"## {name}\n\n{caption}" for name, caption in captions.items()
    )
    write_utf8_lf(ARCHIVE / "CAPTIONS.md", caption_text + "\n")
    write_utf8_lf(
        ARCHIVE / "README.md",
        "# Issue 81 ocean-region provenance repair\n\nStart with [REPORT.md](REPORT.md). Large input geometry and node products remain outside Git.\n",
    )
    tracked = [ARCHIVE / "README.md", ARCHIVE / "REPORT.md", ARCHIVE / "CAPTIONS.md"]
    tracked += sorted((ARCHIVE / "figures").glob("*.png"))
    tracked += sorted(table_dir.glob("*.csv"))
    figure_sources = {
        "fig01_corrected_region_map.png": ["table01_region_node_counts.csv"],
        "fig02_legacy_basin_map.png": ["table02_legacy_new_confusion.csv"],
        "fig03_changed_nodes_map.png": ["table07_changed_fraction_by_region.csv"],
        "fig04_japan_corrected_map.png": ["table03_boundary_fixtures.csv"],
        "fig05_confusion_matrix.png": ["table02_legacy_new_confusion.csv"],
        "fig06_region_node_counts.png": ["table01_region_node_counts.csv"],
        "fig07_unresolved_nodes.png": ["table04_unresolved_blocks.csv"],
        "fig08_shared_boundary_nodes.png": ["table05_shared_boundary_blocks.csv"],
    }
    manifest = {
        "experiment_id": "p2_ocean_region_provenance_v2.5",
        "training_git_commit": git_commit(),
        "data_manifest_sha256": sha256_tracked_text(config),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "label-free geographic provenance over the complete v2.4 product nodes",
        "decision": "adopt_goas_v1_for_provenance_only_and_prohibit_basin_release_switches",
        "figure_source_data": figure_sources,
        "figure_captions": captions,
        "files_sha256": {
            str(path.relative_to(ARCHIVE)).replace("\\", "/"): sha256(path) for path in tracked
        },
        "archive_builder_sha256": sha256_tracked_text(Path(__file__)),
        "analysis_script_sha256": sha256_tracked_text(Path(__file__)),
        "source_artifacts_sha256": {
            str(config.relative_to(ROOT)).replace("\\", "/"): sha256_tracked_text(config),
            str(geometry): sha256(geometry),
            str(product): sha256(product),
            str(output_product): sha256(output_product),
        },
    }
    write_utf8_lf(ARCHIVE / "archive_manifest.json", json.dumps(manifest, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--product", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_ocean_region_contract(args.config)
    geometry = args.geometry or contract.default_external_geometry
    geometries = load_region_geometries(geometry, contract)
    longitude, latitude, legacy = load_nodes(args.product)
    assignment = assign_ocean_regions(longitude, latitude, geometries, contract)
    args.output.mkdir(parents=True, exist_ok=True)
    output_product = args.output / "ocean_region_provenance_v2.5.nc"
    write_product(
        output_product,
        longitude,
        latitude,
        legacy,
        assignment,
        sha256_tracked_text(args.config),
        sha256(geometry),
    )
    frame = node_frame(longitude, latitude, legacy, assignment)
    tables = build_tables(frame, contract, geometries)
    for filename, table in tables.items():
        write_table(table, args.output / "tables" / filename)
    build_archive(frame, tables, contract, args.config, geometry, args.product, output_product)
    summary = {
        "nodes": len(frame),
        "unresolved": int(frame.unresolved.sum()),
        "shared_boundary": int(frame.shared_boundary.sum()),
        "changed": int(frame.macro_basin_id.ne(frame.legacy_basin_id).sum()),
        "product": str(output_product),
        "product_sha256": sha256(output_product),
    }
    write_utf8_lf(args.output / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
