"""Download, verify, and sample GEBCO-2025 onto the native ReCAD grid.

The eight 90-degree GeoTIFFs are processed sequentially so that a machine with
limited disk space needs room for only one source tile. Generated GeoTIFF and
NPZ data belong outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BASE_URL = "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2025/ice_surface_elevation/geotiff"
RESOLUTION_DEG = 0.125


@dataclass(frozen=True)
class Tile:
    """One official GEBCO-2025 90-degree GeoTIFF and its target-grid bounds."""

    name: str
    md5: str
    west: float
    east: float
    south: float
    north: float


TILES = (
    Tile(
        "gebco_2025_n0.0_s-90.0_w-180.0_e-90.0",
        "4fc4ce0052dc91463a9ab43521cbc7d8",
        -180,
        -90,
        -78,
        0,
    ),
    Tile("gebco_2025_n0.0_s-90.0_w-90.0_e0.0", "39b88b46c36e999021d6bf38e21a794c", -90, 0, -78, 0),
    Tile("gebco_2025_n0.0_s-90.0_w0.0_e90.0", "cc7820f063dc547c8711c367b9dbb06c", 0, 90, -78, 0),
    Tile(
        "gebco_2025_n0.0_s-90.0_w90.0_e180.0", "da7235029d598c53c509e56e7472239d", 90, 180, -78, 0
    ),
    Tile(
        "gebco_2025_n90.0_s0.0_w-180.0_e-90.0", "5339d30e2a4f4ae766a436d33ab185af", -180, -90, 0, 84
    ),
    Tile("gebco_2025_n90.0_s0.0_w-90.0_e0.0", "add2bd85c08106e47d67a1802ae8355a", -90, 0, 0, 84),
    Tile("gebco_2025_n90.0_s0.0_w0.0_e90.0", "e93523f35c40d65e14e15724960cbe2d", 0, 90, 0, 84),
    Tile("gebco_2025_n90.0_s0.0_w90.0_e180.0", "5ef662c016a17d6eaf03841f840d629c", 90, 180, 0, 84),
)


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    """Hash a file in bounded memory."""

    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tile_coordinates(tile: Tile) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact ReCAD grid nodes assigned to one tile."""

    global_lon = np.arange(-180.0, 180.0, RESOLUTION_DEG, dtype=np.float64)
    global_lat = np.arange(-78.0, 84.0 + RESOLUTION_DEG / 2, RESOLUTION_DEG)
    longitude = global_lon[(global_lon >= tile.west) & (global_lon < tile.east)]
    north_operator = np.less_equal if tile.south >= 0 else np.less
    latitude = global_lat[(global_lat >= tile.south) & north_operator(global_lat, tile.north)]
    return longitude, latitude


def download(url: str, output: Path, *, attempts: int = 5) -> None:
    """Download to a temporary file and atomically replace the destination."""

    partial = output.with_suffix(output.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=16 << 20)
            partial.replace(output)
            return
        except OSError:
            partial.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(min(3 * attempt, 15))


def sample_tile(source: Path, output: Path, tile: Tile) -> None:
    """Nearest-sample one verified GeoTIFF at native ReCAD grid nodes."""

    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - real-data dependency
        raise RuntimeError("install the coastal-mask optional dependencies") from exc

    longitude, latitude = tile_coordinates(tile)
    lon2d, lat2d = np.meshgrid(longitude, latitude)
    points = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    with rasterio.open(source) as dataset:
        values = np.fromiter(
            (sample[0] for sample in dataset.sample(points)),
            dtype=np.int16,
            count=len(points),
        ).reshape(latitude.size, longitude.size)
        metadata = {
            "source_crs": str(dataset.crs),
            "source_transform": tuple(dataset.transform),
            "source_width": dataset.width,
            "source_height": dataset.height,
            "sampling": "nearest source pixel at each ReCAD grid node",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        lon=longitude.astype(np.float32),
        lat=latitude.astype(np.float32),
        elevation_m=values,
        source_sha256=np.array(file_digest(source)),
        metadata=np.array(str(metadata)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--keep-geotiff", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    work_root = args.work_root or args.output_root / "source_geotiff"
    work_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    for tile in TILES:
        source = work_root / f"{tile.name}.tif"
        output = args.output_root / f"{tile.name}.npz"
        if output.is_file() and not args.force:
            print(f"SKIP {output}")
            continue
        download(f"{BASE_URL}/{tile.name}.tif?download=1", source)
        actual_md5 = file_digest(source, "md5")
        if actual_md5 != tile.md5:
            source.unlink(missing_ok=True)
            raise ValueError(f"official MD5 mismatch for {tile.name}: {actual_md5}")
        sample_tile(source, output, tile)
        if not args.keep_geotiff:
            source.unlink()
        print(f"WROTE {output} {file_digest(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
