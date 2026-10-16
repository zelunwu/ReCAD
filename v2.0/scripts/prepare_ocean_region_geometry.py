"""Download and topology-simplify the frozen Marine Regions GOaS geometry."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

from shapely import coverage_simplify
from shapely.geometry import mapping, shape

from recad.data.ocean_regions import load_ocean_region_contract, sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "frozen" / "ocean_region_provenance_v2.5.in"


def write_utf8_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def download(url: str, destination: Path) -> None:
    """Stream a public WFS response to an external data directory."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
    temporary.replace(destination)


def simplify_geometry(raw: Path, output: Path, tolerance: float) -> None:
    """Preserve shared boundaries while reducing coastline vertex density."""

    payload = json.loads(raw.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    geometries = [shape(feature["geometry"]) for feature in features]
    if len(geometries) != 10 or any(
        geometry.is_empty or not geometry.is_valid for geometry in geometries
    ):
        raise ValueError("expected ten valid Global Oceans and Seas polygons")
    simplified = coverage_simplify(geometries, tolerance=tolerance)
    output_features = []
    for feature, geometry in zip(features, simplified, strict=True):
        output_features.append(
            {
                "type": "Feature",
                "id": feature.get("id"),
                "geometry": mapping(geometry),
                "properties": feature["properties"],
            }
        )
    compact = json.dumps(
        {"type": "FeatureCollection", "features": output_features},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    write_utf8_lf(output, compact + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--reuse-raw", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_ocean_region_contract(args.config)
    if not args.reuse_raw or not args.raw.is_file():
        download(contract.wfs_url, args.raw)
    raw_hash = sha256(args.raw)
    if raw_hash != contract.raw_sha256:
        raise ValueError(f"raw WFS SHA256 drifted: {raw_hash}")
    simplify_geometry(args.raw, args.output, contract.simplification_tolerance_degrees)
    simplified_hash = sha256(args.output)
    if simplified_hash != contract.simplified_sha256:
        raise ValueError(f"simplified geometry SHA256 drifted: {simplified_hash}")
    manifest = {
        "contract_id": contract.contract_id,
        "source_title": contract.source_title,
        "source_doi": contract.source_doi,
        "wfs_url": contract.wfs_url,
        "raw_path": str(args.raw.resolve()),
        "raw_sha256": raw_hash,
        "simplified_path": str(args.output.resolve()),
        "simplified_sha256": simplified_hash,
        "simplification_tolerance_degrees": contract.simplification_tolerance_degrees,
    }
    if args.manifest:
        write_utf8_lf(args.manifest, json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
