from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import shapely
from shapely.geometry import mapping, shape

pytestmark = pytest.mark.l2

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_ocean_region_geometry", ROOT / "scripts" / "prepare_ocean_region_geometry.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_topology_simplification_preserves_shared_boundary_coverage(tmp_path: Path) -> None:
    raw = tmp_path / "raw.geojson"
    output = tmp_path / "simplified.geojson"
    source_geometries = [shapely.box(index, 0, index + 1, 1) for index in range(10)]
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": f"test.{index}",
                "geometry": mapping(geometry),
                "properties": {"name": str(index)},
            }
            for index, geometry in enumerate(source_geometries)
        ],
    }
    raw.write_text(json.dumps(payload), encoding="utf-8")

    MODULE.simplify_geometry(raw, output, tolerance=0.25)

    result = json.loads(output.read_text(encoding="utf-8"))
    geometries = [shape(feature["geometry"]) for feature in result["features"]]
    boundary = shapely.Point(1.0, 0.5)
    assert geometries[0].covers(boundary)
    assert geometries[1].covers(boundary)
    assert all(geometry.is_valid for geometry in geometries)
