from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import mapping

from recad.data.ocean_regions import (
    assign_ocean_regions,
    load_ocean_region_contract,
    load_region_geometries,
)

CONFIG = Path(__file__).parents[2] / "configs" / "frozen" / "ocean_region_provenance_v2.5.in"


def _synthetic_geometry(path: Path) -> None:
    contract = load_ocean_region_contract(CONFIG)
    features = []
    for index, region in enumerate(contract.regions):
        geometry = shapely.box(index * 10.0, 0.0, (index + 1) * 10.0, 10.0)
        features.append(
            {
                "type": "Feature",
                "id": f"test.{index}",
                "geometry": mapping(geometry),
                "properties": {"name": region.source_name},
            }
        )
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))


def test_frozen_contract_has_complete_stable_regions() -> None:
    contract = load_ocean_region_contract(CONFIG)
    assert contract.contract_id == "ocean_region_provenance_v2.5"
    assert len(contract.regions) == 10
    assert [region.region_id for region in contract.regions] == list(range(10))
    assert contract.regions[4].canonical_name == "North Pacific Ocean"
    assert contract.regions[9].canonical_name == "South China and Eastern Archipelagic Seas"
    assert "never a direct quality or release-status switch" in contract.release_policy


def test_polygon_assignment_exposes_boundaries_and_unresolved_points(tmp_path: Path) -> None:
    source = tmp_path / "regions.geojson"
    _synthetic_geometry(source)
    contract = load_ocean_region_contract(CONFIG)
    geometries = load_region_geometries(source, contract, verify_hash=False)
    result = assign_ocean_regions(
        np.array([5.0, 15.0, 10.0, -5.0, 365.0]),
        np.array([5.0, 5.0, 5.0, 5.0, 5.0]),
        geometries,
        contract,
    )
    np.testing.assert_array_equal(result.region_id, np.array([0, 1, 0, -1, 0], dtype=np.int8))
    assert result.shared_boundary.tolist() == [False, False, True, False, False]
    assert result.unresolved.tolist() == [False, False, False, True, False]


def test_geometry_hash_and_source_name_drift_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "regions.geojson"
    _synthetic_geometry(source)
    contract = load_ocean_region_contract(CONFIG)
    with pytest.raises(ValueError, match="SHA256"):
        load_region_geometries(source, contract)

    payload = json.loads(source.read_text())
    payload["features"][0]["properties"]["name"] = "Renamed Ocean"
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source names drifted"):
        load_region_geometries(source, contract, verify_hash=False)
