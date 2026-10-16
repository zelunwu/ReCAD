from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from recad.data.ocean_regions import load_ocean_region_contract

pytestmark = pytest.mark.l3

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "frozen" / "ocean_region_provenance_v2.5.in"
ARCHIVE = ROOT / "docs" / "experiment_archive" / "p2_ocean_region_provenance_v2.5"
GEOMETRY_KEY = "C:/backup/phd/data/raw/marine_regions/GOaS_v1_20211214_simplified_0p005deg.geojson"


def tracked_text_sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def test_frozen_regions_are_stable_reporting_provenance_only() -> None:
    contract = load_ocean_region_contract(CONFIG)
    assert [region.region_id for region in contract.regions] == list(range(10))
    assert set(contract.macro_basins) == set(range(8))
    assert contract.release_policy == (
        "provenance and reporting only; never a direct quality or release-status switch"
    )
    assert contract.regions[4].canonical_name == "North Pacific Ocean"
    assert contract.regions[9].canonical_name == "South China and Eastern Archipelagic Seas"


def test_archive_records_canonical_hashes_and_label_free_scope() -> None:
    contract = load_ocean_region_contract(CONFIG)
    manifest = json.loads((ARCHIVE / "archive_manifest.json").read_text(encoding="utf-8"))
    assert manifest["locked_test_opened"] is False
    assert manifest["external_independent_opened"] is False
    assert manifest["data_manifest_sha256"] == tracked_text_sha256(CONFIG)
    script = ROOT / "scripts" / "run_ocean_region_provenance.py"
    assert manifest["archive_builder_sha256"] == tracked_text_sha256(script)
    assert manifest["analysis_script_sha256"] == tracked_text_sha256(script)
    assert manifest["source_artifacts_sha256"][GEOMETRY_KEY] == contract.simplified_sha256
    assert len(manifest["figure_captions"]) == 8
    assert len(manifest["figure_source_data"]) == 8
