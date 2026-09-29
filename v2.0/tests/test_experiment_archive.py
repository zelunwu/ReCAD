from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.l3

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_experiment_archive", ROOT / "scripts" / "verify_experiment_archive.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("experiment_id", ["p1_sss_viability_v2.2", "p1_fco2_viability_v2.2"])
def test_p1_reviewer_archive_is_complete_and_hash_verified(experiment_id: str) -> None:
    result = MODULE.verify_archive(ROOT / "docs/experiment_archive" / experiment_id)
    assert result["verified"], result["errors"]
    assert result["figures"] >= 8
    assert result["captions"] == result["figures"]
    assert result["source_tables"] >= 8
