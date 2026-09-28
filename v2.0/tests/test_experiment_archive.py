from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_experiment_archive", ROOT / "scripts" / "verify_experiment_archive.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_p1_sss_reviewer_archive_is_complete_and_hash_verified() -> None:
    result = MODULE.verify_archive(ROOT / "docs/experiment_archive/p1_sss_viability_v2.2")
    assert result["verified"], result["errors"]
    assert result["figures"] >= 8
    assert result["source_tables"] >= 8
