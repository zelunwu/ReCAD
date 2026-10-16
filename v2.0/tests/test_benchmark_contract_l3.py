from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from recad.evaluate.benchmark_contract import load_benchmark_contract, primary_cell_budget

pytestmark = pytest.mark.l3

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/frozen/model_family_benchmark_v2.5.in"
ARCHIVE = ROOT / "docs/experiment_archive/p2_model_family_benchmark_v2.5"

SPEC = importlib.util.spec_from_file_location(
    "verify_model_family_benchmark_archive", ROOT / "scripts/verify_experiment_archive.py"
)
assert SPEC and SPEC.loader
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def tracked_text_sha256(path: Path) -> str:
    canonical = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_benchmark_preregistration_is_complete_and_hash_verified() -> None:
    contract = load_benchmark_contract(CONFIG)
    budget = primary_cell_budget(contract)
    assert budget.iloc[-1].primary_cells == 68
    assert len(contract.families) == 14

    result = VERIFIER.verify_archive(ARCHIVE)
    assert result["verified"], result["errors"]
    assert result["figures"] == 8
    assert result["captions"] == 8
    assert result["source_tables"] == 8

    manifest = json.loads((ARCHIVE / "archive_manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_manifest_sha256"] == tracked_text_sha256(CONFIG)
    assert manifest["decision"]["models_ranked"] is False
    assert manifest["decision"]["primary_cell_upper_bound"] == 68
    assert manifest["archive_builder_sha256"] == tracked_text_sha256(
        ROOT / "scripts/build_model_family_benchmark_contract_archive.py"
    )
    assert manifest["analysis_script_sha256"] == tracked_text_sha256(
        ROOT / "scripts/validate_model_family_benchmark.py"
    )
