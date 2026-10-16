from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from recad.evaluate.residual_bank import load_residual_bank_contract

pytestmark = pytest.mark.l3

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "frozen" / "end_to_end_residual_bank_v2.5.in"
ARCHIVE = ROOT / "docs" / "experiment_archive" / "p2_end_to_end_residual_bank_v2.5"
SCRIPT = ROOT / "scripts" / "run_end_to_end_residual_bank.py"

SPEC = importlib.util.spec_from_file_location(
    "verify_residual_bank_archive", ROOT / "scripts" / "verify_experiment_archive.py"
)
assert SPEC and SPEC.loader
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def tracked_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_residual_bank_archive_is_complete_and_fail_closed() -> None:
    result = VERIFIER.verify_archive(ARCHIVE)
    assert result["verified"], result["errors"]
    assert result["figures"] == 8
    assert result["captions"] == 8
    assert result["source_tables"] == 9

    contract = load_residual_bank_contract(CONFIG)
    assert set(contract.required_protocols) == set(contract.available_protocols)
    decision = (ARCHIVE / "tables" / "table08_decision.csv").read_text(encoding="utf-8")
    assert "final_grading_allowed" in decision
    assert "False" in decision
    assert "thresholds_remain_unset" in decision


def test_manifest_uses_portable_tracked_hashes_and_no_independent_labels() -> None:
    manifest = json.loads((ARCHIVE / "archive_manifest.json").read_text(encoding="utf-8"))
    assert manifest["locked_test_opened"] is False
    assert manifest["external_independent_opened"] is False
    assert manifest["data_manifest_sha256"] == tracked_text_sha256(CONFIG)
    assert manifest["archive_builder_sha256"] == tracked_text_sha256(SCRIPT)
    assert manifest["analysis_script_sha256"] == tracked_text_sha256(SCRIPT)
