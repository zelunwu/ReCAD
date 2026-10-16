from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from recad.evaluate.error_tier_contract import (
    ErrorReason,
    EvidenceStatus,
    load_error_flag_contract,
)

pytestmark = pytest.mark.l3

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "frozen" / "absolute_error_flag_contract_v2.5.in"
ARCHIVE = ROOT / "docs" / "experiment_archive" / "p2_absolute_error_flag_contract_v2.5"

SPEC = importlib.util.spec_from_file_location(
    "verify_error_tier_contract_archive", ROOT / "scripts" / "verify_experiment_archive.py"
)
assert SPEC and SPEC.loader
ARCHIVE_VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARCHIVE_VERIFIER)


def test_repository_contract_has_complete_unambiguous_release_schema() -> None:
    contract = load_error_flag_contract(CONFIG)
    assert set(contract.targets) == {"SSS", "fCO2"}
    assert contract.status == "methodology_frozen_thresholds_pending_scientific_derivation"
    assert {status.value for status in EvidenceStatus} == {"VD", "VT", "VU", "VF", "VI", "NA"}
    assert contract.reason_bits == {reason.name: int(reason) for reason in ErrorReason}
    assert int(ErrorReason.NUMERICAL_EXECUTION_FAILURE) == 2048
    assert "INTERVAL_UNDERCOVERAGE" not in contract.reason_bits
    assert len(contract.reason_bits) == 15


def test_preregistered_contract_archive_is_hash_verified() -> None:
    result = ARCHIVE_VERIFIER.verify_archive(ARCHIVE)
    assert result["verified"], result["errors"]
    assert result["figures"] == 1
    assert result["captions"] == 1
    assert result["source_tables"] == 1

    manifest = json.loads((ARCHIVE / "archive_manifest.json").read_text(encoding="utf-8"))

    def tracked_text_sha256(path: Path) -> str:
        canonical = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    assert manifest["data_manifest_sha256"] == tracked_text_sha256(CONFIG)
    assert manifest["archive_builder_sha256"] == tracked_text_sha256(
        ROOT / "scripts" / "build_absolute_error_flag_contract_archive.py"
    )
    assert manifest["analysis_script_sha256"] == tracked_text_sha256(
        ROOT / "scripts" / "validate_error_flag_contract.py"
    )
