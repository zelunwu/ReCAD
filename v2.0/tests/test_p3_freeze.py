from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.l1

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "configs/frozen/p3_lock_v2.4.json"
SPEC = importlib.util.spec_from_file_location(
    "prepare_p3_independent_freeze", ROOT / "scripts/prepare_p3_independent_freeze.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_socat_version_inventory_uses_names_only(tmp_path: Path) -> None:
    (tmp_path / "SOCATv2026_Coastal.tsv.zip").write_bytes(b"labels must not be read")
    (tmp_path / "SOCATv2027_Coastal.tsv.zip").write_bytes(b"labels must not be read")
    assert MODULE._socat_versions_from_names(tmp_path) == [2026, 2027]


def test_normalized_text_hash_is_newline_portable(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"
    path.write_bytes(b"one\r\ntwo\r\n")
    windows_hash = MODULE.sha256_normalized_text(path)
    path.write_bytes(b"one\ntwo\n")
    assert MODULE.sha256_normalized_text(path) == windows_hash


def test_frozen_protocol_excludes_spent_and_optional_targets() -> None:
    protocol = json.loads(
        (ROOT / "configs/frozen/p3_independent_evaluation_protocol_v2.4.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["targets"] == ["SSS", "fCO2"]
    assert set(protocol["excluded_targets"]) == {"TA", "DIC"}
    assert "Issue 25 SSS locked set" in protocol["ineligible_evidence"]
    assert protocol["one_time_policy"]["zero_alias_overlap_required"] is True


def test_release_contract_records_completed_conditional_scope() -> None:
    contract = json.loads(
        (ROOT / "configs/frozen/p3_release_contract_v2.4.json").read_text(encoding="utf-8")
    )
    assert contract["status"] == "release_complete_flagged_scope"
    assert "Issue 5" in contract["release_gate"]
    assert contract["qualified_scope"]["SSS"].startswith("grades A/B")
    assert contract["cross_dataset_claim"] is False


def test_lock_blocks_access_without_exact_independent_set() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["status"] == "prepared_waiting_for_eligible_independent_set"
    assert lock["label_access_authorized"] is False
    assert lock["label_access_count"] == 0
    assert lock["metadata_only_eligibility"]["available_versions_from_filenames"] == [2026]
    assert lock["metadata_only_eligibility"]["prospective_versions"] == []
    assert lock["metadata_only_eligibility"]["target_labels_read"] is False


def test_lock_repository_hashes_are_portable() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    for relative, expected in lock["repository_artifact_sha256"].items():
        assert MODULE.sha256_normalized_text(ROOT / relative) == expected
