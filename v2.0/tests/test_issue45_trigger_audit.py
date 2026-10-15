from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.l1

ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "configs/frozen/p2_na_carbonate_trigger_audit_v2.4.json"


def test_issue45_stays_untriggered_without_qualified_inputs() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    assert audit["decision"] == "untriggered_non_blocking"
    assert audit["entry_condition_met"] is False
    assert not any(audit["actions"].values())


def test_issue45_evidence_hashes_are_frozen() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    for evidence in audit["evidence"]:
        path = ROOT / evidence["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == evidence["sha256"]
