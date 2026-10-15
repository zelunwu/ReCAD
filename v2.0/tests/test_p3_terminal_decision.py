from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.l1

ROOT = Path(__file__).resolve().parents[1]


def test_p3_closes_without_relabeling_development_data() -> None:
    decision = json.loads(
        (ROOT / "configs/frozen/p3_no_independent_source_decision_v2.4.json").read_text(
            encoding="utf-8"
        )
    )
    assert decision["decision"] == "terminal_no_eligible_independent_source"
    assert decision["label_access"]["authorized"] is False
    assert decision["label_access"]["count"] == 0
    assert decision["scientific_consequence"]["trusted_release"] is False
    assert decision["scientific_consequence"]["development_candidate_status"] == "diagnostic_only"


def test_p3_reopen_requires_a_new_source() -> None:
    decision = json.loads(
        (ROOT / "configs/frozen/p3_no_independent_source_decision_v2.4.json").read_text(
            encoding="utf-8"
        )
    )
    assert "genuinely new independent observation source" in decision["reopen_rule"]
