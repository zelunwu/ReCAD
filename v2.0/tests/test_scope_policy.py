from __future__ import annotations

import copy
from pathlib import Path

import pytest

from recad.governance.scope_policy import load_scope_policy, validate_scope_policy

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/frozen/product_scope_v2.3.json"


@pytest.mark.l1
def test_used_locked_cannot_be_reused_as_blind() -> None:
    policy = load_scope_policy(POLICY)
    changed = copy.deepcopy(policy)
    row = next(
        item for item in changed["label_exposure_ledger"] if item["exposure"] == "used_locked"
    )
    row["reusable_as_blind"] = True

    errors = validate_scope_policy(changed, ROOT)

    assert any("used locked labels cannot be reused as blind" in error for error in errors)


@pytest.mark.l1
def test_global_projection_cannot_create_global_evidence() -> None:
    policy = load_scope_policy(POLICY)
    changed = copy.deepcopy(policy)
    changed["target_region_status"][0]["direct_global_evidence"] = True

    errors = validate_scope_policy(changed, ROOT)

    assert any(
        "global atlas projection cannot create direct global evidence" in error for error in errors
    )


@pytest.mark.l2
def test_frozen_scope_policy_is_internally_consistent() -> None:
    policy = load_scope_policy(POLICY)
    assert validate_scope_policy(policy, ROOT) == []
