from __future__ import annotations

import copy
from pathlib import Path

import pytest

from recad.governance.benchmark_contract import (
    load_benchmark_contract,
    validate_benchmark_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/frozen/benchmark_contract_v2.3.json"


@pytest.mark.l1
def test_random_fold_literature_score_cannot_become_primary_gate() -> None:
    contract = load_benchmark_contract(CONTRACT)
    changed = copy.deepcopy(contract)
    row = next(
        item
        for item in changed["literature_benchmarks"]
        if "random five-fold" in item["validation_design"]
    )
    row["validation_design"] = "random fold"
    row["primary_gate"] = True
    assert "random-fold literature scores cannot be primary gates" in validate_benchmark_contract(
        changed
    )


@pytest.mark.l1
def test_target_tiers_must_tighten() -> None:
    contract = load_benchmark_contract(CONTRACT)
    changed = copy.deepcopy(contract)
    row = next(
        item
        for item in changed["performance_targets"]
        if item["target"] == "SSS" and item["scheme"] == "cruise" and item["tier"] == "ideal"
    )
    row["rmse_ceiling"] = 2.0
    assert any(
        "RMSE ceilings must tighten" in error for error in validate_benchmark_contract(changed)
    )


@pytest.mark.l2
def test_frozen_benchmark_contract_is_valid() -> None:
    contract = load_benchmark_contract(CONTRACT)
    assert validate_benchmark_contract(contract) == []
