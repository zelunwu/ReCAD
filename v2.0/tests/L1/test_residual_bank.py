from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from recad.evaluate.residual_bank import (
    empirical_quantile,
    hierarchical_group_quantile_bound,
    load_residual_bank_contract,
)

pytestmark = pytest.mark.l1

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs" / "frozen" / "end_to_end_residual_bank_v2.5.in"


def test_frozen_contract_preserves_separate_transfer_banks() -> None:
    contract = load_residual_bank_contract(CONFIG)
    assert contract.targets == ("SSS", "fCO2")
    assert contract.available_protocols == (
        "cruise",
        "geographic",
        "whole_region",
        "forward",
        "early_era",
    )
    assert contract.required_protocols[-1] == "early_era"
    assert contract.early_era_partition_root.name == "partitions"
    assert contract.pool_protocols is False
    assert contract.basin_may_change_error_tier is False
    assert contract.thresholds_may_be_derived_from_this_bank is False


def test_higher_empirical_quantile_is_conservative() -> None:
    assert empirical_quantile(np.arange(10.0), 0.9) == 9.0


def test_hierarchical_bound_uses_complete_groups_as_evidence_units() -> None:
    errors = np.repeat(np.arange(1.0, 31.0), 3)
    groups = np.repeat([f"cruise-{index}" for index in range(30)], 3)
    result = hierarchical_group_quantile_bound(
        errors,
        groups,
        quantile=0.9,
        confidence=0.95,
        minimum_groups=29,
    )
    assert result["groups"] == 30
    assert result["group_q90"] == 28.0
    assert result["group_q90_upper95"] == 30.0
    assert result["upper_order_rank"] == 30


def test_hierarchical_bound_fails_closed_with_too_few_groups() -> None:
    result = hierarchical_group_quantile_bound(
        np.arange(10.0),
        np.repeat(["a", "b"], 5),
        quantile=0.9,
        confidence=0.95,
        minimum_groups=29,
    )
    assert np.isnan(result["group_q90_upper95"])
