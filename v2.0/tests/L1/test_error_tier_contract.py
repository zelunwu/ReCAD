from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from recad.evaluate.error_tier_contract import (
    ErrorReason,
    ErrorThresholds,
    ErrorTier,
    EvidenceStatus,
    assign_error_tier,
    load_error_flag_contract,
)

CONFIG = Path(__file__).parents[2] / "configs" / "frozen" / "absolute_error_flag_contract_v2.5.in"


def test_repository_contract_loads_without_silent_threshold_defaults() -> None:
    contract = load_error_flag_contract(CONFIG)
    assert contract.primary_statistic == "absolute_error_q90_upper95"
    assert contract.absolute_error_quantile == 0.9
    assert contract.confidence_level == 0.95
    assert all(not profile.thresholds.frozen for profile in contract.targets.values())
    assert contract.reason_bits == {reason.name: int(reason) for reason in ErrorReason}


def test_unknown_and_invalid_states_do_not_require_thresholds() -> None:
    profile = load_error_flag_contract(CONFIG).target("fCO2")
    assert (
        assign_error_tier(None, profile=profile, evidence=EvidenceStatus.UNVALIDATED)
        is ErrorTier.UNKNOWN
    )
    assert (
        assign_error_tier(None, profile=profile, evidence=EvidenceStatus.FAILED)
        is ErrorTier.UNKNOWN
    )
    assert (
        assign_error_tier(
            None,
            profile=profile,
            evidence=EvidenceStatus.INVALID,
            prediction_valid=False,
        )
        is ErrorTier.NO_PREDICTION
    )
    assert assign_error_tier(None, profile=profile, evidence=EvidenceStatus.OUTSIDE_DOMAIN) is None


def test_fixed_physical_thresholds_allow_empty_or_full_tiers() -> None:
    profile = load_error_flag_contract(CONFIG).target("fCO2")
    profile = replace(profile, thresholds=ErrorThresholds(a=10.0, b=20.0, c=40.0))
    errors = [0.0, 10.0, 10.1, 20.0, 20.1, 40.0, 40.1]
    tiers = [
        assign_error_tier(error, profile=profile, evidence=EvidenceStatus.DIRECT)
        for error in errors
    ]
    assert tiers == [
        ErrorTier.A,
        ErrorTier.A,
        ErrorTier.B,
        ErrorTier.B,
        ErrorTier.C,
        ErrorTier.C,
        ErrorTier.D,
    ]
    assert all(
        assign_error_tier(error, profile=profile, evidence=EvidenceStatus.TRANSFER) is ErrorTier.A
        for error in [1.0, 2.0, 3.0]
    )


def test_supported_grade_fails_closed_until_thresholds_are_frozen() -> None:
    profile = load_error_flag_contract(CONFIG).target("SSS")
    with pytest.raises(ValueError, match="thresholds are not frozen"):
        assign_error_tier(0.5, profile=profile, evidence=EvidenceStatus.DIRECT)


def test_partial_or_non_monotonic_thresholds_are_rejected(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    partial = tmp_path / "partial.in"
    partial.write_text(
        text.replace("threshold_a = UNSET", "threshold_a = 1.0", 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="all frozen or all UNSET"):
        load_error_flag_contract(partial)

    non_monotonic = tmp_path / "non_monotonic.in"
    configured = text.replace("threshold_a = UNSET", "threshold_a = 2.0", 1)
    configured = configured.replace("threshold_b = UNSET", "threshold_b = 1.0", 1)
    configured = configured.replace("threshold_c = UNSET", "threshold_c = 3.0", 1)
    non_monotonic.write_text(configured, encoding="utf-8")
    with pytest.raises(ValueError, match="A < B < C"):
        load_error_flag_contract(non_monotonic)
