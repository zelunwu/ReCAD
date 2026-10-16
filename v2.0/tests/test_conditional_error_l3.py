from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.l3

ROOT = Path(__file__).parents[1]
ARCHIVE = ROOT / "docs/experiment_archive/p2_conditional_physical_error_v2.5"
SPEC = importlib.util.spec_from_file_location(
    "verify_conditional_error_archive", ROOT / "scripts/verify_experiment_archive.py"
)
assert SPEC and SPEC.loader
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def test_conditional_error_archive_preserves_failed_cohorts_and_unset_thresholds() -> None:
    result = VERIFIER.verify_archive(ARCHIVE)
    assert result["verified"], result["errors"]
    assert result["figures"] == 8
    decision = pd.read_csv(ARCHIVE / "tables/table09_decision.csv").iloc[0]
    assert bool(decision.thresholds_remain_unset)
    assert not bool(decision.grades_emitted)
    assert decision.qualified_region_cohorts < decision.adequate_region_cohorts
    assert decision.qualified_support_cohorts < decision.adequate_support_cohorts
