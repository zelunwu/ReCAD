from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.l3

ROOT = Path(__file__).parents[1]
ARCHIVE = ROOT / "docs/experiment_archive/p2_early_era_validation_v2.5"
SPEC = importlib.util.spec_from_file_location(
    "verify_early_era_archive", ROOT / "scripts/verify_experiment_archive.py"
)
assert SPEC and SPEC.loader
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def test_early_era_archive_and_zero_leakage() -> None:
    result = VERIFIER.verify_archive(ARCHIVE)
    assert result["verified"], result["errors"]
    assert result["figures"] == 8
    audit = pd.read_csv(ARCHIVE / "tables/table03_leakage_audit.csv")
    assert not audit.group_overlap.any()
    assert (audit.earliest_model_fit_year > audit.latest_held_year).all()
