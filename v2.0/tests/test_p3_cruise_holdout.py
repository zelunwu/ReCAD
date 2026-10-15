from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.l1

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_p3_cruise_holdout", ROOT / "scripts/select_p3_cruise_holdout_v2_4.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_groups() -> pd.DataFrame:
    rows = []
    for basin in range(5):
        for index in range(20):
            rows.append(
                {
                    "group_key": f"b{basin}-g{index}",
                    "basin_id": basin,
                    "year_max": 2025 if index < 10 else 2015,
                    "block_group_count": 1 + index % 4,
                }
            )
    return pd.DataFrame(rows)


def test_holdout_is_deterministic_and_basin_balanced() -> None:
    groups = synthetic_groups()
    first = MODULE.select_holdout(groups, total=50, salt="test")
    second = MODULE.select_holdout(groups, total=50, salt="test")
    assert first.group_key.tolist() == second.group_key.tolist()
    assert len(first) == first.group_key.nunique() == 50
    assert first.groupby("basin_id").size().min() >= 6
    assert set(first.selection_reason) == {"recent_2022_2025", "sparse_5deg_block"}


def test_metadata_columns_exclude_target_values() -> None:
    assert "fco2" not in MODULE.METADATA_COLUMNS
    assert "salinity" not in MODULE.METADATA_COLUMNS
    assert {"fco2_n", "sss_n"} <= set(MODULE.METADATA_COLUMNS)
