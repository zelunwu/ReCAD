"""L1/L2 tests for the Issue #52 predictor-availability audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recad.data.predictor_availability import (
    PredictorSpec,
    aggregate_time_blocks,
    coverage_classification,
    month_index,
    time_block,
    weighted_fraction,
)


def test_calendar_contract() -> None:
    assert month_index(1993, 1) == 0
    assert month_index(2025, 12) == 395
    assert time_block(1997) == "1993-1997"
    assert time_block(1998) == "1998-2001"
    assert time_block(2025) == "2021-2025"
    with pytest.raises(ValueError):
        month_index(1993, 13)


def test_validity_and_area_weighting() -> None:
    spec = PredictorSpec("sss", "sss", 0.0, 50.0)
    values = np.array([[35.0, np.nan], [-1.0, 34.0]])
    valid = spec.valid(values)
    mask = np.ones((2, 2), dtype=bool)
    assert valid.tolist() == [[True, False], [False, True]]
    assert weighted_fraction(valid, mask, np.array([1.0, 2.0])) == pytest.approx(0.5)


@pytest.mark.l2
def test_block_aggregation_and_classification() -> None:
    monthly = pd.DataFrame(
        {
            "predictor": ["sst", "sst", "sst"],
            "year": [1993, 1997, 2025],
            "area_fraction": [1.0, 0.98, 0.95],
        }
    )
    result = aggregate_time_blocks(monthly, ["area_fraction"])
    assert len(result) == 2
    assert coverage_classification(0.995) == "deployable"
    assert coverage_classification(0.95) == "deployable_with_missingness"
    assert coverage_classification(0.80) == "repair_required"
    assert coverage_classification(float("nan")) == "unavailable"
