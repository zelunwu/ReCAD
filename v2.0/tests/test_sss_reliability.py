"""L1/L2 tests for the Issue #25 SSS reliability contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recad.evaluate.sss_reliability import (
    BinnedIntervalModel,
    apply_shrinkage,
    assign_reliability_grade,
    finite_quantile,
    shrinkage_weight,
)


def test_shrinkage_returns_to_background_as_risk_increases():
    risk = np.array([0.0, 2.0, 3.5, 5.0, np.inf])
    weight = shrinkage_weight(risk, "linear_2_5")
    assert np.all(np.diff(weight) <= 0)
    prediction = apply_shrinkage(np.full(5, 30.0), np.full(5, 34.0), risk, "linear_2_5")
    assert prediction.tolist() == pytest.approx([34.0, 34.0, 32.0, 30.0, 30.0])


def test_finite_quantile_uses_conformal_higher_order_statistic():
    assert finite_quantile([1, 2, 3, 4], 0.5) == 3
    assert finite_quantile([1, 2, 3, 4], 0.9) == 4


@pytest.mark.l2
def test_binned_interval_model_never_reads_evaluation_truth():
    calibration = pd.DataFrame(
        {
            "absolute_error": np.linspace(0.01, 2.0, 1000),
            "risk_environment_k64": np.linspace(0.0, 5.0, 1000),
            "background": np.tile([15.0, 25.0, 31.0, 34.0, 37.0], 200),
        }
    )
    model = BinnedIntervalModel.fit(calibration, minimum_cell=10)
    evaluation = pd.DataFrame(
        {
            "risk_environment_k64": [0.2, 4.8],
            "background": [34.0, 15.0],
            "truth": [-99999.0, 99999.0],
        }
    )
    first = model.predict(evaluation)
    evaluation["truth"] *= -1
    pd.testing.assert_frame_equal(first, model.predict(evaluation))


def test_grade_rules_suppress_unsupported_and_respect_widths():
    frame = pd.DataFrame(
        {
            "width90": [0.4, 0.8, 1.8, 0.4, 0.4],
            "risk_environment_k64": [1.0, 2.0, 3.0, 6.0, 1.0],
            "unique_cruises": [5, 5, 5, 5, 1],
            "effective_groups": [3.0, 3.0, 3.0, 3.0, 3.0],
            "cell_supported": [True] * 5,
            "background": [34.0] * 5,
        }
    )
    assert assign_reliability_grade(frame).tolist() == ["A", "B", "C", "D", "D"]


def test_unknown_shrinkage_method_fails_closed():
    with pytest.raises(ValueError, match="unknown shrinkage"):
        shrinkage_weight([1.0], "invented")
