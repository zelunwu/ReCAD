import numpy as np
import pandas as pd

from recad.evaluate.ta_mab_reliability import (
    assign_grade,
    calibration_groups,
    finite_sample_quantile,
    regression_metrics,
)


def test_finite_sample_quantile_uses_higher_order_statistic():
    assert finite_sample_quantile([1, 2, 3, 4], 0.5) == 3


def test_calibration_groups_are_deterministic_and_group_safe():
    first = calibration_groups(["b", "a", "b", "c", "d", "e"])
    second = calibration_groups(["e", "d", "c", "b", "a"])
    assert first == second
    assert 0 < len(first) < 5


def test_grade_thresholds_and_ood_override():
    grade = assign_grade([40, 60, 100, 40], [50, 150, 300, 50], [8, 5, 3, 8], [0, 0, 0, 1])
    assert grade.tolist() == ["A", "B", "C", "D"]


def test_regression_metrics_include_interval_coverage():
    frame = pd.DataFrame(
        {
            "truth": [1.0, 2.0],
            "prediction": [1.0, 4.0],
            "q50": [0.5, 0.5],
            "q90": [1.0, 3.0],
            "group_key": ["a", "b"],
            "year": [2000, 2001],
            "grade": ["A", "C"],
        }
    )
    metrics = regression_metrics(frame)
    assert np.isclose(metrics["coverage50"], 0.5)
    assert np.isclose(metrics["coverage90"], 1.0)
    assert np.isclose(metrics["grade_ab_fraction"], 0.5)
