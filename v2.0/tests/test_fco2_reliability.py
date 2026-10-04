import numpy as np
import pandas as pd

from recad.evaluate.fco2_reliability import (
    BinnedIntervalModel,
    apply_shrinkage,
    assign_reliability_grade,
    finite_quantile,
)


def test_shrinkage_returns_background_outside_support():
    result = apply_shrinkage([300.0, 400.0], [350.0, 500.0], [1.0, 6.0], "linear_2_5")
    np.testing.assert_allclose(result, [350.0, 400.0])


def test_finite_quantile_uses_conformal_higher_order_statistic():
    assert finite_quantile([1, 2, 3, 4], 0.5) == 3.0


def test_interval_fallback_and_grade_boundaries():
    calibration = pd.DataFrame(
        {
            "risk_environment_k64": np.linspace(0.1, 1.0, 20),
            "background": np.full(20, 400.0),
            "absolute_error": np.arange(1.0, 21.0),
        }
    )
    model = BinnedIntervalModel.fit(calibration, minimum_cell=100)
    query = pd.DataFrame(
        {
            "risk_environment_k64": [1.0, 1.0, 6.0],
            "background": [400.0] * 3,
            "unique_cruises": [3, 3, 3],
            "effective_groups": [2.0, 2.0, 2.0],
        }
    )
    widths = model.predict(query)
    assert not widths.cell_supported.any()
    query = pd.concat([query, widths], axis=1)
    query["cell_supported"] = [True, True, True]
    query["width90"] = [20.0, 35.0, 10.0]
    assert list(assign_reliability_grade(query).astype(str)) == ["A", "B", "D"]
