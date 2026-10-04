import pandas as pd

from recad.evaluate.v11_parity import parity_gate, regression_metrics


def test_regression_metrics_match_definitions():
    result = regression_metrics([1.0, 2.0, 3.0], [1.0, 3.0, 2.0])
    assert result["RMSE"] == (2.0 / 3.0) ** 0.5
    assert result["MAE"] == 2.0 / 3.0
    assert result["MBE"] == 0.0


def test_parity_gate_distinguishes_minimum_and_strong():
    metrics = pd.DataFrame(
        {
            "Region": ["NAACOM", "NAACOM"],
            "Type": ["Test", "Validation"],
            "RMSE": [17.0, 25.0],
        }
    )
    config = {
        "minimum_parity": {
            "naacom_test_rmse_lt": 17.64,
            "naacom_validation_rmse_lt": 28.97,
        },
        "strong_product_target": {
            "naacom_test_rmse_lte": 15.88,
            "naacom_validation_rmse_lte": 26.08,
        },
    }
    result = parity_gate(metrics, config)
    assert result["minimum_parity_passed"]
    assert not result["strong_product_target_passed"]
