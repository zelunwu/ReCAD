from pathlib import Path

import numpy as np
import pandas as pd

from recad.evaluate.conditional_error import (
    calibration_offset,
    cruise_equal_weights,
    load_conditional_error_contract,
    prepare_features,
    stable_group_folds,
)


def test_frozen_conditional_error_contract_keeps_protocols_separate() -> None:
    path = Path(__file__).parents[2] / "configs/frozen/conditional_physical_error_v2.5.in"
    contract = load_conditional_error_contract(path)
    assert contract.protocols[-1] == "early_era"
    assert contract.candidates == ("constant", "stratified", "catboost_quantile")


def test_group_folds_and_weights_are_cruise_safe() -> None:
    groups = pd.Series(["a", "a", "b", "b", "b", "c"])
    assignment = stable_group_folds(groups, salt="test", folds=5)
    assert assignment[0] == assignment[1]
    assert assignment[2] == assignment[3] == assignment[4]
    weights = cruise_equal_weights(groups)
    totals = pd.Series(weights).groupby(groups).sum()
    assert np.allclose(totals, totals.iloc[0])


def test_calibration_offset_uses_group_evidence_units() -> None:
    groups = np.repeat([f"g{index}" for index in range(40)], 10)
    error = np.tile(np.arange(10.0), 40)
    offset, count, rank = calibration_offset(
        error,
        np.zeros_like(error),
        groups,
        quantile=0.9,
        confidence=0.95,
        minimum_groups=29,
    )
    assert offset == 9.0
    assert count == 40
    assert rank <= count


def test_feature_preparation_is_periodic_and_finite() -> None:
    frame = pd.DataFrame(
        {
            "prediction": [1.0],
            "geographic_km": [10.0],
            "latitude": [30.0],
            "longitude": [350.0],
            "mon": [1],
        }
    )
    transformed = prepare_features(frame, tuple(frame.columns))
    assert "log1p_geographic_km" in transformed
    assert "longitude" not in transformed
    assert np.isfinite(transformed.to_numpy()).all()
