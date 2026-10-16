"""Tests for frozen target parameterizations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recad.evaluate.benchmark_targets import fit_target_estimator, predict_target_estimator

FEATURES = ("feature", "sss_bg", "sss_bg_available")


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": np.linspace(-1, 1, 20),
            "sss_bg": [34.0] * 10 + [np.nan] * 10,
            "sss_bg_available": [1] * 10 + [0] * 10,
            "truth": np.linspace(33, 36, 20),
        }
    )


@pytest.mark.l1
def test_sss_uses_residual_and_direct_paths() -> None:
    data = frame()
    model = fit_target_estimator("ridge", {"alpha": 1.0}, "SSS", data, FEATURES, seed=89)
    prediction = predict_target_estimator(model, data)
    assert prediction.shape == (20,)
    assert np.isfinite(prediction).all()
    assert model.residual is not None


@pytest.mark.l1
def test_fco2_uses_one_direct_path() -> None:
    data = frame().fillna(0)
    model = fit_target_estimator("ridge", {"alpha": 1.0}, "fCO2", data, FEATURES, seed=89)
    prediction = predict_target_estimator(model, data)
    assert prediction.shape == (20,)
    assert model.residual is None
