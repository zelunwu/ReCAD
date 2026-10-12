"""L1/L2 tests for the global SSS bake-off helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from recad.evaluate.sss_bakeoff import (
    GroupedTokenST96,
    NumericTransform,
    RobustRegionalCorrection,
    interval_scores,
)


def test_numeric_transform_is_training_fitted_and_nan_safe() -> None:
    train = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [3.0, 3.0, 3.0]})
    transform = NumericTransform.fit(train, ("a", "b"))
    values = transform.apply(pd.DataFrame({"a": [np.nan, 4.0], "b": [3.0, 3.0]}))
    assert np.isfinite(values).all()
    assert np.allclose(values[:, 1], 0)


def test_regional_correction_falls_back_for_unseen_lme() -> None:
    fit = pd.DataFrame(
        {
            "truth": [31.0, 32.0, 34.0, 35.0],
            "sss_bg": [30.0, 31.0, 33.0, 34.0],
            "lme_id": [1, 1, 2, 2],
            "mon": [1, 1, 1, 1],
        }
    )
    model = RobustRegionalCorrection(prior_rows=0).fit(fit)
    held = pd.DataFrame({"sss_bg": [30.0], "lme_id": [99], "mon": [1]})
    assert model.predict(held)[0] == pytest.approx(31.0)


def test_st96_shape_and_gradients() -> None:
    model = GroupedTokenST96((3, 8, 3, 2))
    values = torch.randn(5, 16, requires_grad=True)
    prediction = model(values)
    assert prediction.shape == (5,)
    prediction.mean().backward()
    assert values.grad is not None


@pytest.mark.l2
def test_interval_scores_reward_calibrated_narrow_predictions() -> None:
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    prediction = truth.copy()
    result = interval_scores(truth, prediction, np.ones(4), np.ones(4) * 2)
    assert result["coverage50"] == 1
    assert result["coverage90"] == 1
    assert result["wis"] > 0
