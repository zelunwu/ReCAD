"""Metrics tests (v1.1-compatible definitions) and uncertainty math."""

from __future__ import annotations

import numpy as np
import pytest

from recad.train.metrics import bias, mae, metrics_summary, per_year_metrics, r2, rmse
from recad.uncertainty.decompose import combine_uncertainties


def test_r2_matches_v11_definition():
    y = np.linspace(10, 40, 50)
    pred = y + 1.0  # perfect correlation, constant offset
    corr = np.corrcoef(y, pred)[0, 1]
    assert r2(y, pred) == pytest.approx(corr**2)
    assert r2(y, y) == pytest.approx(1.0)


def test_rmse_hand_computed():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([2.0, 0.0, 3.0])
    # squared errors: 1, 4, 0 -> mean 5/3
    assert rmse(y, p) == pytest.approx(np.sqrt(5.0 / 3.0))


def test_metrics_ignore_nan_pairs():
    y = np.array([1.0, np.nan, 3.0, 4.0])
    p = np.array([1.0, 9.0, np.nan, 4.0])
    m = metrics_summary(y, p)
    assert m["n"] == 2
    assert m["r2"] == pytest.approx(1.0)
    assert m["rmse"] == 0.0
    assert m["bias"] == 0.0


def test_bias_mae():
    y = np.array([10.0, 20.0, 30.0])
    p = np.array([12.0, 19.0, 33.0])
    assert bias(y, p) == pytest.approx(4.0 / 3.0)
    assert mae(y, p) == pytest.approx(2.0)


def test_per_year_table(prepared_small):
    y_true = prepared_small.target_values()
    y_pred = np.nan_to_num(y_true, nan=0.0) + 0.0  # perfect
    table = per_year_metrics(y_true, y_pred, prepared_small.years)
    assert list(table["year"]) == [1993, 1994, 1995]
    assert (table["r2"] > 0.99).all()


def test_combine_rss():
    a = np.full((2, 3), 1.0)  # aleatoric
    e = np.full((2, 3), 2.0)  # epistemic
    i = np.full((2, 3), 2.0)  # input
    total = combine_uncertainties(a, e, i, mode="rss")
    assert np.allclose(total, np.sqrt(1 + 4 + 4))


def test_combine_none_ignores_input():
    a = np.full((2, 3), 1.0)
    e = np.full((2, 3), 0.0)
    i = np.full((2, 3), 5.0)
    assert np.allclose(combine_uncertainties(a, e, i, mode="none"), 1.0)
