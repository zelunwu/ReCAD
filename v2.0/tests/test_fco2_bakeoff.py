from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recad.evaluate.fco2_bakeoff import (
    LMEResidualAdapter,
    SeasonalTrendBackground,
    fco2_band,
    interval_scores,
)


def frame() -> pd.DataFrame:
    rows = []
    for year in range(2000, 2005):
        for month in range(1, 13):
            for lme in (1, 2):
                truth = 350 + 1.5 * (year - 2000) + 8 * np.sin(2 * np.pi * (month - 1) / 12)
                truth += 15 if lme == 2 else -5
                rows.append((year, month, lme, f"{year}-{month}-{lme}", truth))
    return pd.DataFrame(rows, columns=["yr", "mon", "lme_id", "group_key", "truth"])


def test_seasonal_trend_learns_time_season_and_region() -> None:
    values = frame()
    prediction = SeasonalTrendBackground(shrinkage_groups=0).fit(values).predict(values)
    assert np.sqrt(np.mean((prediction - values.truth) ** 2)) < 1e-8


def test_lme_adapter_uses_zero_for_unseen_lme() -> None:
    values = frame()
    base = values.truth.to_numpy() - np.where(values.lme_id.eq(1), 4.0, -6.0)
    adapter = LMEResidualAdapter(shrinkage_groups=0).fit(values, base)
    held = pd.DataFrame({"lme_id": [1, 2, 99], "mon": [1, 1, 1]})
    correction = adapter.predict_correction(held)
    assert correction == pytest.approx([4.0, -6.0, 0.0])


def test_fco2_bands_are_stable() -> None:
    assert list(fco2_band([200, 300, 400, 500, 600]).astype(str)) == [
        "lt250",
        "250_350",
        "350_450",
        "450_550",
        "ge550",
    ]


def test_interval_scores_match_simple_case() -> None:
    result = interval_scores(
        np.array([0.0, 2.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([2.0, 2.0]),
    )
    assert result["coverage50"] == 0.5
    assert result["coverage90"] == 1.0
    assert result["mean_width90"] == 4.0
