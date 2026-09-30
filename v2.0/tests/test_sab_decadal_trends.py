from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_sab_decadal_trends", ROOT / "scripts/analyze_sab_decadal_trends.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_sab_mask_uses_canonical_half_open_bounds() -> None:
    frame = pd.DataFrame(
        {
            "lme_id": [6, 6, 6, 7],
            "latitude": [28.45, 35.299, 35.3, 31.0],
        }
    )

    selected = MODULE.sab_mask(frame)

    assert selected.latitude.tolist() == [28.45, 35.299]


def test_clustered_trend_recovers_cruise_equal_synthetic_slope() -> None:
    rows = []
    for cruise, year, count in (("a", 2000, 3), ("b", 2005, 8), ("c", 2010, 2), ("d", 2015, 12)):
        for _ in range(count):
            rows.append(
                {
                    "group_key": cruise,
                    "year": year,
                    "month": 6,
                    "latitude": 31.0,
                    "longitude": -79.0,
                    "truth": 35.0 + 0.02 * (year - 2000),
                }
            )
    frame = pd.DataFrame(rows)

    result = MODULE.fit_clustered_trend(
        frame,
        method="synthetic",
        adjust_season_space=False,
    )

    assert np.isclose(result["slope_per_decade"], 0.2)
    assert result["cruises"] == 4
