from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_p1_fco2_viability", ROOT / "scripts/run_p1_fco2_viability.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_seasonal_trend_climatology_has_finite_group_fallback() -> None:
    train = pd.DataFrame(
        {
            "year": [2000, 2001, 2002, 2003],
            "month": [1, 1, 2, 2],
            "lme_id": ["1", "1", "1", "1"],
            "truth": [350.0, 352.0, 360.0, 362.0],
        }
    )
    query = pd.DataFrame({"year": [2004], "month": [1], "lme_id": ["unseen"]})
    model = MODULE.SeasonalTrendClimatology().fit(train)
    prediction = model.predict(query)
    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()
