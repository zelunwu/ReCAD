from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.l1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "run_p3_cruise_holdout", ROOT / "scripts/run_p3_cruise_holdout_v2_4.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_metrics_reports_background_skill_and_coverage() -> None:
    frame = pd.DataFrame(
        {
            "group_key": ["a", "a", "b", "b"],
            "truth": [0.0, 1.0, 2.0, 3.0],
            "prediction": [0.0, 1.5, 2.0, 2.5],
            "background_prediction": [1.0, 2.0, 3.0, 4.0],
            "width50": [0.1, 0.6, 0.1, 0.6],
            "width90": [0.2, 1.0, 0.2, 1.0],
        }
    )
    result = MODULE.metrics(frame)
    assert result["rows"] == 4
    assert result["cruises"] == 2
    assert result["coverage50"] == pytest.approx(1.0)
    assert result["coverage90"] == pytest.approx(1.0)
    assert result["background_skill"] > 0


def test_split_fit_calibration_has_disjoint_complete_groups() -> None:
    frame = pd.DataFrame(
        {
            "group_key": np.repeat([f"g{i}" for i in range(20)], 3),
            "truth": np.arange(60, dtype=float),
        }
    )
    fit, calibration = MODULE.split_fit_calibration(frame, "SSS")
    assert set(fit.group_key).isdisjoint(calibration.group_key)
    assert len(fit) + len(calibration) == len(frame)
