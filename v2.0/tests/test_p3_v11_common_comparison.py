from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.l1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "run_p3_v11_common_comparison", ROOT / "scripts/run_p3_v11_common_comparison.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_comparison_metrics_use_complete_cruise_groups() -> None:
    frame = pd.DataFrame(
        {
            "group_key": ["a", "a", "b", "b"],
            "truth": [1.0, 2.0, 3.0, 4.0],
            "prediction": [1.0, 2.5, 2.5, 4.0],
        }
    )
    result = MODULE.metrics(frame, "prediction")
    assert result["rows"] == 4
    assert result["cruises"] == 2
    assert result["rmse"] == pytest.approx((0.5 / 4) ** 0.5)
