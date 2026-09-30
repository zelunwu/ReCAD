from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_p1_ta_viability", ROOT / "scripts/run_p1_ta_viability.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def synthetic_frame() -> pd.DataFrame:
    rows = []
    for lme in (1, 2):
        for cruise in range(6):
            for month in (2, 6, 10):
                sss = 30 + lme + 0.2 * cruise
                rows.append(
                    {
                        "truth": 500 + 50 * sss + 8 * lme,
                        "sss": sss,
                        "salinity": sss - 0.1,
                        "month": month,
                        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
                        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
                        "lme_id": lme,
                        "regime_id": lme,
                        "group_key": f"g{lme}_{cruise}",
                        "latitude": 30 + lme,
                        "longitude": -70 - cruise,
                    }
                )
    return pd.DataFrame(rows)


def test_hierarchical_model_learns_varying_ta_sss_relation() -> None:
    frame = synthetic_frame()
    model = MODULE.RobustPartialPooling(alpha=1.0, groups=True).fit(frame)
    prediction = model.predict(frame)
    assert np.sqrt(np.mean((prediction - frame.truth) ** 2)) < 1.0


def test_support_distance_is_recomputed_from_training_coordinates() -> None:
    fit = pd.DataFrame({"latitude": [0.0], "longitude": [0.0]})
    target = pd.DataFrame({"latitude": [0.0, 1.0], "longitude": [0.0, 0.0]})
    distance = MODULE.nearest_support_km(fit, target)
    assert distance[0] == 0.0
    assert 110.0 < distance[1] < 112.0


def test_safe_forward_never_uses_primary_locked_rows() -> None:
    train = pd.DataFrame(
        {
            "group_key": ["train", "late_train"],
            "forward_split": ["train", "development"],
        }
    )
    development = pd.DataFrame(
        {
            "group_key": ["dev", "early_dev"],
            "forward_split": ["development", "train"],
        }
    )
    fit, held = MODULE.safe_forward_frames(train, development)
    assert fit.group_key.tolist() == ["train"]
    assert held.group_key.tolist() == ["dev"]


def test_numpy_gate_values_can_be_normalized_for_json() -> None:
    checks = {"passed": np.bool_(True), "failed": np.bool_(False)}
    normalized = {name: bool(value) for name, value in checks.items()}
    assert json.loads(json.dumps(normalized)) == {"passed": True, "failed": False}
