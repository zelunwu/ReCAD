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


def test_stable_id_preserves_distinct_observations_at_same_location() -> None:
    frame = pd.DataFrame(
        {
            "obs_id": ["a", "b"],
            "group_key": ["cruise", "cruise"],
            "year": [2020, 2020],
            "month": [1, 1],
            "latitude": [40.0, 40.0],
            "longitude": [-70.0, -70.0],
        }
    )
    assert MODULE.stable_id(frame).tolist() == ["a", "b"]


def test_sparse_populated_support_bin_cannot_be_dropped_from_gate() -> None:
    strata = pd.DataFrame(
        {
            "stratum": ["support_distance", "support_distance"],
            "n": [100, 1],
            "skill_vs_carter": [0.5, -0.1],
        }
    )
    assert not MODULE.all_populated_support_bins_positive(strata)


def test_bight_definitions_use_frozen_lme_and_endpoint_masks() -> None:
    frame = pd.DataFrame(
        {
            "lme_id": [6, 6, 6, 7, 7, 7],
            "latitude": [28.45, 33.0, 35.30, 35.20, 40.0, 41.76],
        }
    )

    sab = MODULE.apply_bight_definition(frame, "sab")
    mab = MODULE.apply_bight_definition(frame, "mab")

    assert sab.latitude.tolist() == [28.45, 33.0]
    assert sab.subregion.tolist() == ["south", "north"]
    assert mab.latitude.tolist() == [35.20, 40.0]
    assert mab.subregion.tolist() == ["south", "central"]
    assert sab.source_lme_id.eq(6).all()
    assert mab.source_lme_id.eq(7).all()


def test_sab_sensitivity_changes_only_southern_boundary() -> None:
    frame = pd.DataFrame(
        {
            "lme_id": [6, 6, 6, 6, 7],
            "latitude": [25.9, 26.5, 27.5, 31.0, 27.5],
        }
    )

    lat26 = MODULE.apply_bight_definition(frame, "sab", sab_lat_min=26.0)
    lat28 = MODULE.apply_bight_definition(frame, "sab", sab_lat_min=28.45)

    assert lat26.latitude.tolist() == [26.5, 27.5, 31.0]
    assert lat26.subregion.tolist() == ["26_27", "27_28.45", "30.5_33"]
    assert lat28.latitude.tolist() == [31.0]
