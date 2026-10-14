import numpy as np
import pandas as pd

from recad.evaluate.reliability_contract import (
    DeployableSupportIndex,
    assign_grade,
    scaled_widths,
    stable_group_fold,
)


def frame():
    return pd.DataFrame(
        {
            "latitude": [0.0, 0.0, 1.0, 1.0],
            "longitude": [0.0, 1.0, 0.0, 1.0],
            "group_key": ["a", "b", "c", "d"],
            "yr": [2000, 2001, 2002, 2003],
            "sst": [20.0, 21.0, 22.0, 23.0],
            "wspd": [5.0, 5.0, 6.0, 6.0],
            "xco2air": [370.0, 371.0, 372.0, 373.0],
        }
    )


def test_support_index_is_training_only_and_deployable():
    source = frame()
    query = source.iloc[[0]].copy()
    query["yr"] = 2005
    result = DeployableSupportIndex(("sst", "wspd", "xco2air"), k=4).fit(source).query(query)
    assert result.geographic_km.iloc[0] == 0
    assert result.unique_cruises.iloc[0] == 4
    assert result.time_extrapolation_years.iloc[0] == 2


def test_scaled_widths_are_finite_sample_and_scale_aware():
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    prediction = np.zeros(4)
    width50, width90 = scaled_widths(truth, prediction, np.ones(4), np.array([2.0]))
    assert width90[0] == 6.0
    assert width50[0] == 4.0


def test_grade_hard_conditions_override_low_risk():
    grade, reason = assign_grade(
        np.array([0.1, 0.2]),
        thresholds=(0.5, 1.0, 2.0),
        geographic_km=np.array([10.0, 500.0]),
        environment_k16=np.array([0.1, 0.1]),
        effective_groups=np.array([3.0, 3.0]),
        missing=np.array([False, False]),
        geographic_max=300.0,
        environment_max=3.0,
    )
    assert grade.tolist() == ["A", "D"]
    assert reason[1] & 1


def test_group_fold_keeps_groups_together():
    groups = pd.Series(["a", "a", "b", "b"])
    assignment = stable_group_fold(groups, salt="test")
    assert assignment[0] == assignment[1]
    assert assignment[2] == assignment[3]
