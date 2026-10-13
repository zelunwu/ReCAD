"""Regression tests for the P2.7 validation repair."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recad.evaluate.validation_repair import (
    assert_split_integrity,
    conformal_widths,
    geographic_splits,
    grouped_fit_calibration,
    normalize_group_key,
    rolling_forward_splits,
)


def sample_frame() -> pd.DataFrame:
    rows = []
    record_id = 0
    for group, year, latitude, longitude in (
        ("A", 2010, 0.0, 0.0),
        ("B", 2011, 0.0, 10.0),
        ("C", 2014, 10.0, 0.0),
        ("D", 2017, 10.0, 10.0),
        ("E", 2022, 20.0, 20.0),
    ):
        for offset in range(2):
            rows.append(
                {
                    "record_id": record_id,
                    "group_key": group,
                    "yr": year,
                    "grid_flat": record_id,
                    "latitude": latitude + offset * 0.25,
                    "longitude": longitude,
                    "lme_id": offset,
                }
            )
            record_id += 1
    return pd.DataFrame(rows)


def test_normalize_group_key_removes_alias_punctuation() -> None:
    assert normalize_group_key(" 33-gc_2012/01 ") == "33GC201201"


def test_geographic_split_removes_complete_held_groups() -> None:
    frame = sample_frame()
    splits = geographic_splits(frame, n_folds=2, block_degrees=5.0, buffer_km=0.0)
    assert splits
    for split in splits:
        checks = assert_split_integrity(frame, split)
        assert checks == {
            "group_overlap": 0,
            "record_overlap": 0,
            "forbidden_time_rows": 0,
        }


def test_rolling_forward_uses_only_past_groups() -> None:
    frame = sample_frame()
    split = rolling_forward_splits(frame, [(2013, 2016), (2021, 2023)])
    assert len(split) == 2
    for item in split:
        assert_split_integrity(frame, item)
        assert frame.iloc[item.fit_index].yr.max() < int(item.metadata["held_start"])


def test_conformal_widths_use_prediction_bands() -> None:
    calibration_prediction = np.concatenate([np.full(300, 15.0), np.full(300, 35.0)])
    calibration_truth = calibration_prediction + np.concatenate(
        [np.full(300, 2.0), np.full(300, 0.2)]
    )
    held_prediction = np.array([15.0, 35.0])
    width50, width90, bands = conformal_widths(
        calibration_truth,
        calibration_prediction,
        held_prediction,
        target="SSS",
    )
    assert bands.tolist() == ["lt20", "34_36"]
    assert width50[0] > width50[1]
    assert width90[0] > width90[1]


def test_grouped_fit_calibration_has_no_group_overlap() -> None:
    frame = sample_frame()
    fit, calibration = grouped_fit_calibration(frame, salt="test-calibration")
    assert set(frame.iloc[fit].group_key).isdisjoint(frame.iloc[calibration].group_key)
