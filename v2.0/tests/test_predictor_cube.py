"""L1/L2 tests for the Issue #53 predictor-cube contract."""

from __future__ import annotations

import numpy as np
import pytest

from recad.data.predictor_cube import (
    CORE_MONTHS,
    STATUS_MISSING,
    STATUS_RANGE_REJECTED,
    STATUS_SOURCE_VALID,
    CubeVariableSpec,
    calendar_axis,
    compressed_grid_indices,
    deterministic_time_features,
    readiness_masks,
)


def test_status_preserves_missing_and_rejects_ranges() -> None:
    spec = CubeVariableSpec("sss_bg", "sss", 0.0, 50.0, "masked_background", "1e-3")
    status = spec.status(np.array([35.0, np.nan, -1.0, 51.0]))
    assert status.tolist() == [
        STATUS_SOURCE_VALID,
        STATUS_MISSING,
        STATUS_RANGE_REJECTED,
        STATUS_RANGE_REJECTED,
    ]


def test_frozen_calendar_and_cyclic_features() -> None:
    year, month, days = calendar_axis()
    assert len(year) == CORE_MONTHS == 396
    assert (int(year[0]), int(month[0]), int(days[0])) == (1993, 1, 0)
    assert (int(year[-1]), int(month[-1])) == (2025, 12)
    sine, cosine = deterministic_time_features(np.array([1, 4, 7, 10]))
    assert sine == pytest.approx([0, 1, 0, -1], abs=1e-6)
    assert cosine == pytest.approx([1, 0, -1, 0], abs=1e-6)


def test_compressed_grid_indices_are_row_major() -> None:
    mask = np.array([[False, True, True], [True, False, False]])
    flat, lat_index, lon_index = compressed_grid_indices(mask)
    assert flat.tolist() == [1, 2, 3]
    assert lat_index.tolist() == [0, 0, 1]
    assert lon_index.tolist() == [1, 2, 0]


@pytest.mark.l2
def test_readiness_keeps_optional_adt_from_blocking_base() -> None:
    good = np.array([STATUS_SOURCE_VALID, STATUS_SOURCE_VALID], dtype=np.uint8)
    missing = np.array([STATUS_MISSING, STATUS_SOURCE_VALID], dtype=np.uint8)
    mandatory, strict = readiness_masks(
        {"sst": good, "sss_bg": good, "adt": missing, "wspd": good, "xco2air": good}
    )
    assert mandatory.tolist() == [True, True]
    assert strict.tolist() == [False, True]
