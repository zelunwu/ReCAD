from __future__ import annotations

import numpy as np
import pytest

from recad.evaluate.dic_reliability import (
    REASON_BITS,
    assign_dic_grade,
    chemistry_reason_bits,
    covariance_from_sigmas,
    finite_difference_jacobian,
    jacobian_interval,
    weakest_grade,
)

pytestmark = pytest.mark.l1


def test_weakest_grade_and_dic_width_gate_never_upgrade_upstream() -> None:
    inherited = weakest_grade(["A", "B", "C", "D"], ["B", "A", "D", "A"])
    assert inherited.tolist() == ["B", "B", "D", "D"]
    assert assign_dic_grade([90, 90, 90, 90], inherited).tolist() == ["B", "B", "D", "D"]


def test_chemistry_reason_bits_are_composable() -> None:
    bits = chemistry_reason_bits(
        [2300, np.nan, 3500],
        [400, 400, 20],
        [35, 35, 50],
        [20, 20, 50],
        ranges={
            "ta": [500, 3000],
            "fco2": [50, 1500],
            "salinity": [0.1, 45],
            "temperature": [-2.5, 40],
        },
    )
    assert bits[0] == 0
    assert bits[1] & REASON_BITS["missing_input"]
    expected = (
        REASON_BITS["ta_outside"]
        | REASON_BITS["fco2_outside"]
        | REASON_BITS["salinity_outside"]
        | REASON_BITS["temperature_outside"]
    )
    assert bits[2] == expected


def test_covariance_modes_are_psd_and_jacobian_interval_is_finite() -> None:
    sigmas = np.array([[40.0, 20.0, 0.5], [60.0, 30.0, 1.0]])
    for mode in ("independent", "bounded_negative", "bounded_positive"):
        covariance = covariance_from_sigmas(sigmas, mode=mode)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-8
        interval = jacobian_interval(np.ones((2, 3)), covariance)
        assert np.isfinite(interval["halfwidth90"]).all()


def test_pyco2sys_finite_difference_jacobian_has_expected_signs() -> None:
    jacobian = finite_difference_jacobian([2300], [400], [35], [20])
    assert jacobian.shape == (1, 3)
    assert jacobian[0, 0] > 0
    assert jacobian[0, 1] > 0
