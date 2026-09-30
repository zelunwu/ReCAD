from __future__ import annotations

import numpy as np
import pytest

from recad.chem.inverse_dic import (
    forward_fco2,
    interval_summary,
    inverse_dic,
    measured_fco2_mask,
    region_masks,
)


@pytest.mark.l1
def test_measured_fco2_excludes_calculated_method_three() -> None:
    mask = measured_fco2_mask(
        np.array([400.0, 410.0, 420.0, np.nan]),
        np.array([2, 2, 1, 2]),
        np.array([1, 3, 2, 1]),
    )
    assert mask.tolist() == [True, False, False, False]


@pytest.mark.l1
def test_region_boundary_conventions_are_frozen() -> None:
    masks = region_masks(
        np.array([6, 6, 7, 7]),
        np.array([28.45, 35.30, 35.20, 41.75]),
    )
    assert masks["SAB"].tolist() == [True, False, False, False]
    assert masks["MAB"].tolist() == [False, False, True, True]


@pytest.mark.l1
def test_interval_summary_uses_central_intervals() -> None:
    draws = np.arange(100, dtype=float).reshape(20, 5)
    summary = interval_summary(np.array([50, 51, 52, 53, 54]), draws)
    assert summary["median"].shape == (5,)
    assert summary["covered_90"].all()


@pytest.mark.l2
def test_exact_co2sys_round_trip() -> None:
    ta = np.array([2100.0, 2300.0, 2450.0])
    dic = np.array([1950.0, 2050.0, 2200.0])
    salinity = np.array([31.0, 35.0, 37.0])
    temperature = np.array([5.0, 18.0, 28.0])
    fco2 = forward_fco2(ta, dic, salinity, temperature)
    recovered = inverse_dic(ta, fco2, salinity, temperature)
    np.testing.assert_allclose(recovered, dic, atol=1e-6)
