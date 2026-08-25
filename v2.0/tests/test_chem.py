"""Carbonate-chemistry conversion tests."""

from __future__ import annotations

import numpy as np
import pytest

from recad.constants import FUGACITY_COEFF_A, FUGACITY_COEFF_B
from recad.utils.chem import fco2_to_pco2, has_pyco2sys, xco2air_to_pco2air


def test_fco2_to_pco2_formula():
    fco2 = np.array([400.0, 400.0, np.nan])
    sst = np.array([10.0, 25.0, 25.0])
    pco2 = fco2_to_pco2(fco2, sst)
    expected = fco2 * (FUGACITY_COEFF_A - FUGACITY_COEFF_B * sst)
    assert np.allclose(pco2[:2], expected[:2], rtol=1e-9)
    assert np.isnan(pco2[2])
    # the fugacity factor (A - B*SST) decreases with SST, so for a fixed fCO2
    # the converted pCO2 is slightly lower at higher temperature (v1.1 formula)
    assert pco2[1] < pco2[0]


def test_fco2_to_pco2_wraps_v11_value():
    # v1.1: pCO2 = fCO2 * (1.00436 - 4.669e-5 * SST)
    pco2 = fco2_to_pco2(np.array([400.0]), np.array([20.0]))
    assert pco2[0] == pytest.approx(400.0 * (1.00436 - 4.669e-5 * 20.0))


def test_xco2air_conversion_requires_pyco2sys():
    if not has_pyco2sys():
        pytest.skip("PyCO2SYS not installed")
    xco2 = np.array([400.0, 400.0])
    sst = np.array([20.0, 20.0])
    sss = np.array([35.0, 35.0])
    pco2air = xco2air_to_pco2air(xco2, sst, sss)
    # pCO2air at ~20C, S=35 from 400 ppm dry air: roughly 400-410 µatm
    assert np.all(pco2air > 350) and np.all(pco2air < 450)
    assert not np.isnan(pco2air).any()


def test_xco2air_nan_propagation():
    if not has_pyco2sys():
        pytest.skip("PyCO2SYS not installed")
    out = xco2air_to_pco2air(
        np.array([400.0, np.nan]), np.array([20.0, 20.0]), np.array([35.0, 35.0])
    )
    assert np.isnan(out[1]) and not np.isnan(out[0])
