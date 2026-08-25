"""Carbonate-chemistry conversions used by the pipeline.

Two-legged design inherited from v1.1:
 * atmospheric xCO2 -> pCO2(air) at in-situ SST/SSS via PyCO2SYS
   (v1.1 Data_readalldata.ipynb cell 8);
 * SOCAT fCO2 -> pCO2 via the Wanninkhof (1992) fugacity correction
   (v1.1 Data_readalldata.ipynb cell 3):
     pCO2 = fCO2 * (1.00436 - 4.669e-5 * SST)

PyCO2SYS is optional at import time (the module degrades gracefully if the
package is missing), but required for the full pipeline.
"""

from __future__ import annotations

import numpy as np

from recad.constants import FUGACITY_COEFF_A, FUGACITY_COEFF_B

try:  # pragma: no cover - environment-dependent
    import PyCO2SYS as pyco2

    _HAS_PYCO2SYS = True
except ImportError:  # pragma: no cover
    _HAS_PYCO2SYS = False


def has_pyco2sys() -> bool:
    """Whether the PyCO2SYS extension is importable."""
    return _HAS_PYCO2SYS


def fco2_to_pco2(fco2: np.ndarray, sst: np.ndarray) -> np.ndarray:
    """Convert seawater fugacity fCO2 (µatm) to pCO2 (µatm) at SST (°C).

    Wanninkhof (1992) correction, identical to v1.1.
    """
    fco2 = np.asarray(fco2, dtype=np.float64)
    sst = np.asarray(sst, dtype=np.float64)
    pco2 = fco2 * (FUGACITY_COEFF_A - FUGACITY_COEFF_B * sst)
    return np.where(np.isnan(fco2), np.nan, pco2)


def xco2air_to_pco2air(
    xco2air: np.ndarray,
    sst: np.ndarray,
    sss: np.ndarray,
    *,
    pressure: float = 0.0,
) -> np.ndarray:
    """Convert atmospheric dry-mole-fraction xCO2 (µmol mol-1) to pCO2 (µatm).

    The conversion evaluates the in-vivo seawater pCO2 at equilibrium with the
    given atmosphere (par1_type=9 in PyCO2SYS), following v1.1. Returns NaN
    where PyCO2SYS is unavailable.
    """
    xco2air = np.asarray(xco2air, dtype=np.float64)
    sst = np.asarray(sst, dtype=np.float64)
    sss = np.asarray(sss, dtype=np.float64)
    if not _HAS_PYCO2SYS:
        return np.full_like(xco2air, np.nan)
    result = pyco2.sys(
        par1=xco2air,
        par1_type=9,
        temperature=sst,
        temperature_out=sst,
        salinity=sss,
        pressure=pressure,
        pressure_out=0,
    )
    out = np.asarray(result["pCO2_out"], dtype=np.float64)
    return np.where(np.isnan(xco2air) | np.isnan(sst) | np.isnan(sss), np.nan, out)
