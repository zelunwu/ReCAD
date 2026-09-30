"""Exact TA + fCO2 -> DIC calculations and uncertainty summaries."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

CO2SYS_OPTIONS: dict[str, int | float] = {
    "par1_type": 1,
    "par2_type": 5,
    "pressure": 0,
    "pressure_out": 0,
    "total_silicate": 0,
    "total_phosphate": 0,
    "opt_pH_scale": 1,
    "opt_k_carbonic": 10,
    "opt_k_bisulfate": 1,
    "opt_total_borate": 1,
}


def inverse_dic(
    ta: np.ndarray,
    fco2: np.ndarray,
    salinity: np.ndarray,
    temperature: np.ndarray,
) -> np.ndarray:
    """Solve DIC exactly with the frozen Issue-10 PyCO2SYS options."""
    import PyCO2SYS as pyco2

    out = pyco2.sys(
        par1=np.asarray(ta, dtype=np.float64),
        par2=np.asarray(fco2, dtype=np.float64),
        salinity=np.asarray(salinity, dtype=np.float64),
        temperature=np.asarray(temperature, dtype=np.float64),
        temperature_out=np.asarray(temperature, dtype=np.float64),
        **CO2SYS_OPTIONS,
    )
    return np.asarray(out["dic"], dtype=np.float64)


def forward_fco2(
    ta: np.ndarray,
    dic: np.ndarray,
    salinity: np.ndarray,
    temperature: np.ndarray,
) -> np.ndarray:
    """Calculate reference fCO2 from observed TA and DIC with matching options."""
    import PyCO2SYS as pyco2

    options = dict(CO2SYS_OPTIONS)
    options["par2_type"] = 2
    out = pyco2.sys(
        par1=np.asarray(ta, dtype=np.float64),
        par2=np.asarray(dic, dtype=np.float64),
        salinity=np.asarray(salinity, dtype=np.float64),
        temperature=np.asarray(temperature, dtype=np.float64),
        temperature_out=np.asarray(temperature, dtype=np.float64),
        **options,
    )
    return np.asarray(out["fCO2"], dtype=np.float64)


def measured_fco2_mask(
    fco2: np.ndarray,
    qc: np.ndarray,
    method: np.ndarray,
) -> np.ndarray:
    """Return the preregistered strict measured-fCO2 eligibility mask."""
    return (
        np.isfinite(np.asarray(fco2, dtype=float))
        & (np.asarray(qc) == 2)
        & np.isin(np.asarray(method), (1, 2))
    )


def region_masks(lme_id: np.ndarray, latitude: np.ndarray) -> Mapping[str, np.ndarray]:
    """Return the frozen North America, SAB, and MAB reporting scopes."""
    lme = np.asarray(lme_id)
    lat = np.asarray(latitude, dtype=float)
    return {
        "North America": np.ones(lat.shape, dtype=bool),
        "SAB": (lme == 6) & (lat >= 28.45) & (lat < 35.30),
        "MAB": (lme == 7) & (lat >= 35.20) & (lat <= 41.75),
    }


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Compute finite-pair error metrics used by the archived analysis."""
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    keep = np.isfinite(truth) & np.isfinite(prediction)
    truth, prediction = truth[keep], prediction[keep]
    error = prediction - truth
    denominator = np.sum((truth - truth.mean()) ** 2)
    return {
        "n": len(truth),
        "bias": float(np.mean(error)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1 - np.sum(error**2) / denominator) if denominator > 0 else np.nan,
    }


def interval_summary(truth: np.ndarray, draws: np.ndarray) -> dict[str, np.ndarray]:
    """Summarize row-wise 50% and 90% intervals from ``(draw, row)`` values."""
    truth = np.asarray(truth, dtype=float)
    draws = np.asarray(draws, dtype=float)
    quantiles = np.quantile(draws, (0.05, 0.25, 0.50, 0.75, 0.95), axis=0)
    return {
        "q05": quantiles[0],
        "q25": quantiles[1],
        "median": quantiles[2],
        "q75": quantiles[3],
        "q95": quantiles[4],
        "covered_50": (truth >= quantiles[1]) & (truth <= quantiles[3]),
        "covered_90": (truth >= quantiles[0]) & (truth <= quantiles[4]),
    }
