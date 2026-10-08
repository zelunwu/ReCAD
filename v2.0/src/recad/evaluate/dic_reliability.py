"""Reliability utilities for DIC derived with PyCO2SYS.

These functions are label agnostic. Direct DIC observations are consumed only
by the experiment runner when it evaluates already generated predictions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from recad.chem.inverse_dic import inverse_dic

GRADE_RANK = {"A": 0, "B": 1, "C": 2, "D": 3}
REASON_BITS: dict[str, int] = {
    "missing_input": 1 << 0,
    "salinity_outside": 1 << 1,
    "temperature_outside": 1 << 2,
    "ta_outside": 1 << 3,
    "fco2_outside": 1 << 4,
    "solver_nonfinite": 1 << 5,
    "upstream_sss_d": 1 << 6,
    "upstream_fco2_d": 1 << 7,
    "upstream_ta_d": 1 << 8,
    "exact_mc_domain": 1 << 9,
    "covariance_unidentified": 1 << 10,
    "interval_too_wide": 1 << 11,
}


def weakest_grade(*grades: Sequence[str]) -> np.ndarray:
    """Return the weakest row-wise grade, enforcing upstream inheritance."""

    arrays = [np.asarray(values).astype("U1") for values in grades]
    if not arrays or any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("grade arrays must be non-empty and have matching shapes")
    ranks = np.stack([[GRADE_RANK.get(value, 3) for value in array] for array in arrays])
    labels = np.asarray(["A", "B", "C", "D"])
    return labels[ranks.max(axis=0)]


def chemistry_reason_bits(
    ta: Sequence[float],
    fco2: Sequence[float],
    salinity: Sequence[float],
    temperature: Sequence[float],
    *,
    ranges: Mapping[str, Sequence[float]],
) -> np.ndarray:
    """Encode preregistered chemistry validity failures as integer bit masks."""

    values = [np.asarray(item, dtype=float) for item in (ta, fco2, salinity, temperature)]
    if any(item.shape != values[0].shape for item in values):
        raise ValueError("chemistry arrays must have matching shapes")
    ta_array, fco2_array, salinity_array, temperature_array = values
    bits = np.zeros(ta_array.shape, dtype=np.uint16)
    missing = ~np.isfinite(np.stack(values)).all(axis=0)
    bits[missing] |= REASON_BITS["missing_input"]
    checks = (
        (salinity_array, "salinity", "salinity_outside"),
        (temperature_array, "temperature", "temperature_outside"),
        (ta_array, "ta", "ta_outside"),
        (fco2_array, "fco2", "fco2_outside"),
    )
    for array, key, reason in checks:
        lower, upper = ranges[key]
        invalid = np.isfinite(array) & ((array < lower) | (array > upper))
        bits[invalid] |= REASON_BITS[reason]
    return bits


def finite_difference_jacobian(
    ta: Sequence[float],
    fco2: Sequence[float],
    salinity: Sequence[float],
    temperature: Sequence[float],
    *,
    steps: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Return central finite-difference PyCO2SYS derivatives for TA/fCO2/SSS."""

    step = {"ta": 1.0, "fco2": 1.0, "sss": 0.01, **(steps or {})}
    ta_array = np.asarray(ta, dtype=float)
    fco2_array = np.asarray(fco2, dtype=float)
    salinity_array = np.asarray(salinity, dtype=float)
    temperature_array = np.asarray(temperature, dtype=float)
    result = np.empty((len(ta_array), 3), dtype=float)
    base = [ta_array, fco2_array, salinity_array]
    for column, name in enumerate(("ta", "fco2", "sss")):
        delta = float(step[name])
        high = [item.copy() for item in base]
        low = [item.copy() for item in base]
        high[column] += delta
        low[column] -= delta
        result[:, column] = (
            inverse_dic(high[0], high[1], high[2], temperature_array)
            - inverse_dic(low[0], low[1], low[2], temperature_array)
        ) / (2.0 * delta)
    return result


def covariance_from_sigmas(
    sigmas: np.ndarray,
    *,
    mode: str,
    bounded_rho: float = 0.5,
    ta_sss_slope: Sequence[float] | float = 0.0,
    fco2_sss_slope: Sequence[float] | float = 0.0,
) -> np.ndarray:
    """Construct row-wise PSD covariance matrices for registered scenarios."""

    scales = np.asarray(sigmas, dtype=float)
    if scales.ndim != 2 or scales.shape[1] != 3:
        raise ValueError("sigmas must have shape (row, 3) for TA, fCO2, SSS")
    result = np.zeros((len(scales), 3, 3), dtype=float)
    result[:, np.arange(3), np.arange(3)] = scales**2
    if mode == "independent":
        return result
    if mode in {"bounded_negative", "bounded_positive"}:
        rho = float(bounded_rho) * (-1 if mode == "bounded_negative" else 1)
        correlation = np.full((3, 3), rho, dtype=float)
        np.fill_diagonal(correlation, 1.0)
        # rho=-0.5 is the PSD boundary for three variables; clip round-off.
        covariance = scales[:, :, None] * correlation * scales[:, None, :]
        eigenvalue, eigenvector = np.linalg.eigh(covariance)
        eigenvalue = np.maximum(eigenvalue, 0.0)
        return np.einsum("nij,nj,nkj->nik", eigenvector, eigenvalue, eigenvector)
    if mode != "common_sss_covariance":
        raise ValueError(f"unknown covariance mode: {mode}")
    ta_slope = np.broadcast_to(np.asarray(ta_sss_slope, dtype=float), len(scales))
    fco2_slope = np.broadcast_to(np.asarray(fco2_sss_slope, dtype=float), len(scales))
    shared = np.stack([ta_slope, fco2_slope, np.ones(len(scales))], axis=1)
    shared *= scales[:, 2, None]
    independent_variance = scales**2 - shared**2
    independent_variance[:, 2] = 0.0
    independent_variance = np.maximum(independent_variance, 0.0)
    result = shared[:, :, None] * shared[:, None, :]
    result[:, np.arange(3), np.arange(3)] += independent_variance
    return result


def jacobian_interval(jacobian: np.ndarray, covariance: np.ndarray) -> dict[str, np.ndarray]:
    """Propagate covariance through a row-wise Jacobian under a normal approximation."""

    jac = np.asarray(jacobian, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    variance = np.einsum("ni,nij,nj->n", jac, cov, jac)
    sd = np.sqrt(np.maximum(variance, 0.0))
    return {"sd": sd, "halfwidth50": 0.67448975 * sd, "halfwidth90": 1.64485363 * sd}


def monte_carlo_dic(
    means: np.ndarray,
    covariance: np.ndarray,
    temperature: Sequence[float],
    *,
    draws: int,
    seed: int,
    chunk_size: int = 128,
) -> dict[str, np.ndarray]:
    """Run exact PyCO2SYS draws in chunks and return row-wise quantiles."""

    center = np.asarray(means, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    temperature_array = np.asarray(temperature, dtype=float)
    if center.ndim != 2 or center.shape[1] != 3 or cov.shape != (len(center), 3, 3):
        raise ValueError("means/covariance shapes must be (row,3)/(row,3,3)")
    rng = np.random.default_rng(seed)
    chol = np.empty_like(cov)
    for index, matrix in enumerate(cov):
        eigenvalue, eigenvector = np.linalg.eigh(matrix)
        chol[index] = eigenvector @ np.diag(np.sqrt(np.maximum(eigenvalue, 0.0)))
    samples = np.empty((draws, len(center)), dtype=np.float32)
    for start in range(0, draws, chunk_size):
        stop = min(draws, start + chunk_size)
        normal = rng.standard_normal((stop - start, len(center), 3))
        values = center[None, :, :] + np.einsum("dni,nji->dnj", normal, chol)
        values[:, :, 0] = np.clip(values[:, :, 0], 500.0, 3000.0)
        values[:, :, 1] = np.clip(values[:, :, 1], 50.0, 1500.0)
        values[:, :, 2] = np.clip(values[:, :, 2], 0.1, 45.0)
        samples[start:stop] = inverse_dic(
            values[:, :, 0].ravel(),
            values[:, :, 1].ravel(),
            values[:, :, 2].ravel(),
            np.broadcast_to(temperature_array, values.shape[:2]).ravel(),
        ).reshape(stop - start, len(center))
    quantile = np.quantile(samples, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
    return {
        "q05": quantile[0],
        "q25": quantile[1],
        "q50": quantile[2],
        "q75": quantile[3],
        "q95": quantile[4],
        "sd": samples.std(axis=0, ddof=1),
    }


def assign_dic_grade(width90: Sequence[float], inherited: Sequence[str]) -> np.ndarray:
    """Apply DIC width gates without allowing improvement over upstream grades."""

    width = np.asarray(width90, dtype=float)
    interval_grade = np.full(len(width), "D", dtype="U1")
    interval_grade[width <= 220] = "C"
    interval_grade[width <= 150] = "B"
    interval_grade[width <= 100] = "A"
    return weakest_grade(interval_grade, inherited)
