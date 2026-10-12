"""Schema helpers for the compressed global-coastal predictor cube."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CORE_START_YEAR = 1993
CORE_END_YEAR = 2025
CORE_MONTHS = (CORE_END_YEAR - CORE_START_YEAR + 1) * 12

STATUS_MISSING = np.uint8(0)
STATUS_SOURCE_VALID = np.uint8(1)
STATUS_RANGE_REJECTED = np.uint8(2)
STATUS_IMPUTED = np.uint8(3)


@dataclass(frozen=True)
class CubeVariableSpec:
    """Frozen source and physical-range contract for one dynamic field."""

    output_name: str
    source_variable: str
    minimum: float
    maximum: float
    role: str
    units: str

    def status(self, values: np.ndarray) -> np.ndarray:
        """Encode missing, accepted-source, and rejected-source values."""

        values = np.asarray(values)
        finite = np.isfinite(values)
        accepted = finite & (values >= self.minimum) & (values <= self.maximum)
        status = np.full(values.shape, STATUS_MISSING, dtype=np.uint8)
        status[finite & ~accepted] = STATUS_RANGE_REJECTED
        status[accepted] = STATUS_SOURCE_VALID
        return status


def calendar_axis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return integer year/month and mid-month day offsets for 1993-2025."""

    month_index = np.arange(CORE_MONTHS, dtype=np.int32)
    year = (CORE_START_YEAR + month_index // 12).astype(np.int16)
    month = (month_index % 12 + 1).astype(np.uint8)
    dates = np.array(
        [np.datetime64(f"{int(y):04d}-{int(m):02d}-15") for y, m in zip(year, month, strict=True)]
    )
    origin = np.datetime64("1993-01-15")
    days = (dates - origin).astype("timedelta64[D]").astype(np.int32)
    return year, month, days


def deterministic_time_features(month: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return cyclic month features without fitting any data."""

    month = np.asarray(month, dtype=np.float64)
    angle = 2 * np.pi * (month - 1) / 12
    return np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)


def compressed_grid_indices(product_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flat, latitude, and longitude indices for a 2-D product mask."""

    mask = np.asarray(product_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("product_mask must be two-dimensional")
    flat = np.flatnonzero(mask.reshape(-1)).astype(np.int32)
    lat_index, lon_index = np.unravel_index(flat, mask.shape)
    return flat, lat_index.astype(np.int16), lon_index.astype(np.int16)


def readiness_masks(status_by_name: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return mandatory-base and exact five-field strict support masks."""

    required = {"sst", "sss_bg", "adt", "wspd", "xco2air"}
    missing = required.difference(status_by_name)
    if missing:
        raise ValueError(f"missing status arrays: {sorted(missing)}")
    available = {
        name: np.asarray(status) == STATUS_SOURCE_VALID for name, status in status_by_name.items()
    }
    shape = available["sst"].shape
    if any(field.shape != shape for field in available.values()):
        raise ValueError("all status arrays must have the same shape")
    mandatory = available["sst"] & available["wspd"] & available["xco2air"]
    strict = mandatory & available["sss_bg"] & available["adt"]
    return mandatory, strict
