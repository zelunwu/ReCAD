"""Deployable process-proxy features for the Issue 63 controlled ablation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def cardinal_water_neighbours(
    latitude_index: np.ndarray,
    longitude_index: np.ndarray,
    *,
    latitude_cells: int | None = None,
    longitude_cells: int = 2880,
) -> np.ndarray:
    """Return node indices for north, south, west, and east water neighbours.

    The input contains only product-domain water nodes.  Therefore a missing
    lookup represents land or a node outside the product geometry. Longitude
    wraps at the dateline, while latitude never wraps.
    """

    row = np.asarray(latitude_index, dtype=np.int64)
    column = np.asarray(longitude_index, dtype=np.int64)
    if row.ndim != 1 or column.ndim != 1 or row.shape != column.shape:
        raise ValueError("grid indices must be aligned one-dimensional arrays")
    if latitude_cells is None:
        latitude_cells = int(row.max(initial=-1)) + 1
    if np.any(row < 0) or np.any(row >= latitude_cells):
        raise ValueError("latitude index outside declared grid")
    if np.any(column < 0) or np.any(column >= longitude_cells):
        raise ValueError("longitude index outside declared grid")

    flat = row * longitude_cells + column
    order = np.argsort(flat)
    sorted_flat = flat[order]
    result = np.full((len(row), 4), -1, dtype=np.int32)
    for slot, (delta_row, delta_column) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
        candidate_row = row + delta_row
        candidate_column = (column + delta_column) % longitude_cells
        valid = (candidate_row >= 0) & (candidate_row < latitude_cells)
        candidate = candidate_row[valid] * longitude_cells + candidate_column[valid]
        position = np.searchsorted(sorted_flat, candidate)
        found = (position < len(sorted_flat)) & (
            sorted_flat[np.minimum(position, len(sorted_flat) - 1)] == candidate
        )
        source = np.flatnonzero(valid)[found]
        result[source, slot] = order[position[found]].astype(np.int32)
    return result


def local_contrast(values: np.ndarray, neighbours: np.ndarray) -> dict[str, np.ndarray]:
    """Compute same-time water-neighbour contrast with explicit support count."""

    array = np.asarray(values, dtype=float)
    links = np.asarray(neighbours, dtype=np.int64)
    if array.ndim != 1 or links.shape != (len(array), 4):
        raise ValueError("values and four-neighbour matrix are not aligned")
    neighbour_values = np.full(links.shape, np.nan, dtype=float)
    valid_link = links >= 0
    neighbour_values[valid_link] = array[links[valid_link]]
    difference = np.abs(neighbour_values - array[:, None])
    finite = np.isfinite(difference)
    count = finite.sum(axis=1).astype(np.uint8)
    total = np.where(finite, difference, 0.0).sum(axis=1)
    mean_absolute = np.divide(total, count, out=np.zeros(len(array)), where=count > 0)
    combined = np.column_stack([array, neighbour_values])
    has_value = np.isfinite(combined).any(axis=1)
    local_min = np.zeros(len(array))
    local_max = np.zeros(len(array))
    local_min[has_value] = np.nanmin(combined[has_value], axis=1)
    local_max[has_value] = np.nanmax(combined[has_value], axis=1)
    value_range = np.where(count > 0, local_max - local_min, 0.0)
    return {
        "mean_absolute_difference": mean_absolute.astype("float32"),
        "range": value_range.astype("float32"),
        "valid_neighbour_count": count,
    }


def local_contrast_subset(
    values: np.ndarray, neighbours: np.ndarray, selected: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute contrasts only for selected nodes while reading a full field."""

    array = np.asarray(values, dtype=float)
    links = np.asarray(neighbours, dtype=np.int64)[np.asarray(selected, dtype=np.int64)]
    centre = array[np.asarray(selected, dtype=np.int64)]
    neighbour_values = np.full(links.shape, np.nan, dtype=float)
    valid_link = links >= 0
    neighbour_values[valid_link] = array[links[valid_link]]
    difference = np.abs(neighbour_values - centre[:, None])
    finite = np.isfinite(difference)
    count = finite.sum(axis=1).astype(np.uint8)
    mean_absolute = np.divide(
        np.where(finite, difference, 0.0).sum(axis=1),
        count,
        out=np.zeros(len(links)),
        where=count > 0,
    )
    combined = np.column_stack([centre, neighbour_values])
    has_value = np.isfinite(combined).any(axis=1)
    local_min = np.zeros(len(links))
    local_max = np.zeros(len(links))
    local_min[has_value] = np.nanmin(combined[has_value], axis=1)
    local_max[has_value] = np.nanmax(combined[has_value], axis=1)
    return {
        "mean_absolute_difference": mean_absolute.astype("float32"),
        "range": np.where(count > 0, local_max - local_min, 0.0).astype("float32"),
        "valid_neighbour_count": count,
    }


def continuous_background_design(frame: pd.DataFrame) -> np.ndarray:
    """Build the frozen spherical, seasonal, and trend background design."""

    latitude = np.deg2rad(frame.latitude.to_numpy(float))
    longitude = np.deg2rad(frame.longitude.to_numpy(float))
    month = frame.mon.to_numpy(float)
    year = frame.yr.to_numpy(float)
    phase = 2.0 * np.pi * (month - 1.0) / 12.0
    trend = (year - 2009.0) / 16.0
    sphere = np.column_stack(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ]
    )
    seasonal = np.column_stack([np.sin(phase), np.cos(phase), np.sin(2 * phase), np.cos(2 * phase)])
    return np.column_stack(
        [sphere, seasonal, trend, sphere * trend[:, None], seasonal[:, :2] * trend[:, None]]
    ).astype("float32")


@dataclass
class ContinuousBackground:
    """Outer-training-only smooth fCO2 background."""

    estimator: Ridge

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        target: np.ndarray,
        sample_weight: np.ndarray | None = None,
        *,
        alpha: float = 10.0,
    ) -> ContinuousBackground:
        estimator = Ridge(alpha=alpha)
        estimator.fit(continuous_background_design(frame), target, sample_weight=sample_weight)
        return cls(estimator)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            self.estimator.predict(continuous_background_design(frame)), dtype="float32"
        )
