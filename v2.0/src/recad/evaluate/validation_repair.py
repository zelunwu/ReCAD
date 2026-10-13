"""Leakage-safe split and calibration helpers for the P2.7 evidence repair."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from recad.evaluate.predictor_ablation import stable_fold

EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class RepairedSplit:
    """One outer split with explicit fit and evaluation rows."""

    scheme: str
    fold: int
    fit_index: np.ndarray
    held_index: np.ndarray
    held_groups: tuple[str, ...]
    metadata: dict[str, object]


def normalize_group_key(value: object) -> str:
    """Normalize a cruise/platform identifier for cross-product lineage checks."""

    return "".join(character for character in str(value).upper() if character.isalnum())


def geographic_block_id(
    latitude: pd.Series | np.ndarray,
    longitude: pd.Series | np.ndarray,
    *,
    block_degrees: float = 5.0,
) -> np.ndarray:
    """Return deterministic global block identifiers from row coordinates."""

    lat = np.asarray(latitude, dtype=float)
    lon = np.mod(np.asarray(longitude, dtype=float), 360.0)
    lat_index = np.floor((np.clip(lat, -90.0, 90.0 - 1e-9) + 90.0) / block_degrees).astype(np.int16)
    lon_index = np.floor(lon / block_degrees).astype(np.int16)
    return np.asarray(
        [
            f"{int(lat_value)}:{int(lon_value)}"
            for lat_value, lon_value in zip(lat_index, lon_index, strict=True)
        ],
        dtype=object,
    )


def _buffered_grid_cells(
    frame: pd.DataFrame,
    candidate_mask: np.ndarray,
    held_mask: np.ndarray,
    *,
    buffer_km: float,
) -> set[int]:
    """Return candidate grid cells within ``buffer_km`` of held coordinates."""

    if buffer_km <= 0:
        return set()
    held = frame.loc[held_mask, ["grid_flat", "latitude", "longitude"]].drop_duplicates("grid_flat")
    candidate = frame.loc[candidate_mask, ["grid_flat", "latitude", "longitude"]].drop_duplicates(
        "grid_flat"
    )
    if held.empty or candidate.empty:
        return set()
    held_coordinates = np.deg2rad(held[["latitude", "longitude"]].to_numpy(float))
    candidate_coordinates = np.deg2rad(candidate[["latitude", "longitude"]].to_numpy(float))
    tree = BallTree(held_coordinates, metric="haversine")
    count = tree.query_radius(
        candidate_coordinates,
        r=float(buffer_km) / EARTH_RADIUS_KM,
        count_only=True,
    )
    return set(candidate.loc[np.asarray(count) > 0, "grid_flat"].astype(int))


def geographic_splits(
    frame: pd.DataFrame,
    *,
    n_folds: int = 5,
    block_degrees: float = 5.0,
    buffer_km: float = 100.0,
    salt: str = "recad-v2.4-geographic",
) -> list[RepairedSplit]:
    """Hold actual geographic blocks and remove groups plus a spatial buffer from fitting."""

    blocks = geographic_block_id(frame.latitude, frame.longitude, block_degrees=block_degrees)
    unique_blocks = np.unique(blocks)
    block_fold = {block: stable_fold(block, salt, n_folds=n_folds) for block in unique_blocks}
    assignment = np.fromiter((block_fold[block] for block in blocks), dtype=np.int8)
    groups = frame.group_key.astype(str).to_numpy()
    grids = frame.grid_flat.to_numpy(int)
    splits: list[RepairedSplit] = []
    for fold in range(n_folds):
        held = assignment == fold
        held_groups = tuple(sorted(set(groups[held])))
        fit = (assignment != fold) & ~np.isin(groups, held_groups)
        buffered = _buffered_grid_cells(frame, fit, held, buffer_km=buffer_km)
        if buffered:
            fit &= ~np.isin(grids, list(buffered))
        if not held.any() or not fit.any():
            continue
        splits.append(
            RepairedSplit(
                scheme="geographic",
                fold=fold,
                fit_index=np.flatnonzero(fit),
                held_index=np.flatnonzero(held),
                held_groups=held_groups,
                metadata={
                    "block_degrees": block_degrees,
                    "buffer_km": buffer_km,
                    "held_blocks": len(set(blocks[held])),
                    "buffered_fit_cells": len(buffered),
                },
            )
        )
    return splits


def cruise_splits(
    frame: pd.DataFrame,
    *,
    n_folds: int = 5,
    salt: str = "recad-v2.4-cruise",
) -> list[RepairedSplit]:
    """Hold complete normalized cruise/platform groups."""

    groups = frame.group_key.astype(str).to_numpy()
    unique = np.unique(groups)
    group_fold = {group: stable_fold(group, salt, n_folds=n_folds) for group in unique}
    assignment = np.fromiter((group_fold[group] for group in groups), dtype=np.int8)
    return [
        RepairedSplit(
            scheme="cruise",
            fold=fold,
            fit_index=np.flatnonzero(assignment != fold),
            held_index=np.flatnonzero(assignment == fold),
            held_groups=tuple(sorted(set(groups[assignment == fold]))),
            metadata={},
        )
        for fold in range(n_folds)
        if np.any(assignment == fold) and np.any(assignment != fold)
    ]


def _region_keys(frame: pd.DataFrame) -> np.ndarray:
    """Use LMEs where defined and neutral geographic regions elsewhere."""

    lme = frame.lme_id.fillna(-1).to_numpy(int)
    latitude = frame.latitude.to_numpy(float)
    longitude = np.mod(frame.longitude.to_numpy(float), 360.0)
    keys: list[str] = []
    for lme_value, lat_value, lon_value in zip(lme, latitude, longitude, strict=True):
        if lme_value >= 0:
            keys.append(f"lme:{lme_value}")
        else:
            lat_band = int(np.floor((lat_value + 90.0) / 20.0))
            lon_band = int(np.floor(lon_value / 40.0))
            keys.append(f"neutral:{lat_band}:{lon_band}")
    return np.asarray(keys, dtype=object)


def whole_region_splits(
    frame: pd.DataFrame,
    *,
    n_folds: int = 5,
    salt: str = "recad-v2.4-whole-region",
) -> list[RepairedSplit]:
    """Hold complete LMEs or fixed neutral geographic regions."""

    regions = _region_keys(frame)
    unique = np.unique(regions)
    region_fold = {region: stable_fold(region, salt, n_folds=n_folds) for region in unique}
    assignment = np.fromiter((region_fold[region] for region in regions), dtype=np.int8)
    groups = frame.group_key.astype(str).to_numpy()
    splits: list[RepairedSplit] = []
    for fold in range(n_folds):
        held = assignment == fold
        held_groups = tuple(sorted(set(groups[held])))
        fit = (assignment != fold) & ~np.isin(groups, held_groups)
        if held.any() and fit.any():
            splits.append(
                RepairedSplit(
                    scheme="whole_region",
                    fold=fold,
                    fit_index=np.flatnonzero(fit),
                    held_index=np.flatnonzero(held),
                    held_groups=held_groups,
                    metadata={"held_regions": len(set(regions[held]))},
                )
            )
    return splits


def rolling_forward_splits(
    frame: pd.DataFrame,
    windows: Iterable[tuple[int, int]],
) -> list[RepairedSplit]:
    """Return past-only rolling-origin splits with complete held groups removed from fit."""

    years = frame.yr.to_numpy(int)
    groups = frame.group_key.astype(str).to_numpy()
    splits: list[RepairedSplit] = []
    for fold, (held_start, held_end) in enumerate(windows):
        held = (years >= held_start) & (years <= held_end)
        held_groups = tuple(sorted(set(groups[held])))
        fit = (years < held_start) & ~np.isin(groups, held_groups)
        if held.any() and fit.any():
            splits.append(
                RepairedSplit(
                    scheme="forward",
                    fold=fold,
                    fit_index=np.flatnonzero(fit),
                    held_index=np.flatnonzero(held),
                    held_groups=held_groups,
                    metadata={
                        "train_end": held_start - 1,
                        "held_start": held_start,
                        "held_end": held_end,
                    },
                )
            )
    return splits


def inner_group_folds(
    frame: pd.DataFrame,
    *,
    n_folds: int = 2,
    salt: str,
) -> np.ndarray:
    """Assign complete groups to deterministic inner folds."""

    groups = frame.group_key.astype(str)
    mapping = {group: stable_fold(group, salt, n_folds=n_folds) for group in groups.unique()}
    return groups.map(mapping).to_numpy(np.int8)


def prediction_band(values: np.ndarray, target: str) -> pd.Categorical:
    """Create deployable interval strata from predictions rather than target truth."""

    if target == "SSS":
        edges = [-np.inf, 20.0, 30.0, 34.0, 36.0, np.inf]
        labels = ["lt20", "20_30", "30_34", "34_36", "ge36"]
    elif target == "fCO2":
        edges = [-np.inf, 300.0, 350.0, 400.0, 450.0, np.inf]
        labels = ["lt300", "300_350", "350_400", "400_450", "ge450"]
    else:
        raise ValueError(f"unknown target: {target}")
    return pd.cut(values, edges, labels=labels, right=False)


def conformal_widths(
    calibration_truth: np.ndarray,
    calibration_prediction: np.ndarray,
    held_prediction: np.ndarray,
    *,
    target: str,
    minimum_band_rows: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return prediction-stratified symmetric 50/90 conformal half-widths."""

    truth = np.asarray(calibration_truth, dtype=float)
    calibration = np.asarray(calibration_prediction, dtype=float)
    held = np.asarray(held_prediction, dtype=float)
    finite = np.isfinite(truth) & np.isfinite(calibration)
    if not finite.any():
        raise ValueError("calibration contains no finite residuals")
    absolute = np.abs(calibration[finite] - truth[finite])
    global_width = np.quantile(absolute, [0.5, 0.9])
    calibration_band = prediction_band(calibration[finite], target)
    held_band = prediction_band(held, target)
    width50 = np.full(len(held), global_width[0], dtype=float)
    width90 = np.full(len(held), global_width[1], dtype=float)
    for band in calibration_band.categories:
        selected = np.asarray(calibration_band == band)
        if selected.sum() < minimum_band_rows:
            continue
        band_width = np.quantile(absolute[selected], [0.5, 0.9])
        destination = np.asarray(held_band == band)
        width50[destination] = band_width[0]
        width90[destination] = band_width[1]
    return width50, width90, np.asarray(held_band.astype(str))


def assert_split_integrity(frame: pd.DataFrame, split: RepairedSplit) -> dict[str, int]:
    """Raise on group, row, or forbidden forward-time overlap."""

    fit = frame.iloc[split.fit_index]
    held = frame.iloc[split.held_index]
    group_overlap = set(fit.group_key.astype(str)) & set(held.group_key.astype(str))
    record_overlap = set(fit.record_id.astype(int)) & set(held.record_id.astype(int))
    if group_overlap or record_overlap:
        raise AssertionError(
            f"{split.scheme}/{split.fold} leaks groups={len(group_overlap)} "
            f"records={len(record_overlap)}"
        )
    forbidden_time_rows = 0
    if split.scheme == "forward":
        train_end = int(split.metadata["train_end"])
        forbidden_time_rows = int((fit.yr > train_end).sum())
        if forbidden_time_rows:
            raise AssertionError("forward fit includes future rows")
    return {
        "group_overlap": len(group_overlap),
        "record_overlap": len(record_overlap),
        "forbidden_time_rows": forbidden_time_rows,
    }
