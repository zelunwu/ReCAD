"""Nested support, interval, and grade utilities for the Issue 44 contract."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_RADIUS_KM = 6371.0088


def _sphere(frame: pd.DataFrame) -> np.ndarray:
    lat = np.deg2rad(frame.latitude.to_numpy(float))
    lon = np.deg2rad(frame.longitude.to_numpy(float))
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def _km(chord: np.ndarray) -> np.ndarray:
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord, 0, 2) / 2)


@dataclass
class DeployableSupportIndex:
    """Training-only geographic, diversity, and environmental support index."""

    environmental_columns: tuple[str, ...]
    k: int = 16

    def fit(self, frame: pd.DataFrame) -> DeployableSupportIndex:
        self.frame = frame.reset_index(drop=True)
        self.geo = cKDTree(_sphere(self.frame))
        environment = self.frame[list(self.environmental_columns)].to_numpy(float)
        self.median = np.nanmedian(environment, axis=0)
        environment = np.where(np.isfinite(environment), environment, self.median)
        self.mean = environment.mean(axis=0)
        self.scale = environment.std(axis=0)
        self.scale = np.where(self.scale > 1e-8, self.scale, 1.0)
        self.environment = cKDTree((environment - self.mean) / self.scale)
        self.group_codes, _ = pd.factorize(self.frame.group_key.astype(str), sort=True)
        self.max_year = int(self.frame.yr.max())
        self.min_year = int(self.frame.yr.min())
        return self

    def query(self, frame: pd.DataFrame) -> pd.DataFrame:
        usable = min(self.k, len(self.frame))
        distance, neighbour = self.geo.query(_sphere(frame), k=usable, workers=-1)
        if usable == 1:
            distance, neighbour = distance[:, None], neighbour[:, None]
        distance = _km(distance)
        result = pd.DataFrame(index=frame.index)
        result["geographic_km"] = distance[:, 0]
        unique, effective, entropy = [], [], []
        for codes in self.group_codes[neighbour]:
            _, count = np.unique(codes, return_counts=True)
            probability = count / count.sum()
            unique.append(len(count))
            effective.append(1 / np.sum(probability**2))
            entropy.append(-np.sum(probability * np.log(probability)) / np.log(max(2, usable)))
        result["unique_cruises"] = unique
        result["effective_groups"] = effective
        result["support_entropy"] = entropy
        environment = frame[list(self.environmental_columns)].to_numpy(float)
        result["missing_mandatory_input"] = ~np.isfinite(environment[:, :3]).all(axis=1)
        environment = np.where(np.isfinite(environment), environment, self.median)
        environmental_distance, _ = self.environment.query(
            (environment - self.mean) / self.scale, k=usable, workers=-1
        )
        if usable == 1:
            environmental_distance = environmental_distance[:, None]
        result["environment_k16"] = environmental_distance.mean(axis=1)
        year = frame.yr.to_numpy(int)
        result["time_extrapolation_years"] = np.maximum(
            np.maximum(year - self.max_year, self.min_year - year), 0
        )
        return result.reset_index(drop=True)


def stable_group_fold(groups: pd.Series, *, salt: str, folds: int = 3) -> np.ndarray:
    mapping = {}
    for group in groups.astype(str).unique():
        digest = hashlib.sha256(f"{salt}:{group}".encode()).digest()
        mapping[group] = int.from_bytes(digest[:8], "little") % folds
    return groups.astype(str).map(mapping).to_numpy(np.int8)


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    finite = np.sort(np.asarray(scores, float)[np.isfinite(scores)])
    if len(finite) == 0:
        raise ValueError("no finite conformity scores")
    rank = min(len(finite) - 1, max(0, int(np.ceil((len(finite) + 1) * coverage)) - 1))
    return float(finite[rank])


def scaled_widths(
    truth: np.ndarray,
    prediction: np.ndarray,
    calibration_scale: np.ndarray,
    destination_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scale = np.maximum(np.asarray(calibration_scale, float), 1e-3)
    score = np.abs(np.asarray(truth, float) - np.asarray(prediction, float)) / scale
    destination = np.maximum(np.asarray(destination_scale, float), 1e-3)
    return conformal_quantile(score, 0.5) * destination, conformal_quantile(
        score, 0.9
    ) * destination


def interval_wis(
    truth: np.ndarray,
    prediction: np.ndarray,
    width50: np.ndarray,
    width90: np.ndarray,
) -> float:
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    score = 0.5 * np.abs(truth - prediction)
    for alpha, weight, width in ((0.5, 0.25, width50), (0.1, 0.05, width90)):
        width = np.asarray(width, float)
        lower, upper = prediction - width, prediction + width
        interval = 2 * width + 2 / alpha * np.maximum(lower - truth, 0)
        interval += 2 / alpha * np.maximum(truth - upper, 0)
        score += weight * interval
    return float(np.mean(score / 0.8))


def assign_grade(
    risk: np.ndarray,
    *,
    thresholds: tuple[float, float, float],
    geographic_km: np.ndarray,
    environment_k16: np.ndarray,
    effective_groups: np.ndarray,
    missing: np.ndarray,
    geographic_max: float,
    environment_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    risk = np.asarray(risk, float)
    grade = np.full(len(risk), "D", dtype="U1")
    grade[risk <= thresholds[2]] = "C"
    grade[risk <= thresholds[1]] = "B"
    grade[risk <= thresholds[0]] = "A"
    reason = np.zeros(len(risk), dtype=np.uint16)
    geographic = np.asarray(geographic_km, float)
    environment = np.asarray(environment_k16, float)
    effective = np.asarray(effective_groups, float)
    reason[geographic > 200] |= 1
    reason[environment > environment_max] |= 16
    reason[effective < 2] |= 32
    reason[np.asarray(missing, bool)] |= 64
    hard = (
        (geographic > geographic_max) | (environment > environment_max) | np.asarray(missing, bool)
    )
    grade[hard] = "D"
    return grade, reason
