"""Cruise-grouped conditional physical-error estimation utilities."""

from __future__ import annotations

import configparser
import hashlib
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom


@dataclass(frozen=True)
class ConditionalErrorContract:
    contract_id: str
    targets: tuple[str, ...]
    protocols: tuple[str, ...]
    residual_bank_root: Path
    residual_bank_pattern: str
    output_root: Path
    quantile: float
    confidence: float
    minimum_calibration_groups: int
    group_folds: int
    calibration_lag: int
    split_salt: str
    features: tuple[str, ...]
    candidates: tuple[str, ...]
    stratified_minimum_rows: int
    stratified_minimum_groups: int
    stratified_edges: dict[str, tuple[float, ...]]
    prediction_edges: dict[str, tuple[float, ...]]
    catboost_config: tuple[int | float, ...]
    pooled_row_coverage_min: float
    minimum_fold_row_coverage: float
    pooled_group_q90_coverage_min: float


def _required(parser: configparser.ConfigParser, section: str, key: str) -> str:
    if not parser.has_option(section, key):
        raise ValueError(f"missing required setting [{section}] {key}")
    value = parser.get(section, key).strip()
    if not value:
        raise ValueError(f"empty required setting [{section}] {key}")
    return value


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError("CSV settings must contain unique values")
    return values


def _edges(value: str) -> tuple[float, ...]:
    edges = tuple(float(item.strip()) for item in value.split(","))
    if len(edges) < 2 or edges[0] != 0 or not math.isinf(edges[-1]):
        raise ValueError("stratified edges must start at zero and end at inf")
    if any(right <= left for left, right in itertools.pairwise(edges)):
        raise ValueError("stratified edges must increase")
    return edges


def load_conditional_error_contract(path: Path | str) -> ConditionalErrorContract:
    parser = configparser.ConfigParser(interpolation=None)
    if not parser.read(Path(path), encoding="utf-8"):
        raise FileNotFoundError(path)
    contract = ConditionalErrorContract(
        contract_id=_required(parser, "contract", "contract_id"),
        targets=_csv(_required(parser, "contract", "targets")),
        protocols=_csv(_required(parser, "contract", "protocols")),
        residual_bank_root=Path(_required(parser, "source", "residual_bank_root")),
        residual_bank_pattern=_required(parser, "source", "residual_bank_pattern"),
        output_root=Path(_required(parser, "source", "output_root")),
        quantile=parser.getfloat("estimand", "conditional_quantile"),
        confidence=parser.getfloat("estimand", "confidence_level"),
        minimum_calibration_groups=parser.getint("estimand", "minimum_calibration_groups"),
        group_folds=parser.getint("crossfit", "group_folds"),
        calibration_lag=parser.getint("crossfit", "calibration_lag"),
        split_salt=_required(parser, "crossfit", "split_salt"),
        features=_csv(_required(parser, "features", "fields")),
        candidates=_csv(_required(parser, "candidates", "names")),
        stratified_minimum_rows=parser.getint("candidates", "stratified_minimum_rows"),
        stratified_minimum_groups=parser.getint("candidates", "stratified_minimum_groups"),
        stratified_edges={
            "geographic_km": _edges(_required(parser, "candidates", "stratified_geographic_edges")),
            "environment_k16": _edges(
                _required(parser, "candidates", "stratified_environment_edges")
            ),
            "unique_cruises": _edges(_required(parser, "candidates", "stratified_cruise_edges")),
            "time_extrapolation_years": _edges(
                _required(parser, "candidates", "stratified_time_edges")
            ),
        },
        prediction_edges={
            target: tuple(
                float(item.strip())
                for item in _required(parser, f"target:{target}", "prediction_edges").split(",")
            )
            for target in ("SSS", "fCO2")
        },
        catboost_config=(
            parser.getint("candidates", "catboost_iterations"),
            parser.getint("candidates", "catboost_depth"),
            parser.getfloat("candidates", "catboost_learning_rate"),
            parser.getfloat("candidates", "catboost_l2_leaf_reg"),
        ),
        pooled_row_coverage_min=parser.getfloat("selection", "pooled_row_coverage_min"),
        minimum_fold_row_coverage=parser.getfloat("selection", "minimum_fold_row_coverage"),
        pooled_group_q90_coverage_min=parser.getfloat("selection", "pooled_group_q90_coverage_min"),
    )
    if contract.targets != ("SSS", "fCO2") or len(contract.protocols) != 5:
        raise ValueError("the frozen contract requires two targets and five protocols")
    if contract.candidates != ("constant", "stratified", "catboost_quantile"):
        raise ValueError("candidate set drift")
    for edges in contract.prediction_edges.values():
        if not math.isinf(edges[0]) or edges[0] > 0 or not math.isinf(edges[-1]):
            raise ValueError("prediction edges must span negative to positive infinity")
        if any(right <= left for left, right in itertools.pairwise(edges)):
            raise ValueError("prediction edges must increase")
    if not 0 < contract.quantile < 1 or not 0 < contract.confidence < 1:
        raise ValueError("quantile and confidence must lie between zero and one")
    if contract.group_folds < 3 or not 0 < contract.calibration_lag < contract.group_folds:
        raise ValueError("invalid group cross-fitting settings")
    prohibited = (
        parser.getboolean("policy", "pool_protocols_for_training")
        or parser.getboolean("policy", "basin_is_model_feature")
        or parser.getboolean("policy", "thresholds_may_be_derived_from_this_experiment")
        or parser.getboolean("policy", "independent_labels_opened")
        or parser.getboolean("policy", "labels_2026_opened")
    )
    if prohibited or not parser.getboolean("policy", "combine_protocols_by_upper_envelope"):
        raise ValueError("conditional-error safety policy was weakened")
    return contract


def stable_group_folds(groups: pd.Series, *, salt: str, folds: int) -> np.ndarray:
    mapping = {}
    for group in groups.astype(str).unique():
        digest = hashlib.sha256(f"{salt}:{group}".encode()).digest()
        mapping[group] = int.from_bytes(digest[:8], "little") % folds
    return groups.astype(str).map(mapping).to_numpy(np.int8)


def cruise_equal_weights(groups: pd.Series) -> np.ndarray:
    count = groups.astype(str).groupby(groups.astype(str)).transform("size").to_numpy(float)
    weights = 1.0 / count
    return weights / weights.mean()


def prepare_features(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    output = frame.loc[:, features].copy()
    for field in (
        "geographic_km",
        "environment_k16",
        "model_disagreement",
        "background_disagreement",
    ):
        if field in output:
            output[f"log1p_{field}"] = np.log1p(np.maximum(output.pop(field), 0))
    if "latitude" in output:
        latitude = np.deg2rad(output.pop("latitude").to_numpy(float))
        output["latitude_sin"] = np.sin(latitude)
        output["latitude_cos"] = np.cos(latitude)
    if "longitude" in output:
        longitude = np.deg2rad(output.pop("longitude").to_numpy(float))
        output["longitude_sin"] = np.sin(longitude)
        output["longitude_cos"] = np.cos(longitude)
    if "mon" in output:
        month = output.pop("mon").to_numpy(float)
        output["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
        output["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    return output.astype("float32")


def calibration_offset(
    absolute_error: np.ndarray,
    raw_bound: np.ndarray,
    groups: np.ndarray,
    *,
    quantile: float,
    confidence: float,
    minimum_groups: int,
) -> tuple[float, int, int]:
    frame = pd.DataFrame(
        {
            "group": np.asarray(groups).astype(str),
            "score": np.asarray(absolute_error, float) - np.asarray(raw_bound, float),
        }
    )
    group_scores = frame.groupby("group", observed=True).score.quantile(
        quantile, interpolation="higher"
    )
    ordered = np.sort(group_scores.to_numpy(float))
    count = len(ordered)
    rank = int(binom.ppf(confidence, count, quantile)) + 1 if count else 0
    if count < minimum_groups or rank > count:
        return float("nan"), count, rank
    return max(float(ordered[rank - 1]), 0.0), count, rank


def error_bound_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    error = frame.absolute_error.to_numpy(float)
    bound = frame.error_bound.to_numpy(float)
    groups = frame.group_key.astype(str)
    group_covered = (
        pd.DataFrame({"group": groups, "covered": error <= bound})
        .groupby("group", observed=True)
        .covered.mean()
        .ge(0.9)
    )
    return {
        "rows": len(frame),
        "groups": int(groups.nunique()),
        "row_coverage": float(np.mean(error <= bound)),
        "group_q90_coverage": float(group_covered.mean()),
        "mean_bound": float(np.mean(bound)),
        "median_bound": float(np.median(bound)),
        "q90_bound": float(np.quantile(bound, 0.9)),
    }
