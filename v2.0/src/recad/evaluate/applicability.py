"""Leakage-safe applicability indices and nested outer-fold evaluation.

The module deliberately separates three objects:

* a prediction model fitted with an inner grouped selection loop;
* label-free support indices fitted from the corresponding outer-training rows;
* error/risk diagnostics evaluated only after outer predictions are frozen.

It is therefore safe to use the resulting residuals for later calibration.  No
locked-test or external-independent label access is implemented here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntFlag
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EARTH_RADIUS_KM = 6371.0088
SCHEMA_VERSION = "recad_applicability_v1"
DEFAULT_K = (8, 16, 32, 64)
DEFAULT_RADII_KM = (25.0, 50.0, 100.0, 200.0)


class ReasonBit(IntFlag):
    """Stable bit vocabulary for grid-month applicability records."""

    NONE = 0
    NO_TRAINING_WITHIN_200KM = 1 << 0
    UNSEEN_LME = 1 << 1
    UNSEEN_REGIME = 1 << 2
    DISCONNECTED_COASTAL_GRAPH = 1 << 3
    ENVIRONMENTAL_EXTRAPOLATION = 1 << 4
    LOW_GROUP_DIVERSITY = 1 << 5
    MISSING_PREDICTOR = 1 << 6
    CALIBRATION_SUPPORT_SPARSE = 1 << 7
    INDEPENDENT_SUPPORT_SEALED = 1 << 8


RELIABILITY_COLUMNS = (
    "target",
    "year",
    "month",
    "coastal_node",
    "latitude",
    "longitude",
    "training_support_km",
    "calibration_support_km",
    "independent_support_km",
    "water_connected_support_km",
    "environmental_dissimilarity",
    "unique_cruises",
    "effective_groups",
    "sampled_years",
    "sampled_months",
    "support_entropy",
    "applicability_flag",
    "reason_bits",
    "model_version",
    "calibration_version",
    "evidence_stage",
)


def reliability_schema() -> dict[str, object]:
    """Return the versioned schema written with every grid-month index."""

    return {
        "schema_version": SCHEMA_VERSION,
        "columns": list(RELIABILITY_COLUMNS),
        "reason_bits": {item.name: int(item) for item in ReasonBit if item is not ReasonBit.NONE},
        "grades": {
            "A": "lowest cross-fitted predicted risk; suitable for primary product use",
            "B": "moderate calibrated risk; publish with uncertainty",
            "C": "high risk or sparse support; diagnostic use only",
            "D": "outside demonstrated applicability; suppress product value",
        },
        "independent_support_policy": (
            "independent_support_km remains null until the external validation asset is opened"
        ),
    }


def _unit_sphere(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(np.asarray(latitude, dtype=float))
    lon = np.deg2rad(np.asarray(longitude, dtype=float))
    cos_lat = np.cos(lat)
    return np.column_stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)))


def _chord_to_km(chord: np.ndarray) -> np.ndarray:
    chord = np.clip(np.asarray(chord, dtype=float), 0.0, 2.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(chord / 2.0)


def _stable_bucket(value: object, buckets: int) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % buckets


def assign_spatial_blocks(
    frame: pd.DataFrame,
    *,
    block_degrees: float = 5.0,
    n_folds: int = 5,
) -> pd.Series:
    """Assign complete fixed geographic blocks to deterministic outer folds."""

    lat_bin = np.floor((frame.latitude.to_numpy(float) + 90.0) / block_degrees).astype(int)
    lon = np.mod(frame.longitude.to_numpy(float), 360.0)
    lon_bin = np.floor(lon / block_degrees).astype(int)
    block = pd.Series(
        [f"{a}:{b}" for a, b in zip(lat_bin, lon_bin, strict=True)], index=frame.index
    )
    return block.map(lambda value: _stable_bucket(value, n_folds)).astype(int)


def assign_whole_region_folds(
    frame: pd.DataFrame,
    *,
    column: str = "lme_id",
    n_folds: int = 5,
) -> pd.Series:
    """Assign each complete region to one and only one outer fold."""

    if column not in frame:
        raise ValueError(f"missing region column: {column}")
    values = frame[column].fillna(-9999)
    return values.map(lambda value: _stable_bucket(value, n_folds)).astype(int)


@dataclass(frozen=True)
class OuterSplit:
    scheme: str
    fold: int
    fit_index: np.ndarray
    held_index: np.ndarray


def outer_splits(frame: pd.DataFrame, scheme: str, *, n_folds: int = 5) -> list[OuterSplit]:
    """Create cruise, spatial-block, whole-LME, or forward outer partitions."""

    if scheme == "cruise":
        assignment = frame.cv_fold.astype(int)
    elif scheme == "spatial_block":
        assignment = assign_spatial_blocks(frame, n_folds=n_folds)
    elif scheme == "whole_lme":
        assignment = assign_whole_region_folds(frame, n_folds=n_folds)
    elif scheme == "forward":
        if "forward_split" not in frame:
            raise ValueError("forward split requires forward_split")
        fit = np.flatnonzero(frame.forward_split.eq("train").to_numpy())
        held = np.flatnonzero(frame.forward_split.eq("development").to_numpy())
        if not len(fit) or not len(held):
            raise ValueError("forward split has an empty fit or held partition")
        return [OuterSplit(scheme, 0, fit, held)]
    else:
        raise ValueError(f"unknown outer scheme: {scheme}")

    result: list[OuterSplit] = []
    for fold in range(n_folds):
        held = np.flatnonzero(assignment.to_numpy() == fold)
        fit = np.flatnonzero(assignment.to_numpy() != fold)
        if len(fit) and len(held):
            result.append(OuterSplit(scheme, fold, fit, held))
    return result


def _cruise_equal_rmse(truth: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> float:
    values = []
    for group in np.unique(groups):
        mask = groups == group
        values.append(mean_squared_error(truth[mask], prediction[mask]))
    return float(np.sqrt(np.mean(values)))


def _prediction_pipeline(
    numeric: Sequence[str], categorical: Sequence[str], alpha: float
) -> Pipeline:
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transform = ColumnTransformer(
        [
            ("numeric", numeric_pipe, list(numeric)),
            ("categorical", categorical_pipe, list(categorical)),
        ]
    )
    return Pipeline([("transform", transform), ("ridge", Ridge(alpha=float(alpha), solver="lsqr"))])


@dataclass
class NestedRidgeResult:
    prediction: np.ndarray
    selected_alpha: float
    inner_scores: pd.DataFrame
    model: Pipeline


def nested_grouped_ridge(
    fit: pd.DataFrame,
    held: pd.DataFrame,
    *,
    numeric: Sequence[str],
    categorical: Sequence[str],
    alphas: Sequence[float],
    baseline_column: str | None = None,
    inner_splits: int = 3,
) -> NestedRidgeResult:
    """Select alpha inside ``fit`` and predict untouched ``held`` rows.

    Held labels are never read.  This property is regression-tested by
    perturbing them and requiring identical predictions and selections.
    """

    if len(alphas) < 2:
        raise ValueError("nested selection requires at least two alpha candidates")
    group_count = fit.group_key.nunique()
    if group_count < 2:
        raise ValueError("nested selection requires at least two training groups")
    n_splits = min(int(inner_splits), int(group_count))
    splitter = GroupKFold(n_splits=n_splits)
    x = fit[[*numeric, *categorical]]
    baseline = (
        fit[baseline_column].to_numpy(float) if baseline_column else np.zeros(len(fit), dtype=float)
    )
    response = fit.truth.to_numpy(float) - baseline
    rows: list[dict[str, float | int]] = []
    for alpha in alphas:
        for inner_fold, (inner_fit, inner_held) in enumerate(
            splitter.split(x, response, groups=fit.group_key)
        ):
            model = _prediction_pipeline(numeric, categorical, alpha)
            model.fit(x.iloc[inner_fit], response[inner_fit])
            prediction = model.predict(x.iloc[inner_held]) + baseline[inner_held]
            score = _cruise_equal_rmse(
                fit.truth.to_numpy(float)[inner_held],
                prediction,
                fit.group_key.to_numpy()[inner_held],
            )
            rows.append(
                {"alpha": float(alpha), "inner_fold": inner_fold, "cruise_equal_rmse": score}
            )
    scores = pd.DataFrame(rows)
    selected = float(scores.groupby("alpha").cruise_equal_rmse.mean().idxmin())
    model = _prediction_pipeline(numeric, categorical, selected)
    model.fit(x, response)
    held_baseline = (
        held[baseline_column].to_numpy(float)
        if baseline_column
        else np.zeros(len(held), dtype=float)
    )
    prediction = model.predict(held[[*numeric, *categorical]]) + held_baseline
    return NestedRidgeResult(np.asarray(prediction), selected, scores, model)


def _effective_group_statistics(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = np.empty(len(codes), dtype=np.int16)
    effective = np.empty(len(codes), dtype=np.float32)
    entropy = np.empty(len(codes), dtype=np.float32)
    for row, values in enumerate(codes):
        _, counts = np.unique(values, return_counts=True)
        probabilities = counts / counts.sum()
        unique[row] = len(counts)
        effective[row] = 1.0 / np.sum(probabilities**2)
        entropy[row] = -np.sum(probabilities * np.log(probabilities)) / math.log(len(values))
    return unique, effective, entropy


class SupportIndex:
    """Training-only geographic, environmental, regional, and diversity index."""

    def __init__(
        self,
        *,
        environmental_columns: Sequence[str],
        k_values: Sequence[int] = DEFAULT_K,
        radii_km: Sequence[float] = DEFAULT_RADII_KM,
    ):
        self.environmental_columns = tuple(environmental_columns)
        self.k_values = tuple(sorted({int(value) for value in k_values}))
        self.radii_km = tuple(sorted({float(value) for value in radii_km}))
        if min(self.k_values) < 1:
            raise ValueError("k must be positive")

    def fit(self, frame: pd.DataFrame) -> SupportIndex:
        required = {
            "latitude",
            "longitude",
            "group_key",
            "year",
            "month",
            "lme_id",
            "basin_id",
            "regime_id",
            *self.environmental_columns,
        }
        missing = required - set(frame)
        if missing:
            raise ValueError(f"support index is missing {sorted(missing)}")
        self.frame = frame.reset_index(drop=True).copy()
        self.geo_tree = cKDTree(_unit_sphere(self.frame.latitude, self.frame.longitude))
        environment = self.frame[list(self.environmental_columns)].to_numpy(float)
        self.environment_median = np.nanmedian(environment, axis=0)
        environment = np.where(np.isfinite(environment), environment, self.environment_median)
        self.environment_mean = environment.mean(axis=0)
        self.environment_scale = environment.std(axis=0)
        self.environment_scale[self.environment_scale < 1e-8] = 1.0
        standardized = (environment - self.environment_mean) / self.environment_scale
        self.environment_tree = cKDTree(standardized)
        self.group_codes, _ = pd.factorize(self.frame.group_key, sort=True)
        self.year_codes = self.frame.year.to_numpy(int)
        self.month_codes = self.frame.month.to_numpy(int)
        self.region_counts: dict[str, dict[tuple[int, ...], int]] = {}
        for name, columns in {
            "lme_month": ["lme_id", "month"],
            "basin_month": ["basin_id", "month"],
            "regime_month": ["regime_id", "month"],
        }.items():
            counts = self.frame.groupby(columns, dropna=False).size()
            self.region_counts[name] = {
                tuple(map(int, key)): int(value) for key, value in counts.items()
            }
        return self

    def query(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "frame"):
            raise RuntimeError("fit the support index before query")
        max_k = min(max(self.k_values), len(self.frame))
        chord, neighbours = self.geo_tree.query(
            _unit_sphere(frame.latitude, frame.longitude), k=max_k, workers=-1
        )
        if max_k == 1:
            chord, neighbours = chord[:, None], neighbours[:, None]
        distances = _chord_to_km(chord)
        result = pd.DataFrame(index=frame.index)
        result["geographic_km"] = distances[:, 0]
        for radius in self.radii_km:
            result[f"count_within_{int(radius)}km"] = np.sum(distances <= radius, axis=1)
        for k in self.k_values:
            usable = min(k, max_k)
            result[f"geographic_mean_k{k}_km"] = distances[:, :usable].mean(axis=1)

        diversity_k = min(16, max_k)
        local = neighbours[:, :diversity_k]
        unique, effective, entropy = _effective_group_statistics(self.group_codes[local])
        result["unique_cruises"] = unique
        result["effective_groups"] = effective
        result["support_entropy"] = entropy
        result["sampled_years"] = [len(np.unique(values)) for values in self.year_codes[local]]
        result["sampled_months"] = [len(np.unique(values)) for values in self.month_codes[local]]

        environment = frame[list(self.environmental_columns)].to_numpy(float)
        result["missing_predictor"] = ~np.isfinite(environment).all(axis=1)
        environment = np.where(np.isfinite(environment), environment, self.environment_median)
        standardized = (environment - self.environment_mean) / self.environment_scale
        env_distance, _ = self.environment_tree.query(standardized, k=max_k, workers=-1)
        if max_k == 1:
            env_distance = env_distance[:, None]
        for k in self.k_values:
            usable = min(k, max_k)
            result[f"environment_k{k}"] = env_distance[:, :usable].mean(axis=1)

        for name, columns in {
            "lme_month": ["lme_id", "month"],
            "basin_month": ["basin_id", "month"],
            "regime_month": ["regime_id", "month"],
        }.items():
            mapping = self.region_counts[name]
            keys = zip(
                *(frame[column].fillna(-9999).astype(int) for column in columns), strict=True
            )
            result[f"{name}_count"] = [mapping.get(tuple(key), 0) for key in keys]
        return result.reset_index(drop=True)


class CoastalGraphIndex:
    """Weighted shortest paths on the frozen no-land-crossing coastal graph."""

    def __init__(self, graph: sparse.csr_matrix, latitude: np.ndarray, longitude: np.ndarray):
        graph = graph.tocsr()
        if graph.shape != (len(latitude), len(latitude)):
            raise ValueError("graph and coordinates have inconsistent sizes")
        rows = np.repeat(np.arange(graph.shape[0]), np.diff(graph.indptr))
        cols = graph.indices
        source = _unit_sphere(np.asarray(latitude)[rows], np.asarray(longitude)[rows])
        target = _unit_sphere(np.asarray(latitude)[cols], np.asarray(longitude)[cols])
        weights = _chord_to_km(np.linalg.norm(source - target, axis=1))
        self.graph = sparse.csr_matrix((weights, cols, graph.indptr.copy()), shape=graph.shape)
        self.latitude = np.asarray(latitude, dtype=float)
        self.longitude = np.asarray(longitude, dtype=float)

    @classmethod
    def from_frozen(cls, graph_path: str | Path, support_path: str | Path) -> CoastalGraphIndex:
        with np.load(graph_path) as payload:
            graph = sparse.csr_matrix(
                (payload["data"], payload["indices"], payload["indptr"]),
                shape=tuple(payload["shape"]),
            )
        with xr.open_dataset(support_path) as dataset:
            latitude = dataset.latitude.load().to_numpy()
            longitude = dataset.longitude.load().to_numpy()
        return cls(graph, latitude, longitude)

    def distance_to_sources(self, source_nodes: Iterable[int]) -> np.ndarray:
        sources = np.unique(np.fromiter((int(node) for node in source_nodes), dtype=int))
        sources = sources[(sources >= 0) & (sources < self.graph.shape[0])]
        if not len(sources):
            return np.full(self.graph.shape[0], np.inf)
        return np.asarray(
            dijkstra(self.graph, directed=False, indices=sources, min_only=True), dtype=float
        )


def add_label_free_risk_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Add comparable high-is-risky scores without consulting residual labels."""

    result = frame.copy()
    result["risk_geographic"] = np.log1p(result.geographic_km.clip(lower=0))
    regional_count = result.lme_month_count + result.basin_month_count + result.regime_month_count
    result["risk_region"] = 1.0 / np.sqrt(regional_count.clip(lower=0) + 1.0)
    result["risk_graph"] = np.log1p(result.water_connected_km.replace(np.inf, np.nan))
    result["risk_graph"] = result.risk_graph.fillna(result.risk_graph.max() + 1.0)
    for k in DEFAULT_K:
        result[f"risk_environment_k{k}"] = result[f"environment_k{k}"]
    components = ["risk_geographic", "risk_region", "risk_graph", "risk_environment_k16"]
    ranks = [result[column].rank(pct=True, method="average") for column in components]
    result["risk_hybrid"] = np.mean(np.column_stack(ranks), axis=1)
    return result


def risk_coverage_curve(
    truth: Sequence[float],
    prediction: Sequence[float],
    risk: Sequence[float],
    *,
    coverages: Sequence[float] = tuple(np.linspace(0.1, 1.0, 10)),
) -> pd.DataFrame:
    """Evaluate selective RMSE after retaining the lowest-risk fraction."""

    truth_array = np.asarray(truth, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    risk_array = np.asarray(risk, dtype=float)
    valid = np.isfinite(truth_array) & np.isfinite(prediction_array) & np.isfinite(risk_array)
    truth_array, prediction_array, risk_array = (
        truth_array[valid],
        prediction_array[valid],
        risk_array[valid],
    )
    order = np.argsort(risk_array, kind="stable")
    squared_error = (prediction_array[order] - truth_array[order]) ** 2
    rows = []
    for coverage in coverages:
        n = max(1, math.ceil(len(order) * float(coverage)))
        rows.append(
            {
                "coverage": float(coverage),
                "retained": n,
                "rmse": float(np.sqrt(np.mean(squared_error[:n]))),
                "risk_threshold": float(risk_array[order[n - 1]]),
            }
        )
    return pd.DataFrame(rows)


def risk_curve_auc(curve: pd.DataFrame) -> float:
    """Trapezoidal area under the normalized selective-RMSE curve."""

    ordered = curve.sort_values("coverage")
    scale = float(ordered.loc[ordered.coverage.idxmax(), "rmse"])
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    x = np.concatenate(([0.0], ordered.coverage.to_numpy(float)))
    y = np.concatenate(([ordered.rmse.iloc[0]], ordered.rmse.to_numpy(float))) / scale
    return float(np.trapezoid(y, x))


def support_error_summary(frame: pd.DataFrame, risk_columns: Sequence[str]) -> pd.DataFrame:
    """Summarize ranking, monotonicity, and risk-coverage for each method."""

    rows = []
    absolute_error = np.abs(frame.prediction.to_numpy(float) - frame.truth.to_numpy(float))
    for column in risk_columns:
        valid = np.isfinite(frame[column]) & np.isfinite(absolute_error)
        subset = frame.loc[valid]
        error = absolute_error[valid]
        if len(subset) < 10:
            continue
        rank_error = pd.Series(error).rank().to_numpy()
        rank_risk = subset[column].rank().to_numpy()
        spearman = float(np.corrcoef(rank_error, rank_risk)[0, 1])
        strata = pd.qcut(subset[column].rank(method="first"), 5, labels=False)
        stratum_rmse = []
        for stratum in range(5):
            mask = strata.to_numpy() == stratum
            stratum_rmse.append(float(np.sqrt(np.mean(error[mask] ** 2))))
        monotonic_steps = float(np.mean(np.diff(stratum_rmse) >= 0))
        curve = risk_coverage_curve(subset.truth, subset.prediction, subset[column])
        rows.append(
            {
                "method": column.removeprefix("risk_"),
                "n": len(subset),
                "spearman_abs_error": spearman,
                "monotonic_steps": monotonic_steps,
                "risk_curve_auc": risk_curve_auc(curve),
                **{f"q{i + 1}_rmse": value for i, value in enumerate(stratum_rmse)},
            }
        )
    return pd.DataFrame(rows)


def validate_reliability_frame(frame: pd.DataFrame) -> None:
    missing = set(RELIABILITY_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"reliability frame is missing {sorted(missing)}")
    invalid = set(frame.applicability_flag.dropna()) - {"A", "B", "C", "D"}
    if invalid:
        raise ValueError(f"invalid applicability flags: {sorted(invalid)}")
    if frame.loc[frame.evidence_stage.eq("development"), "independent_support_km"].notna().any():
        raise ValueError("development records cannot expose independent-validation support")


def applicability_gate_passed(decisions: pd.DataFrame) -> bool:
    """Apply Issue #24's global gate while preserving target-level failures."""

    if "monotonic_method_exists" not in decisions:
        raise ValueError("decision table lacks monotonic_method_exists")
    return bool(decisions.monotonic_method_exists.astype(bool).any())


def schema_json() -> str:
    return json.dumps(reliability_schema(), indent=2, sort_keys=True)
