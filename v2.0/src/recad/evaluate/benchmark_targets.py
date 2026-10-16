"""Target parameterizations shared by every Issue 89 tabular family."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recad.evaluate.benchmark_backends import (
    BenchmarkEstimator,
    fit_tabular_estimator,
    predict_tabular_estimator,
)
from recad.evaluate.model_zoo import TabularTransform


@dataclass
class TransformedEstimator:
    """Training-only preprocessing paired with one backend estimator."""

    transform: TabularTransform
    model: BenchmarkEstimator


@dataclass
class TargetEstimator:
    """A direct model and optional background-residual model for one target."""

    target: str
    direct: TransformedEstimator
    residual: TransformedEstimator | None = None


def _fit_one(
    family: str,
    parameters: dict[str, object],
    train: pd.DataFrame,
    features: tuple[str, ...],
    target: np.ndarray,
    seed: int,
) -> TransformedEstimator:
    transform = TabularTransform.fit(train, features)
    model = fit_tabular_estimator(
        family,
        transform.apply(train),
        np.asarray(target, dtype=np.float32),
        parameters,
        seed=seed,
    )
    return TransformedEstimator(transform, model)


def fit_target_estimator(
    family: str,
    parameters: dict[str, object],
    target: str,
    train: pd.DataFrame,
    features: tuple[str, ...],
    *,
    seed: int,
) -> TargetEstimator:
    """Fit direct fCO2 or direct-plus-background-residual SSS."""

    direct = _fit_one(
        family,
        parameters,
        train,
        features,
        train.truth.to_numpy(float),
        seed,
    )
    if target != "SSS":
        if target != "fCO2":
            raise ValueError(f"unsupported target: {target}")
        return TargetEstimator(target, direct)
    available = train.sss_bg.notna() & train.sss_bg_available.eq(1)
    residual_train = train.loc[available].reset_index(drop=True)
    residual = _fit_one(
        family,
        parameters,
        residual_train,
        features,
        residual_train.truth.to_numpy(float) - residual_train.sss_bg.to_numpy(float),
        seed + 1,
    )
    return TargetEstimator(target, direct, residual)


def _predict_one(estimator: TransformedEstimator, frame: pd.DataFrame) -> np.ndarray:
    return predict_tabular_estimator(estimator.model, estimator.transform.apply(frame))


def predict_target_estimator(model: TargetEstimator, frame: pd.DataFrame) -> np.ndarray:
    """Predict with the frozen target parameterization and identical row support."""

    prediction = _predict_one(model.direct, frame)
    if model.target == "SSS":
        if model.residual is None:
            raise RuntimeError("SSS estimator misses its background-residual path")
        available = frame.sss_bg.notna().to_numpy() & frame.sss_bg_available.eq(1).to_numpy()
        residual = _predict_one(model.residual, frame.loc[available])
        prediction[available] = frame.loc[available, "sss_bg"].to_numpy(float) + residual
    return prediction.astype(np.float32)
