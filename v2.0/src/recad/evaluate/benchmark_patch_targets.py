"""Streaming patch target parameterizations for Issue 89."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recad.evaluate.benchmark_backends import (
    BenchmarkEstimator,
    fit_patch_dataset_estimator,
    predict_patch_dataset_estimator,
)
from recad.evaluate.benchmark_search import stable_seed
from recad.evaluate.coastal_patches import (
    CoastalPatchMemmap,
    PatchFrameDataset,
    patch_channel_statistics,
)

COMMON_SCALARS = ("year_centered", "month_sin", "month_cos")


@dataclass
class PatchTargetEstimator:
    """Patch models and training-only normalization for one target."""

    target: str
    parameters: dict[str, object]
    scalar_features: tuple[str, ...]
    channel_mean: np.ndarray
    channel_scale: np.ndarray
    direct: BenchmarkEstimator
    residual: BenchmarkEstimator | None


def scalar_features(target: str) -> tuple[str, ...]:
    if target == "SSS":
        return COMMON_SCALARS
    if target == "fCO2":
        return (*COMMON_SCALARS, "sss_crossfit")
    raise ValueError(f"unsupported target: {target}")


def _dataset(
    cache: CoastalPatchMemmap,
    frame: pd.DataFrame,
    parameters: dict[str, object],
    target: str,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> PatchFrameDataset:
    return PatchFrameDataset(
        cache,
        frame,
        sequence=int(parameters["sequence_months"]),
        patch_size=int(parameters["patch_size"]),
        scalar_features=scalar_features(target),
        channel_mean=mean,
        channel_scale=scale,
    )


def fit_patch_target_estimator(
    family: str,
    parameters: dict[str, object],
    target: str,
    train: pd.DataFrame,
    cache: CoastalPatchMemmap,
    *,
    seed: int,
) -> PatchTargetEstimator:
    """Fit direct fCO2 or direct-plus-background-residual SSS from lazy patches."""

    raw = _dataset(cache, train, parameters, target)
    keys = train.record_id.map(
        lambda record: stable_seed("issue89", target, "patch-normalization", record)
    ).to_numpy(np.int64)
    count = min(len(train), 4096)
    indices = np.argpartition(keys, count - 1)[:count]
    mean, scale = patch_channel_statistics(raw, indices)
    direct_dataset = _dataset(cache, train, parameters, target, mean, scale)
    direct = fit_patch_dataset_estimator(family, direct_dataset, parameters, seed=seed)
    residual: BenchmarkEstimator | None = None
    if target == "SSS":
        available = train.sss_bg.notna() & train.sss_bg_available.eq(1)
        residual_frame = train.loc[available].reset_index(drop=True).copy()
        residual_frame["truth"] = residual_frame.truth - residual_frame.sss_bg
        residual_dataset = _dataset(cache, residual_frame, parameters, target, mean, scale)
        residual = fit_patch_dataset_estimator(family, residual_dataset, parameters, seed=seed + 1)
    return PatchTargetEstimator(
        target,
        parameters,
        scalar_features(target),
        mean,
        scale,
        direct,
        residual,
    )


def predict_patch_target_estimator(
    model: PatchTargetEstimator,
    frame: pd.DataFrame,
    cache: CoastalPatchMemmap,
) -> np.ndarray:
    """Predict through the frozen patch target path."""

    direct_dataset = _dataset(
        cache,
        frame,
        model.parameters,
        model.target,
        model.channel_mean,
        model.channel_scale,
    )
    prediction = predict_patch_dataset_estimator(model.direct, direct_dataset)
    if model.target == "SSS":
        if model.residual is None:
            raise RuntimeError("SSS patch model misses its residual path")
        available = frame.sss_bg.notna().to_numpy() & frame.sss_bg_available.eq(1).to_numpy()
        residual_frame = frame.loc[available].reset_index(drop=True)
        residual_dataset = _dataset(
            cache,
            residual_frame,
            model.parameters,
            model.target,
            model.channel_mean,
            model.channel_scale,
        )
        residual = predict_patch_dataset_estimator(model.residual, residual_dataset)
        prediction[available] = frame.loc[available, "sss_bg"].to_numpy(float) + residual
    return prediction.astype(np.float32)
