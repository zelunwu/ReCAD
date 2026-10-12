"""Frozen split, metric, and gate helpers for predictor ablations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AblationVariant:
    """One controlled feature set with an optional training-only dropout rule."""

    name: str
    features: tuple[str, ...]
    dropout_sss: float = 0.0
    dropout_adt: float = 0.0


@dataclass(frozen=True)
class FrozenSplit:
    """One outer fit/held partition."""

    scheme: str
    fold: int
    fit_index: np.ndarray
    held_index: np.ndarray


def stable_fold(value: object, salt: str, n_folds: int = 5) -> int:
    """Map a value to a stable hash fold."""

    payload = f"{salt}:{value}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % n_folds


def frozen_splits(frame: pd.DataFrame, scheme: str, n_folds: int = 5) -> list[FrozenSplit]:
    """Return cruise, spatial, whole-LME, or forward partitions without leakage."""

    if scheme in {"cruise", "spatial"}:
        column = "cruise_fold" if scheme == "cruise" else "spatial_fold"
        assignment = frame[column].to_numpy(int)
        groups = frame.group_key.astype(str).to_numpy()
        result = []
        for fold in range(n_folds):
            held = assignment == fold
            held_groups = set(groups[held])
            fit = (assignment != fold) & ~np.isin(groups, list(held_groups))
            if held.any() and fit.any():
                result.append(FrozenSplit(scheme, fold, np.flatnonzero(fit), np.flatnonzero(held)))
        return result
    if scheme == "whole_lme":
        lme = frame.lme_id.fillna(-1).to_numpy(int)
        unique_lme, inverse = np.unique(lme, return_inverse=True)
        unique_fold = np.array(
            [stable_fold(value, "recad-v2.3-whole-lme", n_folds) for value in unique_lme],
            dtype=int,
        )
        lme_fold = unique_fold[inverse]
        groups = frame.group_key.astype(str).to_numpy()
        result = []
        for fold in range(n_folds):
            held = lme_fold == fold
            held_groups = set(groups[held])
            fit = (lme_fold != fold) & ~np.isin(groups, list(held_groups))
            if held.any() and fit.any():
                result.append(FrozenSplit(scheme, fold, np.flatnonzero(fit), np.flatnonzero(held)))
        return result
    if scheme == "forward":
        role = frame.forward_role.astype(str)
        held_mask = role.isin(["forward_development", "late_development"]).to_numpy()
        held_groups = set(frame.loc[held_mask, "group_key"].astype(str))
        fit_mask = role.eq("forward_train").to_numpy() & ~frame.group_key.astype(str).isin(
            held_groups
        )
        fit = np.flatnonzero(fit_mask)
        held = np.flatnonzero(held_mask)
        if not len(fit) or not len(held):
            raise ValueError("forward split has an empty partition")
        return [FrozenSplit(scheme, 0, fit, held)]
    raise ValueError(f"unknown split scheme: {scheme}")


def balanced_sample_weight(frame: pd.DataFrame) -> np.ndarray:
    """Equalize LME then cruise influence while retaining every training row."""

    region = frame.lme_id.fillna(-1)
    region_count = max(region.nunique(), 1)
    cruises = frame.groupby(region, observed=True).group_key.transform("nunique")
    rows = frame.groupby([region, frame.group_key], observed=True).group_key.transform("size")
    weight = 1.0 / (region_count * cruises.to_numpy(float) * rows.to_numpy(float))
    return weight / weight.mean()


def regression_metrics(
    truth: np.ndarray, prediction: np.ndarray, groups: np.ndarray
) -> dict[str, float | int]:
    """Return pooled and cruise-equal error metrics."""

    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - truth
    denominator = np.sum((truth - truth.mean()) ** 2)
    codes, unique_groups = pd.factorize(groups, sort=False)
    group_sum = np.bincount(codes, weights=error**2, minlength=len(unique_groups))
    group_count = np.bincount(codes, minlength=len(unique_groups))
    group_mse = group_sum / group_count
    return {
        "n": len(truth),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1 - np.sum(error**2) / denominator) if denominator else np.nan,
        "cruise_equal_rmse": float(np.sqrt(np.mean(group_mse))),
    }


def paired_group_bootstrap(
    frame: pd.DataFrame,
    candidate: str,
    reference: str,
    *,
    replicates: int = 2000,
    seed: int = 54023,
) -> dict[str, float | int]:
    """Bootstrap paired squared-error differences at cruise level."""

    work = frame[["group_key", "truth", candidate, reference]].dropna()
    work["difference"] = (work[candidate] - work.truth) ** 2 - (work[reference] - work.truth) ** 2
    group = work.groupby("group_key", observed=True).difference.mean().to_numpy(float)
    if not len(group):
        raise ValueError("paired bootstrap has no groups")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 100):
        count = min(100, replicates - start)
        index = rng.integers(0, len(group), size=(count, len(group)))
        draws[start : start + count] = group[index].mean(axis=1)
    return {
        "groups": len(group),
        "replicates": replicates,
        "mean_delta_mse": float(group.mean()),
        "ci_low_delta_mse": float(np.quantile(draws, 0.025)),
        "ci_high_delta_mse": float(np.quantile(draws, 0.975)),
        "probability_improvement": float(np.mean(draws < 0)),
    }


def qualifies_addition(effect: pd.DataFrame) -> tuple[bool, list[str]]:
    """Apply the frozen deployable single-factor gate to one predictor group."""

    by_scheme = effect.set_index("scheme").relative_rmse_improvement.to_dict()
    reasons = []
    if float(np.median(list(by_scheme.values()))) < 0.005:
        reasons.append("median_improvement_below_0.5pct")
    for scheme in ("cruise", "spatial"):
        if by_scheme.get(scheme, -np.inf) <= 0:
            reasons.append(f"{scheme}_not_improved")
    for scheme in ("whole_lme", "forward_2019_2021", "forward_2022_2025"):
        if by_scheme.get(scheme, -np.inf) < -0.01:
            reasons.append(f"{scheme}_degrades_over_1pct")
    primary = effect.loc[effect.scheme.isin(["cruise", "spatial"])]
    if (primary.common_relative_rmse_improvement <= 0).any():
        reasons.append("common_support_not_improved")
    if (primary.bootstrap_ci_high_delta_mse >= 0).any():
        reasons.append("paired_bootstrap_not_significant")
    return not reasons, reasons
