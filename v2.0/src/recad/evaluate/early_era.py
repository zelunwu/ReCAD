"""Strict configuration and complete-cruise splits for early-era validation."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EarlyEraFold:
    fold: int
    held_start: int
    held_end: int
    train_start: int


@dataclass(frozen=True)
class EarlyEraContract:
    contract_id: str
    targets: tuple[str, ...]
    matchup: Path
    process_features: Path
    output_root: Path
    maximum_label_year: int
    folds: tuple[EarlyEraFold, ...]
    calibration_folds: int
    calibration_fold: int
    upstream_sss_folds: int
    sss_config: tuple[float, ...]
    sss_seeds: tuple[int, ...]
    fco2_background_config: tuple[float, ...]
    fco2_catboost_config: tuple[float, ...]
    fco2_catboost_weight: float
    interval_method: str


def _required(parser: configparser.ConfigParser, section: str, key: str) -> str:
    if not parser.has_option(section, key):
        raise ValueError(f"missing required setting [{section}] {key}")
    value = parser.get(section, key).strip()
    if not value:
        raise ValueError(f"empty required setting [{section}] {key}")
    return value


def _csv_float(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(","))


def load_early_era_contract(path: Path | str) -> EarlyEraContract:
    parser = configparser.ConfigParser(interpolation=None)
    if not parser.read(Path(path), encoding="utf-8"):
        raise FileNotFoundError(path)
    folds = []
    for fold in range(3):
        section = f"fold_{fold}"
        folds.append(
            EarlyEraFold(
                fold=fold,
                held_start=parser.getint(section, "held_start"),
                held_end=parser.getint(section, "held_end"),
                train_start=parser.getint(section, "train_start"),
            )
        )
    contract = EarlyEraContract(
        contract_id=_required(parser, "contract", "contract_id"),
        targets=tuple(item.strip() for item in _required(parser, "contract", "targets").split(",")),
        matchup=Path(_required(parser, "source", "matchup")),
        process_features=Path(_required(parser, "source", "process_features")),
        output_root=Path(_required(parser, "source", "output_root")),
        maximum_label_year=parser.getint("source", "maximum_label_year"),
        folds=tuple(folds),
        calibration_folds=parser.getint("split", "calibration_folds"),
        calibration_fold=parser.getint("split", "calibration_fold"),
        upstream_sss_folds=parser.getint("split", "upstream_sss_folds"),
        sss_config=_csv_float(_required(parser, "model", "sss_config")),
        sss_seeds=tuple(int(item) for item in _required(parser, "model", "sss_seeds").split(",")),
        fco2_background_config=_csv_float(_required(parser, "model", "fco2_background_config")),
        fco2_catboost_config=_csv_float(_required(parser, "model", "fco2_catboost_config")),
        fco2_catboost_weight=parser.getfloat("model", "fco2_catboost_weight"),
        interval_method=_required(parser, "model", "interval_method"),
    )
    if contract.targets != ("SSS", "fCO2"):
        raise ValueError("early-era validation requires SSS and fCO2")
    if contract.calibration_fold >= contract.calibration_folds:
        raise ValueError("calibration fold is outside the configured range")
    previous_end = None
    for fold in contract.folds:
        if fold.held_start > fold.held_end or fold.train_start <= fold.held_end:
            raise ValueError("every fold requires an early held window and later training")
        if previous_end is not None and fold.held_start <= previous_end:
            raise ValueError("held windows must be ordered and disjoint")
        previous_end = fold.held_end
    forbidden = (
        parser.getboolean("policy", "independent_labels_opened")
        or parser.getboolean("policy", "labels_2026_opened")
        or parser.getboolean("policy", "thresholds_may_be_derived_from_this_experiment")
    )
    if forbidden:
        raise ValueError("early-era development validation cannot open or define frozen labels")
    if not parser.getboolean("policy", "later_labels_only"):
        raise ValueError("later-label-only training is mandatory")
    if not parser.getboolean("policy", "strict_nested_sss_for_fco2"):
        raise ValueError("strict nested SSS is mandatory for fCO2")
    return contract


def complete_group_indices(
    frame: pd.DataFrame, fold: EarlyEraFold
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return later train, early held, and excluded row indices without splitting cruises."""

    summary = frame.groupby("group_key", observed=True).yr.agg(["min", "max"])
    held_groups = summary.index[
        summary["min"].ge(fold.held_start) & summary["max"].le(fold.held_end)
    ]
    train_groups = summary.index[summary["min"].ge(fold.train_start)]
    held = frame.group_key.isin(held_groups).to_numpy()
    train = frame.group_key.isin(train_groups).to_numpy()
    if np.any(held & train):
        raise AssertionError("early held and later train groups overlap")
    excluded = ~(held | train)
    return np.flatnonzero(train), np.flatnonzero(held), np.flatnonzero(excluded)
