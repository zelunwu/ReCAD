"""Frozen residual-bank contract and distribution-free hierarchical bounds."""

from __future__ import annotations

import configparser
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import binom


@dataclass(frozen=True)
class ResidualBankContract:
    contract_id: str
    status: str
    targets: tuple[str, ...]
    required_protocols: tuple[str, ...]
    available_protocols: tuple[str, ...]
    partition_root: Path
    partition_pattern: str
    geographic_provenance: Path
    output_root: Path
    truth_column: str
    prediction_column: str
    group_column: str
    node_column: str
    quantile: float
    confidence: float
    minimum_groups_for_finite_bound: int
    minimum_rows_per_cohort: int
    minimum_groups_per_cohort: int
    strata_edges: dict[str, tuple[float, ...]]
    pool_protocols: bool
    basin_may_change_error_tier: bool
    missing_protocol_blocks_final_grading: bool
    thresholds_may_be_derived_from_this_bank: bool
    independent_labels_opened: bool


def _required(parser: configparser.ConfigParser, section: str, key: str) -> str:
    if not parser.has_option(section, key):
        raise ValueError(f"missing required setting [{section}] {key}")
    value = parser.get(section, key).strip()
    if not value:
        raise ValueError(f"empty required setting [{section}] {key}")
    return value


def _csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items or len(items) != len(set(items)):
        raise ValueError(f"expected a non-empty unique CSV list, got {value!r}")
    return items


def _edges(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(","))
    if len(parsed) < 2 or parsed[0] != 0 or not math.isinf(parsed[-1]):
        raise ValueError("stratum edges must start at zero and end at inf")
    if any(right <= left for left, right in itertools.pairwise(parsed)):
        raise ValueError("stratum edges must be strictly increasing")
    return parsed


def load_residual_bank_contract(path: Path | str) -> ResidualBankContract:
    parser = configparser.ConfigParser(interpolation=None)
    if not parser.read(Path(path), encoding="utf-8"):
        raise FileNotFoundError(path)
    required = _csv(_required(parser, "contract", "required_protocols"))
    available = _csv(_required(parser, "contract", "available_protocols"))
    if not set(available).issubset(required):
        raise ValueError("available protocols must be a subset of required protocols")
    quantile = parser.getfloat("statistic", "absolute_error_quantile")
    confidence = parser.getfloat("statistic", "confidence_level")
    if not 0 < quantile < 1 or not 0 < confidence < 1:
        raise ValueError("quantile and confidence must lie strictly between zero and one")
    strata = {name.removesuffix("_edges"): _edges(value) for name, value in parser.items("strata")}
    contract = ResidualBankContract(
        contract_id=_required(parser, "contract", "contract_id"),
        status=_required(parser, "contract", "status"),
        targets=_csv(_required(parser, "contract", "targets")),
        required_protocols=required,
        available_protocols=available,
        partition_root=Path(_required(parser, "source", "partition_root")),
        partition_pattern=_required(parser, "source", "partition_pattern"),
        geographic_provenance=Path(_required(parser, "source", "geographic_provenance")),
        output_root=Path(_required(parser, "source", "output_root")),
        truth_column=_required(parser, "source", "truth_column"),
        prediction_column=_required(parser, "source", "prediction_column"),
        group_column=_required(parser, "source", "group_column"),
        node_column=_required(parser, "source", "node_column"),
        quantile=quantile,
        confidence=confidence,
        minimum_groups_for_finite_bound=parser.getint(
            "statistic", "minimum_groups_for_finite_bound"
        ),
        minimum_rows_per_cohort=parser.getint("statistic", "minimum_rows_per_cohort"),
        minimum_groups_per_cohort=parser.getint("statistic", "minimum_groups_per_cohort"),
        strata_edges=strata,
        pool_protocols=parser.getboolean("policy", "pool_protocols_for_primary_error"),
        basin_may_change_error_tier=parser.getboolean("policy", "basin_may_change_error_tier"),
        missing_protocol_blocks_final_grading=parser.getboolean(
            "policy", "missing_protocol_blocks_final_grading"
        ),
        thresholds_may_be_derived_from_this_bank=parser.getboolean(
            "policy", "thresholds_may_be_derived_from_this_bank"
        ),
        independent_labels_opened=parser.getboolean("policy", "independent_labels_opened"),
    )
    if contract.minimum_groups_for_finite_bound < 1:
        raise ValueError("minimum_groups_for_finite_bound must be positive")
    if contract.basin_may_change_error_tier or contract.pool_protocols:
        raise ValueError("basin grading and pooled primary protocols are prohibited")
    if contract.thresholds_may_be_derived_from_this_bank:
        raise ValueError("this development residual bank cannot define intended-use thresholds")
    if contract.independent_labels_opened:
        raise ValueError("this experiment must not open independent labels")
    return contract


def empirical_quantile(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan")
    return float(np.quantile(finite, quantile, method="higher"))


def hierarchical_group_quantile_bound(
    absolute_error: np.ndarray,
    groups: np.ndarray,
    *,
    quantile: float,
    confidence: float,
    minimum_groups: int,
) -> dict[str, float | int]:
    """Bound the q-quantile of within-group q-quantiles.

    Each complete cruise is one independent evidence unit.  The returned
    order-statistic bound is distribution-free: with ``confidence`` one-sided
    confidence, at least ``quantile`` of the cruise-level q-quantiles are no
    larger than the bound.  This deliberately does not treat dense rows within
    one cruise as independent evidence.
    """
    error = np.asarray(absolute_error, dtype=float)
    group = np.asarray(groups)
    valid = np.isfinite(error) & np.asarray([str(value) != "" for value in group])
    error = error[valid]
    group = group[valid].astype(str)
    order = np.argsort(group, kind="stable")
    error = error[order]
    group = group[order]
    starts = np.r_[0, np.flatnonzero(group[1:] != group[:-1]) + 1]
    ends = np.r_[starts[1:], len(group)]
    group_quantiles = [
        empirical_quantile(error[start:end], quantile)
        for start, end in zip(starts, ends, strict=True)
    ]
    ordered = np.sort(np.asarray(group_quantiles, dtype=float))
    ordered = ordered[np.isfinite(ordered)]
    count = len(ordered)
    estimate = empirical_quantile(ordered, quantile)
    rank = int(binom.ppf(confidence, count, quantile)) + 1 if count else 0
    finite_bound = count >= minimum_groups and rank <= count
    upper = float(ordered[rank - 1]) if finite_bound else float("nan")
    return {
        "groups": count,
        "group_q90": estimate,
        "group_q90_upper95": upper,
        "upper_order_rank": rank,
    }
