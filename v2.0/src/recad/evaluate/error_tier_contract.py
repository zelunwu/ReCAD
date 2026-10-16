"""Absolute reconstruction-error tiers and validation-evidence states.

The contract intentionally separates a fixed physical error tier from the
evidence supporting that error estimate. Relative risk ranks must not be
passed to this module as physical error estimates.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from enum import Enum, IntFlag
from pathlib import Path


class ErrorTier(str, Enum):
    """Published reconstruction-error tiers."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    UNKNOWN = "U"
    NO_PREDICTION = "N"


class EvidenceStatus(str, Enum):
    """Evidence supporting a reconstruction-error statement."""

    DIRECT = "VD"
    TRANSFER = "VT"
    UNVALIDATED = "VU"
    FAILED = "VF"
    INVALID = "VI"
    OUTSIDE_DOMAIN = "NA"


class ErrorReason(IntFlag):
    """Stable, non-overlapping reason bits for the absolute-error contract."""

    NO_MATCHING_LOCAL_OBSERVATIONS = 1 << 0
    NO_VALIDATED_ANALOGUE = 1 << 1
    OUTSIDE_VALIDATED_ENVIRONMENT = 1 << 2
    GEOGRAPHIC_TRANSFER_UNVALIDATED = 1 << 3
    TEMPORAL_TRANSFER_UNVALIDATED = 1 << 4
    INSUFFICIENT_CALIBRATION_GROUPS = 1 << 5
    CALIBRATION_AUDIT_FAILED = 1 << 6
    REPRESENTATIVENESS_UNAVAILABLE = 1 << 7
    INPUT_GAPS_OR_IMPUTATION = 1 << 8
    INPUT_MISSINGNESS_UNVALIDATED = 1 << 9
    SEA_ICE_REGIME_UNVALIDATED = 1 << 10
    NUMERICAL_EXECUTION_FAILURE = 1 << 11
    PHYSICAL_VALIDITY_FAILURE = 1 << 12
    TARGET_SUPPORT_UNRESOLVED = 1 << 13
    DATA_LINEAGE_OR_LEAKAGE_FAILURE = 1 << 14


@dataclass(frozen=True)
class ErrorThresholds:
    """Fixed target-specific thresholds in the target's physical unit."""

    a: float | None
    b: float | None
    c: float | None

    @property
    def frozen(self) -> bool:
        return self.a is not None and self.b is not None and self.c is not None

    def require_frozen(self) -> tuple[float, float, float]:
        if not self.frozen:
            raise ValueError("physical error thresholds are not frozen")
        assert self.a is not None and self.b is not None and self.c is not None
        return self.a, self.b, self.c


@dataclass(frozen=True)
class TargetErrorProfile:
    """One target's scientific units and absolute tier thresholds."""

    name: str
    variable_definition: str
    unit: str
    thresholds: ErrorThresholds


@dataclass(frozen=True)
class ErrorFlagContract:
    """Versioned absolute-error flag contract loaded from an external file."""

    contract_id: str
    schema_version: str
    status: str
    primary_statistic: str
    absolute_error_quantile: float
    confidence_level: float
    targets: dict[str, TargetErrorProfile]
    reason_bits: dict[str, int]

    def target(self, name: str) -> TargetErrorProfile:
        try:
            return self.targets[name]
        except KeyError as exc:
            raise KeyError(f"unknown target {name!r}") from exc


def _threshold(value: str) -> float | None:
    cleaned = value.strip()
    if cleaned.upper() == "UNSET":
        return None
    result = float(cleaned)
    if result <= 0:
        raise ValueError("physical error thresholds must be positive")
    return result


def _required(parser: configparser.ConfigParser, section: str, key: str) -> str:
    if not parser.has_option(section, key):
        raise ValueError(f"missing [{section}] {key}")
    return parser.get(section, key)


def _load_target(parser: configparser.ConfigParser, name: str) -> TargetErrorProfile:
    section = f"target:{name}"
    if not parser.has_section(section):
        raise ValueError(f"missing [{section}] section")
    thresholds = ErrorThresholds(
        a=_threshold(_required(parser, section, "threshold_a")),
        b=_threshold(_required(parser, section, "threshold_b")),
        c=_threshold(_required(parser, section, "threshold_c")),
    )
    values = (thresholds.a, thresholds.b, thresholds.c)
    if any(value is None for value in values) and not all(value is None for value in values):
        raise ValueError(f"[{section}] thresholds must be all frozen or all UNSET")
    if thresholds.frozen:
        a, b, c = thresholds.require_frozen()
        if not a < b < c:
            raise ValueError(f"[{section}] thresholds must satisfy A < B < C")
    return TargetErrorProfile(
        name=name,
        variable_definition=_required(parser, section, "variable_definition"),
        unit=_required(parser, section, "unit"),
        thresholds=thresholds,
    )


def _load_reason_bits(parser: configparser.ConfigParser) -> dict[str, int]:
    if not parser.has_section("reason_bits"):
        raise ValueError("missing [reason_bits] section")
    configured = {name: int(value) for name, value in parser.items("reason_bits")}
    expected = {reason.name: int(reason) for reason in ErrorReason}
    if configured != expected:
        raise ValueError("reason bits do not match the immutable code definition")
    if len(set(configured.values())) != len(configured):
        raise ValueError("reason bits must be unique")
    if any(value <= 0 or value & (value - 1) for value in configured.values()):
        raise ValueError("every reason bit must be a positive power of two")
    return configured


def load_error_flag_contract(path: str | Path) -> ErrorFlagContract:
    """Load and strictly validate an absolute-error contract from ``.in``."""

    source = Path(path)
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    with source.open(encoding="utf-8") as handle:
        parser.read_file(handle)
    quantile = float(_required(parser, "uncertainty", "absolute_error_quantile"))
    confidence = float(_required(parser, "uncertainty", "confidence_level"))
    if not 0 < quantile < 1:
        raise ValueError("absolute_error_quantile must be between zero and one")
    if not 0 < confidence < 1:
        raise ValueError("confidence_level must be between zero and one")
    target_names = [value.strip() for value in _required(parser, "contract", "targets").split(",")]
    if not target_names or any(not value for value in target_names):
        raise ValueError("[contract] targets must contain at least one target")
    return ErrorFlagContract(
        contract_id=_required(parser, "contract", "contract_id"),
        schema_version=_required(parser, "contract", "schema_version"),
        status=_required(parser, "contract", "status"),
        primary_statistic=_required(parser, "uncertainty", "primary_statistic"),
        absolute_error_quantile=quantile,
        confidence_level=confidence,
        targets={name: _load_target(parser, name) for name in target_names},
        reason_bits=_load_reason_bits(parser),
    )


def assign_error_tier(
    error_q90_upper95: float | None,
    *,
    profile: TargetErrorProfile,
    evidence: EvidenceStatus,
    prediction_valid: bool = True,
) -> ErrorTier | None:
    """Assign a tier without manufacturing a physical error statement.

    ``None`` is returned outside the declared product domain. Unvalidated or
    failed uncertainty estimates receive ``U`` even when a numeric model
    prediction exists.
    """

    if evidence is EvidenceStatus.OUTSIDE_DOMAIN:
        return None
    if not prediction_valid or evidence is EvidenceStatus.INVALID:
        return ErrorTier.NO_PREDICTION
    if evidence in {EvidenceStatus.UNVALIDATED, EvidenceStatus.FAILED}:
        return ErrorTier.UNKNOWN
    if evidence not in {EvidenceStatus.DIRECT, EvidenceStatus.TRANSFER}:
        raise ValueError(f"unsupported evidence state {evidence.value}")
    if error_q90_upper95 is None:
        raise ValueError("supported evidence requires a finite physical error estimate")
    error = float(error_q90_upper95)
    if not 0 <= error < float("inf"):
        raise ValueError("physical error estimate must be finite and non-negative")
    a, b, c = profile.thresholds.require_frozen()
    if error <= a:
        return ErrorTier.A
    if error <= b:
        return ErrorTier.B
    if error <= c:
        return ErrorTier.C
    return ErrorTier.D
