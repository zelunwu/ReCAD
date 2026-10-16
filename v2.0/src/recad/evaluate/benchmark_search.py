"""Deterministic search utilities for the frozen Issue 89 benchmark."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd


def stable_seed(*parts: object) -> int:
    """Return a platform-stable positive seed from one experiment identity."""

    value = ":".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "little") & 0x7FFFFFFF


def common_fidelity_rows(
    frame: pd.DataFrame,
    maximum_rows: int | None,
    *,
    target: str,
    scheme: str,
    fold: int,
    role: str,
) -> pd.DataFrame:
    """Choose the exact same deterministic rows for every family in a target cell."""

    if maximum_rows is None or len(frame) <= maximum_rows:
        return frame.reset_index(drop=True)
    keys = frame.record_id.map(
        lambda record: stable_seed("issue89", target, scheme, fold, role, record)
    ).to_numpy(np.int64)
    selected = np.argpartition(keys, maximum_rows - 1)[:maximum_rows]
    return frame.iloc[np.sort(selected)].reset_index(drop=True)


def suggest_parameters(trial: Any, search_space: dict[str, dict[str, object]]) -> dict[str, Any]:
    """Materialize one Optuna trial from the versioned generic search schema."""

    result: dict[str, Any] = {}
    for name, specification in search_space.items():
        kind = specification["type"]
        if kind == "choice":
            result[name] = trial.suggest_categorical(name, specification["values"])
        elif kind == "int":
            result[name] = trial.suggest_int(
                name, int(specification["low"]), int(specification["high"])
            )
        elif kind == "float":
            result[name] = trial.suggest_float(
                name, float(specification["low"]), float(specification["high"])
            )
        elif kind == "log_float":
            result[name] = trial.suggest_float(
                name,
                float(specification["low"]),
                float(specification["high"]),
                log=True,
            )
        else:
            raise ValueError(f"unsupported search parameter type: {kind}")
    return result


def canonical_parameters(parameters: dict[str, Any]) -> str:
    """Return a stable serialized configuration identity."""

    return json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False)


def capacity_tail_improvement(
    objective: list[float], capacity: list[int], *, tail_fraction: float
) -> float:
    """Measure gain confined to the largest preregistered model-capacity tail."""

    if len(objective) != len(capacity) or not objective:
        raise ValueError("objective and capacity need the same non-zero length")
    ordered = sorted(zip(capacity, objective, strict=True))
    if len({item[0] for item in ordered}) < 2:
        return 0.0
    tail_count = max(1, math.ceil(len(ordered) * tail_fraction))
    boundary = max(1, len(ordered) - tail_count)
    best_before = min(value for _, value in ordered[:boundary])
    best_final = min(value for _, value in ordered)
    return max((best_before - best_final) / max(abs(best_before), 1e-12), 0.0)
