"""Tests for deterministic Issue 89 search behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recad.evaluate.benchmark_search import (
    canonical_parameters,
    capacity_tail_improvement,
    common_fidelity_rows,
    stable_seed,
    suggest_parameters,
)


class Trial:
    def suggest_categorical(self, _name: str, values: list[object]) -> object:
        return values[0]

    def suggest_int(self, _name: str, low: int, _high: int) -> int:
        return low

    def suggest_float(self, _name: str, low: float, _high: float, **_: object) -> float:
        return low


@pytest.mark.l1
def test_common_rows_do_not_depend_on_family() -> None:
    frame = pd.DataFrame({"record_id": np.arange(1000), "value": np.arange(1000)})
    first = common_fidelity_rows(frame, 80, target="SSS", scheme="cruise", fold=1, role="train")
    second = common_fidelity_rows(frame, 80, target="SSS", scheme="cruise", fold=1, role="train")
    assert first.record_id.tolist() == second.record_id.tolist()
    assert len(first) == 80
    assert len(set(first.record_id)) == 80


@pytest.mark.l1
def test_search_schema_materializes_all_types() -> None:
    space = {
        "choice": {"type": "choice", "values": [None, 4]},
        "integer": {"type": "int", "low": 2, "high": 9},
        "floating": {"type": "float", "low": 0.1, "high": 0.9},
        "log": {"type": "log_float", "low": 1e-4, "high": 1e-1},
    }
    assert suggest_parameters(Trial(), space) == {
        "choice": None,
        "integer": 2,
        "floating": 0.1,
        "log": 1e-4,
    }


@pytest.mark.l1
def test_identities_are_stable_and_order_independent() -> None:
    assert stable_seed("SSS", 1) == stable_seed("SSS", 1)
    assert stable_seed("SSS", 1) != stable_seed("SSS", 2)
    assert canonical_parameters({"b": 2, "a": 1}) == canonical_parameters({"a": 1, "b": 2})


@pytest.mark.l1
def test_capacity_tail_uses_model_size_not_training_row_count() -> None:
    assert capacity_tail_improvement(
        [1.0, 0.9, 0.8, 0.7], [1, 2, 3, 4], tail_fraction=0.25
    ) == pytest.approx(0.125)
    assert capacity_tail_improvement([1.0, 0.9], [1, 1], tail_fraction=0.25) == 0.0
