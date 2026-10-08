"""Governance contracts for evidence, product scope, and label exposure."""

from recad.governance.benchmark_contract import (
    load_benchmark_contract,
    validate_benchmark_contract,
)
from recad.governance.scope_policy import load_scope_policy, validate_scope_policy

__all__ = [
    "load_benchmark_contract",
    "load_scope_policy",
    "validate_benchmark_contract",
    "validate_scope_policy",
]
