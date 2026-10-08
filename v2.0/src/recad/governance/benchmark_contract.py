"""Validation for the frozen P2 benchmark and improvement contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_benchmark_contract(path: Path) -> dict[str, Any]:
    """Load a benchmark contract from JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def validate_benchmark_contract(contract: dict[str, Any]) -> list[str]:
    """Return every contract violation."""
    errors: list[str] = []
    panels = {row.get("id"): row for row in contract.get("comparison_panels", [])}
    required_panels = {"native_context", "common_grid", "common_support", "strict_method"}
    if set(panels) != required_panels:
        errors.append(
            "comparison panels must be native, common-grid, common-support, and strict-method"
        )
    for panel_id in ("native_context", "common_grid"):
        if panels.get(panel_id, {}).get("can_select_model"):
            errors.append(f"{panel_id} cannot select a model")
    if not panels.get("strict_method", {}).get("identical_splits"):
        errors.append("strict method comparison requires identical splits")

    references = contract.get("literature_benchmarks", [])
    if any(
        row.get("validation_design") == "random fold" and row.get("primary_gate")
        for row in references
    ):
        errors.append("random-fold literature scores cannot be primary gates")
    if any(
        row.get("target_basis") == "pCO2" and not row.get("conversion_required")
        for row in references
    ):
        errors.append("pCO2 references require an explicit conversion policy")

    rows = contract.get("performance_targets", [])
    tiers = ("minimum", "competitive", "ideal")
    for target in ("SSS", "fCO2"):
        for scheme in ("cruise", "spatial", "forward"):
            selected = {
                row["tier"]: row
                for row in rows
                if row["target"] == target and row["scheme"] == scheme
            }
            if set(selected) != set(tiers):
                errors.append(f"missing tiers for {target}/{scheme}")
                continue
            ceilings = [float(selected[tier]["rmse_ceiling"]) for tier in tiers]
            improvements = [float(selected[tier]["min_improvement_fraction"]) for tier in tiers]
            if ceilings != sorted(ceilings, reverse=True):
                errors.append(f"RMSE ceilings must tighten by tier for {target}/{scheme}")
            if improvements != sorted(improvements):
                errors.append(f"improvement fractions must increase by tier for {target}/{scheme}")

    bootstrap = contract.get("paired_bootstrap", {})
    if int(bootstrap.get("replicates", 0)) < 2000:
        errors.append("paired bootstrap requires at least 2000 replicates")
    if bootstrap.get("unit") not in {"cruise", "spatial_block"}:
        errors.append("paired bootstrap unit must preserve grouped dependence")
    if contract.get("historical_2004_2005", {}).get("can_select_model"):
        errors.append("2004-2005 may not select a model")
    return errors
