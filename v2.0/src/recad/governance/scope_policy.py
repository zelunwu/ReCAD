"""Validation helpers for the frozen P2 product-scope policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ALLOWED_STATUSES = {"pass_global", "pass_regional", "diagnostic_only", "fail"}
ALLOWED_EXPOSURES = {"development", "historical", "used_locked", "sealed_locked", "external"}


def load_scope_policy(path: Path) -> dict[str, Any]:
    """Load a JSON scope policy."""
    return json.loads(path.read_text(encoding="utf-8"))


def validate_scope_policy(policy: dict[str, Any], repository_root: Path) -> list[str]:
    """Return all policy violations without stopping at the first error."""
    errors: list[str] = []
    hierarchy = policy.get("product_hierarchy", {})
    if hierarchy.get("core_targets") != ["SSS", "fCO2"]:
        errors.append("core_targets must be exactly SSS and fCO2")
    if hierarchy.get("optional_targets") != ["TA", "DIC"]:
        errors.append("optional_targets must be exactly TA and DIC")
    if hierarchy.get("optional_region") != "North American coastal waters":
        errors.append("TA/DIC optional region must be North American coastal waters")

    rows = policy.get("target_region_status", [])
    keys: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.get("target", ""), row.get("region", ""))
        if key in keys:
            errors.append(f"duplicate target-region status: {key}")
        keys.add(key)
        status = row.get("status")
        if status not in ALLOWED_STATUSES:
            errors.append(f"invalid status for {key}: {status}")
        if row.get("global_qualified") and status != "pass_global":
            errors.append(f"only pass_global may set global_qualified for {key}")
        if row.get("projection_domain") == "global coastal atlas" and row.get(
            "direct_global_evidence"
        ):
            errors.append(f"global atlas projection cannot create direct global evidence for {key}")
        for field in ("report", "manifest"):
            relative = row.get(field)
            if not relative or not (repository_root / relative).is_file():
                errors.append(f"missing immutable {field} for {key}: {relative}")

    ledger = policy.get("label_exposure_ledger", [])
    for row in ledger:
        name = row.get("dataset", "unknown")
        exposure = row.get("exposure")
        if exposure not in ALLOWED_EXPOSURES:
            errors.append(f"invalid exposure for {name}: {exposure}")
        if exposure == "used_locked" and row.get("reusable_as_blind"):
            errors.append(f"used locked labels cannot be reused as blind: {name}")
        if exposure in {"sealed_locked", "external"} and row.get("labels_opened"):
            errors.append(f"sealed labels must remain unopened: {name}")
        evidence = row.get("evidence")
        if evidence and not (repository_root / evidence).is_file():
            errors.append(f"missing exposure evidence for {name}: {evidence}")

    if not any(
        row.get("target") == "SSS" and row.get("exposure") == "used_locked" for row in ledger
    ):
        errors.append("SSS used-locked exposure must be recorded")
    if any(
        row.get("target") == "SSS"
        and row.get("exposure") == "used_locked"
        and row.get("reusable_as_blind")
        for row in ledger
    ):
        errors.append("old SSS locked set must be spent")
    return errors
