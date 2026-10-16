"""Regression tests for the resumable Issue 89 benchmark controller."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.l1

ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"


def load_controller():
    sys.path.insert(0, str(SCRIPTS))
    try:
        specification = importlib.util.spec_from_file_location(
            "issue89_pipeline_for_test",
            SCRIPTS / "run_issue89_tabular_pipeline.py",
        )
        assert specification and specification.loader
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


def test_memory_failure_is_preserved_as_non_advancing_terminal_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = load_controller()
    monkeypatch.setattr(controller, "OUTPUT", tmp_path)
    failure = tmp_path / "failures/SSS_cruise_fold0_extra_trees.json"
    failure.parent.mkdir(parents=True)
    failure.write_text(
        json.dumps({"error_type": "MemoryError", "error": "allocation failed"}),
        encoding="utf-8",
    )

    assert controller.preserve_resource_failure("SSS", "cruise", 0, "extra_trees")
    assert controller.complete("SSS", "cruise", 0, "extra_trees")
    status = json.loads(
        (tmp_path / "family_status/SSS_cruise_fold0_extra_trees.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "resource_infeasible"
    assert status["advancement_eligible"] is False
    assert status["search_space_changed"] is False


def test_non_memory_failure_remains_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = load_controller()
    monkeypatch.setattr(controller, "OUTPUT", tmp_path)
    failure = tmp_path / "failures/SSS_cruise_fold0_extra_trees.json"
    failure.parent.mkdir(parents=True)
    failure.write_text(json.dumps({"error_type": "ValueError"}), encoding="utf-8")

    assert not controller.preserve_resource_failure("SSS", "cruise", 0, "extra_trees")
    assert not controller.complete("SSS", "cruise", 0, "extra_trees")
