from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.l2

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "frozen" / "absolute_error_flag_contract_v2.5.in"
SCRIPT = ROOT / "scripts" / "validate_error_flag_contract.py"


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return environment


def test_validator_serializes_frozen_methodology_and_pending_thresholds(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(CONFIG), "--output-json", str(output)],
        cwd=ROOT,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["status"] == "methodology_frozen_thresholds_pending_scientific_derivation"
    assert not summary["targets"]["SSS"]["thresholds_frozen"]
    assert not summary["targets"]["fCO2"]["thresholds_frozen"]
    assert len(summary["sha256"]) == 64


def test_production_gate_fails_until_physical_thresholds_are_frozen() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(CONFIG), "--require-thresholds"],
        cwd=ROOT,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert '"thresholds_frozen": false' in result.stdout
