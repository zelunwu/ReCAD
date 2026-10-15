from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.l1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "build_recad_v2_release", ROOT / "scripts/build_recad_v2_release.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_sss_release_status_follows_grade() -> None:
    grade = np.array([1, 2, 3, 4])
    basin = np.array([3, 3, 3, 3])
    assert MODULE.release_status("SSS", grade, basin).tolist() == [1, 1, 2, 3]


def test_fco2_pacific_ab_is_diagnostic() -> None:
    grade = np.array([1, 2, 3, 4, 1])
    basin = np.array([3, 3, 3, 3, 2])
    assert MODULE.release_status("fCO2", grade, basin).tolist() == [2, 2, 2, 3, 1]
