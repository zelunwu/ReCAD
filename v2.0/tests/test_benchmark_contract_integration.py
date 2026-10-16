from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from recad.evaluate.benchmark_contract import load_benchmark_contract

pytestmark = pytest.mark.l2

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "frozen" / "model_family_benchmark_v2.5.in"


def test_contract_rejects_result_dependent_search_space_changes(tmp_path: Path) -> None:
    changed = tmp_path / CONFIG.name
    shutil.copyfile(CONFIG, changed)
    text = changed.read_text(encoding="utf-8").replace(
        "result_dependent_search_space_change = false",
        "result_dependent_search_space_change = true",
    )
    changed.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="safety policy"):
        load_benchmark_contract(changed)


def test_contract_rejects_family_specific_external_subsampling(tmp_path: Path) -> None:
    changed = tmp_path / CONFIG.name
    shutil.copyfile(CONFIG, changed)
    text = changed.read_text(encoding="utf-8").replace(
        "external_family_specific_subsample = false",
        "external_family_specific_subsample = true",
    )
    changed.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="safety policy"):
        load_benchmark_contract(changed)
