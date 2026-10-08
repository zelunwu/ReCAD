from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.l3

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_experiment_archive", ROOT / "scripts" / "verify_experiment_archive.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    "experiment_id",
    [
        "p1_sss_viability_v2.2",
        "p1_fco2_viability_v2.2",
        "p1_ta_viability_v2.2",
        "p1_ta_bights_v2.2",
        "p1_ta_mab_reliability_v2.2",
        "p1_ta_sab_latitude_sensitivity_v2.2",
        "p1_ta_sss_sab_decadal_trends_v2.2",
        "p1_ta_sss_mab_decadal_trends_v2.2",
        "p1_dic_inverse_co2sys_v2.2",
        "p1_dic_reliability_v2.2",
        "p2_scope_reconciliation_v2.3",
        "p2_benchmark_freeze_v2.3",
        "p2_global_coastal_cache_v2.3",
    ],
)
def test_p1_reviewer_archive_is_complete_and_hash_verified(experiment_id: str) -> None:
    result = MODULE.verify_archive(ROOT / "docs/experiment_archive" / experiment_id)
    assert result["verified"], result["errors"]
    assert result["figures"] >= 8
    assert result["captions"] == result["figures"]
    assert result["source_tables"] >= 8


def test_archive_rejects_caption_missing_from_report(tmp_path: Path) -> None:
    source = ROOT / "docs/experiment_archive/p1_fco2_viability_v2.2"
    archive = tmp_path / source.name
    shutil.copytree(source, archive)
    manifest = json.loads((archive / "archive_manifest.json").read_text(encoding="utf-8"))
    figure_name, caption = next(iter(manifest["figure_captions"].items()))
    report_path = archive / "REPORT.md"
    report = report_path.read_text(encoding="utf-8").replace(caption, "", 1)
    report_path.write_text(report, encoding="utf-8")

    result = MODULE.verify_archive(archive)

    assert not result["verified"]
    assert f"REPORT.md does not place caption with figure: {figure_name}" in result["errors"]
