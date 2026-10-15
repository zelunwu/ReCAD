from __future__ import annotations

import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

pytestmark = pytest.mark.l3

OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_4\p2_global_reliability_contract_v2.4")


def _require_external_issue44_products() -> None:
    if not (OUTPUT / "calibration_contract.json").is_file():
        pytest.skip("Issue 44 external products are not available on this host")


def test_issue44_complete_domain_products_match_frozen_contract() -> None:
    _require_external_issue44_products()
    contract = json.loads((OUTPUT / "calibration_contract.json").read_text(encoding="utf-8"))
    required = {
        "prediction",
        "width50",
        "width90",
        "risk_score",
        "applicability_grade",
        "status",
        "reason_bits",
        "training_support_km",
        "environmental_dissimilarity",
        "effective_groups",
        "model_disagreement",
    }
    for target in ("SSS", "fCO2"):
        path = Path(contract["products"][target]["path"])
        assert path.is_file()
        with netCDF4.Dataset(path) as dataset:
            assert len(dataset.dimensions["time"]) == 396
            assert len(dataset.dimensions["node"]) == 838_661
            assert required <= set(dataset.variables)
            grade = np.asarray(dataset.variables["applicability_grade"][-1, :])
            status = np.asarray(dataset.variables["status"][-1, :])
            prediction = np.asarray(dataset.variables["prediction"][-1, :])
            assert set(np.unique(grade)) <= {1, 2, 3, 4}
            assert set(np.unique(status)) <= {1, 2, 3}
            assert np.isfinite(prediction).all()
            assert np.mean(grade == 4) > 0.5


def test_issue44_all_corrected_partitions_are_present() -> None:
    _require_external_issue44_products()
    for target in ("SSS", "fCO2"):
        paths = list((OUTPUT / "partitions").glob(f"{target}_*.parquet"))
        assert len(paths) == 18
    assert (OUTPUT / "calibration.complete").is_file()
    assert (OUTPUT / "product.complete").is_file()
