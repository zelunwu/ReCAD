import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_weighted_carbonate.py"
SPEC = importlib.util.spec_from_file_location("weighted_carbonate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_balanced_selection_penalizes_dense_and_sparse_degradation():
    scales_dense = {"sss": 1.0, "fco2": 50.0}
    scales_sparse = {"ta": 100.0, "dic": 100.0}
    reference = MODULE.combine_selection_rmse(
        {"sss": 0.5, "fco2": 25.0}, {"ta": 50.0, "dic": 50.0},
        scales_dense, scales_sparse,
    )
    dense_worse = MODULE.combine_selection_rmse(
        {"sss": 1.0, "fco2": 50.0}, {"ta": 50.0, "dic": 50.0},
        scales_dense, scales_sparse,
    )
    sparse_worse = MODULE.combine_selection_rmse(
        {"sss": 0.5, "fco2": 25.0}, {"ta": 100.0, "dic": 100.0},
        scales_dense, scales_sparse,
    )
    assert dense_worse > reference
    assert sparse_worse > reference


def test_non_finite_checkpoint_is_rejected():
    score = MODULE.combine_selection_rmse(
        {"sss": float("nan"), "fco2": 1.0}, {"ta": 1.0, "dic": 1.0},
        {"sss": 1.0, "fco2": 1.0}, {"ta": 1.0, "dic": 1.0},
    )
    assert score == float("inf")
