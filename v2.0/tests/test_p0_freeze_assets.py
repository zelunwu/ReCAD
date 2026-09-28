import importlib.util
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_p0_freeze_assets.py"
SPEC = importlib.util.spec_from_file_location("p0_assets", SCRIPT)
P0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P0)


def test_group_split_is_deterministic_and_label_free():
    keys = ["33WA20220122", "18KF20240625", "TEST-CRUISE"]
    first = [(P0.split_for_group(k), P0.cv_for_group(k)) for k in keys]
    second = [(P0.split_for_group(k), P0.cv_for_group(k)) for k in keys]
    assert first == second
    assert all(s in {"train", "development", "locked_test"} for s, _ in first)
    assert all(0 <= fold < 5 for _, fold in first)


def test_coastal_graph_does_not_cross_land(tmp_path):
    mask = np.array([[1, 0, 1, 0], [1, 0, 1, 0]], dtype=bool)
    flats = np.flatnonzero(mask.ravel())
    audit = P0.build_graph(mask, flats, tmp_path / "graph.npz")
    graph = load_npz(tmp_path / "graph.npz")
    # Two vertical water columns are disconnected by the land column.
    assert audit["cardinal_water_edges_only"] is True
    assert graph[0, 1] == 0
    assert graph[2, 3] == 0
