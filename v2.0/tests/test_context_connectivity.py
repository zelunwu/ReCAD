import numpy as np
import pytest
import torch

from recad.evaluate.context_connectivity import (
    ArrayStandardizer,
    ContextResidualNet,
    adjacent_residual_signal,
    count_parameters,
    water_neighbors,
)


def test_water_neighbors_respects_land_and_wraps_longitude() -> None:
    # 3 x 4 grid: nodes 0 and 3 touch across the dateline; node 1 is land.
    flat = np.array([0, 3, 4, 7], dtype=np.int64)
    neighbours = water_neighbors(flat, grid_rows=3, grid_columns=4)
    assert 1 in neighbours[0]
    assert 0 in neighbours[1]
    assert 2 in neighbours[0]
    assert -1 in neighbours[0]


def test_standardizer_is_training_only_and_preserves_missingness() -> None:
    train = np.array([[[1.0, np.nan], [3.0, 5.0]], [[5.0, 7.0], [7.0, 9.0]]])
    transform = ArrayStandardizer.fit(train)
    values, available = transform.apply(np.array([[[3.0, np.nan], [5.0, 7.0]]]))
    assert np.isfinite(values).all()
    assert available[0, 0, 1] == 0
    assert values[0, 0, 1] == 0


@pytest.mark.parametrize("mode", ["gru", "st96_local", "st96_graph"])
def test_context_network_shapes_and_gradients(mode: str) -> None:
    model = ContextResidualNet(sequence_features=4, static_features=6, mode=mode)
    own = torch.randn(8, 6, 4)
    available = torch.ones_like(own)
    static = torch.randn(8, 6)
    neighbour = torch.randn_like(own) if mode == "st96_graph" else None
    prediction = model(own, available, static, neighbour, available)
    assert prediction.shape == (8,)
    prediction.mean().backward()
    assert count_parameters(model) > 1_000


def test_adjacent_residual_signal() -> None:
    signal = adjacent_residual_signal(np.arange(10.0), np.arange(10.0) + 0.1)
    assert signal["pairs"] == 10
    assert signal["correlation"] > 0.99
    assert signal["same_sign"] > 0.8
