import numpy as np
import pandas as pd

from recad.evaluate.conditional_models import (
    SeasonalNearestResidual,
    finite_sample_quantile,
    water_graph,
)


def test_graph_adapter_respects_land_gap_and_decay():
    neighbours = np.array([[1, -1], [0, 2], [1, -1], [4, -1], [3, -1]])
    graph = water_graph(neighbours)
    source = pd.DataFrame({"coastal_node": [0], "mon": [1], "residual": [8.0]})
    fitted = SeasonalNearestResidual.fit(graph, source, maximum_hops=4)
    query = pd.DataFrame({"coastal_node": [0, 1, 2, 3], "mon": [1, 1, 1, 1]})
    got = fitted.predict(query, decay_hops=2, shrinkage=0.5)
    np.testing.assert_allclose(got[:3], 4 * np.exp(-np.arange(3) / 2), rtol=1e-6)
    assert got[3] == 0


def test_graph_adapter_is_season_specific():
    graph = water_graph(np.array([[1], [0]]))
    source = pd.DataFrame({"coastal_node": [0], "mon": [7], "residual": [3.0]})
    fitted = SeasonalNearestResidual.fit(graph, source, maximum_hops=2)
    query = pd.DataFrame({"coastal_node": [1, 1], "mon": [7, 1]})
    np.testing.assert_allclose(
        fitted.predict(query, decay_hops=10, shrinkage=1), [3 * np.exp(-0.1), 0]
    )


def test_finite_sample_quantile_uses_higher_rank():
    assert finite_sample_quantile(np.arange(1, 11), 0.9) == 10
