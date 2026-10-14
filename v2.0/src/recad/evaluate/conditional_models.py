"""Leakage-safe conditional adapters used by the Issue 64 experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import dijkstra


def water_graph(neighbours: np.ndarray) -> sparse.csr_matrix:
    """Build an undirected, unit-cost graph from a water-neighbour table."""

    table = np.asarray(neighbours, dtype=np.int64)
    source = np.repeat(np.arange(len(table), dtype=np.int64), table.shape[1])
    target = table.reshape(-1)
    valid = target >= 0
    graph = sparse.coo_matrix(
        (np.ones(valid.sum(), dtype=np.float32), (source[valid], target[valid])),
        shape=(len(table), len(table)),
    )
    return graph.maximum(graph.T).tocsr()


def season_id(month: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(month, dtype=np.int16)
    if np.any((values < 1) | (values > 12)):
        raise ValueError("month must be in 1..12")
    return ((values - 1) // 3).astype(np.int8)


@dataclass
class SeasonalNearestResidual:
    """Nearest labelled residual on a water-only graph for each season."""

    distance: np.ndarray
    residual: np.ndarray

    @classmethod
    def fit(
        cls, graph: sparse.csr_matrix, frame: pd.DataFrame, *, maximum_hops: int
    ) -> SeasonalNearestResidual:
        required = {"coastal_node", "mon", "residual"}
        if missing := required - set(frame):
            raise ValueError(f"missing columns: {sorted(missing)}")
        n_nodes = graph.shape[0]
        distance = np.full((4, n_nodes), np.inf, dtype=np.float32)
        residual = np.zeros((4, n_nodes), dtype=np.float32)
        work = frame.assign(season=season_id(frame.mon))
        work = work[np.isfinite(work.residual)].copy()
        work = work.groupby(["season", "coastal_node"], as_index=False).residual.mean()
        for season in range(4):
            subset = work[work.season == season]
            if subset.empty:
                continue
            nodes = subset.coastal_node.to_numpy(np.int64)
            if np.any((nodes < 0) | (nodes >= n_nodes)):
                raise ValueError("coastal_node outside graph")
            values = subset.residual.to_numpy(np.float32)
            result = dijkstra(
                graph,
                directed=False,
                indices=nodes,
                unweighted=True,
                limit=float(maximum_hops),
                min_only=True,
                return_predecessors=True,
            )
            dist, _predecessor, nearest = result
            finite = np.isfinite(dist) & (nearest >= 0)
            lookup = np.full(n_nodes, np.nan, dtype=np.float32)
            lookup[nodes] = values
            distance[season, finite] = np.asarray(dist[finite], dtype=np.float32)
            residual[season, finite] = lookup[np.asarray(nearest[finite], dtype=np.int64)]
        return cls(distance=distance, residual=residual)

    def predict(self, frame: pd.DataFrame, *, decay_hops: float, shrinkage: float) -> np.ndarray:
        node = frame.coastal_node.to_numpy(np.int64)
        season = season_id(frame.mon)
        valid = (node >= 0) & (node < self.distance.shape[1])
        result = np.zeros(len(frame), dtype=np.float32)
        dist = np.full(len(frame), np.inf, dtype=np.float32)
        base = np.zeros(len(frame), dtype=np.float32)
        dist[valid] = self.distance[season[valid], node[valid]]
        base[valid] = self.residual[season[valid], node[valid]]
        reachable = np.isfinite(dist)
        result[reachable] = (
            float(shrinkage) * base[reachable] * np.exp(-dist[reachable] / float(decay_hops))
        )
        return result


def finite_sample_quantile(values: np.ndarray, coverage: float) -> float:
    """Higher finite-sample conformal quantile."""

    finite = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    if len(finite) == 0:
        raise ValueError("no finite conformity scores")
    rank = min(len(finite) - 1, int(np.ceil((len(finite) + 1) * coverage)) - 1)
    return float(finite[max(rank, 0)])
