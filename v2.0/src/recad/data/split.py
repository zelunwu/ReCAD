"""Train/validation/test split logic.

Two schemes are supported:

* ``random_80_20`` - reproduces the v1.1 protocol exactly: random 80/20
  train/validation over all valid samples of the training pool
  (``create4DSplitIndices.m`` with 80%);
* ``blocked_spatiotemporal`` (v2.0 default) - splits the training pool along
  both space (square blocks of ``spatial_block_deg``) and time (blocks of
  ``temporal_block_months``), preventing leakage across spatial/temporal
  autocorrelation.

In both schemes the years listed in ``SplitConfig.test_holdout_years``
(default ``(2004, 2005)``, matching v1.1) are held out entirely as the
independent test set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recad.config import SplitConfig
from recad.data.features import PreparedData
from recad.utils.seed import seed_everything


@dataclass
class SplitMasks:
    """Boolean masks over the 4-D domain for each partition."""

    train: np.ndarray  # [n_year, 12, n_lat, n_lon]
    val: np.ndarray  # [n_year, 12, n_lat, n_lon]
    test: np.ndarray  # [n_year, 12, n_lat, n_lon]

    @property
    def names(self) -> tuple[str, str, str]:
        return ("train", "val", "test")

    def get(self, name: str) -> np.ndarray:
        return getattr(self, name)

    def counts(self) -> dict[str, int]:
        return {name: int(np.count_nonzero(self.get(name))) for name in self.names}

    def validate(self) -> "SplitMasks":
        """Assert the partitions are disjoint and cover the valid samples."""
        for a in self.names:
            for b in self.names:
                if a >= b:
                    continue
                overlap = np.count_nonzero(self.get(a) & self.get(b))
                if overlap:
                    raise ValueError(f"split masks '{a}' and '{b}' overlap")
        return self


def _build_empty_masks(prepared: PreparedData) -> dict[str, np.ndarray]:
    shape = prepared.shape4d
    return {name: np.zeros(shape, dtype=bool) for name in ("train", "val", "test")}


def _block_ids(
    prepared: PreparedData, spatial_block_deg: float, temporal_block_months: int
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each (year, month, lat, lon) sample a (spatial, temporal) block id."""
    n_year, _, n_lat, n_lon = prepared.shape4d
    lat_edges = np.floor(
        (prepared.grid.lat - prepared.grid.lat[0]) / spatial_block_deg
    ).astype(np.int64)
    lon_edges = np.floor(
        (prepared.grid.lon - prepared.grid.lon[0]) / spatial_block_deg
    ).astype(np.int64)

    spatial = lat_edges[:, None] * (n_lon + 1) + lon_edges[None, :]  # [n_lat, n_lon]
    spatial = np.broadcast_to(spatial[None, None, :, :], prepared.shape4d).copy()

    temporal = np.zeros((n_year, 12), dtype=np.int64)
    month_idx = np.arange(12)
    for y in range(n_year):
        temporal[y] = (y * 12 + month_idx) // temporal_block_months
    temporal = np.broadcast_to(temporal[:, :, None, None], prepared.shape4d).copy()
    return spatial, temporal


def make_split_masks(cfg: SplitConfig, prepared: PreparedData) -> SplitMasks:
    """Build the three partition masks for the given prepared data."""
    seed_everything(cfg.seed)
    masks = _build_empty_masks(prepared)
    valid = prepared.valid_mask()

    # Independent test set: the configured holdout years (v1.1: 2004-2005).
    test_years = np.array(cfg.test_holdout_years, dtype=np.int64)
    is_test_year = np.isin(prepared.years, test_years)[:, None, None, None]
    masks["test"] = is_test_year & valid
    pool = valid & ~masks["test"]

    if cfg.scheme == "random_80_20":
        _assign_random_80_20(cfg, pool, masks)
    elif cfg.scheme == "blocked_spatiotemporal":
        _assign_blocked(cfg, pool, prepared, masks)
    else:  # pragma: no cover - guarded by Config validation
        raise ValueError(f"unknown split scheme: {cfg.scheme}")

    return SplitMasks(**masks).validate()


def _assign_random_80_20(
    cfg: SplitConfig, pool: np.ndarray, masks: dict[str, np.ndarray]
) -> None:
    rng = np.random.default_rng(cfg.seed)
    flat = np.flatnonzero(pool)
    perm = rng.permutation(flat.size)
    n_train = int(round(cfg.train_fraction * flat.size))
    train_flat = flat[perm[:n_train]]
    val_flat = flat[perm[n_train:]]
    masks["train"].ravel()[train_flat] = True
    masks["val"].ravel()[val_flat] = True


def _assign_blocked(
    cfg: SplitConfig,
    pool: np.ndarray,
    prepared: PreparedData,
    masks: dict[str, np.ndarray],
) -> None:
    spatial, temporal = _block_ids(
        prepared, cfg.spatial_block_deg, cfg.temporal_block_months
    )
    # Only consider blocks that contain at least one pooled sample.
    pooled = pool
    block_ids = np.stack(
        [spatial[pooled], temporal[pooled]], axis=1
    )  # [n_pooled, 2]
    keys = np.unique(block_ids, axis=0)
    # Deterministic, reproducible assignment of whole blocks to train/val.
    # (Python's builtin hash() is salted per process, so a stable mix is used.)
    hashes = ((keys[:, 0].astype(np.int64) * 73856093) ^ (keys[:, 1].astype(np.int64) * 19349663)) % (2**31)
    order = np.argsort(hashes)
    keys_sorted = keys[order]
    n_train_blocks = int(round(cfg.train_fraction * keys_sorted.shape[0]))
    train_keys = set(
        (int(a), int(b)) for a, b in keys_sorted[:n_train_blocks]
    )
    val_sel = np.zeros(pooled.shape, dtype=bool)
    train_sel = np.zeros(pooled.shape, dtype=bool)
    for (a, b) in keys_sorted:
        sel = (spatial == a) & (temporal == b) & pooled
        if (a, b) in train_keys:
            train_sel |= sel
        else:
            val_sel |= sel
    masks["train"] |= train_sel
    masks["val"] |= val_sel