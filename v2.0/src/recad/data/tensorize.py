"""PyTorch dataset that turns prepared fields into transformer inputs.

Design (see docs/design.md §Model input):
 * The domain is tiled into spatial patches (``grid.patch_size_cells`` cells
   per side). Patches that intersect the coastal mask become *tokens* for the
   spatial transformer stage. Token features are per-patch aggregations of the
   predictor fields (mean over valid coastal cells) plus a "coverage" channel
   (fraction of coastal cells with valid predictor data). Aggregation calls
   the native ``recad_cuda`` kernel (``PatchAggregate``) when the DLL is
   built, falling back to NumPy otherwise.
 * Per-cell features (for the output head) concatenate normalised lon/lat,
   cyclic month encoding, z-scored predictors (NaN replaced by 0) and per
   predictor a missing-flag channel.
 * Each dataset item is a *temporal window* (default 12 consecutive months =
   one year) for a fixed spatial layout; the temporal transformer stage
   attends across the window for every token.
 * ``tile_cells`` limits the per-cell block processed per item (constant
   stride sampling over coastal cells) so the same code runs on a laptop
   smoke test and on a global 1/8-deg run (see docs on scaling).

Masking: the train/val/test Boolean fields are baked into the dataset via
``SplitMasks``; the trainer gates the loss with ``mask[y, m]`` so the exact
same dataset class serves training, validation and prediction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from recad.config import ModelConfig
from recad.data.features import FEATURE_NAMES, PreparedData
from recad.data.split import SplitMasks
from recad.utils.native import recad_native

MONTHS_PER_YEAR = 12


@dataclass(frozen=True)
class TensorShapes:
    """Structural dimensions shared by the dataset and the model."""

    n_tokens: int  # number of active (coastal) spatial tokens
    n_cells: int  # cells per month in one item
    n_token_features: int  # per-token input width
    n_cell_features: int  # per-cell input width
    n_predictors: int = len(FEATURE_NAMES)


def build_token_layout(prepared: PreparedData) -> tuple[np.ndarray, np.ndarray]:
    """Return (patch_id_map [n_lat,n_lon] int32, active_token_ids int32[]).

    ``active_token_ids`` lists the patch ids of patches intersecting the
    coastal mask, in ascending order; the model indexes tokens by position in
    this list (0-based), so ``P`` in the model equals its length.
    """
    patch_map = prepared.grid.patch_id_map()
    coast = prepared.coastal_mask
    active = np.unique(patch_map[coast]).astype(np.int32)
    return patch_map, active


def _cell_coastal_counts(prepared: PreparedData, patch_map: np.ndarray) -> np.ndarray:
    """Number of coastal cells in each patch (all patches), int64."""
    counts, _ = recad_native.patch_aggregate(
        np.ones(patch_map.size, dtype=np.float32),
        patch_map.reshape(-1).astype(np.int32),
        prepared.grid.n_patches,
    )
    return counts


class CoastalPatchDataset(Dataset):
    """Produces (year-window) items of token and cell tensors."""

    def __init__(
        self,
        prepared: PreparedData,
        masks: SplitMasks | None,
        split: str | None,
        model_cfg: ModelConfig,
        *,
        tile_cells: int | None = None,
        seed: int = 100,
        require_target: bool = True,
        year_repeats: Mapping[int, int] | None = None,
    ) -> None:
        """Build a dataset.

        Args:
            prepared: model-ready fields.
            masks: split masks; ``None`` selects every year (reconstruction).
            split: one of masks.names, required iff ``masks`` is not None.
            model_cfg: transformer hyper-parameters (temporal window etc.).
            tile_cells: optional deterministic subsample of coastal cells.
            seed: seeding for the (deterministic) stride-based tiling.
            require_target: if True, a cell counts as observable only where
                the target exists (training/validation); if False every
                coastal cell is observable (product reconstruction).
            year_repeats: optional per-year multiplicity (ensemble-bootstrap),
                mapping year index -> number of times its windows appear.
        """
        if masks is not None and (split is None or split not in masks.names):
            raise ValueError(f"split must be one of {masks.names}, got {split}")
        self.prepared = prepared
        self.masks = masks
        self.split = split
        self.model_cfg = model_cfg
        self.require_target = require_target
        self.year_repeats = year_repeats or {}
        self.window = model_cfg.temporal_window_months
        if self.window < 1 or self.window > MONTHS_PER_YEAR:
            raise ValueError("temporal_window_months must be in [1, 12]")

        self.patch_map, self.active_tokens = build_token_layout(prepared)
        self.token_pos = np.full(prepared.grid.n_patches, -1, dtype=np.int64)
        self.token_pos[self.active_tokens] = np.arange(self.active_tokens.size)
        self.coast = prepared.coastal_mask
        self.cell_idx_global = np.flatnonzero(self.coast)  # flat index of every coastal cell
        self.rng = np.random.default_rng(seed)

        # ---- tile (deterministic subsample) of coastal cells ----
        if tile_cells is None or tile_cells >= self.cell_idx_global.size:
            self.cell_idx_global_sel = self.cell_idx_global
        else:
            step = int(np.ceil(self.cell_idx_global.size / tile_cells))
            self.cell_idx_global_sel = self.cell_idx_global[::step]

        # ---- per-patch denominators ----
        self.patch_coast_counts = _cell_coastal_counts(prepared, self.patch_map)
        self.patch_coast_counts_sel = self.patch_coast_counts[self.active_tokens]

        # ---- cell feature template (static part: lon/lat curvature) ----
        self.cell_lon, self.cell_lat = self._cell_lonlat()
        self.n_cells = int(self.cell_idx_global_sel.size)

        # ---- per-month normed token features and coverage ----
        mean, std = prepared.feature_stats()
        self.pred_mean = mean
        self.pred_std = std
        self.zscored: dict[str, np.ndarray] = {}
        for name in FEATURE_NAMES:
            arr = prepared.require(name).astype(np.float64)
            self.zscored[name] = (arr - mean[FEATURE_NAMES.index(name)]) / std[
                FEATURE_NAMES.index(name)
            ]
        # ---- standardized target (same-learner scaling as the predictors) ----
        self.t_mean, self.t_std = prepared.target_stats()
        self._build_token_features()

        # ---- item list: (year, start_month) windows ----
        n_windows = MONTHS_PER_YEAR - self.window + 1
        self.items: list[tuple[int, int]] = []
        if masks is None:
            year_range = range(prepared.n_year)
        else:
            split_mask = masks.get(split)
            year_range = [y for y in range(prepared.n_year) if split_mask[y].any()]
        for y in year_range:
            repeats = self.year_repeats.get(y, 1)
            for _ in range(repeats):
                for start in range(n_windows):
                    self.items.append((y, start))

    # ------------------------------------------------------------------
    # Static geometry
    # ------------------------------------------------------------------
    def _cell_lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        lon_grid, lat_grid = np.meshgrid(self.prepared.grid.lon, self.prepared.grid.lat)
        lon = lon_grid.ravel()[self.cell_idx_global_sel]
        lat = lat_grid.ravel()[self.cell_idx_global_sel]
        return lon.astype(np.float32), lat.astype(np.float32)

    # ------------------------------------------------------------------
    # Token features (per year AND month), using the native patch kernel
    # ------------------------------------------------------------------
    def _build_token_features(self) -> None:
        """Aggregate predictors into per-(year, month) patch tokens.

        Tokens are year-specific so the temporal stage can learn *interannual*
        patch-level anomalies (e.g., a warm anomaly propagating along the
        coast), not just the climatological seasonal cycle. Aggregation uses
        the native ``PatchAggregate`` kernel (CUDA when built), one call per
        (year, month, predictor) over the flat domain.
        """
        n_pred = len(FEATURE_NAMES)
        n_active = self.active_tokens.size
        n_year = self.prepared.n_year
        # [n_year, 12, n_active, n_pred] aggregated means; [n_year, 12, n_active] coverage
        self.token_feats_ym = np.full(
            (n_year, MONTHS_PER_YEAR, n_active, n_pred), np.nan, dtype=np.float32
        )
        self.coverage_ym = np.zeros((n_year, MONTHS_PER_YEAR, n_active), dtype=np.float32)
        flat_ids = self.patch_map.ravel().astype(np.int32)

        for y in range(n_year):
            for m in range(MONTHS_PER_YEAR):
                for j, name in enumerate(FEATURE_NAMES):
                    field = self.zscored[name][y, m, :, :].reshape(-1)
                    counts, means = recad_native.patch_aggregate(
                        field.astype(np.float32), flat_ids, self.prepared.grid.n_patches
                    )
                    sel = means[self.active_tokens]
                    self.token_feats_ym[y, m, :, j] = sel
                    counts_sel = counts[self.active_tokens]
                    self.coverage_ym[y, m, :] = np.maximum(
                        self.coverage_ym[y, m, :],
                        counts_sel / np.maximum(self.patch_coast_counts_sel, 1),
                    )

    # ------------------------------------------------------------------
    # Item construction
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        y, start = self.items[idx]
        T = self.window
        months = np.arange(start, start + T)
        n_pred = len(FEATURE_NAMES)
        P = self.active_tokens.size
        C = self.n_cells
        prepared = self.prepared

        # ---- per cell: month cyclic encoding + z-scored predictors ----
        mon_sin = np.sin(2 * np.pi * (months + 1) / 12).astype(np.float32)
        mon_cos = np.cos(2 * np.pi * (months + 1) / 12).astype(np.float32)
        lon_norm = (self.cell_lon / 360.0).astype(np.float32)
        lat_norm = ((self.cell_lat + 90.0) / 180.0).astype(np.float32)

        cell_feats = np.empty((T, C, 4 + 2 * n_pred), dtype=np.float32)
        targets = np.empty((T, C), dtype=np.float32)
        cell_mask = np.zeros((T, C), dtype=bool)
        cell_patch = np.empty((T, C), dtype=np.int64)

        zc = self.zscored
        flat_sel = self.cell_idx_global_sel
        for t, m in enumerate(months):
            cell_feats[t, :, 0] = lon_norm
            cell_feats[t, :, 1] = lat_norm
            cell_feats[t, :, 2] = mon_sin[t]
            cell_feats[t, :, 3] = mon_cos[t]
            for j, name in enumerate(FEATURE_NAMES):
                arr = zc[name][y, m, :, :].reshape(-1)[flat_sel]
                missing = np.isnan(arr)
                arr = np.where(missing, 0.0, arr)
                cell_feats[t, :, 4 + j] = arr
                cell_feats[t, :, 4 + n_pred + j] = missing.astype(np.float32)
            tgt = prepared.target_values()[y, m, :, :].reshape(-1)[flat_sel]
            targets[t] = (tgt - self.t_mean) / self.t_std  # standardized target
            valid = ~np.isnan(tgt) if self.require_target else np.ones(tgt.shape, dtype=bool)
            cell_mask[t] = valid
            cell_patch[t] = self.token_pos[self.patch_map.ravel()[flat_sel]]
            cell_patch[t][cell_patch[t] < 0] = -1

        # ---- token tensors for this year ----
        token_feats = np.full((T, P, n_pred + 1), np.nan, dtype=np.float32)
        token_valid = np.zeros((T, P), dtype=bool)
        for t, m in enumerate(months):
            token_feats[t, :, :n_pred] = self.token_feats_ym[y, m]
            token_feats[t, :, n_pred] = self.coverage_ym[y, m]
            token_valid[t] = ~np.isnan(self.token_feats_ym[y, m]).all(axis=1)

        coord_lon, coord_lat = self.prepared.grid.patch_centers()
        coord = np.stack(
            [
                (coord_lon[self.active_tokens] / 360.0),
                ((coord_lat[self.active_tokens] + 90.0) / 180.0),
            ],
            axis=1,
        ).astype(np.float32)  # [P, 2]

        return {
            "token_feats": torch.from_numpy(token_feats),  # [T, P, F_tok]
            "token_valid": torch.from_numpy(token_valid),  # [T, P]
            "coord": torch.from_numpy(np.repeat(coord[None, :, :], T, axis=0)),  # [T, P, 2]
            "month": torch.from_numpy(
                np.repeat(np.stack([mon_sin, mon_cos], axis=-1)[:, None, :], P, axis=1)
            ),  # [T, P, 2]
            "cell_feats": torch.from_numpy(cell_feats),  # [T, C, F_cell]
            "cell_mask": torch.from_numpy(cell_mask),  # [T, C]
            "cell_patch": torch.from_numpy(cell_patch),  # [T, C]
            "targets": torch.from_numpy(targets),  # [T, C]
            "year_id": torch.tensor(int(y)),
            "month_ids": torch.from_numpy((months + 1).astype(np.int64)),  # [T]
        }

    # ------------------------------------------------------------------
    # Structural info for the model
    # ------------------------------------------------------------------
    def shapes(self) -> TensorShapes:
        n_pred = len(FEATURE_NAMES)
        return TensorShapes(
            n_tokens=self.active_tokens.size,
            n_cells=self.n_cells,
            n_token_features=n_pred + 1,
            n_cell_features=4 + 2 * n_pred,
            n_predictors=n_pred,
        )

    @property
    def cell_flat_indices(self) -> np.ndarray:
        """Flat (lat*lon) indices of the cells produced per month."""
        return self.cell_idx_global_sel

    @property
    def token_positions(self) -> np.ndarray:
        """Per-patch token position map: patch id -> token index (or -1)."""
        return self.token_pos


def collate_items(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate dataset items into a batch (all shapes already uniform)."""
    out: dict[str, torch.Tensor] = {}
    for key in items[0]:
        stacked = torch.stack([item[key] for item in items], dim=0)
        if key in ("cell_mask", "token_valid"):
            stacked = stacked.bool()
        if key in ("targets",):
            stacked = stacked.float()
        out[key] = stacked
    return out
