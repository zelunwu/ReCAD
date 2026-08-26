"""Prepared-data container: the model-ready feature/target tensors.

``PreparedData`` is the single in-memory structure that the data pipeline
produces and that the model/ensemble/uncertainty stages consume. It holds the
target grid, the coastal mask, and every physical field as 4-D arrays of
shape ``[n_year, 12, n_lat, n_lon]`` (NaN = missing). No machine-learning
specific encoding happens here - that is the job of
``recad.data.tensorize`` - but per-predictor normalisation statistics are
computed *only on training samples* to avoid leakage, and exposed through
:meth:`PreparedData.fit_feature_stats`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from recad.data.grid import DomainGrid
from recad.utils.logging import get_logger

_LOG = get_logger(__name__)

# Predictor order is fixed (and matches required_variables in specs.py).
FEATURE_NAMES: tuple[str, ...] = ("sst", "sss", "adt", "pco2air", "wspd")


@dataclass
class PreparedData:
    """Model-ready 4-D fields on the target mesh."""

    grid: DomainGrid
    coastal_mask: np.ndarray  # [n_lat, n_lon], bool
    years: np.ndarray  # [n_year]
    target: str = "fco2"  # which variable is the reconstruction target
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    # each entry shape [n_year, 12, n_lat, n_lon], float32/float64, NaN = missing
    _feature_stats: tuple[np.ndarray, np.ndarray] | None = None  # (mean, std)
    _target_stats: tuple[float, float] | None = None  # (mean, std) of the target

    @property
    def n_year(self) -> int:
        return int(self.years.size)

    @property
    def n_lat(self) -> int:
        return self.grid.n_lat

    @property
    def n_lon(self) -> int:
        return self.grid.n_lon

    @property
    def shape4d(self) -> tuple[int, int, int, int]:
        return (self.n_year, 12, self.n_lat, self.n_lon)

    def require(self, name: str) -> np.ndarray:
        """Return a field array, raising if it is absent."""
        if name not in self.arrays:
            raise KeyError(f"field '{name}' not present; have: {sorted(self.arrays)}")
        return self.arrays[name]

    def target_values(self) -> np.ndarray:
        """Return the reconstruction target field."""
        return self.require(self.target)

    def valid_mask(self) -> np.ndarray:
        """Cells that are coastal AND have a valid target observation."""
        valid = self.coastal_mask[None, None, :, :]
        valid = np.broadcast_to(valid, self.shape4d).copy()
        valid &= ~np.isnan(self.target_values())
        return valid

    # ------------------------------------------------------------------
    # Feature normalisation (fitted on training samples only)
    # ------------------------------------------------------------------
    def fit_feature_stats(self, train_mask: np.ndarray) -> None:
        """Compute per-predictor AND target mean/std over training samples.

        ``train_mask`` is a bool array shaped like the 4-D fields. All stats
        must be fitted on the training split only (no leakage). If a
        predictor is entirely missing in training, its std is set to NaN and
        a warning is logged (the tensorizer replaces NaN std with 1.0).
        """
        mean = np.full(len(FEATURE_NAMES), np.nan)
        std = np.full(len(FEATURE_NAMES), np.nan)
        for i, name in enumerate(FEATURE_NAMES):
            arr = self.require(name)
            sel = arr[train_mask]
            valid = sel[~np.isnan(sel)]
            if valid.size == 0:
                _LOG.warning("predictor '%s' has no valid training samples", name)
                continue
            mean[i] = float(np.mean(valid))
            std[i] = float(np.std(valid))
        if np.isnan(mean).any() or np.isnan(std).any():
            _LOG.warning(
                "some predictors are fully missing in training; "
                "they will be treated as zero-mean/unit-std"
            )
        self._feature_stats = (mean, std)

        tgt = self.target_values()[train_mask]
        tgt_valid = tgt[~np.isnan(tgt)]
        if tgt_valid.size == 0:
            _LOG.warning("no valid target samples in training")
            self._target_stats = (0.0, 1.0)
        else:
            self._target_stats = (float(np.mean(tgt_valid)), float(np.std(tgt_valid)))

    def feature_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) per predictor; fit first, or fallback to 0/1."""
        if self._feature_stats is None:
            n = len(FEATURE_NAMES)
            self._feature_stats = (np.zeros(n), np.ones(n))
        mean, std = self._feature_stats
        return mean.copy(), np.where(np.isnan(std), 1.0, std).copy()

    def target_stats(self) -> tuple[float, float]:
        """Return (mean, std) of the target; fallback to (0, 1) if unfitted."""
        if self._target_stats is None:
            self._target_stats = (0.0, 1.0)
        mean, std = self._target_stats
        return (float(mean), float(std) if std == std and std > 0 else 1.0)
