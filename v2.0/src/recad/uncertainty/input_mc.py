"""Monte-Carlo propagation of input measurement errors (v1.1 protocol).

Mirrors ``v1.1/u_inputs_Monte_Carlo.mlx``:

  * draw N Gaussian perturbations of the predictor fields with the per-input
    sigmas of ``UncertaintyConfig.input_uncertainties`` (v1.1 defaults:
    u_sst=0.23 °C, u_sss=0.6 PSU, u_ssh=0.018 m, u_u10=0.901 m/s, u_pco2air
    = 0.22 % relative);
  * re-tokenize and re-predict with the *frozen* ensemble for each draw (the
    honest, rigorous route - perturbed inputs propagate into the patch tokens
    they define);
  * the standard deviation of the ensemble-mean predictions across draws is
    the input-error contribution.

The computational cost scales as n_mc_draws x ensemble cost; ``tile_cells``
subsamples cells deterministically for large domains (documented in
docs/design.md §Uncertainty).
"""

from __future__ import annotations

import numpy as np

from recad.config import UncertaintyConfig
from recad.data.features import FEATURE_NAMES, PreparedData
from recad.model.ensemble import DeepEnsemble, pick_device
from recad.utils.logging import get_logger
from recad.utils.seed import seed_everything

_LOG = get_logger(__name__)

# Predictors that are perturbed, and their keys in input_uncertainties.
# All perturbations are additive with the v1.1 sigmas (u_inputs_Monte_Carlo.mlx):
#   sst += N(0, u_sst); sss += N(0, u_sss); adt += N(0, u_ssh);
#   pco2air += N(0, u_pco2air); wspd += N(0, u_u10).
PERTURBED = (
    ("sst", "u_sst"),
    ("sss", "u_sss"),
    ("adt", "u_ssh"),
    ("pco2air", "u_pco2air"),
    ("wspd", "u_u10"),
)


def _perturbed_arrays(
    prepared: PreparedData, rng: np.random.Generator, cfg: UncertaintyConfig
) -> dict[str, np.ndarray]:
    """One MC draw: return copies of the perturbed predictor fields."""
    arrays = {name: prepared.require(name).copy() for name in FEATURE_NAMES}
    u = cfg.input_uncertainties
    for name, key in PERTURBED:
        sigma = u.get(key)
        if sigma is None or sigma <= 0:
            continue
        arrays[name] = arrays[name] + rng.normal(0.0, sigma, size=arrays[name].shape)
    return arrays


def propagate_input_uncertainty(
    ensemble: DeepEnsemble,
    prepared: PreparedData,
    cfg: UncertaintyConfig,
    model_cfg,
    *,
    tile_cells: int | None = None,
    batch_size: int = 1,
    device=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mc_std, mc_mean) fields of shape [n_year, 12, n_lat, n_lon].

    ``mc_std`` is the draw-to-draw standard deviation of the ensemble-mean
    prediction; ``mc_mean`` is the mean prediction across draws (useful for
    sanity reports).
    """
    device = device or pick_device("auto")
    seed_everything(42)  # deterministic MC across runs
    rng = np.random.default_rng(42)
    shape = prepared.shape4d
    accum = np.zeros(shape, dtype=np.float64)
    accum_sq = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.int64)

    for _ in range(cfg.n_mc_draws):
        arrays = _perturbed_arrays(prepared, rng, cfg)
        pert = _clone_prepared(prepared, arrays)
        moments = ensemble.predict_field(
            pert,
            model_cfg,
            tile_cells=tile_cells,
            batch_size=batch_size,
            use_amp=False,
            progress=False,
        )
        pred = moments.mean
        valid = ~np.isnan(pred)
        accum[valid] += pred[valid]
        accum_sq[valid] += pred[valid] ** 2
        counts[valid] += 1

    n = np.maximum(counts, 1)
    mc_mean = accum / n
    variance = np.maximum(accum_sq / n - mc_mean**2, 0.0)
    mc_std = np.sqrt(variance)
    mc_std[counts == 0] = np.nan
    mc_mean[counts == 0] = np.nan
    return mc_std.astype(np.float32), mc_mean.astype(np.float32)


def _clone_prepared(prepared: PreparedData, arrays: dict[str, np.ndarray]) -> PreparedData:
    """Shallow-copy PreparedData with replaced predictor arrays."""
    new_arrays = dict(prepared.arrays)
    new_arrays.update(arrays)
    return PreparedData(
        grid=prepared.grid,
        coastal_mask=prepared.coastal_mask,
        years=prepared.years,
        target=prepared.target,
        arrays=new_arrays,
        _feature_stats=prepared._feature_stats,
    )
