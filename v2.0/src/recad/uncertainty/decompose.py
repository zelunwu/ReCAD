"""Combination of the three uncertainty contributions.

The v1.1 product combined input-error contributions in quadrature
(``u_inputs_Monte_Carlo.mlx``: ``diff_pco2_all = sqrt(sum of squared per-input
stds)``), treating the sources as independent. v2.0 extends the same rule to
combine the model-intrinsic terms (aleatoric, epistemic; see
``recad.model.ensemble``) with the input-error term:

    sigma_total = sqrt(sigma_aleatoric^2 + sigma_epistemic^2 + sigma_input^2)

All inputs are arrays over the 4-D domain; ``mode='rss'`` is the only
supported combination currently (``mode='none'`` returns the model term only).
"""

from __future__ import annotations

import numpy as np


def combine_uncertainties(
    aleatoric_std: np.ndarray,
    epistemic_std: np.ndarray,
    input_std: np.ndarray | None,
    *,
    mode: str = "rss",
) -> np.ndarray:
    """Combine uncertainty fields (same 4-D shape) into a total std field."""
    model_var = aleatoric_std**2 + epistemic_std**2
    if mode == "none":
        return np.sqrt(model_var)
    if mode == "rss":
        inp = np.zeros_like(model_var) if input_std is None else input_std**2
        return np.sqrt(model_var + inp)
    raise ValueError(f"unknown combine mode: {mode}")
