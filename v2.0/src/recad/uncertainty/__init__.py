"""Uncertainty quantification (v1.1-compatible + v2.0 additions).

v1.1 estimated the product uncertainty by Monte-Carlo propagation of input
measurement errors through the frozen model (``u_inputs_Monte_Carlo.mlx``):
100 draws of Gaussian perturbations to SST/SSS/ADT/pCO2air (with fixed or
per-pixel sigmas) and the std of the resulting predictions. v2.0 keeps this
protocol (``input_mc.py``) and *adds* the two model-intrinsic terms provided
by the deep ensemble - aleatoric (Gaussian-head variance) and epistemic
(ensemble spread) - and combines them with the input term in quadrature
(``decompose.py``), the same variance-sum rule v1.1 used for its final
combined uncertainty.
"""

from recad.uncertainty.decompose import combine_uncertainties
from recad.uncertainty.input_mc import propagate_input_uncertainty

__all__ = ["combine_uncertainties", "propagate_input_uncertainty"]
