"""Model package: transformer architecture, heads, losses, ensembles."""

from recad.model.attention import (
    MultiHeadSelfAttention,
    Sinusoidal2DPosEnc,
    TransformerBlock,
    TransformerEncoder,
)
from recad.model.ensemble import DeepEnsemble, EnsembleMoments, pick_device
from recad.model.factory import build_model
from recad.model.heads import build_head
from recad.model.losses import (
    build_loss,
    gaussian_nll_loss,
    mse_loss_masked,
    quantile_loss,
)
from recad.model.st_transformer import SpatioTemporalTransformer, count_parameters

__all__ = [
    "DeepEnsemble",
    "EnsembleMoments",
    "MultiHeadSelfAttention",
    "Sinusoidal2DPosEnc",
    "SpatioTemporalTransformer",
    "TransformerBlock",
    "TransformerEncoder",
    "build_head",
    "build_loss",
    "build_model",
    "count_parameters",
    "gaussian_nll_loss",
    "mse_loss_masked",
    "pick_device",
    "quantile_loss",
]
