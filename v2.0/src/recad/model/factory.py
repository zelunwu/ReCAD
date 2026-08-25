"""Model factory: build the ST-Transformer from a config and data shapes."""

from __future__ import annotations

import torch
from torch import nn

from recad.config import ModelConfig
from recad.data.tensorize import TensorShapes
from recad.model.st_transformer import SpatioTemporalTransformer


def build_model(shapes: TensorShapes, cfg: ModelConfig, seed: int | None = None) -> nn.Module:
    """Instantiate the model configured by ``cfg``.

    With ``seed`` set, torch's RNG is reseeded first (member reproducibility).
    """
    if seed is not None:
        torch.manual_seed(seed)
    if cfg.name == "st_transformer":
        return SpatioTemporalTransformer(shapes, cfg)
    raise ValueError(f"unknown model name: {cfg.name}")
