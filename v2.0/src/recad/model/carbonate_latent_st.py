"""ST-Transformer whose carbonate state is parameterized by fCO2 and latent TA."""

from __future__ import annotations

import torch
from torch import nn

from recad.config import ModelConfig
from recad.data.tensorize import TensorShapes
from recad.model.st_transformer import SpatioTemporalTransformer


class CarbonateLatentHead(nn.Module):
    def __init__(self, d_in: int, hidden: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.outputs = nn.ModuleDict({
            name: nn.Linear(hidden, 1) for name in ("sss", "fco2", "ta", "ta_logvar")
        })
        nn.init.constant_(self.outputs["ta_logvar"].bias, -0.5)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {name: layer(h)[..., 0] for name, layer in self.outputs.items()}


class CarbonateLatentST(SpatioTemporalTransformer):
    def __init__(self, shapes: TensorShapes, cfg: ModelConfig) -> None:
        super().__init__(shapes, cfg)
        self.head = CarbonateLatentHead(2 * cfg.embed_dim, cfg.embed_dim)
