"""Joint ST model with small task-specific adapters for sparse carbonate labels."""

from __future__ import annotations

import torch
from torch import nn

from recad.config import ModelConfig
from recad.data.tensorize import TensorShapes
from recad.model.st_transformer import SpatioTemporalTransformer


class RegularizedJointHead(nn.Module):
    """Separate hydrography and carbonate adapters prevent task domination."""

    def __init__(self, d_in: int, hidden: int, chem_width: int = 32) -> None:
        super().__init__()
        self.hydro_adapter = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.chem_adapter = nn.Sequential(
            nn.Linear(d_in, chem_width), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(chem_width, chem_width), nn.GELU(),
        )
        self.sss = nn.Linear(hidden, 1)
        self.fco2 = nn.Linear(hidden, 1)
        self.dic = nn.Linear(chem_width, 1)
        self.ta = nn.Linear(chem_width, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hydro = self.hydro_adapter(x)
        chem = self.chem_adapter(x)
        return {"sss": self.sss(hydro)[..., 0], "fco2": self.fco2(hydro)[..., 0],
                "dic": self.dic(chem)[..., 0], "ta": self.ta(chem)[..., 0]}


class RegularizedJointST(SpatioTemporalTransformer):
    def __init__(self, shapes: TensorShapes, cfg: ModelConfig, chem_width: int = 32) -> None:
        super().__init__(shapes, cfg)
        self.head = RegularizedJointHead(2 * cfg.embed_dim, cfg.embed_dim, chem_width)

    def freeze_for_chemistry_adapter(self) -> None:
        """Freeze dense-data representation; expose only sparse chemistry adapter."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in (self.head.chem_adapter, self.head.dic, self.head.ta):
            for parameter in module.parameters(): parameter.requires_grad_(True)

    def unfreeze_last_blocks(self) -> None:
        """Enable task heads, cell projection and final spatial/temporal blocks."""
        for parameter in self.parameters(): parameter.requires_grad_(False)
        for module in (self.head, self.cell_embed, self.spatial.blocks[-1], self.temporal.blocks[-1]):
            for parameter in module.parameters(): parameter.requires_grad_(True)

