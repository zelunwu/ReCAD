"""ReCAD v2.0 flagship model: spatio-temporal transformer (ST-Transformer).

Scientific rationale (full treatment in ``docs/design.md``):

  * v1.1's random forest predicts each (month, cell) sample independently; it
    cannot exploit spatial context (neighbouring coastal cells covary through
    shared water masses) or temporal context (the seasonal cycle, interannual
    trends). The ST-Transformer adds both, which a "more advanced, more
    appropriate model" for a coastal reconstruction should do.
  * *Spatial stage*: coastal cells are grouped into spatial patches; a
    transformer encoder attends over patches within each month, so local and
    remote spatial dependencies across the coastal margin are learned.
  * *Temporal stage*: for every patch, a second transformer encoder attends
    across the temporal window (default = 12 months = one seasonal cycle),
    capturing seasonal/interannual co-variation.
  * *Decoder head*: per cell, the head concatenates the cell's own (z-scored,
    missing-flagged) feature vector with its patch's temporal context and
    emits a predictive distribution (heteroscedastic Gaussian by default).

The model is device-agnostic (``torch.nn.Module``), uses explicit attention
masking, NaN-safe input handling (missing predictors are zero-filled with an
explicit missing-flag channel, matching v1.1's NaN handling), and is small
enough by default to smoke-test on CPU while scaling to a global 1/8-deg run
on GPU (see ``configs/global_1over8.yaml``).
"""

from __future__ import annotations

import torch
from torch import nn

from recad.config import ModelConfig
from recad.data.tensorize import TensorShapes
from recad.model.attention import Sinusoidal2DPosEnc, TransformerEncoder
from recad.model.heads import build_head


class SpatioTemporalTransformer(nn.Module):
    """Two-stage transformer over coastal patches and months."""

    def __init__(self, shapes: TensorShapes, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.shapes = shapes
        D = cfg.embed_dim

        # ---- token embedding (spatial stage input) ----
        self.token_embed = nn.Linear(shapes.n_token_features, D)
        self.mask_token = nn.Parameter(torch.zeros(D))
        nn.init.normal_(self.mask_token, std=0.02)
        self.month_proj = nn.Linear(2, D)
        if cfg.pos_encoding == "sinusoidal":
            self.pos_enc = Sinusoidal2DPosEnc(D)
        elif cfg.pos_encoding == "learned":
            self.pos_enc = nn.Linear(2, D)
        else:
            raise ValueError(f"unknown pos_encoding: {cfg.pos_encoding}")

        # ---- spatial stage: attention over patches within each month ----
        self.spatial = TransformerEncoder(
            D, cfg.n_heads, cfg.spatial_layers, cfg.mlp_ratio, cfg.dropout
        )

        # ---- temporal stage: attention over months for each patch ----
        self.temporal = TransformerEncoder(
            D, cfg.n_heads, cfg.temporal_layers, cfg.mlp_ratio, cfg.dropout
        )

        # ---- per-cell decoder ----
        self.cell_embed = nn.Linear(shapes.n_cell_features, D)
        self.head = build_head(cfg.head, d_in=2 * D, hidden=D)

    # ------------------------------------------------------------------
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the full ST-Transformer on one batch.

        Batch keys (from ``recad.data.tensorize.CoastalPatchDataset``):
            token_feats [B, T, P, F_token]
            token_valid [B, T, P]   (bool, attendable)
            coord       [B, T, P, 2] (patch-centre lon/lat, normalised)
            month       [B, T, P, 2] (cyclic month encoding)
            cell_feats  [B, T, C, F_cell]
            cell_patch  [B, T, C]   (token index per cell, -1 if none)

        Returns a dict with 'mean' [B, T, C] and (for the gaussian head)
        'logvar' [B, T, C].
        """
        token_feats = batch["token_feats"]
        token_valid = batch["token_valid"]
        coord = batch["coord"]
        month = batch["month"]
        B, T, P, _ = token_feats.shape

        # ---- spatial stage ----
        feats = torch.where(torch.isnan(token_feats), torch.zeros_like(token_feats), token_feats)
        tok = self.token_embed(feats)  # [B, T, P, D]
        tok = torch.where(
            token_valid[..., None], tok, self.mask_token.view(1, 1, 1, -1).expand_as(tok)
        )
        tok = tok + self.pos_enc(coord) + self.month_proj(month)

        spatial = self.spatial(
            tok.reshape(B * T, P, -1), mask=token_valid.reshape(B * T, P)
        )  # [B*T, P, D]

        # ---- temporal stage: per patch over months ----
        # [B, T, P, D] -> [B, P, T, D] -> [B*P, T, D]
        temporal_in = spatial.reshape(B, T, P, -1).transpose(1, 2)
        temporal = self.temporal(
            temporal_in.reshape(B * P, T, -1),
            mask=token_valid.transpose(1, 2).reshape(B * P, T),
        )  # [B*P, T, D]
        ctx = temporal.reshape(B, P, T, -1).transpose(1, 2)  # [B, T, P, D]

        # ---- per-cell decoder ----
        cell_feats = batch["cell_feats"]
        cell_feats = torch.where(torch.isnan(cell_feats), torch.zeros_like(cell_feats), cell_feats)
        cell_emb = self.cell_embed(cell_feats)  # [B, T, C, D]

        patch_idx = batch["cell_patch"]  # [B, T, C], long, -1 = outside
        valid_cell = patch_idx >= 0  # [B, T, C]
        safe_idx = patch_idx.clamp(min=0)  # clamp for the gather
        gather_idx = safe_idx.unsqueeze(-1).expand(-1, -1, -1, ctx.shape[-1])  # [B,T,C,D]
        ctx_cell = torch.gather(ctx, dim=2, index=gather_idx)  # [B, T, C, D]
        ctx_cell = torch.where(valid_cell[..., None], ctx_cell, torch.zeros_like(ctx_cell))

        hidden = torch.cat([cell_emb, ctx_cell], dim=-1)  # [B, T, C, 2D]
        return self.head(hidden)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters (for run reports)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
