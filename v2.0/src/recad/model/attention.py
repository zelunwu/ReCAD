"""Transformer building blocks for ReCAD v2.0.

All components are plain ``torch.nn`` modules with explicit attention masking
(``mask[i, j] = True`` means *attendable*). The mask semantics are uniform
across the code base: valid/attendable = True.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with additive padding masking.

    ``mask`` may be ``None`` (no masking) or a bool tensor of shape
    ``[N, S]`` broadcastable to ``[N, q, k]`` where True = attendable.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [N, S, D]; returns [N, S, D]."""
        N, S, _ = x.shape
        qkv = self.qkv(x).reshape(N, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each [N, H, S, hd]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [N, H, S, S]
        if mask is not None:
            if mask.dim() == 2:
                valid = mask[:, None, None, :] & mask[:, None, :, None]  # [N,1,S,S]
            elif mask.dim() == 3:
                valid = mask[:, None, :, :]  # [N,1,q,k]
            else:
                raise ValueError(f"unexpected mask dims {mask.dim()}")
            attn = attn.masked_fill(~valid, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        # A query with no valid keys at all (fully padded row) otherwise gets
        # NaN softmax weights; its output is never used downstream, so zero it.
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(N, S, self.d_model)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-LN transformer encoder block: LN -> MHA -> residual -> LN -> MLP."""

    def __init__(
        self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1
    ) -> None:
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of TransformerBlocks applied over a sequence dimension."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask)
        return x


class Sinusoidal2DPosEnc(nn.Module):
    """Learnable-free sinusoidal positional encoding for normalised lon/lat.

    ``coord`` is in [0, 1] for both axes (lon/360, (lat+90)/180). The two
    axes share the frequency schedule; each axis gets half of ``d_model``.
    """

    def __init__(self, d_model: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        if d_model < 4:
            raise ValueError("d_model must be >= 4 for 2D sinusoidal encoding")
        self.d_model = d_model
        # each of the two axes receives d_model/4 frequencies -> sin+cos gives
        # d_model/2 per axis, and 2 axes fill d_model (odd d_model zero-padded)
        per_axis = d_model // 4
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, per_axis, dtype=torch.float32) / per_axis
        )
        self.register_buffer("freqs", freqs)

    def forward(self, coord: torch.Tensor) -> torch.Tensor:
        """coord: [..., 2] in [0,1]; returns [..., d_model]."""
        lon = coord[..., 0:1] * 2.0 * math.pi  # wrap to [0, 2pi)
        lat = coord[..., 1:2] * 2.0 * math.pi
        enc = torch.cat([self._encode(lon), self._encode(lat)], dim=-1)
        if enc.shape[-1] < self.d_model:  # odd d_model: zero-pad the tail
            pad = torch.zeros(
                *enc.shape[:-1], self.d_model - enc.shape[-1], device=enc.device, dtype=enc.dtype
            )
            enc = torch.cat([enc, pad], dim=-1)
        return enc

    def _encode(self, phase: torch.Tensor) -> torch.Tensor:
        # phase: [..., 1]
        angles = phase * self.freqs  # [..., per_axis]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
