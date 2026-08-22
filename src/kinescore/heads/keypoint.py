"""The trained head: pooled DINO tokens -> 3-D keypoints.

Three stages. :class:`FramePool` collapses each frame's patch tokens to one
vector by attentive pooling; :class:`TemporalEncoder` mixes those vectors
across the clip; a linear layer reads out ``K`` points. The temporal stage is
bidirectional -- judging is offline, the whole clip is on disk -- and enters
through a residual, so a head run per-frame (``use_context=False``) is never
worse than one with no temporal stage at all.

Output is ``(B, T, K, 3)`` metres in the robot-base frame. Nothing downstream
converts it: the violation detectors are written against points.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["FramePool", "TemporalEncoder", "KeypointHead"]


class FramePool(nn.Module):
    """``(B, T, P, D) -> (B, T, d_model)``, one vector per frame.

    LayerNorm over ``D``, learned per-patch scores, softmax over patches, then
    a multi-head weighted sum concatenated with the frame's mean token so the
    global context survives a peaked attention map.

    Args:
        in_dim: Token embed width ``D``.
        d_model: Output width.
        n_heads: Pooling heads.
    """

    def __init__(self, in_dim: int = 1024, d_model: int = 768, n_heads: int = 8
                ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.norm = nn.LayerNorm(self.in_dim)
        self.score = nn.Linear(self.in_dim, self.n_heads)
        self.proj = nn.Linear(self.in_dim * (self.n_heads + 1), self.d_model)
        self.act = nn.SiLU()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """``(B, T, P, D) -> (B, T, d_model)``."""
        if feat.dim() != 4:
            raise ValueError(
                f"FramePool expects (B, T, P, D), got shape {tuple(feat.shape)}")
        x = self.norm(feat.float())
        w = self.score(x).softmax(dim=2)
        pooled = torch.einsum("btph,btpd->bthd", w, x)
        pooled = pooled.reshape(*pooled.shape[:2], -1)
        z = torch.cat([pooled, x.mean(dim=2)], dim=-1)
        return self.act(self.proj(z))


class TemporalEncoder(nn.Module):
    """Bidirectional Transformer over frame tokens.

    Pre-norm encoder with a learned positional table covering ``t_max`` frames
    and no causal mask. ``use_context=False`` returns the input untouched, so
    the same weights serve per-frame reads.

    Args:
        d_model: Token width.
        nhead: Attention heads.
        ff: Feed-forward width.
        n_layers: Encoder layers.
        t_max: Frames the positional table covers.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int = 768, nhead: int = 8, ff: int = 2048,
                 n_layers: int = 4, t_max: int = 64, dropout: float = 0.1
                ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.ff = int(ff)
        self.n_layers = int(n_layers)
        self.t_max = int(t_max)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.n_layers)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.t_max, self.d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, z: torch.Tensor, use_context: bool = True) -> torch.Tensor:
        """``(B, T, d_model) -> (B, T, d_model)``."""
        if not use_context:
            return z
        t = z.shape[1]
        if t > self.t_max:
            raise ValueError(
                f"TemporalEncoder got T={t} frames, above t_max={self.t_max}; "
                f"read the clip in windows of {self.t_max} or raise t_max and "
                f"retrain")
        return self.encoder(z + self.pos_emb[:, :t])


class KeypointHead(nn.Module):
    """:class:`FramePool` -> residual :class:`TemporalEncoder` -> ``K`` points.

    Args:
        in_dim: Token embed width ``D``.
        n_keypoints: ``K``, points predicted per frame.
        d_model: Frame/temporal token width.
        n_heads: Pooling heads.
        temporal_nhead: Temporal-attention heads.
        ff: Temporal feed-forward width.
        n_temporal_layers: Temporal encoder layers.
        t_max: Frames the positional table covers.
        dropout: Dropout probability.

    Shape:
        - Input: ``(B, T, P, D)``.
        - Output: ``(B, T, K, 3)`` metres, robot-base frame.
    """

    def __init__(self, in_dim: int = 1024, n_keypoints: int = 12,
                 d_model: int = 768, n_heads: int = 8, temporal_nhead: int = 8,
                 ff: int = 2048, n_temporal_layers: int = 4, t_max: int = 64,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if int(n_keypoints) < 1:
            raise ValueError(f"n_keypoints must be >= 1, got {n_keypoints}")
        self.in_dim = int(in_dim)
        self.n_keypoints = int(n_keypoints)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.temporal_nhead = int(temporal_nhead)
        self.ff = int(ff)
        self.n_temporal_layers = int(n_temporal_layers)
        self.t_max = int(t_max)
        self.dropout = float(dropout)

        self.pool = FramePool(in_dim=self.in_dim, d_model=self.d_model,
                              n_heads=self.n_heads)
        self.temporal = TemporalEncoder(
            d_model=self.d_model, nhead=self.temporal_nhead, ff=self.ff,
            n_layers=self.n_temporal_layers, t_max=self.t_max, dropout=dropout,
        )
        self.mu_head = nn.Linear(self.d_model, self.n_out)

    @property
    def n_out(self) -> int:
        """Readout width, ``3 * n_keypoints``. The checkpoint records this."""
        return 3 * self.n_keypoints

    def forward(self, feat: torch.Tensor, use_context: bool = True
               ) -> torch.Tensor:
        """``(B, T, P, D) -> (B, T, K, 3)``."""
        z_frame = self.pool(feat)
        z = z_frame + self.temporal(z_frame, use_context=use_context)
        return self.mu_head(z).reshape(*z.shape[:2], self.n_keypoints, 3)
