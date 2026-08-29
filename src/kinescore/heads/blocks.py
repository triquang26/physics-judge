"""Blocks the diffusion head is built from."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["KeypointQueryDecoder", "TemporalEncoder",
           "masked_smooth_l1", "temporal_tracks"]


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor, *, beta: float) -> torch.Tensor:
    """Smooth-L1 between ``(B, T, K, 3)`` tensors over the frames ``mask`` keeps.
    """
    per_frame = F.smooth_l1_loss(
        pred, target, beta=beta, reduction="none").mean(dim=(-1, -2))
    return (per_frame * mask).sum() / mask.sum().clamp_min(1e-8)


class _CrossBlock(nn.Module):
    """Queries read prepared keys and values, then a feed-forward.

    Args:
        d_model: Query width.
        nhead: Attention heads.
        ff: Feed-forward width.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, nhead: int, ff: int, dropout: float
                ) -> None:
        super().__init__()
        self.nhead = int(nhead)
        self.d_head = d_model // self.nhead
        self.q_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff, d_model))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
                ) -> torch.Tensor:
        """``(N, K, d)`` against ``(N, h, P, d_head)`` -> ``(N, K, d)``."""
        n, kp, d = q.shape
        qh = (self.q_proj(self.q_norm(q))
              .reshape(n, kp, self.nhead, self.d_head).transpose(1, 2))
        attended = F.scaled_dot_product_attention(qh, k, v)
        q = q + self.out_proj(attended.transpose(1, 2).reshape(n, kp, d))
        return q + self.ff(self.ff_norm(q))


class KeypointQueryDecoder(nn.Module):
    """``(B, T, P, D) -> (B, T, K, d_model)``, one query per keypoint.

    Args:
        in_dim: Token embed width ``D``.
        n_keypoints: ``K``, queries and points.
        n_views: Views packed into a frame.
        tokens_per_view: Patch tokens one view contributes; must be square.
        d_model: Query and token width.
        nhead: Attention heads.
        ff: Feed-forward width.
        n_layers: Cross-attention blocks.
        dropout: Dropout probability.
    """

    def __init__(self, in_dim: int = 1024, n_keypoints: int = 12,
                 n_views: int = 1, tokens_per_view: int = 576,
                 d_model: int = 512, nhead: int = 8, ff: int = 2048,
                 n_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        grid = int(math.isqrt(int(tokens_per_view)))
        if grid * grid != int(tokens_per_view):
            raise ValueError(
                f"tokens_per_view must be a square patch grid, got "
                f"{tokens_per_view}")
        self.in_dim = int(in_dim)
        self.n_keypoints = int(n_keypoints)
        self.n_views = int(n_views)
        self.tokens_per_view = int(tokens_per_view)
        self.d_model = int(d_model)
        self.grid = grid

        self.norm = nn.LayerNorm(self.in_dim)
        self.proj = nn.Linear(self.in_dim, self.d_model)
        self.view_emb = nn.Parameter(torch.zeros(self.n_views, self.d_model))
        self.row_emb = nn.Parameter(torch.zeros(grid, self.d_model))
        self.col_emb = nn.Parameter(torch.zeros(grid, self.d_model))
        self.query = nn.Parameter(torch.zeros(self.n_keypoints, self.d_model))
        for p in (self.view_emb, self.row_emb, self.col_emb, self.query):
            nn.init.trunc_normal_(p, std=0.02)
        self.nhead = int(nhead)
        self.kv_norm = nn.LayerNorm(self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.blocks = nn.ModuleList(
            [_CrossBlock(self.d_model, nhead, ff, dropout)
             for _ in range(int(n_layers))])
        self.out_norm = nn.LayerNorm(self.d_model)

    @property
    def n_tokens(self) -> int:
        """Tokens one frame carries, ``n_views * tokens_per_view``."""
        return self.n_views * self.tokens_per_view

    def positions(self) -> torch.Tensor:
        """``(P, d_model)`` position vectors, view then row then column."""
        pos = (self.view_emb[:, None, None, :]
               + self.row_emb[None, :, None, :]
               + self.col_emb[None, None, :, :])
        return pos.reshape(self.n_tokens, self.d_model)

    def keys_values(self, feat: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, T, P, D)`` -> keys and values, ``(B*T, h, P, d_head)`` each.

        Raises:
            ValueError: If ``feat`` is not four-dimensional, or carries a
                token count the decoder was not built for.
        """
        if feat.dim() != 4:
            raise ValueError(
                f"KeypointQueryDecoder expects (B, T, P, D), got shape "
                f"{tuple(feat.shape)}")
        b, t, p, _ = feat.shape
        if p != self.n_tokens:
            raise ValueError(
                f"KeypointQueryDecoder is built for {self.n_tokens} tokens "
                f"({self.n_views} views x {self.tokens_per_view}), got {p}")
        n = b * t
        kv = self.proj(self.norm(feat)).reshape(n, p, self.d_model)
        kv = self.kv_norm(kv + self.positions())
        heads = (n, p, self.nhead, self.d_model // self.nhead)
        return (self.k_proj(kv).reshape(heads).transpose(1, 2),
                self.v_proj(kv).reshape(heads).transpose(1, 2))

    def queries(self, n: int) -> torch.Tensor:
        """``(n, K, d_model)``: one learned query per keypoint, per frame."""
        return self.query.expand(n, self.n_keypoints, self.d_model)

    def decode(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
              ) -> torch.Tensor:
        """``(N, K, d)`` queries against :meth:`keys_values` -> ``(N, K, d)``."""
        for block in self.blocks:
            q = block(q, k, v)
        return self.out_norm(q)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """``(B, T, P, D) -> (B, T, K, d_model)``."""
        k, v = self.keys_values(feat)
        b, t = feat.shape[0], feat.shape[1]
        z = self.decode(self.queries(b * t), k, v)
        return z.reshape(b, t, self.n_keypoints, self.d_model)


class TemporalEncoder(nn.Module):
    """Bidirectional Transformer over one track's frames.

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
        """``(N, T, d_model) -> (N, T, d_model)``."""
        if not use_context:
            return z
        t = z.shape[1]
        if t > self.t_max:
            raise ValueError(
                f"TemporalEncoder got T={t} frames, above t_max={self.t_max}; "
                f"read the clip in windows of {self.t_max} or raise t_max and "
                f"retrain")
        return self.encoder(z + self.pos_emb[:, :t])


def temporal_tracks(encoder: TemporalEncoder, z: torch.Tensor) -> torch.Tensor:
    """``(B, T, K, d)`` through ``encoder``, one track per keypoint."""
    b, t, k, d = z.shape
    tracks = encoder(z.permute(0, 2, 1, 3).reshape(b * k, t, d))
    return tracks.reshape(b, k, t, d).permute(0, 2, 1, 3)
