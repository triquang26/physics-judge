"""Unsquashed heteroscedastic *temporal* pose head (``readout-v2``).

Ported from ``models.posendf.readout_v2``. ``mu`` is **raw radians** -- no
sigmoid / limit squash -- so limit violations stay observable (the whole
point of a physics judge: a valid-by-construction head can never report an
out-of-limit pose; see ``heads/ranges.py`` for the D7 writeup and
:func:`~kinescore.heads.ranges.clamp_for_fk`, which turns ``mu`` into an
FK-safe pose plus the violation magnitude). ``sigma`` (from ``logvar``) is the
model's **aleatoric** uncertainty only -- the irreducible per-joint noise it
learned to expect -- it is **not** an OOD detector. The OOD axis is *ensemble
disagreement* (epistemic variance across members), computed in
``readers/ensemble.py``.

What this adds over a per-frame attentive probe
-------------------------------------------------
* **Temporal context.** A per-frame :class:`FramePool` (structurally the same
  attentive-probe pool as :class:`~kinescore.heads.attentive.AttentivePoseHead`,
  minus the multiview ``cam_emb`` -- multiview support for this head is
  composed externally via :class:`~kinescore.heads.views.ViewEmbedding`, see
  ``readers/heteroscedastic.py``) feeds a bidirectional
  :class:`TemporalEncoder`; the two are combined by a residual so a single
  frame is never worse than the per-frame head. ``use_context=False`` bypasses
  the temporal attention entirely (per-frame mode).
* **Heteroscedastic head.** Alongside ``mu`` the head emits ``logvar``
  (clamped), so ``sigma`` tracks the target noise instead of collapsing.

Only the inference-time surface is ported here (:class:`FramePool`,
:class:`TemporalEncoder`, :class:`ReadoutV2Head`). The training-time losses
(``beta_nll_loss``, ``fit_sigma_temperature``) are not part of the pose-reader
contract this package owns and live with the training code that consumes
them.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

__all__ = ["FramePool", "TemporalEncoder", "ReadoutV2Head"]


class FramePool(nn.Module):
    """Per-frame attentive-probe pool ``(B, T, P, D) -> (B, T, d_model)``.

    LayerNorm over ``D`` -> learned per-patch scores -> softmax over patches
    -> multi-head einsum pool, concatenated with the global mean token, then
    a linear projection + SiLU to ``d_model``.

    Args:
        in_dim: Token embed width ``D``.
        d_model: Output width.
        n_heads: Number of attention-pool heads.
    """

    def __init__(self, in_dim: int = 1024, d_model: int = 512, n_heads: int = 4
                ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.norm = nn.LayerNorm(self.in_dim)
        self.score = nn.Linear(self.in_dim, self.n_heads)  # per-patch logits
        self.proj = nn.Linear(self.in_dim * (self.n_heads + 1), self.d_model)
        self.act = nn.SiLU()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """``(B, T, P, D) -> (B, T, d_model)``."""
        if feat.dim() != 4:
            raise ValueError(
                f"FramePool expects (B, T, P, D) input, got shape {tuple(feat.shape)}")
        x = self.norm(feat.float())  # (B,T,P,D)
        w = self.score(x).softmax(dim=2)  # (B,T,P,H) over patches
        pooled = torch.einsum("btph,btpd->bthd", w, x)  # (B,T,H,D)
        pooled = pooled.reshape(*pooled.shape[:2], -1)  # (B,T,H*D)
        glob = x.mean(dim=2)  # (B,T,D) global context
        z = torch.cat([pooled, glob], dim=-1)  # (B,T,(H+1)*D)
        return self.act(self.proj(z))  # (B,T,d_model)


class TemporalEncoder(nn.Module):
    """Bidirectional temporal Transformer over frame tokens.

    A pre-norm (``norm_first=True``) ``nn.TransformerEncoder`` with a learned
    positional embedding (up to ``t_max`` frames). **Bidirectional** -- no
    causal mask, because judging is offline / the whole clip is available.
    When ``use_context`` is ``False`` the attention (and positional embedding)
    is bypassed and ``z`` is returned unchanged (per-frame mode).

    Args:
        d_model: Token width.
        nhead: Attention heads.
        ff: Feed-forward width.
        n_layers: Encoder layers.
        t_max: Max frames the positional table covers.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, ff: int = 1024,
                 n_layers: int = 2, t_max: int = 64, dropout: float = 0.1
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
        """``(B, T, d_model) -> (B, T, d_model)``.

        With ``use_context=False`` returns ``z`` unchanged (no attention, no
        positional embedding).
        """
        if not use_context:
            return z
        t = z.shape[1]
        if t > self.t_max:
            raise ValueError(
                f"TemporalEncoder T={t} exceeds t_max={self.t_max}; raise t_max")
        z = z + self.pos_emb[:, :t]  # learned positions
        return self.encoder(z)  # bidirectional (no mask)


class ReadoutV2Head(nn.Module):
    """Unsquashed heteroscedastic temporal pose head.

    Pipeline: :class:`FramePool` -> per-frame tokens ``z_frame`` -> residual
    with a bidirectional :class:`TemporalEncoder` -> two linear heads
    emitting ``mu`` (raw radians, no squash) and ``logvar`` (clamped to
    ``[logvar_min, logvar_max]``).

    Args:
        in_dim: Token embed width ``D``.
        d_model: Frame/temporal token width.
        n_heads: Attention-pool heads in :class:`FramePool`.
        temporal_nhead: Temporal-attention heads.
        ff: Temporal feed-forward width.
        n_temporal_layers: Temporal encoder layers.
        t_max: Max frames (positional table).
        dropout: Dropout.
        n_out: Output width (e.g. 29 = 17 joints + 12 hand DoF for GR-1; 7 for
            a single arm). Named ``n_out`` here (the source hardcoded 29 for
            its one embodiment); :meth:`split` needs ``n_q`` to divide it.
        logvar_min/max: ``logvar`` clamp bounds.

    Shape:
        - Input ``feat``: ``(B, T, P, D)``.
        - Output: ``{"mu": (B,T,n_out), "logvar": (B,T,n_out)}``.
    """

    def __init__(self, in_dim: int = 1024, d_model: int = 512, n_heads: int = 4,
                 temporal_nhead: int = 8, ff: int = 1024,
                 n_temporal_layers: int = 2, t_max: int = 64, dropout: float = 0.1,
                 n_out: int = 29, logvar_min: float = -10.0,
                 logvar_max: float = 4.0) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.temporal_nhead = int(temporal_nhead)
        self.ff = int(ff)
        self.n_temporal_layers = int(n_temporal_layers)
        self.t_max = int(t_max)
        self.dropout = float(dropout)
        self.n_out = int(n_out)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

        self.pool = FramePool(in_dim=self.in_dim, d_model=self.d_model,
                              n_heads=self.n_heads)
        self.temporal = TemporalEncoder(
            d_model=self.d_model, nhead=self.temporal_nhead, ff=self.ff,
            n_layers=self.n_temporal_layers, t_max=self.t_max, dropout=dropout,
        )
        self.mu_head = nn.Linear(self.d_model, self.n_out)  # RAW radians
        self.logvar_head = nn.Linear(self.d_model, self.n_out)  # clamped logvar

    def forward(self, feat: torch.Tensor, use_context: bool = True
               ) -> Dict[str, torch.Tensor]:
        """``(B, T, P, D) -> {"mu": (B,T,n_out), "logvar": (B,T,n_out)}``.

        ``mu`` is raw radians (no squash). ``logvar`` is clamped into
        ``[logvar_min, logvar_max]``. With ``use_context=False`` the temporal
        attention is bypassed (per-frame mode) but the residual wiring is
        preserved.
        """
        z_frame = self.pool(feat)  # (B,T,d_model)
        z = z_frame + self.temporal(z_frame, use_context=use_context)  # residual
        mu = self.mu_head(z)  # (B,T,n_out) raw radians
        logvar = self.logvar_head(z).clamp(self.logvar_min, self.logvar_max)
        return {"mu": mu, "logvar": logvar}
