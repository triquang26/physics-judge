"""The per-camera additive token bias, and the multiview guard it enforces.

The source (``judge.pixel_judge.AttentivePoseHead``) added a per-camera
"camera-ID" bias to patch tokens only ``if n_cams > 1``, and validated the
token count with a single line: ``if N % self.n_cams != 0: raise``. That check
only catches a *divisibility* mismatch. It does **not** catch a 3-view,
49-tokens-per-view feature cache (``N=147``) being fed to a checkpoint trained
with ``n_cams=1`` -- ``147 % 1 == 0``, so the bias-and-guard branch was never
even entered, and a 1-camera head silently consumed a 3-camera feature grid
for its entire training run (defect D4). The bug is structural: the guard
lived *inside* the ``n_cams > 1`` branch, so the single-camera case -- the
overwhelmingly common one -- had no guard at all.

:class:`ViewEmbedding` fixes this by making the bias **unconditional** (always
constructed, zero-init) and by validating tokens against a full
:class:`~kinescore.core.clip.ViewLayout` -- which, when built with
``tokens_per_view`` set, checks the *exact* expected count
(``n_views * tokens_per_view``) rather than mere divisibility, in both
directions (see :meth:`ViewLayout.assert_tokens`).

Zero-init is what makes this backward compatible: a ``ViewEmbedding`` built
with ``n_views=1`` adds a bias of exactly ``0`` to every token, so routing
single-view features through it is bit-identical to not routing them through
anything (see ``tests/test_multiview_layout.py``). That means new heads (or
readers composing heads that never had multiview support at all --
:class:`~kinescore.heads.heteroscedastic.ReadoutV2Head`,
:class:`~kinescore.heads.disentangled.DisentangledPoseHead`,
:class:`~kinescore.heads.mlp.DinoPoseHead`) can gain the same multiview
capability by composing this module in front of them, without special-casing
the single-view path.

Note: :class:`~kinescore.heads.attentive.AttentivePoseHead` does **not** use
this module internally -- it keeps its own source-verbatim, checkpoint-
load-bearing ``cam_emb`` (conditional on ``n_cams > 1``, exact parameter name
and shape) so real checkpoints (``judge_v3l_mv``) still ``load_state_dict``
with ``strict=True``. It gets the same D4 fix a different way: see its
docstring.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from kinescore.core.clip import ViewLayout

__all__ = ["ViewEmbedding"]


class ViewEmbedding(nn.Module):
    """Unconditional per-camera additive token bias + the D4 token guard.

    Parameters
    ----------
    in_dim:
        Token feature width ``D``.
    view_layout:
        Camera packing this embedding was built for. When
        ``view_layout.tokens_per_view`` is known, :meth:`forward` validates
        the *exact* token count (both directions); otherwise it falls back to
        the weaker divisibility check (see
        :meth:`~kinescore.core.clip.ViewLayout.assert_tokens`).
    """

    def __init__(self, in_dim: int, view_layout: ViewLayout = ViewLayout()) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.view_layout = view_layout
        # Always present, zero-init -- see module docstring. Shape (V, D).
        self.cam_emb = nn.Parameter(
            torch.zeros(self.view_layout.n_views, self.in_dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(..., N, D) -> (..., N, D)``, per-view bias added.

        ``N`` is the concatenation of ``n_views`` contiguous blocks in camera
        order (``[cam0's P tokens, cam1's P tokens, ...]``), matching
        :meth:`~kinescore.backbones.dino.FeatureBackbone.encode`'s view axis
        flattened onto the token axis. Raises via
        :meth:`~kinescore.core.clip.ViewLayout.assert_tokens` if ``N``
        disagrees with this embedding's layout -- checked *before* touching
        the tokens, so a shape mismatch fails loudly instead of broadcasting
        into garbage.
        """
        N, D = tokens.shape[-2], tokens.shape[-1]
        if D != self.in_dim:
            raise ValueError(f"token width {D} != in_dim={self.in_dim}")
        self.view_layout.assert_tokens(N)
        P = N // self.view_layout.n_views
        bias = self.cam_emb.repeat_interleave(P, dim=0)  # (N, D)
        return tokens + bias.reshape(*([1] * (tokens.dim() - 2)), N, D)
