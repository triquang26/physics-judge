"""Multi-head attention-pool over DINO patch tokens -> joints + gripper.

Ported from ``judge.pixel_judge.AttentivePoseHead``. The forward math is
copied **verbatim** -- this is the head three real checkpoints
(``judge_v3l``, ``judge_v3l_mv``, ``judge_reward``) were trained with, so
``AttentivePoseHead(...).load_state_dict(ckpt["head"], strict=True)`` must
keep working exactly as it did in the source. That constrains two things that
would otherwise be tempting to "clean up":

* ``cam_emb`` stays **conditional** on ``n_cams > 1`` (not the unconditional,
  always-present design used by :class:`~kinescore.heads.views.ViewEmbedding`
  for *new* heads). ``judge_v3l`` and ``judge_reward`` have no ``cam_emb`` key
  in their saved ``head`` state dict at all; if this module always created the
  parameter, loading either of those real checkpoints with ``strict=True``
  would fail on a missing key. Legacy compatibility wins here.
* the einsum / concat / MLP order is copied line for line.

What *is* changed is the multiview token-count guard (defect D4). The source
checked ``N % self.n_cams != 0`` **inside** the ``n_cams > 1`` branch, so a
1-camera head (the common case) had no check at all and would silently
consume, say, a 147-token 3-camera feature grid. Here the guard is a
:class:`~kinescore.core.clip.ViewLayout` built from ``n_cams`` (optionally
with ``tokens_per_view`` for the exact-count check), asserted unconditionally
at the top of :meth:`forward` -- before the ``n_cams > 1`` branch, so it fires
for ``n_cams == 1`` too.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from kinescore.core.clip import ViewLayout

__all__ = ["AttentivePoseHead"]


class AttentivePoseHead(nn.Module):
    """Multi-head attention-pool over DINO **patch tokens** -> joints+gripper.

    The CLS-token head is pixel-global: it cannot localise the (small) arm in
    the frame, so distal joints are read poorly. This head instead takes the
    full patch grid ``(N, D)`` and learns ``n_heads`` soft attention maps over
    the patches -- each head can lock onto a different part of the arm --
    then pools and regresses pose. Standard "attentive probe" evaluation of
    frozen DINO features on dense/geometric tasks.

    Parameters
    ----------
    in_dim, hidden, n_joints, dropout, n_heads:
        As the source. Output width is ``n_joints + 1`` (joints + a trailing
        gripper/opening logit).
    n_cams:
        Number of camera views packed into the token axis (``N = n_cams *
        P``). ``n_cams == 1`` (default) creates no ``cam_emb`` parameter at
        all, matching every single-view checkpoint in production.
    tokens_per_view:
        Optional per-camera token count ``P``, used only to strengthen the D4
        guard to an exact-count check (see module docstring). ``None``
        (default) falls back to the source's divisibility check, which is
        what a checkpoint loaded without this information gets.
    """

    def __init__(self, in_dim: int = 768, hidden: int = 512, n_joints: int = 7,
                 dropout: float = 0.1, n_heads: int = 4, n_cams: int = 1,
                 tokens_per_view: Optional[int] = None) -> None:
        super().__init__()
        self.n_joints = int(n_joints)
        self.n_heads = int(n_heads)
        self.n_cams = int(n_cams)
        self.view_layout = ViewLayout(n_views=self.n_cams,
                                      tokens_per_view=tokens_per_view)
        self.norm = nn.LayerNorm(in_dim)
        self.score = nn.Linear(in_dim, self.n_heads)  # per-patch attn logits
        # Zero-init => identity at start; created ONLY for n_cams>1 so
        # single-view checkpoints (no such parameter) still load strictly.
        # See module docstring.
        if self.n_cams > 1:
            self.cam_emb = nn.Parameter(torch.zeros(self.n_cams, in_dim))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * (self.n_heads + 1), hidden),  # heads ⊕ mean
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.n_joints + 1),
        )

    def forward(self, feat: torch.Tensor, return_attn: bool = False
               ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """``(B, T, N, D) -> (B, T, n_joints + 1)`` raw (pre-squash) pose.

        For ``n_cams>1`` the ``N`` patch tokens are the concatenation of the
        per-camera grids in camera order (``[cam0 P tokens, cam1 P tokens,
        ...]``); the camera-ID embedding is added to each contiguous block of
        ``P = N // n_cams`` tokens before pooling. The concat order must
        match the encoder's order (see
        :meth:`kinescore.backbones.dino.FeatureBackbone.encode`).

        ``return_attn=True`` additionally returns the per-patch attention
        weights ``w`` of shape ``(B, T, N, n_heads)`` (softmax over patches) --
        the spatial support the head reads pose from.
        """
        B, T, N, D = feat.shape
        self.view_layout.assert_tokens(N)  # D4 fix: fires even for n_cams==1
        if self.n_cams > 1:
            P = N // self.n_cams
            feat = feat + self.cam_emb.repeat_interleave(P, dim=0).view(1, 1, N, D)
        x = self.norm(feat)  # (B,T,N,D)
        w = self.score(x).softmax(dim=2)  # (B,T,N,H) over patches
        pooled = torch.einsum("btnh,btnd->bthd", w, x)  # (B,T,H,D)
        pooled = pooled.reshape(*pooled.shape[:2], -1)  # (B,T,H*D)
        glob = x.mean(dim=2)  # (B,T,D) global context
        z = torch.cat([pooled, glob], dim=-1)  # (B,T,(H+1)*D)
        out = self.mlp(z)
        if return_attn:
            return out, w
        return out
