"""The legacy CLS-token MLP head.

LEGACY -- not on any wired path
--------------------------------
Superseded by :class:`~kinescore.heads.attentive.AttentivePoseHead` (the live
Franka path -- see ``readers/squashed.py``). This head is kept for
provenance / possible future use, but it is **not** part of either of
kinescore's two live pose-reader paths (Franka -> ``AttentivePoseHead`` ->
:class:`~kinescore.readers.squashed.SquashedPoseReader`; GR-1 ->
:class:`~kinescore.heads.heteroscedastic.ReadoutV2Head` ->
:class:`~kinescore.readers.heteroscedastic.HeteroscedasticPoseReader` /
:class:`~kinescore.readers.checkpoint_v2.ReadoutV2PoseReader`). There is no
checkpoint loader or reader composition for :class:`DinoPoseHead` in
``readers/``, and none of ``cli/``, ``training/`` import this module on a
default (no-flag) code path. See ``docs/PROVENANCE.md`` for the correction
that made ReadoutV2 first-class and this head explicitly secondary.

Ported **verbatim** from ``judge.pixel_judge.DinoPoseHead``. Kept for
checkpoints trained with ``pool="cls"`` and as the simplest possible
comparison point against the attentive probe (:mod:`kinescore.heads.attentive`)
-- it reads a single pooled feature vector per frame rather than the full
patch grid, so it cannot localise the arm and is expected to read distal
joints worse. No multiview handling: it takes one vector per frame, so there
is nothing for :class:`~kinescore.heads.views.ViewEmbedding` to be inserted
in front of unless the caller pre-pools the per-view vectors into one.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["DinoPoseHead"]


class DinoPoseHead(nn.Module):
    """Small trainable MLP: frozen DINO feature ``(D,)`` -> joints+gripper.

    Kept deliberately small (one hidden layer) so it cannot memorise scenes --
    it must lean on the frozen backbone's geometry features.
    """

    def __init__(self, in_dim: int = 768, hidden: int = 512, n_joints: int = 7,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.n_joints = int(n_joints)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.n_joints + 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """``(B, T, D) -> (B, T, n_joints + 1)`` raw (pre-squash) pose."""
        return self.net(feat)
