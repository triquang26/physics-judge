"""Trained heads: patch tokens -> pose. Every head here is CPU-testable with
tiny tensors and needs no backbone -- construct one directly.
"""
from kinescore.heads.attentive import AttentivePoseHead
from kinescore.heads.disentangled import DisentangledPoseHead
from kinescore.heads.heteroscedastic import (FramePool, ReadoutV2Head,
                                             TemporalEncoder)
from kinescore.heads.mlp import DinoPoseHead
from kinescore.heads.ranges import clamp_for_fk, squash_to_limits
from kinescore.heads.views import ViewEmbedding

__all__ = [
    "AttentivePoseHead", "DinoPoseHead", "DisentangledPoseHead",
    "ReadoutV2Head", "FramePool", "TemporalEncoder",
    "ViewEmbedding", "squash_to_limits", "clamp_for_fk",
]
