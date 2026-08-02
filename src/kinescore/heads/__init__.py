"""Trained heads: patch tokens -> pose. Every head here is CPU-testable with
tiny tensors and needs no backbone -- construct one directly.

``AttentivePoseHead`` (multiview attentive probe), ``DinoPoseHead`` (CLS-token
MLP) and ``DisentangledPoseHead`` (per-DoF query attention) have been removed:
none of them could produce a working :class:`~kinescore.core.reader.PoseReader`
any more (their one reader, ``SquashedPoseReader``, was removed -- see
``legacy_docs/PROVENANCE.md``'s D7 addendum) and nothing else in this package
constructed them. :class:`ReadoutV2Head` is the one live head family.
"""
from kinescore.heads.heteroscedastic import FramePool, ReadoutV2Head, TemporalEncoder
from kinescore.heads.ranges import clamp_for_fk
from kinescore.heads.views import ViewEmbedding

__all__ = [
    "ReadoutV2Head", "FramePool", "TemporalEncoder",
    "ViewEmbedding", "clamp_for_fk",
]
