"""Pose readers: frozen backbone + trained head, composed into a
:class:`~kinescore.core.reader.PoseReader`.

For loading a checkpoint into a reader, see ``kinescore.readers.checkpoint``
(``load_reader`` -- the auto-routing entry point for a single ``.pt`` file)
and ``kinescore.readers.checkpoint_v2`` (the ``ReadoutV2Head``/GR-1 loader it
delegates to, plus :class:`ReadoutV2PoseReader`).
"""
from kinescore.readers.checkpoint_v2 import ReadoutV2PoseReader
from kinescore.readers.ensemble import EnsemblePoseReader, variance_decompose
from kinescore.readers.heteroscedastic import HeteroscedasticPoseReader
from kinescore.readers.squashed import SquashedPoseReader

__all__ = [
    "SquashedPoseReader", "HeteroscedasticPoseReader",
    "EnsemblePoseReader", "variance_decompose", "ReadoutV2PoseReader",
]
