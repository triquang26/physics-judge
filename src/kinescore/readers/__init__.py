"""Pose readers: frozen backbone + trained head, composed into a
:class:`~kinescore.core.reader.PoseReader`.
"""
from kinescore.readers.ensemble import EnsemblePoseReader, variance_decompose
from kinescore.readers.heteroscedastic import HeteroscedasticPoseReader
from kinescore.readers.squashed import SquashedPoseReader

__all__ = [
    "SquashedPoseReader", "HeteroscedasticPoseReader",
    "EnsemblePoseReader", "variance_decompose",
]
