"""Pose readers: frozen backbone + trained head, composed into a
:class:`~kinescore.core.reader.PoseReader`.

For loading a checkpoint into a reader, see :func:`load_reader` (re-exported
here from ``kinescore.readers.loader`` -- the auto-routing entry point for a
single ``.pt`` file) and ``kinescore.readers.checkpoint_v2`` (the
``ReadoutV2Head``/GR-1 loader it delegates to, plus
:class:`ReadoutV2PoseReader`, and the direct-keypoint sibling loader,
:func:`~kinescore.readers.checkpoint_v2.load_direct_keypoint_reader`).
``kinescore.readers.checkpoint`` still re-exports ``load_reader`` too, for
callers pinned to that historical submodule path.
"""
from kinescore.readers.checkpoint_v2 import ReadoutV2PoseReader
from kinescore.readers.direct_keypoint import DirectKeypointPoseReader
from kinescore.readers.heteroscedastic import HeteroscedasticPoseReader
from kinescore.readers.loader import load_reader, save_reader

__all__ = [
    "HeteroscedasticPoseReader", "ReadoutV2PoseReader",
    "DirectKeypointPoseReader", "load_reader", "save_reader",
]
