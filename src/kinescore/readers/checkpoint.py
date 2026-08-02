"""Backward-compatible re-export of :func:`kinescore.readers.loader.load_reader`.

This module used to hold both the auto-routing dispatcher (now
``readers/loader.py``) and a full save/load round-trip for
:class:`~kinescore.heads.attentive.AttentivePoseHead`. That head class has
been deleted (dead code -- its one reader, ``SquashedPoseReader``, was
removed in the D7 addendum documented in ``legacy_docs/PROVENANCE.md``, so nothing
could ever turn a saved ``AttentivePoseHead`` back into a working
:class:`~kinescore.core.reader.PoseReader`); this module now only exists so
callers importing the submodule by its historical name
(``from kinescore.readers import checkpoint``) keep working. New code should
import :func:`~kinescore.readers.loader.load_reader` from
``kinescore.readers`` or ``kinescore.readers.loader`` directly.
"""
from __future__ import annotations

from kinescore.readers.loader import load_reader

__all__ = ["load_reader"]
