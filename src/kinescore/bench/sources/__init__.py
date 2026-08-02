"""Source plugins: one class per data layout, each a :class:`~kinescore.bench.sources.base.ClipSource`.

A new data source is a new module here, never a new branch in shared
discovery code. Every source is **pure discovery**: globs paths, yields
:class:`~kinescore.bench.manifest.DiscoveredClip`; probing is centralised in
``kinescore.bench.manifest.build_manifest``.

``ctrlworld``/``dreamdojo``/``dreamgen`` (:class:`~kinescore.bench.sources.ctrlworld.CtrlWorldSource`
/ :class:`~kinescore.bench.sources.dreamdojo.DreamDojoSource` /
:class:`~kinescore.bench.sources.dreamgen.DreamGenSource`) each take a full
:class:`~kinescore.bench.cell.Cell` (they are the ``generator`` axis of the
five-axis matrix) and are registered under their generator name in
:data:`~kinescore.bench.sources.registry.DEFAULT_SOURCES`. Each module's own
docstring records what was verified against the real data for that generator
specifically (including known gaps) -- start there, not here.

Two sources previously lived here that took a subset of the matrix's
parameters rather than a full ``Cell`` (no ``cache``/``generator`` axis
applied to either): ``cosmos`` (the hand-labelled construct-validity set) and
``lerobot`` (the real-video reference distribution). Both are DELETED as of
this port -- neither was reachable from
``kinescore.cli.cmd_bench._GENERATOR_MODULES`` (the only real caller) nor a
valid ``generator`` axis value, so they were orphaned discovery code with no
path to being invoked. If a construct-validity check or a real-video
baseline is needed again, re-add it as its own module (matching the original
docstrings, preserved in version control history) rather than resurrecting
dead code.
"""
from __future__ import annotations

from kinescore.bench.sources.base import ClipSource
from kinescore.bench.sources.ctrlworld import CtrlWorldSource
from kinescore.bench.sources.dreamdojo import DreamDojoSource
from kinescore.bench.sources.dreamgen import DreamGenSource
from kinescore.bench.sources.registry import DEFAULT_SOURCES

__all__ = [
    "ClipSource", "DEFAULT_SOURCES",
    "CtrlWorldSource", "DreamDojoSource", "DreamGenSource",
]
