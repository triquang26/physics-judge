"""``ClipSource``: re-exported from :mod:`kinescore.core.contracts`.

Ported from the earlier ``make_plugin(cell, data_root, config) -> SourcePlugin``
free-function-per-module style to one class per generator, each registered in
:data:`~kinescore.bench.sources.registry.DEFAULT_SOURCES` (a
:class:`kinescore.core.registry.Registry`), so ``cli.cmd_bench``/
``bench.ingest`` dispatch on a generator NAME through one registry lookup
instead of an ``importlib.import_module(_GENERATOR_MODULES[generator])``
table that has to be kept in sync by hand.

The split between "validate eagerly, glob lazily" is preserved exactly:
``ClipSource.make_plugin`` does every check that can be done from
``cell``/``config`` alone (wrong generator, an unresolved ``iter`` pin, an
unrecognised horizon) and raises immediately, mirroring the free functions'
behaviour -- a caller building plugins for 50 cells up front still fails on
cell #3's bad config before touching the filesystem for cell #1, rather than
after. The returned zero-argument :data:`~kinescore.bench.manifest.SourcePlugin`
closure is what actually globs, and only when
:func:`~kinescore.bench.manifest.build_manifest` calls it -- unchanged from
before.

``kinescore.core.contracts`` was checked for an existing ``ClipSource``
definition (see ``bench.layout``'s module docstring for the same check
against ``DataLayout``): it now defines one, matching this module's shape
exactly (``GENERATOR`` class var, ``make_plugin(cell, data_root, config) ->
SourcePlugin``, a ``__call__`` alias) -- so this module imports it from there
instead of keeping a second, identical definition.
"""
from __future__ import annotations

from kinescore.core.contracts import ClipSource

__all__ = ["ClipSource"]
