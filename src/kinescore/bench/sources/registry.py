"""``DEFAULT_SOURCES``: generator name -> :class:`~kinescore.core.contracts.ClipSource`.

Built on :class:`kinescore.core.registry.Registry` (the generic
``name -> lazy factory -> T`` registry the ``core`` agent added, now the
shared mechanism for every extension axis -- robots, metric suites, and this
one). ``ClipSource`` subclasses are stateless (see
:class:`~kinescore.core.contracts.ClipSource`'s docstring), so registering
the CLASS itself as the zero-arg factory (``registry.register(name, Cls)``
-- calling a class constructs an instance, exactly matching
``Callable[[], T]``) is enough; :meth:`~kinescore.core.registry.Registry.get`
then constructs a fresh, equivalent instance per call, which is fine since
there is no per-instance state to share.

Note the exception type this inherits from ``Registry.get``:
:class:`ValueError` (naming every registered generator), not ``KeyError`` --
callers matching on the old shape need updating (see
``kinescore.cli.cmd_bench``).
"""
from __future__ import annotations

from kinescore.core.contracts import ClipSource
from kinescore.core.registry import Registry

__all__ = ["DEFAULT_SOURCES"]


def _build_default_registry() -> Registry[ClipSource]:
    from kinescore.bench.sources.ctrlworld import CtrlWorldSource
    from kinescore.bench.sources.dreamdojo import DreamDojoSource
    from kinescore.bench.sources.dreamgen import DreamGenSource

    registry: Registry[ClipSource] = Registry(kind="clip source")
    registry.register(CtrlWorldSource.GENERATOR, CtrlWorldSource)
    registry.register(DreamDojoSource.GENERATOR, DreamDojoSource)
    registry.register(DreamGenSource.GENERATOR, DreamGenSource)
    return registry


#: The registry every real caller (``cli.cmd_bench``, ``bench.ingest``) uses.
#: A module-level singleton, not a function called fresh each time, so every
#: caller in one process shares the same instance.
DEFAULT_SOURCES: Registry[ClipSource] = _build_default_registry()
