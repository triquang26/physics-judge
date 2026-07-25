"""Named-suite lookup for the ``--suite`` flag, importable without torch.

``kinescore.metrics.suites`` is a heavy import (it imports
``kinescore.metrics``, which imports every metric module, which imports
``torch``): the whole point of the ``INVARIANT_V1`` side-effecting import
(see that module's docstring). This module keeps the *name -> module path*
table torch-free so ``kinescore --help`` never touches it, and defers the
actual import to :func:`get_suite`, which every subcommand calls from inside
its ``run(args)``, never at module scope.
"""
from __future__ import annotations

import importlib

__all__ = ["available_suites", "get_suite"]

#: ``--suite`` name -> ``"module:attribute"``. One entry today
#: (``kinescore.metrics.suites.INVARIANT_V1``); a robot-specific suite added
#: there later (see that module's docstring) just needs a new row here.
_SUITES: dict[str, str] = {
    "invariant_v1": "kinescore.metrics.suites:INVARIANT_V1",
}


def available_suites() -> tuple[str, ...]:
    """Registered ``--suite`` names, in a stable (sorted) order."""
    return tuple(sorted(_SUITES))


def get_suite(name: str):
    """Import and return the named :class:`~kinescore.core.suite.MetricSuite`.

    Raises
    ------
    KeyError
        If ``name`` is not registered -- lists the valid names.
    """
    try:
        target = _SUITES[name]
    except KeyError:
        raise KeyError(
            f"unknown suite {name!r}; available: {list(available_suites())}"
        ) from None
    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, attr)
