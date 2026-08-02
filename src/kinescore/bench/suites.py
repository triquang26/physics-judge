"""Named-suite lookup for the ``--suite`` flag, importable without torch.

A metric suite is a benchmark concept, not a CLI one -- this module used to
live at ``kinescore.cli._suites``, which made ``kinescore.bench.csv_export``
reach *up* into ``cli`` to use it, the one place in this package where the
dependency ran backwards (everywhere else ``cli`` depends on ``bench``, never
the other way -- see ``tests/test_import_layering.py``). Moved here so that
importer is no longer an inversion; ``kinescore.cli`` now imports this
module like every other ``bench`` name it uses.

``kinescore.metrics.suites`` is a heavy import (it imports
``kinescore.metrics``, which imports every metric module, which imports
``torch``): the whole point of the ``INVARIANT_V1`` side-effecting import
(see that module's docstring). This module keeps the *name -> module path*
table torch-free so ``kinescore --help`` never touches it, and defers the
actual import to :func:`get_suite`, which every caller invokes from inside
its own work function, never at module scope.
"""
from __future__ import annotations

import importlib

__all__ = ["available_suites", "get_suite"]

#: ``--suite`` name -> ``"module:attribute"``. A robot-specific suite added in
#: ``kinescore.metrics.suites`` later (see that module's docstring) just needs
#: a new row here.
#:
#: The three differ in what they are *for*, not in quality:
#:
#: * ``invariant_v1`` -- the ported 26-term set. Its ``suite_id`` is what every
#:   golden fixture and every previously published number was computed under,
#:   so it is frozen: score with it when a result has to line up with those.
#: * ``all_metrics`` -- ``invariant_v1`` plus ``torque_frac_rated``, the one ruler with
#:   a real physical ceiling (percent of the motor's URDF-rated N.m). This is
#:   the suite to score a fresh benchmark run with.
#: * ``rate_free`` -- only ``dt_exponent == 0`` metrics, the sole valid basis
#:   for comparing clips recorded at different frame rates (docs/BENCHMARKING.md).
_SUITES: dict[str, str] = {
    "invariant_v1": "kinescore.metrics.suites:INVARIANT_V1",
    "all_metrics": "kinescore.metrics.suites:ALL_METRICS",
    "rate_free": "kinescore.metrics.suites:RATE_FREE",
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
