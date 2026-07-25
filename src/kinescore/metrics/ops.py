"""Shared finite-difference primitives.

Every derivative in this subpackage is built from one function, ported
verbatim from the source's ``_fd``:

    _fd(x, dt) = (x[:, 1:] - x[:, :-1]) / dt

Centralising it here (instead of letting each metric module redefine its own
copy, which is what the source's sibling files did) is what makes the
``dt_exponent`` contract auditable: every derivative in the package divides by
``dt`` exactly once per order, so a metric's declared exponent is simply "how
many times a chain of :func:`fd` was applied to build it" -- countable by
inspection, and verified numerically by
``tests/test_metric_registry_conformance.py``.

Why the conformance test can use a tight tolerance (``rtol=1e-6``)
-------------------------------------------------------------------
:func:`fd` divides by ``dt`` and nothing else. Holding the *sample array*
fixed and only changing the ``dt`` value passed to it therefore scales the
result by a rational factor with no other source of numerical error -- unlike
*resampling* a trajectory (``P[::k]``, ``dt*k``), which changes which samples
are differenced and so introduces real finite-difference truncation error
(that is what ``tests/test_dt_invariance.py`` exercises instead, at a much
looser tolerance). The registry conformance test holds ``P``/``R``/``q`` fixed
and only perturbs the declared ``dt``, which is exactly what turns "declared
``dt_exponent``" into a claim checkable to float32 precision.
"""
from __future__ import annotations

import torch

__all__ = ["fd", "vee", "finite"]


def fd(x: torch.Tensor, dt: float) -> torch.Tensor:
    """First-order finite difference along the time axis (dim=1).

    ``(B, T, ...) -> (B, T-1, ...)`` divided by ``dt``. Verbatim port of the
    source's ``_fd`` in ``judge/physics.py``.
    """
    return (x[:, 1:] - x[:, :-1]) / dt


def vee(skew: torch.Tensor) -> torch.Tensor:
    """Inverse-hat of a batched skew-symmetric matrix ``(...,3,3) -> (...,3)``.

    Verbatim port of ``PhysicsConsistency._vee``.
    """
    return torch.stack(
        [skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1)


def finite(value: float) -> bool:
    """``True`` iff ``value`` is a finite float (not NaN, not +/-inf).

    Metrics use this to enforce the :class:`~kinescore.core.metric.MetricValue`
    invariant "``value`` is ``NaN`` exactly when ``reason`` is set" -- a metric
    whose arithmetic produces a NaN from degenerate input (a stationary
    end-effector feeding ``dimensionless_jerk``'s zero-path-length guard, an
    empty reduction from too few frames slipping past ``min_frames``) must
    convert that into an explicit unavailable reason rather than pass the NaN
    through as if it were a measured "unavailable-looking but technically
    available" value. See :class:`kinescore.metrics._base.SafeMetric`.
    """
    import math
    return math.isfinite(value)
