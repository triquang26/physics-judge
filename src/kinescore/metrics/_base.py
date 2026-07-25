"""Internal helper shared by every metric in this subpackage.

Not part of the public deliverable list in the design doc; it exists to close
one gap between the frozen :mod:`kinescore.core.metric` contract and floating
point reality.

The gap
-------
:class:`~kinescore.core.metric.MetricValue` documents an invariant: ``value``
is ``NaN`` **exactly** when ``reason`` is set. ``BaseMetric._ok`` (in
``core/metric.py``, frozen) does not enforce this -- it just wraps whatever
float you give it. Several of the ported formulas can produce a NaN from
*valid* inputs that nonetheless make the arithmetic degenerate:

* :func:`~kinescore.metrics.smoothness.dimensionless_jerk` returns ``nan`` for
  a stationary end-effector (zero path length divides into the
  ``duration**5`` normalisation);
* :func:`~kinescore.metrics.smoothness.sparc` returns ``nan`` when the
  amplitude-threshold band is empty;
* a temporal std over the minimum permitted frame count can still be zero
  input variance in a degenerate synthetic fixture and propagate oddly through
  a log.

Passing such a NaN through ``_ok`` would violate the frozen contract silently
-- a caller checking ``MetricValue.available`` would see ``True`` for a value
that is not a number. ``SafeMetric`` is the one place that gap is closed: every
metric in this package subclasses it instead of ``BaseMetric`` directly, and
its overridden ``_ok`` demotes a non-finite result to ``MetricValue.unavailable``
with an explicit reason instead of a bare NaN with none.
"""
from __future__ import annotations

import math
from typing import Any

from kinescore.core.metric import BaseMetric, MetricValue

__all__ = ["SafeMetric"]


class SafeMetric(BaseMetric):
    """``BaseMetric`` with a NaN-safe ``_ok``.

    Subclasses call ``self._ok(value, perframe)`` exactly as they would on
    ``BaseMetric``; the only difference is what happens when ``value`` is not
    finite.
    """

    def _ok(self, value: float, perframe: Any = None) -> MetricValue:
        value = float(value)
        if not math.isfinite(value):
            return MetricValue.unavailable(
                self.spec.key, "degenerate_input:non_finite_result")
        return super()._ok(value, perframe)
