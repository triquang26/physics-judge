"""Defect D7: joint-limit violation must never read as a fabricated ``0.0``.

A pose-reader head that sigmoid-squashes its output into the URDF limits
(``q = lo + (hi-lo)*sigmoid(raw)``) can never predict an out-of-limit joint
by construction -- so ``limit_violation_frac``/``limit_excess_rad`` computed
from such a head are always, trivially, exactly ``0.0``, for a reason that
has nothing to do with whether the video shows plausible motion. Reading
that as "this generator never violates its joint limits" is not merely
imprecise, it is actively misleading: it reports the *best possible* score
for a property the head structurally cannot fail. ``kinescore.core.metric``
(frozen) exists to make that distinguishable from a real, measured pass --
this file checks that :mod:`kinescore.metrics.joint_limits` actually uses
that mechanism rather than silently computing (and returning) the
structurally-guaranteed zero.

The squashed pose-reader path itself (``readers/squashed.py::SquashedPoseReader``,
the only reader that ever actually set ``limit_semantics="squashed"``) has
been removed -- see ``legacy_docs/PROVENANCE.md``'s D7 addendum -- so this file no
longer exercises a live "squashed" ``MetricContext``; it covers the
``raw_rad`` half of the contract, which is what every real reader now
produces.
"""
from __future__ import annotations

import pytest
import torch
from _fake_robot import FakeRobot

from kinescore.core.metric import MetricContext
from kinescore.metrics.joint_limits import LimitExcessRad, LimitViolationFrac


def _robot() -> FakeRobot:
    return FakeRobot(
        q_lo=torch.tensor([-1.0, -1.0]), q_hi=torch.tensor([1.0, 1.0]))


def test_missing_q_raw_gives_nan_with_reason_not_zero():
    """Any reader that does not expose ``q_raw`` (the squashed reader always
    did not; a raw_rad reader that simply omits it would too) must resolve
    to NaN (missing_input), never a fabricated pass."""
    robot = _robot()
    T = 5
    ctx = MetricContext(dt=0.1, q=torch.zeros(1, T, 2), q_raw=None, robot=robot)

    frac = LimitViolationFrac().compute(ctx)
    assert not frac.available
    assert frac.reason == "missing_input:q_raw"
    assert not (frac.value == 0.0)


def test_raw_rad_semantics_with_out_of_limit_q_raw_gives_positive_frac():
    """Under raw_rad semantics, an out-of-limit q_raw produces a genuinely
    positive violation fraction/excess -- the metric is observable and
    actually measures something."""
    robot = _robot()
    T = 6
    q_raw = torch.zeros(1, T, 2)
    q_raw[:, 2:4, 0] = 1.5                                          # exceeds hi=1.0 for 2/6 frames
    ctx = MetricContext(
        dt=0.1, q=torch.zeros(1, T, 2), q_raw=q_raw, robot=robot,
        flags={"limit_semantics": "raw_rad"})

    frac = LimitViolationFrac().compute(ctx)
    excess = LimitExcessRad().compute(ctx)

    assert frac.available, frac.reason
    assert frac.value == pytest.approx(2 / 6)
    assert excess.available, excess.reason
    assert excess.value > 0.0


def test_raw_rad_semantics_within_limits_is_a_real_measured_zero():
    """A raw_rad reader that never exceeds its limits also legitimately
    reads 0.0 -- the point of D7 is not "never report zero", it is "zero
    must be a measurement, not a structural guarantee". This confirms the
    metric IS observable (available=True) even though the value happens to
    be zero."""
    robot = _robot()
    T = 5
    q_raw = torch.zeros(1, T, 2)                                    # always within [-1,1]
    ctx = MetricContext(
        dt=0.1, q=torch.zeros(1, T, 2), q_raw=q_raw, robot=robot,
        flags={"limit_semantics": "raw_rad"})

    frac = LimitViolationFrac().compute(ctx)
    assert frac.available, frac.reason
    assert frac.value == 0.0
