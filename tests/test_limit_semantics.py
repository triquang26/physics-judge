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
"""
from __future__ import annotations

import math

import pytest
import torch
from _fake_robot import FakeRobot

from kinescore.core.metric import MetricContext
from kinescore.metrics.joint_limits import LimitExcessRad, LimitViolationFrac


def _robot() -> FakeRobot:
    return FakeRobot(
        q_lo=torch.tensor([-1.0, -1.0]), q_hi=torch.tensor([1.0, 1.0]))


def test_squashed_semantics_yields_nan_with_reason_not_zero():
    """Even feeding q_raw values that are WELL outside the limits, a
    squashed-semantics context must resolve to NaN + reason -- the flag
    gates the metric before the arithmetic ever runs, exactly the D7 fix."""
    robot = _robot()
    T = 5
    q_raw_way_out_of_bounds = torch.full((1, T, 2), 100.0)          # would violate massively
    ctx = MetricContext(
        dt=0.1, q=torch.zeros(1, T, 2), q_raw=q_raw_way_out_of_bounds,
        robot=robot, flags={"limit_semantics": "squashed"})

    frac = LimitViolationFrac().compute(ctx)
    excess = LimitExcessRad().compute(ctx)

    assert not frac.available
    assert frac.reason == "unobservable:limit_semantics=squashed"
    assert math.isnan(frac.value)
    assert not (frac.value == 0.0)          # the exact anti-pattern D7 forbids

    assert not excess.available
    assert excess.reason == "unobservable:limit_semantics=squashed"
    assert math.isnan(excess.value)
    assert not (excess.value == 0.0)


def test_squashed_semantics_still_nan_even_without_flag_if_q_raw_missing():
    """A squashed reader's Readout.q_raw is documented as None -- so even
    without the explicit flag, a context missing q_raw must resolve to NaN
    (missing_input), never a fabricated pass."""
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
