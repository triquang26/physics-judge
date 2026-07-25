"""Rigidity/wobble are purely geometric: identical at EVERY dt, ``rtol=0``.

Unlike every temporal/angular/energy metric in this package,
``rigidity_residual_mm`` and ``rigidity_wobble_mm`` never call
:func:`kinescore.metrics.ops.fd` -- they are built entirely from bone
lengths at each frame, with no finite difference anywhere. There is
therefore no ``dt`` term in the formula at all (``dt_exponent=0`` in the
strongest possible sense: not "scales as ``dt**0``" but "``dt`` is never
read"), so changing the declared ``dt`` must leave the value bit-for-bit
identical -- ``rtol=0``, not merely a tight tolerance. Any nonzero
difference here would mean ``dt`` leaked into a computation that has no
business touching it.
"""
from __future__ import annotations

import torch
from _fake_robot import FakeRobot

from kinescore.core.metric import MetricContext
from kinescore.metrics.rigidity import RigidityResidual, RigidityWobble


def _robot() -> FakeRobot:
    return FakeRobot(
        bone_pairs=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        bone_lengths=torch.tensor([1.0, 0.5], dtype=torch.float32))


def _moving_P(seed: int = 0, T: int = 6) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.cumsum(torch.randn(1, T, 3, 3, generator=g) * 0.1, dim=1)


def test_rigidity_residual_identical_at_every_dt():
    robot = _robot()
    P = _moving_P()
    metric = RigidityResidual(bone_set="rigid")
    values = [metric.compute(MetricContext(dt=dt, P=P, robot=robot)).value
              for dt in (0.01, 0.033, 0.1, 0.2, 1.0, 7.3)]
    assert all(v == values[0] for v in values), values


def test_rigidity_wobble_identical_at_every_dt():
    robot = _robot()
    P = _moving_P()
    metric = RigidityWobble(bone_set="rigid")
    values = [metric.compute(MetricContext(dt=dt, P=P, robot=robot)).value
              for dt in (0.01, 0.033, 0.1, 0.2, 1.0, 7.3)]
    assert all(v == values[0] for v in values), values


def test_rigidity_all_bone_set_also_dt_free():
    """The legacy ``bone_set="all"`` variant is just as dt-free as ``"rigid"``
    -- the fix for D9 (dropping degenerate bones) is orthogonal to dt at all."""
    robot = _robot()
    P = _moving_P()
    res = RigidityResidual(bone_set="all")
    wob = RigidityWobble(bone_set="all")
    res_values = [res.compute(MetricContext(dt=dt, P=P, robot=robot)).value
                  for dt in (0.02, 0.5, 3.0)]
    wob_values = [wob.compute(MetricContext(dt=dt, P=P, robot=robot)).value
                  for dt in (0.02, 0.5, 3.0)]
    assert all(v == res_values[0] for v in res_values), res_values
    assert all(v == wob_values[0] for v in wob_values), wob_values
