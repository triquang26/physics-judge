"""``rigidity_worst_bone_mm``: the two properties it exists for.

This metric was added because the two ported rigidity metrics failed to
separate generated from real video at all -- ``rigidity_residual_mm`` had a
median paired delta of ``+0.013 mm`` and put generated worse in 56.6% of 659
episodes, a coin flip, while two independent smoothness rulers on the *same*
episodes agreed at 88-90%. The cause was not the data: the published rigidity
figure came from a different source function
(``models/physics/feasibility.py::rigidity_field``) that differs in exactly the
two ways tested here. See ``legacy_docs/DECISIONS.md`` D-A.

These are behavioural tests, not conformance tests --
``test_metric_registry_conformance.py`` already auto-parametrizes over the
registry and checks the declared ``dt_exponent`` numerically. What it cannot
check is *why* this metric is in the suite when two similarly-named ones
already are, and that is what would silently rot if someone "simplified" the
reduction back to a mean.
"""
from __future__ import annotations

import torch
from _fake_robot import FakeRobot

import kinescore.metrics  # noqa: F401  (side effect: populates the registry)
from kinescore.core.metric import MetricContext, all_metrics


def _ctx(P: torch.Tensor, robot: FakeRobot) -> MetricContext:
    return MetricContext(dt=0.1, P=P, R=None, q=None, q_raw=None, robot=robot)


def _three_keypoints(l0: torch.Tensor, l1: torch.Tensor) -> torch.Tensor:
    """``(1,T,3,3)`` collinear keypoints with bone lengths ``l0``/``l1`` per frame."""
    T = l0.shape[0]
    z = torch.zeros(T)
    base = torch.stack([z, z, z], dim=-1)
    mid = torch.stack([l0, z, z], dim=-1)
    tip = torch.stack([l0 + l1, z, z], dim=-1)
    return torch.stack([base, mid, tip], dim=-2).unsqueeze(0)


def _robot() -> FakeRobot:
    return FakeRobot(
        bone_pairs=torch.tensor([[0, 1], [1, 2]]),
        rigid_bone_pairs=torch.tensor([[0, 1], [1, 2]]),
        bone_lengths=torch.tensor([1.0, 1.0]),
        rigid_bone_lengths=torch.tensor([1.0, 1.0]),
    )


class TestWorstBoneIsNotDilutedByGoodBones:
    """``amax`` over bones, not ``mean`` -- the sensitivity property."""

    def test_one_rubber_banding_bone_is_not_averaged_away(self) -> None:
        T = 40
        steady = torch.full((T,), 1.0)
        # One bone rubber-bands by +-10 mm; the other is perfectly rigid.
        wobbly = 1.0 + 0.010 * torch.sin(torch.linspace(0, 8 * 3.14159, T))

        worst = all_metrics()["rigidity_worst_bone_mm"]
        mean_reduced = all_metrics()["rigidity_wobble_mm"]
        robot = _robot()

        P = _three_keypoints(wobbly, steady)
        v_worst = worst.compute(_ctx(P, robot)).value
        v_mean = mean_reduced.compute(_ctx(P, robot)).value

        # The two are different statistics -- mean-absolute-deviation-from-
        # median (amax over bones) versus population std (mean over bones) --
        # so no exact ratio is predictable and none is asserted. What matters
        # is the direction and that it is not marginal: on this fixture the
        # max-reduced ruler measures 6.20 mm against the mean-reduced 3.49 mm,
        # a factor of 1.78, because exactly one of the two bones moves. With
        # more rigid bones in the chain (the real GR-1 case) the gap widens,
        # which is the whole reason this metric exists.
        assert v_worst > 1.5 * v_mean, (
            f"worst-bone {v_worst:.4f} mm should dominate mean-reduced "
            f"{v_mean:.4f} mm when exactly one of two bones rubber-bands")

    def test_scales_with_the_worst_bone_not_the_bone_count(self) -> None:
        """Adding rigid bones must not shrink the reading."""
        T = 30
        wobbly = 1.0 + 0.010 * torch.sin(torch.linspace(0, 6 * 3.14159, T))
        steady = torch.full((T,), 1.0)
        worst = all_metrics()["rigidity_worst_bone_mm"]

        two_bone = worst.compute(_ctx(_three_keypoints(wobbly, steady), _robot())).value

        # Same wobbly bone, but now paired with a second *also* rigid bone --
        # under a mean this would fall; under amax it must not move.
        robot3 = FakeRobot(
            bone_pairs=torch.tensor([[0, 1], [1, 2], [2, 3]]),
            rigid_bone_pairs=torch.tensor([[0, 1], [1, 2], [2, 3]]),
            bone_lengths=torch.tensor([1.0, 1.0, 1.0]),
            rigid_bone_lengths=torch.tensor([1.0, 1.0, 1.0]),
        )
        z = torch.zeros(T)
        pts = [
            torch.stack([z, z, z], dim=-1),
            torch.stack([wobbly, z, z], dim=-1),
            torch.stack([wobbly + steady, z, z], dim=-1),
            torch.stack([wobbly + 2 * steady, z, z], dim=-1),
        ]
        P3 = torch.stack(pts, dim=-2).unsqueeze(0)
        three_bone = worst.compute(_ctx(P3, robot3)).value

        assert abs(three_bone - two_bone) < 1e-4, (
            f"amax must ignore extra rigid bones: {two_bone:.6f} -> "
            f"{three_bone:.6f} mm")


class TestMedianReferenceCancelsConstantBias:
    """Deviation from the bone's *own median*, not from the URDF rest length."""

    def test_constant_length_error_reads_zero(self) -> None:
        """A perfectly rigid robot drawn at the wrong scale is still rigid.

        This is the failure mode ``rigidity_residual_mm`` cannot avoid: the
        reader's absolute accuracy floor is tens of mm, so a constant per-bone
        offset is expected and says nothing about the *video*.
        """
        T = 20
        # Both bones constant, but 30 mm longer than the URDF rest length.
        biased = torch.full((T,), 1.030)
        robot = _robot()
        P = _three_keypoints(biased, biased)

        worst = all_metrics()["rigidity_worst_bone_mm"].compute(_ctx(P, robot)).value
        residual = all_metrics()["rigidity_residual_mm"].compute(_ctx(P, robot)).value

        assert worst < 1e-6, (
            f"a constant length offset is not a rigidity defect, got {worst} mm")
        assert residual > 25.0, (
            "rigidity_residual_mm is expected to report the constant bias "
            f"(that is its definition), got {residual} mm")


class TestUnavailableNotZero:
    def test_missing_robot_is_unavailable_with_reason(self) -> None:
        P = _three_keypoints(torch.ones(5), torch.ones(5))
        mv = all_metrics()["rigidity_worst_bone_mm"].compute(
            MetricContext(dt=0.1, P=P, R=None, q=None, q_raw=None, robot=None))
        assert mv.value is None or (mv.value != mv.value)  # None or NaN
        assert mv.reason and "robot" in mv.reason
