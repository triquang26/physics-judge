"""Defect D9: a rigid, motionless arm that merely opens its gripper must NOT
register as a rigidity defect under ``bone_set="rigid"`` -- and the legacy
``bone_set="all"`` must still reproduce the contaminated number, so a
provenance comparison against the source benchmark stays possible.

Three tiers, in increasing fidelity and decreasing portability:

1. :func:`test_degenerate_bone_contamination_and_fix` -- a hand-built
   :class:`~_fake_robot.FakeRobot` with one bone whose rest length is
   exactly 0 (the mechanism of D9, isolated). CPU-only, zero optional
   dependencies, always runs. This is the regression test that cannot be
   skipped by a missing asset.
2. :func:`test_franka_reproduces_measured_15_37mm` -- the literal Franka
   Panda geometry via ``kinescore.robots.franka.spec.FrankaSpec``, checking
   the design doc's own measured figure (``rigidity_residual_mm ~= 15.37``
   under ``bone_set="all"``, ``~= 0`` under ``bone_set="rigid"``). Skipped
   when ``pytorch_kinematics`` / a cached Panda URDF are unavailable.

``Synthetic2R`` is deliberately not used here: its own module docstring
states it has no degenerate bone ("unlike the Franka gripper"), so it cannot
exercise this defect at all -- using it would only demonstrate the fix looks
like a no-op, which is not the same claim as "the fix works".
"""
from __future__ import annotations

import pytest
import torch
from _fake_robot import FakeRobot

from kinescore.core.metric import MetricContext
from kinescore.metrics.rigidity import RigidityResidual, RigidityWobble

# ===========================================================================
# 1. CPU-only, dependency-free reproduction of the mechanism
# ===========================================================================

def _rigid_arm_opening_gripper() -> tuple[FakeRobot, torch.Tensor]:
    """A perfectly rigid, motionless 2-bone arm whose *third* bone is a
    zero-rest-length "gripper" that opens over the clip.

    Keypoints: ``base`` (fixed), ``elbow`` (fixed, 1 m from base),
    ``fingerL``/``fingerR`` (coincide at rest, 0 m apart, then separate
    linearly to 4 cm -- a plausible gripper stroke). Bones:
    ``[base-elbow (1.0 m), fingerL-fingerR (0.0 m rest)]``. The arm itself
    (base->elbow) never moves at all.
    """
    T = 6
    base = torch.zeros(T, 3)
    elbow = torch.tensor([1.0, 0.0, 0.0]).expand(T, 3).clone()
    opening = torch.linspace(0.0, 0.04, T)                          # 0 -> 4 cm
    finger_l = elbow + torch.stack(
        [torch.zeros(T), opening / 2, torch.zeros(T)], dim=-1)
    finger_r = elbow - torch.stack(
        [torch.zeros(T), opening / 2, torch.zeros(T)], dim=-1)
    P = torch.stack([base, elbow, finger_l, finger_r], dim=1).unsqueeze(0)  # (1,T,4,3)

    robot = FakeRobot(
        keypoint_links=("base", "elbow", "fingerL", "fingerR"),
        bone_pairs=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        bone_lengths=torch.tensor([1.0, 0.0], dtype=torch.float32),
        rigid_bone_pairs=torch.tensor([[0, 1]], dtype=torch.long),
        rigid_bone_lengths=torch.tensor([1.0], dtype=torch.float32),
    )
    return robot, P


def test_degenerate_bone_contamination_and_fix():
    robot, P = _rigid_arm_opening_gripper()
    ctx = MetricContext(dt=0.2, P=P, robot=robot)

    residual_all = RigidityResidual(bone_set="all").compute(ctx)
    residual_rigid = RigidityResidual(bone_set="rigid").compute(ctx)
    wobble_all = RigidityWobble(bone_set="all").compute(ctx)
    wobble_rigid = RigidityWobble(bone_set="rigid").compute(ctx)

    # bone_set="all" is contaminated by the gripper: the finger bone's
    # realised length ranges over [0, 4cm], averaging well above its 0 rest
    # length -- a large nonzero residual/wobble despite a perfectly rigid,
    # motionless arm.
    assert residual_all.value > 5.0, residual_all       # mm; opening averages ~2cm >> 0
    assert wobble_all.value > 5.0, wobble_all

    # bone_set="rigid" drops the degenerate bone entirely: the only
    # remaining bone (base->elbow) never moves, so both are exactly zero.
    assert residual_rigid.value == pytest.approx(0.0, abs=1e-6), residual_rigid
    assert wobble_rigid.value == pytest.approx(0.0, abs=1e-6), wobble_rigid


# ===========================================================================
# 2. Literal Franka reproduction of the measured ~15.37 mm figure
# ===========================================================================

def _franka_spec():
    pytest.importorskip("pytorch_kinematics")
    from kinescore.robots.franka.spec import FrankaSpec
    try:
        return FrankaSpec()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Franka URDF / robot_descriptions unavailable: {exc}")


def test_franka_reproduces_measured_15_37mm():
    """The measured D9 figure: a rigid, motionless arm holding its gripper OPEN.

    The 15.37 mm reference value is for a gripper held **fully open for every
    frame**, not one that opens over the clip. That distinction matters: the
    residual is a mean over frames, so a 0 -> 1 ramp averages the contamination
    over its intermediate openings and lands near half the figure. Asserting
    15.37 against a ramp would be comparing two different quantities.

    Both cases are checked here, because the ramp is the more realistic clip and
    its ~2x smaller number is exactly the kind of thing that looks like a
    regression later if nobody wrote down why.
    """
    robot = _franka_spec()
    T = 6
    q = torch.zeros(1, T, robot.n_joints)      # rest pose, perfectly still

    held_open = torch.ones(1, T)
    ramp = torch.linspace(0.0, 1.0, T).view(1, T)

    def residual(gripper, bone_set):
        P = robot.forward_kinematics(q, aux=gripper)
        return RigidityResidual(bone_set=bone_set).compute(
            MetricContext(dt=0.2, P=P, robot=robot)).value

    # Legacy bone set: the arm never moved, yet it "deforms" by 15 mm.
    assert residual(held_open, "all") == pytest.approx(15.37, rel=0.02)
    # A ramp averages over intermediate openings -> roughly half.
    assert residual(ramp, "all") == pytest.approx(7.2, rel=0.10)

    # Rigid bone set: the arm is correctly reported as perfectly rigid in both.
    assert residual(held_open, "rigid") == pytest.approx(0.0, abs=1e-6)
    assert residual(ramp, "rigid") == pytest.approx(0.0, abs=1e-6)
