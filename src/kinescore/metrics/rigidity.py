"""Static geometry: does each frame draw a rigid robot of the right link lengths.

Ports :class:`KinematicConsistency.rigidity_residual_mm` and
``.rigidity_wobble_mm`` from the source verbatim (mean-abs-deviation-from-rest
and temporal-std-of-length, both in mm), restructured into two
:class:`~kinescore.core.metric.BaseMetric` subclasses with one change each,
both defensive rather than arithmetic:

Defect D9 -- the degenerate bone
---------------------------------
The source's ``bone_pairs`` was every *consecutive* keypoint pair, which for
the Franka includes ``panda_leftfinger -> panda_rightfinger``. That bone's
rest length is ``0.0`` m (the fingers coincide when the gripper is fully
closed), so its "realised length" is simply the gripper opening -- not a
rigidity residual. Reproduced measurement: a perfectly rigid, motionless arm
that merely opens its gripper scores ``rigidity_residual_mm ~= 15.37``, i.e.
grasping clips were penalised for grasping.

The fix lives in :mod:`kinescore.core.robot` (frozen): every ``RobotSpec``
exposes both ``bone_pairs`` (the full legacy set) and ``rigid_bone_pairs``
(degenerate bones, defined as rest length ``<= DEGENERATE_BONE_M`` = 1 mm,
dropped). Both metrics here take a ``bone_set`` constructor argument selecting
which one to read off the robot; the default is ``"rigid"`` because that is
the number you want in every suite that scores grasping. ``"all"`` exists
purely to reproduce the legacy (buggy) numbers for provenance comparison --
see ``tests/test_rigidity_gripper_contamination.py``.

Defect D7 -- the poisoned mean
-------------------------------
``rigidity_wobble_mm`` is a temporal std (``L.std(dim=1)``). torch's default
``std`` is Bessel-corrected (``unbiased=True``, i.e. ``ddof=1``), which is
``NaN`` at ``T=1`` -- there is no such thing as a sample variance of one
sample. That NaN does not error; it silently poisons whatever downstream mean
it feeds (a suite aggregate, a per-clip report). :class:`RigidityWobble` fixes
this the same way :class:`~kinescore.metrics._base.SafeMetric` fixes NaN in
general -- by declaring ``min_frames=2`` so a 1-frame clip returns an explicit
``too_few_frames`` reason -- **and** by using the population std
(``unbiased=False``, ``ddof=0``) rather than the sample std, which is the more
defensible statistic here regardless: there is no sampling story where "the
observed T frames are a sample from a larger population of frames" makes
sense, so a Bessel correction is not buying anything except a spurious poles
at small ``T``.
"""
from __future__ import annotations

import torch

from kinescore.core.metric import MetricContext, MetricSpec, MetricValue, register
from kinescore.metrics._base import SafeMetric

__all__ = ["RigidityResidual", "RigidityWobble"]

_BONE_SETS = ("rigid", "all")


def _bone_geometry(robot, bone_set: str) -> tuple[torch.Tensor, torch.Tensor]:
    if bone_set not in _BONE_SETS:
        raise ValueError(f"bone_set must be one of {_BONE_SETS}, got {bone_set!r}")
    if bone_set == "rigid":
        return robot.rigid_bone_pairs, robot.rigid_bone_lengths
    return robot.bone_pairs, robot.bone_lengths


def _bone_lengths(P: torch.Tensor, bone_pairs: torch.Tensor) -> torch.Tensor:
    """Per-frame realised bone lengths ``(B, T, n_bones)`` from ``(B,T,K,3)``.

    Verbatim port of ``KinematicConsistency._bone_lengths``.
    """
    bp = bone_pairs.to(P.device).long()
    i, j = bp[:, 0], bp[:, 1]
    return torch.linalg.norm(P[..., i, :] - P[..., j, :], dim=-1)


class RigidityResidual(SafeMetric):
    """Mean |realised - nominal| bone length (mm) -- verbatim ``rigidity_residual_mm``.

    Parameters
    ----------
    bone_set:
        ``"rigid"`` (default) uses ``robot.rigid_bone_pairs`` /
        ``rigid_bone_lengths`` -- degenerate (near-zero rest length) bones
        excluded, so opening a gripper does not register as a rigidity defect
        (D9). ``"all"`` uses the full ``bone_pairs`` set, reproducing legacy
        numbers exactly, degenerate bones included.
    """

    def __init__(self, bone_set: str = "rigid") -> None:
        if bone_set not in _BONE_SETS:
            raise ValueError(f"bone_set must be one of {_BONE_SETS}, got {bone_set!r}")
        self.bone_set = bone_set
        key = "rigidity_residual_mm" if bone_set == "rigid" else "rigidity_residual_all_mm"
        self.spec = MetricSpec(
            key=key, units="mm", dt_exponent=0, direction="lower_better",
            requires=frozenset({"P"}), min_frames=1,
            description=(
                "Mean absolute deviation of realised bone length from the "
                f"URDF rest length (bone_set={bone_set!r}). Purely geometric "
                "(no derivative), so it does not depend on dt at all."))

    def _compute(self, ctx: MetricContext) -> MetricValue:
        if ctx.robot is None:
            return MetricValue.unavailable(self.spec.key, "missing_input:robot")
        bone_pairs, bone_lengths = _bone_geometry(ctx.robot, self.bone_set)
        L = _bone_lengths(ctx.P, bone_pairs)                       # (B, T, n_bones)
        nominal = bone_lengths.to(L.device, L.dtype).view(1, 1, -1)
        return self._ok((L - nominal).abs().mean() * 1000.0)


class RigidityWobble(SafeMetric):
    """Temporal std of realised bone length (mm) -- verbatim ``rigidity_wobble_mm``.

    Robust to a constant localisation bias in the reader (a systematic offset
    in every keypoint cancels in a length difference), so it isolates
    flicker/wobble of the drawn arm independent of absolute accuracy.

    Parameters
    ----------
    bone_set:
        Same meaning as :class:`RigidityResidual`.
    """

    def __init__(self, bone_set: str = "rigid") -> None:
        if bone_set not in _BONE_SETS:
            raise ValueError(f"bone_set must be one of {_BONE_SETS}, got {bone_set!r}")
        self.bone_set = bone_set
        key = "rigidity_wobble_mm" if bone_set == "rigid" else "rigidity_wobble_all_mm"
        self.spec = MetricSpec(
            key=key, units="mm", dt_exponent=0, direction="lower_better",
            requires=frozenset({"P"}), min_frames=2,
            description=(
                "Temporal std (population, ddof=0) of realised bone length "
                f"(bone_set={bone_set!r}). min_frames=2: an unbiased std at "
                "T=1 is NaN and would poison any mean it feeds (defect D7); "
                "using the population statistic sidesteps the Bessel-"
                "correction pole but T>=2 is still required for the "
                "statistic to mean anything."))

    def _compute(self, ctx: MetricContext) -> MetricValue:
        if ctx.robot is None:
            return MetricValue.unavailable(self.spec.key, "missing_input:robot")
        bone_pairs, _ = _bone_geometry(ctx.robot, self.bone_set)
        L = _bone_lengths(ctx.P, bone_pairs)                       # (B, T, n_bones)
        wobble = L.std(dim=1, unbiased=False).mean() * 1000.0
        return self._ok(wobble)


register(RigidityResidual(bone_set="rigid"))
register(RigidityResidual(bone_set="all"))
register(RigidityWobble(bone_set="rigid"))
register(RigidityWobble(bone_set="all"))
