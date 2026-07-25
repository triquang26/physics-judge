"""Joint-space kinematics/feasibility from ``q`` -- ports ``joint_feasibility``.

``mean_qdot`` / ``mean_qddot`` are the joint-space analogues of
:mod:`kinescore.metrics.temporal`'s ``MeanSpeed`` / ``MeanAccel``, built the
same way (:func:`kinescore.metrics.ops.fd` applied once/twice to ``q``), so
their ``dt_exponent`` is countable the same way: 1 and 2 respectively.

``vel_violation_frac`` and ``effort_proxy`` need robot data that is not one of
the standard ``requires`` inputs (``vel_limits`` / ``effort_limits`` are
optional per-robot attributes on ``RobotSpec``, not a member of the
``Requires`` literal in ``core/metric.py``), so they cannot be gated purely
through ``spec.requires`` the way ``requires={"q_raw"}`` gates the joint-limit
metrics. Both check ``ctx.robot`` and its optional limits array explicitly in
``_compute`` and return an explicit unavailable reason (never fabricate a
limit) when the URDF does not declare one -- the same "NaN with a reason, not
a fabricated pass" discipline as everywhere else in this package, just
enforced by hand instead of by the declarative ``requires`` mechanism because
the input isn't one of the ones that mechanism knows about.
"""
from __future__ import annotations

from kinescore.core.metric import MetricContext, MetricSpec, MetricValue, register
from kinescore.metrics._base import SafeMetric
from kinescore.metrics.ops import fd

__all__ = ["MeanQdot", "MeanQddot", "VelViolationFrac", "EffortProxy"]


class MeanQdot(SafeMetric):
    """Mean |q_dot| (rad/s) -- verbatim ``joint_feasibility()["mean_qdot_radps"]``."""

    spec = MetricSpec(
        key="mean_qdot_radps", units="rad/s", dt_exponent=1,
        direction="lower_better", requires=frozenset({"q"}), min_frames=2,
        description="Mean absolute joint velocity dq/dt over joints and frames.")

    def _compute(self, ctx: MetricContext) -> MetricValue:
        qd = fd(ctx.q, ctx.dt)                                     # (B,T-1,n_joints)
        return self._ok(qd.abs().mean())


class MeanQddot(SafeMetric):
    """Mean |q_ddot| (rad/s^2) -- verbatim ``joint_feasibility()["mean_qddot_radps2"]``."""

    spec = MetricSpec(
        key="mean_qddot_radps2", units="rad/s^2", dt_exponent=2,
        direction="lower_better", requires=frozenset({"q"}), min_frames=3,
        description="Mean absolute joint acceleration d^2q/dt^2 over joints and frames.")

    def _compute(self, ctx: MetricContext) -> MetricValue:
        qd = fd(ctx.q, ctx.dt)
        qdd = fd(qd, ctx.dt)                                       # (B,T-2,n_joints)
        return self._ok(qdd.abs().mean())


class VelViolationFrac(SafeMetric):
    """Fraction of (frame, joint) velocities exceeding the URDF rate limit.

    Verbatim ``joint_feasibility()["vel_violation_frac"]``. Like
    :class:`kinescore.metrics.temporal.AccelViolationFrac`, this thresholds
    against a fixed constant per joint (``robot.vel_limits``, N.m-independent
    of ``dt``), so it is **not** a homogeneous power law in ``dt`` --
    ``dt_exponent=None``, same reasoning as the linear acceleration violation
    fraction.
    """

    spec = MetricSpec(
        key="vel_violation_frac", units="fraction", dt_exponent=None,
        direction="lower_better", requires=frozenset({"q"}), min_frames=2,
        description=(
            "Fraction of (frame,joint) pairs where |dq/dt| exceeds "
            "robot.vel_limits (rad/s, URDF-declared, fixed w.r.t. dt). "
            "dt_exponent=None: a threshold crossing, not a power law. NaN "
            "with reason if the robot has no vel_limits (URDF omitted them) "
            "-- never fabricated as 0.0."))

    def _compute(self, ctx: MetricContext) -> MetricValue:
        if ctx.robot is None:
            return MetricValue.unavailable(self.spec.key, "missing_input:robot")
        vel_limits = getattr(ctx.robot, "vel_limits", None)
        if vel_limits is None:
            return MetricValue.unavailable(
                self.spec.key, "missing_input:vel_limits")
        qd = fd(ctx.q, ctx.dt)                                     # (B,T-1,n_joints)
        vl = vel_limits.to(device=qd.device, dtype=qd.dtype).view(1, 1, -1)
        frac = (qd.abs() > vl).float().mean()
        return self._ok(frac)


class EffortProxy(SafeMetric):
    """Mean sum_j |q_ddot_j| / effort_j -- verbatim ``joint_feasibility()["effort_proxy"]``.

    A relative actuator-load indicator: low-rated-effort joints (typically
    wrist joints) cost more per unit acceleration. **There is no inertia
    model here** -- this is ``|acceleration| / rated_effort``, not
    ``torque = inertia @ acceleration + ...``. Use it comparatively (across
    clips scored with the same robot and the same ``dt``), never as an
    absolute torque estimate; the source's own docstring makes the same
    caveat and this port preserves it verbatim.

    ``dt_exponent=2``: ``q_ddot`` contributes its usual ``dt^-2``, and
    ``effort_j`` is a ``dt``-independent URDF constant, so the ratio is still
    a clean power law (unlike the violation fractions in this module, which
    break homogeneity via an absolute *threshold* rather than a
    dt-independent *divisor* -- dividing by a constant preserves the power
    law, thresholding against one does not).
    """

    spec = MetricSpec(
        key="effort_proxy", units="rad/(N.m.s^2)", dt_exponent=2,
        direction="lower_better", requires=frozenset({"q"}), min_frames=3,
        description=(
            "Mean over frames of sum_j |d^2q_j/dt^2| / effort_limits_j. "
            "Comparative actuator-load proxy, NOT an absolute torque (no "
            "inertia model). NaN with reason if the robot has no "
            "effort_limits."))

    def _compute(self, ctx: MetricContext) -> MetricValue:
        if ctx.robot is None:
            return MetricValue.unavailable(self.spec.key, "missing_input:robot")
        effort_limits = getattr(ctx.robot, "effort_limits", None)
        if effort_limits is None:
            return MetricValue.unavailable(
                self.spec.key, "missing_input:effort_limits")
        qd = fd(ctx.q, ctx.dt)
        qdd = fd(qd, ctx.dt)                                       # (B,T-2,n_joints)
        el = effort_limits.to(device=qdd.device, dtype=qdd.dtype).view(1, 1, -1)
        proxy = (qdd.abs() / el).sum(-1).mean()
        return self._ok(proxy)


register(MeanQdot())
register(MeanQddot())
register(VelViolationFrac())
register(EffortProxy())
