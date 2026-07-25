"""Angular dynamics on per-link rotations -- needs ``R``.

Ports ``PhysicsConsistency._vee`` / ``.angular_velocity`` / ``.angular_stats``
verbatim. ``angular_velocity`` is a **small-angle estimate**: it linearises
the relative rotation ``dR = R_{t+1} R_t^T`` by taking its skew-symmetric part
and reading off the rotation vector, ``omega ~= vee(skew(dR)) / dt``. This is
the first-order approximation of the true angular velocity and is only valid
while the rotation between consecutive frames is small, i.e.
``|omega| * dt << 1`` (equivalently: the robot must not spin by more than a
small fraction of a radian between frames). At a normal control rate
(dt ~ 0.02-0.2s for a Franka/GR-1 joint) this holds comfortably for any
physically actuated link; it would silently degrade -- underestimating the
true angular speed while looking numerically fine -- if fed a badly
subsampled (very large dt) or very fast-spinning trajectory. Nothing in this
module enforces the bound; it is a modelling assumption inherited unchanged
from the source, documented here because the source did not call it out.

``dt_exponent``: ``omega`` divides by ``dt`` once (``dt_exponent=1``);
``alpha = fd(omega, dt)`` divides again (``dt_exponent=2``). ``dR`` itself
does not depend on ``dt`` at all, so -- exactly as with the linear metrics --
holding ``R`` fixed and varying only the declared ``dt`` scales these by a
clean power with no other source of error.
"""
from __future__ import annotations

import torch

from kinescore.core.metric import MetricContext, MetricSpec, MetricValue, register
from kinescore.metrics._base import SafeMetric
from kinescore.metrics.ops import fd, vee

__all__ = ["MeanAngularSpeed", "MaxAngularSpeed", "MeanAngularAccel"]


def _angular_velocity(R: torch.Tensor, dt: float) -> torch.Tensor:
    """``(B,T,K,3,3) -> (B,T-1,K,3)`` (rad/s). Verbatim ``angular_velocity``."""
    dR = R[:, 1:] @ R[:, :-1].transpose(-1, -2)                   # (B,T-1,K,3,3)
    skew = 0.5 * (dR - dR.transpose(-1, -2))
    return vee(skew) / dt


class MeanAngularSpeed(SafeMetric):
    """Mean |omega| over the rollout (rad/s) -- verbatim ``angular_stats()[0]``."""

    spec = MetricSpec(
        key="mean_angvel_radps", units="rad/s", dt_exponent=1,
        direction="lower_better", requires=frozenset({"R"}), min_frames=2,
        description=(
            "Mean small-angle angular speed ||vee(skew(R_{t+1} R_t^T))|| / dt "
            "(rad/s). Valid only while |omega|*dt << 1 per frame."))

    def _compute(self, ctx: MetricContext) -> MetricValue:
        w = torch.linalg.norm(_angular_velocity(ctx.R, ctx.dt), dim=-1)
        return self._ok(w.mean())


class MaxAngularSpeed(SafeMetric):
    """Max |omega| over the rollout (rad/s) -- verbatim ``angular_stats()[1]``."""

    spec = MetricSpec(
        key="max_angvel_radps", units="rad/s", dt_exponent=1,
        direction="lower_better", requires=frozenset({"R"}), min_frames=2,
        description=(
            "Max small-angle angular speed over frames and links (rad/s). "
            "Same small-angle validity bound as mean_angvel_radps."))

    def _compute(self, ctx: MetricContext) -> MetricValue:
        w = torch.linalg.norm(_angular_velocity(ctx.R, ctx.dt), dim=-1)
        return self._ok(w.max())


class MeanAngularAccel(SafeMetric):
    """Mean |alpha| over the rollout (rad/s^2) -- verbatim ``angular_stats()[2]``.

    This is the key that ``INVARIANT_V1`` (``suites.py``) actually uses, since
    it was the only angular term present in the source's own
    ``INVARIANT_KEYS`` -- the mean and max angular *speed* were task-dependent
    magnitudes, not invariant residuals.
    """

    spec = MetricSpec(
        key="mean_angacc_radps2", units="rad/s^2", dt_exponent=2,
        direction="lower_better", requires=frozenset({"R"}), min_frames=3,
        description=(
            "Mean small-angle angular acceleration ||d(omega)/dt|| (rad/s^2), "
            "omega itself a small-angle estimate (valid while |omega|*dt << 1)."))

    def _compute(self, ctx: MetricContext) -> MetricValue:
        w = _angular_velocity(ctx.R, ctx.dt)                      # (B,T-1,K,3)
        alpha = fd(w, ctx.dt)                                     # (B,T-2,K,3)
        return self._ok(torch.linalg.norm(alpha, dim=-1).mean())


register(MeanAngularSpeed())
register(MaxAngularSpeed())
register(MeanAngularAccel())
