"""Mechanical feasibility -- self-collision, CoM balance, no-teleport.

Ports the per-check scorers of Marionette-fkjepa's ``MechanicalFeasibility``
(``self_collision``, ``com_balance``, ``no_teleport``, including the private
``_arm_spheres`` capsule construction they are built on) as four standalone
metrics: :class:`Penetration`, :class:`SelfCollision`, :class:`CoMBalance`,
:class:`NoTeleport`. Two structural changes from the source, both mechanical
(the arithmetic inside each check is untouched):

1. **``default_dt`` is deleted.** The source's ``MechanicalFeasibility.__init__``
   took ``default_dt: float = 0.1`` and fell back to it whenever ``report()``
   was called without an explicit ``dt`` -- exactly the class of bug
   ``kinescore.core.clip.ClipSpec`` exists to make impossible (see its module
   docstring: three places a benchmark's ``dt`` defaulted to a wrong constant
   and nobody noticed). Only :class:`NoTeleport` uses ``dt`` at all (the
   other three are per-frame-geometric); it reads ``ctx.dt``, which has no
   default, full stop.

2. **There is no more standalone ``MechanicalFeasibility`` wrapper class.**
   The source built it from a ``GR1FK`` + ``RobotColliders`` pair passed in
   at construction. ``kinescore.core.robot.RobotSpec`` is frozen and
   deliberately robot-agnostic -- it declares *capabilities*
   (``Capability.COLLIDERS``, ``Capability.SUPPORT_POLYGON``) but not a
   generic collision-geometry field, since that geometry is irreducibly
   per-embodiment (a Franka bolted to a table has no feet, no torso capsule
   set, nothing to check against). ``kinescore.robots.gr1.spec.GR1Spec`` is
   the concrete robot that backs these capabilities today, and its module
   docstring names the exact extension surface a metric may rely on:
   ``robot.fk.keypoints_fk(q, side)`` (the per-arm keypoint chain, for
   capsule construction), ``robot.body_collider_spheres(q)`` (posed
   torso/base/head spheres), ``robot.world_com(q)`` and
   ``robot.support_polygon(q)`` (foot positions). :func:`_arm_capsule_spheres`
   and :func:`_penetration_field` / :func:`_com_margin_field` below rebuild
   exactly the source's own capsule-interpolation and penetration/margin
   arithmetic on top of that surface -- nothing about the geometry itself is
   new, only where it is assembled from.

   The ``side in ("left", "right")`` iteration is a real, documented
   specialisation to bimanual arm layouts (GR-1 today), inherited unchanged
   from the source's own ``_arm_spheres``. A future single-arm
   ``COLLIDERS``-bearing robot would need this generalised; nothing here
   pretends otherwise.

**Never 0.0 for "no colliders".** A table-mounted Franka has neither
capability. The naive failure mode is to report ``0.0`` penetration /
``0.0`` (perfectly centred) CoM margin for a robot that was never checked --
which reads as "flawless balance" for a robot that structurally cannot fall
over in the first place, a much stronger and false claim. Every metric below
is gated through ``spec.requires`` on the matching capability string, so an
ungated robot returns ``missing_input:colliders`` /
``missing_input:support_polygon``, never a fabricated zero. If a robot claims
a capability but the expected attributes are absent (a future, differently-
shaped ``COLLIDERS`` robot), the metric degrades to ``NaN`` + an explicit
``missing_capability_data`` reason instead of raising an ``AttributeError``
past the caller.

:class:`NoTeleport` needs neither capability -- it is a pure function of
keypoints ``P`` and a fixed Cartesian speed cap, so it is available for every
robot that has FK at all (Franka included).
"""
from __future__ import annotations

import torch

from kinescore.core.metric import MetricContext, MetricSpec, MetricValue, register
from kinescore.metrics._base import SafeMetric
from kinescore.metrics.ops import fd

__all__ = ["Penetration", "SelfCollision", "CoMBalance", "NoTeleport"]

#: Penetration depth above which a frame counts as colliding (m). Source default.
_PEN_EPS_M = 1e-3
#: Max per-keypoint Cartesian speed before a frame counts as a teleport (m/s). Source default.
_V_KP_MAX = 2.0
#: Capsule radius (m) for the arm segments. Source default.
_ARM_RADIUS_M = 0.045
#: Spheres per arm segment (denser = tighter capsule). Source default.
_N_SEG_SPHERES = 4
#: Support-polygon margin (m) added around the foot links. Source default.
_FOOT_HALFSIZE_M = 0.12

#: Bimanual arm sides this module's capsule construction iterates over -- see
#: the module docstring's note on the ``("left","right")`` specialisation.
_SIDES = ("left", "right")


def _arm_capsule_spheres(robot, q: torch.Tensor, side: str,
                         n_seg: int, distal_start: int = 0) -> torch.Tensor:
    """FK keypoint chain -> capsule spheres ``(B,T,Na,3)``.

    Verbatim port of the source's ``MechanicalFeasibility._arm_spheres``,
    rebuilt on ``robot.fk.keypoints_fk`` instead of a constructor-injected
    ``self.fk``. ``distal_start`` skips proximal segments (the shoulder sits
    next to the torso, so include it only for arm-vs-arm, not arm-vs-body).
    """
    P = robot.fk.keypoints_fk(q, side)                          # (B,T,K,3)
    s = torch.linspace(0, 1, n_seg, device=P.device, dtype=P.dtype)[:-1]
    pts = []
    for i in range(distal_start, P.shape[2] - 1):
        a, b = P[:, :, i], P[:, :, i + 1]                       # (B,T,3)
        pts.append(a[:, :, None] + s[None, None, :, None] * (b - a)[:, :, None])
    pts.append(P[:, :, -1:])                                    # EE tip
    return torch.cat(pts, dim=2)                                # (B,T,Na,3)


def _penetration_field(robot, q: torch.Tensor, arm_radius: float,
                       n_seg: int) -> torch.Tensor:
    """Max sphere-sphere penetration per frame ``(B,T)`` (m, >= 0).

    Verbatim port of the source's ``MechanicalFeasibility.penetration_field``:
    each arm's distal capsule spheres against the body-core spheres, plus the
    two arms' full capsules against each other.
    """
    body_centers, body_radii = robot.body_collider_spheres(q)   # (B,T,Sb,3),(Sb,)
    r = arm_radius
    pens = []
    for side in _SIDES:
        arm = _arm_capsule_spheres(robot, q, side, n_seg, distal_start=1)
        d = torch.cdist(arm, body_centers)                      # (B,T,Na,Sb)
        pens.append(torch.relu((r + body_radii[None, None, None, :]) - d).flatten(2))
    la = _arm_capsule_spheres(robot, q, _SIDES[0], n_seg, distal_start=0)
    ra = _arm_capsule_spheres(robot, q, _SIDES[1], n_seg, distal_start=0)
    pens.append(torch.relu(2 * r - torch.cdist(la, ra)).flatten(2))
    return torch.cat(pens, dim=2).amax(dim=2)                   # (B,T)


def _com_margin_field(robot, q: torch.Tensor, foot_halfsize: float) -> torch.Tensor:
    """Signed CoM margin per frame ``(B,T)`` (m): >0 inside support, <0 tipping.

    Verbatim arithmetic from the source's ``MechanicalFeasibility.com_margin_field``,
    with one adaptation: the source cached a support-polygon rectangle once at
    construction time from a fixed rest pose (its ``__init__`` ran FK on
    ``q=0``); this port instead reads ``robot.support_polygon(q)`` on every
    call. ``GR1Spec.support_polygon`` documents that the feet are not
    reachable from the 17 predicted joints and are therefore constant across
    ``q`` in practice -- so this produces the identical rectangle, just
    without a robot needing a special one-shot "pose at rest" code path in
    this module.
    """
    foot_xy = robot.support_polygon(q)[..., :2]                 # (B,T,2,2)
    lo = foot_xy.amin(dim=2) - foot_halfsize                    # (B,T,2)
    hi = foot_xy.amax(dim=2) + foot_halfsize                    # (B,T,2)
    com_xy = robot.world_com(q)[..., :2]                        # (B,T,2)
    inside = torch.minimum(com_xy - lo, hi - com_xy)            # (B,T,2)
    return inside.amin(dim=-1)                                  # (B,T)


def _capability_field(ctx: MetricContext, builder, capability: str,
                      *args) -> torch.Tensor | MetricValue:
    """Call ``builder(ctx.robot, ctx.q, *args)``, or a ``MetricValue.unavailable``.

    Every attribute access the geometry builders above need
    (``robot.fk``, ``robot.body_collider_spheres``, ``robot.world_com``,
    ``robot.support_polygon``) is optional as far as the frozen ``RobotSpec``
    Protocol is concerned -- only ``GR1Spec`` implements them today. Catching
    both ``AttributeError`` (a capability-bearing robot shaped differently
    than GR-1) and any arithmetic error keeps this module's promise: never an
    uncaught exception past a well-formed ``MetricContext``, never a
    fabricated number in its place.
    """
    try:
        return builder(ctx.robot, ctx.q, *args)
    except AttributeError as exc:
        return MetricValue.unavailable(
            capability, f"missing_capability_data:{exc}")
    except Exception as exc:                                    # noqa: BLE001
        return MetricValue.unavailable(
            capability, f"error:{type(exc).__name__}:{exc}")


class Penetration(SafeMetric):
    """Mean per-frame max self-collision penetration depth (mm).

    Verbatim ``MechanicalFeasibility.self_collision(q)[0]``. Requires
    ``Capability.COLLIDERS``.

    Parameters
    ----------
    arm_radius_m, n_seg_spheres:
        Capsule geometry for the arm chain -- see :func:`_arm_capsule_spheres`.
    """

    def __init__(self, arm_radius_m: float = _ARM_RADIUS_M,
                 n_seg_spheres: int = _N_SEG_SPHERES) -> None:
        self.arm_radius_m = float(arm_radius_m)
        self.n_seg_spheres = int(n_seg_spheres)
        self.spec = MetricSpec(
            key="penetration_mm", units="mm", dt_exponent=0,
            direction="lower_better", requires=frozenset({"q", "colliders"}),
            min_frames=1,
            description=(
                "Mean over frames of the max arm-capsule/body-sphere (and "
                "arm/arm) penetration depth (mm). No time derivative -> "
                "dt_exponent=0. NaN with reason for a robot lacking "
                "colliders (e.g. a table-mounted Franka), never 0.0."))

    def _compute(self, ctx: MetricContext) -> MetricValue:
        if ctx.robot is None:
            return MetricValue.unavailable(self.spec.key, "missing_input:robot")
        pen = _capability_field(ctx, _penetration_field, self.spec.key,
                                self.arm_radius_m, self.n_seg_spheres)
        if isinstance(pen, MetricValue):
            return pen
        return self._ok(pen.mean() * 1000.0)


class SelfCollision(SafeMetric):
    """Fraction of frames with penetration depth above ``pen_eps_m``.

    Verbatim ``MechanicalFeasibility.self_collision(q)[1]``. Requires
    ``Capability.COLLIDERS``.

    Parameters
    ----------
    pen_eps_m:
        Penetration depth (m) above which a frame counts as colliding.
    arm_radius_m, n_seg_spheres:
        Capsule geometry for the arm chain -- see :func:`_arm_capsule_spheres`.
    """

    def __init__(self, pen_eps_m: float = _PEN_EPS_M,
                 arm_radius_m: float = _ARM_RADIUS_M,
                 n_seg_spheres: int = _N_SEG_SPHERES) -> None:
        self.pen_eps_m = float(pen_eps_m)
        self.arm_radius_m = float(arm_radius_m)
        self.n_seg_spheres = int(n_seg_spheres)
        self.spec = MetricSpec(
            key="self_collision_frac", units="fraction", dt_exponent=0,
            direction="lower_better", requires=frozenset({"q", "colliders"}),
            min_frames=1,
            description=(
                "Fraction of frames whose max penetration depth exceeds "
                f"pen_eps_m={pen_eps_m} m. No time derivative -> "
                "dt_exponent=0 (this is a spatial threshold, unlike the "
                "velocity/acceleration violation fractions elsewhere in the "
                "package, which threshold a dt-dependent quantity)."))

    def _compute(self, ctx: MetricContext) -> MetricValue:
        if ctx.robot is None:
            return MetricValue.unavailable(self.spec.key, "missing_input:robot")
        pen = _capability_field(ctx, _penetration_field, self.spec.key,
                                self.arm_radius_m, self.n_seg_spheres)
        if isinstance(pen, MetricValue):
            return pen
        frac = (pen > self.pen_eps_m).float().mean()
        return self._ok(frac)


class CoMBalance(SafeMetric):
    """Mean signed centre-of-mass margin to the support polygon (m).

    Verbatim ``MechanicalFeasibility.com_balance(q)[0]``. Positive = inside
    the support polygon (stable); negative = tipping.
    ``direction="higher_better"`` -- more margin means further from the
    tipping boundary. Requires ``Capability.SUPPORT_POLYGON`` (feet/base
    geometry a table-mounted arm does not have).

    The source also reported a ``balance_violation_frac`` (fraction of frames
    with negative margin) from the same field; this port exposes only the
    continuous margin as the registered metric, matching the choice made for
    ``LimitHeadroomRad`` over a violation fraction -- a continuous margin
    carries strictly more information and degrades gracefully, whereas a
    frac is a coarser derived view of the same field.

    Parameters
    ----------
    foot_halfsize_m:
        Support-polygon margin (m) added around the foot links.
    """

    def __init__(self, foot_halfsize_m: float = _FOOT_HALFSIZE_M) -> None:
        self.foot_halfsize_m = float(foot_halfsize_m)
        self.spec = MetricSpec(
            key="com_margin_m", units="m", dt_exponent=0,
            direction="higher_better",
            requires=frozenset({"q", "support_polygon"}), min_frames=1,
            description=(
                "Mean over frames of the signed CoM-to-support-polygon "
                "margin (m); positive = stable. No time derivative -> "
                "dt_exponent=0. NaN with reason for a robot lacking a "
                "support polygon (e.g. a table-mounted Franka), never a "
                "fabricated perfect-balance 0.0."))

    def _compute(self, ctx: MetricContext) -> MetricValue:
        if ctx.robot is None:
            return MetricValue.unavailable(self.spec.key, "missing_input:robot")
        margin = _capability_field(ctx, _com_margin_field, self.spec.key,
                                   self.foot_halfsize_m)
        if isinstance(margin, MetricValue):
            return margin
        return self._ok(margin.mean())


class NoTeleport(SafeMetric):
    """Fraction of frames whose max keypoint Cartesian speed exceeds ``v_kp_max``.

    Verbatim ``MechanicalFeasibility.no_teleport``. Needs no robot capability
    -- only keypoints ``P`` -- so it is available for every embodiment,
    including a table-mounted Franka. Like ``AccelViolationFrac`` /
    ``VelViolationFrac`` elsewhere in this package, it thresholds a
    dt-dependent quantity (Cartesian speed) against a fixed physical constant
    (``v_kp_max``, m/s), so it is not a homogeneous power law in ``dt`` --
    ``dt_exponent=None``.

    Parameters
    ----------
    v_kp_max:
        Max physically plausible per-keypoint Cartesian speed (m/s).
    """

    def __init__(self, v_kp_max: float = _V_KP_MAX) -> None:
        self.v_kp_max = float(v_kp_max)
        self.spec = MetricSpec(
            key="no_teleport_frac", units="fraction", dt_exponent=None,
            direction="lower_better", requires=frozenset({"P"}), min_frames=2,
            description=(
                "Fraction of FRAMES where the worst (max) keypoint's "
                f"Cartesian speed ||dP/dt|| exceeds v_kp_max={v_kp_max} m/s "
                "(fixed physical constant) -- verbatim reduction order from "
                "the source: amax over keypoints per frame, then a frac "
                "over frames. dt_exponent=None: a threshold crossing "
                "against an absolute bound, same reasoning as "
                "accel_violation_frac."))

    def _compute(self, ctx: MetricContext) -> MetricValue:
        v = torch.linalg.norm(fd(ctx.P, ctx.dt), dim=-1)            # (B,T-1,K)
        v_frame = v.amax(dim=-1)                                    # (B,T-1)
        frac = (v_frame > self.v_kp_max).float().mean()
        return self._ok(frac)


register(Penetration())
register(SelfCollision())
register(CoMBalance())
register(NoTeleport())
