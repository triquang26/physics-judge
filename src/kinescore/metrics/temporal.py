"""Linear dynamics on keypoint trajectories: speed, acceleration, jerk.

Ports the ``velocity`` / ``acceleration`` / ``jerk`` finite-difference chain
and the scalar metrics built on it (``speed_stats``, ``accel_stats``,
``mean_jerk``, ``accel_violation_frac``) from ``PhysicsConsistency`` verbatim.
All linear quantities assume keypoints are in metres, so speeds are m/s,
accelerations m/s^2, jerk m/s^3 -- exactly the source's convention.

``dt_exponent`` for these is countable by construction: each is built from
:func:`kinescore.metrics.ops.fd` applied ``n`` times to position, and ``fd``
divides by ``dt`` exactly once per application, so speed is
``dt_exponent=1``, acceleration ``2``, jerk ``3``. See ``ops.py`` for why this
makes the declaration checkable to float32 precision rather than merely
"probably right".

``accel_violation_frac`` is the odd one out
--------------------------------------------
Every other metric in this module is a *homogeneous* function of ``1/dt`` --
scale ``dt`` and the value scales by a clean power. ``accel_violation_frac``
is not: it thresholds the (dt-dependent) acceleration magnitude against
``accel_bound``, a **fixed physical constant** in m/s^2 that does not move
when ``dt`` does. Measured: a 2x dt error (the exact D1 scenario from
``core/clip.py`` -- scoring stride-2-decimated frames as if they were
stride-1) inflates acceleration 4x and drives this fraction from
``0.000`` straight to ``0.3875`` on a trajectory whose true accelerations sit
just under the bound (see ``tests/test_dt_invariance.py::test_wrong_dt_is_detected``).
That is precisely why it carries ``units="fraction"`` and
``dt_exponent=None`` -- it is the metric most sensitive to a wrong dt of any
in this package, and declaring a dt_exponent for it would be a lie the
registry conformance test cannot catch (a threshold crossing is not a power
law), so it is explicitly excluded from that check instead.
"""
from __future__ import annotations

import torch

from kinescore.core.metric import MetricContext, MetricSpec, MetricValue, register
from kinescore.metrics._base import SafeMetric
from kinescore.metrics.ops import fd

__all__ = ["MeanSpeed", "MeanAccel", "MaxAccel", "MeanJerk", "AccelViolationFrac"]


def _velocity(P: torch.Tensor, dt: float) -> torch.Tensor:
    """``(B,T,K,3) -> (B,T-1,K,3)``. Verbatim ``PhysicsConsistency.velocity``."""
    return fd(P, dt)


def _acceleration(P: torch.Tensor, dt: float) -> torch.Tensor:
    """``(B,T,K,3) -> (B,T-2,K,3)``. Verbatim ``PhysicsConsistency.acceleration``."""
    return fd(_velocity(P, dt), dt)


def _jerk(P: torch.Tensor, dt: float) -> torch.Tensor:
    """``(B,T,K,3) -> (B,T-3,K,3)``. Verbatim ``PhysicsConsistency.jerk``."""
    return fd(_acceleration(P, dt), dt)


class MeanSpeed(SafeMetric):
    """Mean keypoint speed magnitude (m/s) -- verbatim ``speed_stats()[0]``."""

    spec = MetricSpec(
        key="mean_speed_mps", units="m/s", dt_exponent=1,
        direction="lower_better", requires=frozenset({"P"}), min_frames=2,
        perframe=True,
        description="Mean over frames and keypoints of ||dP/dt|| (m/s). "
                     "perframe=True: per-frame mean-over-keypoints speed "
                     "trace. One finite difference drops 1 frame, from the "
                     "front (see bench/traces.py's ALIGNMENTS['front']): "
                     "trace length == n_frames - 1, trace[0] is frame 1.")

    def _compute(self, ctx: MetricContext) -> MetricValue:
        s = torch.linalg.norm(_velocity(ctx.P, ctx.dt), dim=-1)
        perframe = s.mean(dim=-1).reshape(-1).detach().cpu().numpy()
        return self._ok(s.mean(), perframe=perframe)


class MeanAccel(SafeMetric):
    """Mean keypoint acceleration magnitude (m/s^2) -- verbatim ``accel_stats()[0]``."""

    spec = MetricSpec(
        key="mean_accel_mps2", units="m/s^2", dt_exponent=2,
        direction="lower_better", requires=frozenset({"P"}), min_frames=3,
        perframe=True,
        description="Mean over frames and keypoints of ||d^2P/dt^2|| (m/s^2). "
                     "perframe=True: per-frame mean-over-keypoints "
                     "acceleration trace. Two finite differences drop 2 "
                     "frames, from the front: trace length == n_frames - 2, "
                     "trace[0] is frame 2.")

    def _compute(self, ctx: MetricContext) -> MetricValue:
        a = torch.linalg.norm(_acceleration(ctx.P, ctx.dt), dim=-1)
        perframe = a.mean(dim=-1).reshape(-1).detach().cpu().numpy()
        return self._ok(a.mean(), perframe=perframe)


class MaxAccel(SafeMetric):
    """Max keypoint acceleration magnitude (m/s^2) -- verbatim ``accel_stats()[1]``."""

    spec = MetricSpec(
        key="max_accel_mps2", units="m/s^2", dt_exponent=2,
        direction="lower_better", requires=frozenset({"P"}), min_frames=3,
        description="Max over frames and keypoints of ||d^2P/dt^2|| (m/s^2).")

    def _compute(self, ctx: MetricContext) -> MetricValue:
        a = torch.linalg.norm(_acceleration(ctx.P, ctx.dt), dim=-1)
        return self._ok(a.max())


class MeanJerk(SafeMetric):
    """Mean keypoint jerk magnitude (m/s^3) -- verbatim ``mean_jerk``.

    The single most sensitive un-physicality detector in the source: a
    generator that pops the arm between frames has bounded velocity but
    explosive jerk. Emits a per-frame array (``perframe=True``) so the
    sidecar NPZ carries the full jerk-magnitude trace, not just its mean --
    useful for the distributional (W1) comparisons in ``kinescore.bench``.
    """

    spec = MetricSpec(
        key="mean_jerk_mps3", units="m/s^3", dt_exponent=3,
        direction="lower_better", requires=frozenset({"P"}), min_frames=4,
        perframe=True,
        description="Mean over frames and keypoints of ||d^3P/dt^3|| (m/s^3). "
                     "Three finite differences drop 3 frames, from the "
                     "front: trace length == n_frames - 3, trace[0] is "
                     "frame 3.")

    def _compute(self, ctx: MetricContext) -> MetricValue:
        j = torch.linalg.norm(_jerk(ctx.P, ctx.dt), dim=-1)        # (B,T-3,K)
        perframe = j.mean(dim=-1).reshape(-1).detach().cpu().numpy()
        return self._ok(j.mean(), perframe=perframe)


class AccelViolationFrac(SafeMetric):
    """Fraction of (frame, keypoint) accelerations exceeding ``accel_bound``.

    Verbatim ``accel_violation_frac``. See the module docstring for why this
    carries ``dt_exponent=None`` rather than a declared power law: it
    thresholds against a fixed physical constant, so a wrong ``dt`` does not
    scale this metric cleanly -- it can snap it from 0 to a large fraction
    almost discontinuously, which is exactly what makes it a good detector of
    a timebase bug and a bad candidate for the power-law conformance check.

    Parameters
    ----------
    accel_bound:
        Physical per-keypoint acceleration ceiling (m/s^2). Source default
        (5.0) is conservative for a Franka end-effector under normal teleop.
    """

    def __init__(self, accel_bound: float = 5.0) -> None:
        self.accel_bound = float(accel_bound)
        self.spec = MetricSpec(
            key="accel_violation_frac", units="fraction", dt_exponent=None,
            direction="lower_better", requires=frozenset({"P"}), min_frames=3,
            perframe=True,
            description=(
                "Fraction of (frame,keypoint) pairs whose acceleration "
                f"magnitude exceeds accel_bound={accel_bound} m/s^2 (fixed "
                "physical constant, not scaled by dt). dt_exponent=None: "
                "this is a threshold crossing, not a power law -- it is the "
                "metric most sensitive to a wrong dt (measured 0.000 -> "
                "0.3875 under a 2x dt error). perframe=True: per-frame "
                "fraction of KEYPOINTS violating at that frame (mean over "
                "the keypoint axis only) -- a continuous [0,1] trace whose "
                "own mean over frames reproduces the scalar exactly (the "
                "scalar means over frame AND keypoint jointly; this is that "
                "same reduction split into two steps so time is visible). "
                "Two finite differences drop 2 frames, from the front: "
                "trace length == n_frames - 2, trace[0] is frame 2."))

    def _compute(self, ctx: MetricContext) -> MetricValue:
        a = torch.linalg.norm(_acceleration(ctx.P, ctx.dt), dim=-1)
        frac = (a > self.accel_bound).float().mean()
        perframe = (a > self.accel_bound).float().mean(
            dim=-1).reshape(-1).detach().cpu().numpy()
        return self._ok(frac, perframe=perframe)


register(MeanSpeed())
register(MeanAccel())
register(MaxAccel())
register(MeanJerk())
register(AccelViolationFrac())
