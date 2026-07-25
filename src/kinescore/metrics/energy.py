"""Energy and momentum continuity (unit point masses on keypoints).

Ports ``PhysicsConsistency.kinetic_energy``, ``.energy`` and
``.momentum_continuity`` verbatim. All three model each keypoint as a unit
point mass, so "energy" and "momentum" here are *proxies*, not physical
quantities of the real robot (whose links have real, non-uniform mass) -- the
source used them exactly this way, as smoothness/continuity signals rather
than absolute physical energies, and this port preserves that framing.

``min_frames`` here is not a restructuring choice, it is what makes the
frozen ``MetricValue`` invariant ("NaN exactly when reason is set") hold. Each
of these three ends in a ``.std(dim=1)`` or ``.mean()`` over an array of
``T-1`` (kinetic/total energy) or ``T-2`` (momentum jump) elements built from
*default* (Bessel-corrected) torch statistics -- unlike
:class:`kinescore.metrics.rigidity.RigidityWobble`, the source did not flag
these as needing the population variant, so the arithmetic here is untouched.
But an unbiased std over fewer than 2 samples, or a mean over zero samples,
is NaN by construction regardless of the estimator -- so ``min_frames`` is set
to the smallest ``T`` that keeps every such reduction non-degenerate
(``T>=3`` for all three), and :class:`~kinescore.metrics._base.SafeMetric`
catches anything that still slips through as unavailable rather than a silent
NaN "success".

``TotalEnergyTStd`` is the headline non-homogeneous metric in this package
--------------------------------------------------------------------------
``total = kinetic_energy + g * sum(z)``. The kinetic term is built from
velocity (``dt_exponent`` contribution ``+2``, since it is squared); the
potential term is a pure sum of positions with **no** finite difference in it
at all (``dt_exponent`` contribution ``0``). Adding a ``dt^-2``-scaling
quantity to a ``dt^0`` one is not homogeneous in ``dt`` -- there is no single
exponent ``n`` for which ``total(k*dt) == total(dt) / k**n`` holds, because
the two terms would need *different* ``n``. Measured on a representative
trajectory: a 2x dt error shifts ``total_energy_tstd`` by **1.397x**, not the
clean 4x a pure ``dt_exponent=2`` metric would show and not the 1x a pure
``dt_exponent=0`` metric would show either -- a genuine trap for anyone
skimming the output and assuming every residual obeys a power law. This is
exactly why the field exists: ``dt_exponent=None`` says "do not try to verify
this numerically as a power law", and this docstring is the recorded reason.
"""
from __future__ import annotations

import torch

from kinescore.core.metric import MetricContext, MetricSpec, MetricValue, register
from kinescore.metrics._base import SafeMetric
from kinescore.metrics.ops import fd

__all__ = ["KineticEnergyTStd", "TotalEnergyTStd", "MomentumDpMean"]

#: Standard gravity (m/s^2), source default.
_G = 9.81


def _velocity(P: torch.Tensor, dt: float) -> torch.Tensor:
    return fd(P, dt)


class KineticEnergyTStd(SafeMetric):
    """Temporal std of unit-mass kinetic energy -- verbatim ``kinetic_energy()[1]``.

    ``KE_t = 0.5 * sum_k ||v_k||^2`` per frame; a physical motion changes
    kinetic energy smoothly, so a flickering/popping generator spikes this.
    """

    spec = MetricSpec(
        key="kinetic_energy_tstd", units="(m/s)^2", dt_exponent=2,
        direction="lower_better", requires=frozenset({"P"}), min_frames=3,
        description=(
            "Temporal std (default/unbiased) of per-frame unit-mass kinetic "
            "energy 0.5*sum_k||v_k||^2. min_frames=3 keeps the underlying "
            "T-1-length KE sequence at >=2 samples so the unbiased std is "
            "not NaN by construction."))

    def _compute(self, ctx: MetricContext) -> MetricValue:
        v2 = (_velocity(ctx.P, ctx.dt) ** 2).sum(-1)                # (B,T-1,K)
        ke = 0.5 * v2.sum(-1)                                       # (B,T-1)
        return self._ok(ke.std(dim=1).mean())


class TotalEnergyTStd(SafeMetric):
    """Temporal std of unit-mass total (KE+PE) energy -- verbatim ``energy()[1]``.

    See the module docstring for why this declares ``dt_exponent=None``: it
    mixes a ``dt^-2``-scaling kinetic term with a ``dt^0`` potential term and
    is therefore not homogeneous in ``dt``.
    """

    def __init__(self, g: float = _G) -> None:
        self.g = float(g)
        self.spec = MetricSpec(
            key="total_energy_tstd", units="(m/s)^2", dt_exponent=None,
            direction="lower_better", requires=frozenset({"P"}), min_frames=3,
            description=(
                "Temporal std of per-frame unit-mass total mechanical energy "
                "KE+PE, PE=g*sum_k(z_k). dt_exponent=None: KE scales dt^-2, "
                "PE is dt-independent, so the sum is not a power law in dt "
                "(measured 2x dt error -> 1.397x shift, not a clean 4x)."))

    def _compute(self, ctx: MetricContext) -> MetricValue:
        v2 = (_velocity(ctx.P, ctx.dt) ** 2).sum(-1)                # (B,T-1,K)
        ke = 0.5 * v2.sum(-1)                                       # (B,T-1)
        pe = self.g * ctx.P[:, :-1, :, 2].sum(-1)                   # (B,T-1)
        tot = ke + pe
        return self._ok(tot.std(dim=1).mean())


class MomentumDpMean(SafeMetric):
    """Mean |d(momentum)/dt| -- verbatim ``momentum_continuity()[1]``.

    ``p = sum_k v_k`` (unit-mass linear momentum). ``mean |dp/dt|`` is a
    net-force-magnitude proxy: large jumps in system momentum without contact
    are unphysical. ``dt_exponent=2``: ``p`` inherits velocity's single power
    of ``1/dt``, and the finite difference that builds ``dp`` divides by
    ``dt`` again.
    """

    spec = MetricSpec(
        key="momentum_dp_mean", units="m/s^2", dt_exponent=2,
        direction="lower_better", requires=frozenset({"P"}), min_frames=3,
        description=(
            "Mean magnitude of the frame-to-frame change in unit-mass linear "
            "momentum p=sum_k v_k, divided by dt -- a net-force proxy."))

    def _compute(self, ctx: MetricContext) -> MetricValue:
        v = _velocity(ctx.P, ctx.dt)                                # (B,T-1,K,3)
        p = v.sum(2)                                                # (B,T-1,3)
        dp = torch.linalg.norm(fd(p, ctx.dt), dim=-1)                # (B,T-2)
        return self._ok(dp.mean())


register(KineticEnergyTStd())
register(TotalEnergyTStd())
register(MomentumDpMean())
