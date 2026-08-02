"""Joint-limit feasibility -- and why it needs ``q_raw``, not ``q``.

Ports ``PhysicsConsistency.limit_violation`` verbatim (``excess = relu(q-hi) +
relu(lo-q)``, ``frac = (excess.sum(-1) > 0).mean()``), but retargets it at
``q_raw`` instead of ``q``. That retargeting is the whole point of this
module.

Defect D7, restated precisely
------------------------------
``kinescore.core.reader.Readout.q`` is documented as "guaranteed inside the
URDF limits and therefore safe to feed to FK" -- true for every reader,
because ``q`` is a clamped copy of ``q_raw`` used only to keep FK numerically
sane. That means limit violation computed from ``ctx.q`` is **identically
zero for every reader**, so checking ``q`` instead of ``q_raw`` would make
this metric look "perfect" while measuring nothing (the same shape of defect
that made joint-limit violation unmeasurable under the now-removed sigmoid-
squashed head family -- see ``legacy_docs/PROVENANCE.md``'s D7 addendum). ``q_raw``
is unclamped by construction, so ``q_raw - clamp(q_raw)`` is the actual
violation signal. Both :class:`LimitViolationFrac` and :class:`LimitExcessRad`
therefore declare ``requires={"q_raw"}`` and are unavailable
(``missing_input:q_raw``) for any reader that does not populate it.

:class:`LimitHeadroomRad` answers a different, complementary question using
the always-safe ``q``: not "did it violate its limits" but "how close does it
get" -- a metric that stays computable even for a reader whose ``q_raw`` is
absent.
"""
from __future__ import annotations

import torch

from kinescore.core.metric import MetricContext, MetricSpec, MetricValue, register
from kinescore.metrics._base import SafeMetric

__all__ = ["LimitViolationFrac", "LimitExcessRad", "LimitHeadroomRad"]


def _limits(robot, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    lo = robot.q_lo.to(device=device, dtype=dtype).view(1, 1, -1)
    hi = robot.q_hi.to(device=device, dtype=dtype).view(1, 1, -1)
    return lo, hi


def _excess(q: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """``(B,T,n_joints) -> (B,T,n_joints)``. Verbatim per-joint excess term."""
    return torch.relu(q - hi) + torch.relu(lo - q)


class LimitViolationFrac(SafeMetric):
    """Fraction of frames with any joint outside its URDF limit.

    Verbatim ``limit_violation(q_raw)[1]``, retargeted at ``q_raw`` (see
    module docstring for why ``q`` cannot carry this signal at all).
    """

    spec = MetricSpec(
        key="limit_violation_frac", units="fraction", dt_exponent=0,
        direction="lower_better", requires=frozenset({"q_raw"}),
        min_frames=1,
        description=(
            "Fraction of frames where any joint's UNCLAMPED reading q_raw "
            "exceeds [q_lo, q_hi]. Purely a per-frame comparison against "
            "static limits -- no derivative, hence dt_exponent=0."))

    def _compute(self, ctx: MetricContext) -> MetricValue:
        if ctx.robot is None:
            return MetricValue.unavailable(self.spec.key, "missing_input:robot")
        lo, hi = _limits(ctx.robot, ctx.q_raw.device, ctx.q_raw.dtype)
        excess = _excess(ctx.q_raw, lo, hi)                        # (B,T,n_joints)
        frac = (excess.sum(-1) > 0).float().mean()
        return self._ok(frac)


class LimitExcessRad(SafeMetric):
    """Mean summed excess (rad) over frames -- verbatim ``limit_violation(q_raw)[0]``."""

    spec = MetricSpec(
        key="limit_excess_rad", units="rad", dt_exponent=0,
        direction="lower_better", requires=frozenset({"q_raw"}),
        min_frames=1,
        description=(
            "Mean per-frame sum over joints of relu(q_raw-hi)+relu(lo-q_raw) "
            "(rad). The clamp magnitude on the unclamped reading IS the "
            "signal under raw_rad semantics."))

    def _compute(self, ctx: MetricContext) -> MetricValue:
        if ctx.robot is None:
            return MetricValue.unavailable(self.spec.key, "missing_input:robot")
        lo, hi = _limits(ctx.robot, ctx.q_raw.device, ctx.q_raw.dtype)
        excess = _excess(ctx.q_raw, lo, hi)                        # (B,T,n_joints)
        return self._ok(excess.sum(-1).mean())


class LimitHeadroomRad(SafeMetric):
    """Distance (rad) from the nearest joint limit, using the safe ``q``.

    Uses the always-safe ``q`` (never ``q_raw``), so it stays computable even
    for a reader that does not populate ``q_raw`` at all. Per frame, takes the
    tightest margin across joints (``min(hi-q, q-lo)`` per joint, then the
    minimum over joints); reports the mean of that worst-case-per-frame margin
    over the clip. ``direction="higher_better"``: this is the one metric in
    the package where more is better -- more headroom means the trajectory
    stays further from a mechanical extreme.
    """

    spec = MetricSpec(
        key="limit_headroom_rad", units="rad", dt_exponent=0,
        direction="higher_better", requires=frozenset({"q"}), min_frames=1,
        description=(
            "Mean over frames of min_j(min(hi_j-q_j, q_j-lo_j)) (rad): how "
            "much slack remains to the tightest joint limit at each frame, "
            "using the always-limit-safe q. Computable even when "
            "limit_violation_frac/limit_excess_rad are not (no q_raw)."))

    def _compute(self, ctx: MetricContext) -> MetricValue:
        if ctx.robot is None:
            return MetricValue.unavailable(self.spec.key, "missing_input:robot")
        lo, hi = _limits(ctx.robot, ctx.q.device, ctx.q.dtype)
        margin = torch.minimum(hi - ctx.q, ctx.q - lo)              # (B,T,n_joints)
        worst = margin.amin(dim=-1)                                 # (B,T)
        return self._ok(worst.mean())


register(LimitViolationFrac())
register(LimitExcessRad())
register(LimitHeadroomRad())
