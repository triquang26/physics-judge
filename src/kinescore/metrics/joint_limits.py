"""Joint-limit feasibility -- and why it needs ``q_raw``, not ``q``.

Ports ``PhysicsConsistency.limit_violation`` verbatim (``excess = relu(q-hi) +
relu(lo-q)``, ``frac = (excess.sum(-1) > 0).mean()``), but retargets it at
``q_raw`` instead of ``q``. That retargeting is the whole point of this
module.

Defect D7, restated precisely
------------------------------
``kinescore.core.reader.Readout.q`` is documented as "guaranteed inside the
URDF limits and therefore safe to feed to FK" -- true for *every* reader,
squashed or not, because a raw_rad reader's ``q`` is a clamped copy of its
``q_raw`` used only to keep FK numerically sane. That means limit violation
computed from ``ctx.q`` is **identically zero for every reader**, not just
squashed ones -- checking ``q`` instead of ``q_raw`` would silently
reintroduce D7 in a form even the ``limit_semantics`` flag could not catch
(a raw_rad reader would look "perfect" too). ``q_raw`` -- present only when
``limit_semantics == "raw_rad"`` -- is unclamped by construction, so
``q_raw - clamp(q_raw)`` is the actual violation signal. Both
:class:`LimitViolationFrac` and :class:`LimitExcessRad` therefore declare
``requires={"q_raw"}``.

That requirement alone already makes both metrics unavailable
(``missing_input:q_raw``) for a squashed reader, since
:class:`~kinescore.core.reader.Readout` documents ``q_raw`` as ``None`` in
that case. Both *also* declare ``unobservable_when=("limit_semantics=squashed",)``
as defence in depth: the reason should read ``unobservable:...`` (a
structural, head-architecture fact) rather than ``missing_input:...`` (which
reads like a data-quality accident) whenever the true cause is "this head
family cannot violate its limits by construction", and a future reader that
happens to populate ``q_raw`` alongside a squashed ``q`` should not
accidentally make this metric look observable.

:class:`LimitHeadroomRad` is the substitute that *is* observable under
squashing: distance from the (safe, clamped) ``q`` to the nearest limit. It
answers a different, complementary question -- not "did it violate its
limits" (structurally impossible to answer from a squashed head) but "how
close does it get" (answerable from any head, and arguably the more useful
signal for a squashed head specifically, since near-total absence of
violation is guaranteed and the interesting information is all in the margin).
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
        min_frames=1, unobservable_when=("limit_semantics=squashed",),
        description=(
            "Fraction of frames where any joint's UNSQUASHED reading q_raw "
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
        min_frames=1, unobservable_when=("limit_semantics=squashed",),
        description=(
            "Mean per-frame sum over joints of relu(q_raw-hi)+relu(lo-q_raw) "
            "(rad). The clamp magnitude on the unsquashed reading IS the "
            "signal under raw_rad semantics."))

    def _compute(self, ctx: MetricContext) -> MetricValue:
        if ctx.robot is None:
            return MetricValue.unavailable(self.spec.key, "missing_input:robot")
        lo, hi = _limits(ctx.robot, ctx.q_raw.device, ctx.q_raw.dtype)
        excess = _excess(ctx.q_raw, lo, hi)                        # (B,T,n_joints)
        return self._ok(excess.sum(-1).mean())


class LimitHeadroomRad(SafeMetric):
    """Distance (rad) from the nearest joint limit -- the squashed-head substitute.

    Uses the always-safe ``q`` (never ``q_raw``), so it is observable
    regardless of ``limit_semantics``. Per frame, takes the tightest margin
    across joints (``min(hi-q, q-lo)`` per joint, then the minimum over
    joints); reports the mean of that worst-case-per-frame margin over the
    clip. ``direction="higher_better"``: this is the one metric in the
    package where more is better -- more headroom means the trajectory stays
    further from a mechanical extreme, which is the closest a squashed head
    can get to expressing "did this look like it was straining against its
    limits" even though it can never literally cross them.
    """

    spec = MetricSpec(
        key="limit_headroom_rad", units="rad", dt_exponent=0,
        direction="higher_better", requires=frozenset({"q"}), min_frames=1,
        description=(
            "Mean over frames of min_j(min(hi_j-q_j, q_j-lo_j)) (rad): how "
            "much slack remains to the tightest joint limit at each frame, "
            "using the always-limit-safe q. Observable under squashed "
            "semantics, unlike limit_violation_frac/limit_excess_rad."))

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
