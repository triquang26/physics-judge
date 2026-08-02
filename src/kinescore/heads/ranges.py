"""How a head's raw output becomes a joint-limit-respecting pose for FK.

:func:`clamp_for_fk` is the mechanical core of the ``"raw_rad"``
``limit_semantics`` (see ``core/reader.py``): a head emits unconstrained
radians; this clamps a *copy* into ``[lo, hi]`` for FK and returns the clamp
magnitude (``relu(q-hi) + relu(lo-q) >= 0``) as an explicit, observable
signal -- ``0`` when in range, the exact overshoot otherwise. A reader built
on this exposes ``q_raw`` and declares ``limit_semantics="raw_rad"`` (see
``readers/heteroscedastic.py``).

This module used to also carry ``squash_to_limits`` -- ``q = lo + (hi-lo) *
sigmoid(raw)``, ported verbatim from ``PixelPhysicsJudge.predict_pose`` --
which made every prediction land inside the URDF limits *by construction*.
That is also why joint-limit violation was unmeasurable under it: the squash
is a bijection onto the open interval ``(lo, hi)``, so ``clamp(q) == q``
always and the violation metric was identically ``0.0`` for every clip ever
scored with a squashed head, reading as "perfect" while measuring nothing
about the video (defect D7). The function, its head
(``heads/attentive.py::AttentivePoseHead``), its reader
(``readers/squashed.py::SquashedPoseReader``), and the training loop built
on it (``training/trainer.py``, ``kinescore train``) were all removed once
every real embodiment had a ``raw_rad`` reader -- see ``legacy_docs/PROVENANCE.md``'s
D7 addendum for when and why. ``clamp_for_fk`` was never the problem: it
*reports* the overshoot rather than hiding it, which is the opposite of a
squash.

A pure tensor function (no learned parameters), so it lives alongside the
heads rather than as a method on any one of them --
:mod:`~kinescore.heads.heteroscedastic`, the one live head family, is paired
with it in ``readers/heteroscedastic.py::HeteroscedasticPoseReader.read``.
"""
from __future__ import annotations

import torch

__all__ = ["clamp_for_fk"]


def _broadcast_limits(t: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
    """Reshape a ``(n_joints,)`` limit vector to broadcast against ``t``."""
    limit = limit.to(device=t.device, dtype=t.dtype)
    return limit.view(*([1] * (t.dim() - 1)), -1)


def clamp_for_fk(q_raw: torch.Tensor, q_lo: torch.Tensor, q_hi: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Hard-clamp a raw pose into ``[lo, hi]`` for FK, and report the overshoot.

    The raw head can (deliberately) emit out-of-limit radians. Forward
    kinematics, however, must be fed a physically realisable pose, so we
    clamp. The magnitude we clamped by -- ``relu(q - hi) + relu(lo - q) >=
    0`` -- is the observable limit-violation signal a downstream physics
    metric scores; it is ``0`` in-range and equals the overshoot out-of-range.
    Differentiable w.r.t. ``q`` (clamp / relu have a subgradient; the identity
    ``q_clamped = q - relu(q-hi) + relu(lo-q)`` keeps the autograd path
    intact).

    Args:
        q_raw: Raw joint angles ``(..., n_joints)`` (radians, possibly out of
            limit).
        q_lo, q_hi: Limits ``(n_joints,)``.

    Returns:
        ``(q_clamped (..., n_joints) in [lo,hi], clamp_rad (..., n_joints) >= 0)``.
    """
    lo = _broadcast_limits(q_raw, q_lo)
    hi = _broadcast_limits(q_raw, q_hi)
    over = torch.relu(q_raw - hi)  # >0 above upper limit
    under = torch.relu(lo - q_raw)  # >0 below lower limit
    q_clamped = q_raw - over + under  # == clamp(q, lo, hi), autograd-friendly
    clamp_rad = over + under  # observable violation magnitude (>=0)
    return q_clamped, clamp_rad
