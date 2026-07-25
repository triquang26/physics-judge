"""How a head's raw output becomes a joint-limit-respecting pose -- two ways.

These two functions are the mechanical core of ``limit_semantics`` (see
``core/reader.py``), and they trade off the same thing in opposite
directions:

* :func:`squash_to_limits` -- ``q = lo + (hi-lo) * sigmoid(raw)`` (ported
  verbatim from ``PixelPhysicsJudge.predict_pose`` /
  ``GR1PixelPhysicsJudge.predict_pose``). Every prediction is inside the URDF
  limits *by construction*, which sounds like a feature. It is also why
  joint-limit violation is unmeasurable: the squash is a bijection onto the
  open interval ``(lo, hi)``, so ``clamp(q) == q`` always and the violation
  metric is identically ``0.0`` for every clip ever scored with a squashed
  head -- reading as "perfect" while measuring nothing about the video
  (defect D7). A reader built on this function must declare
  ``limit_semantics="squashed"`` and ``q_raw=None`` (see
  ``readers/squashed.py``) so downstream metrics know to report the
  violation metric as unobservable rather than as a suspiciously perfect
  zero.

* :func:`clamp_for_fk` -- ported verbatim from
  ``models.posendf.readout_v2.clamp_for_fk``. The head emits unconstrained
  radians; this clamps a *copy* into ``[lo, hi]`` for FK and returns the
  clamp magnitude (``relu(q-hi) + relu(lo-q) >= 0``) as an explicit,
  observable signal -- ``0`` when in range, the exact overshoot otherwise.
  A reader built on this exposes ``q_raw`` and declares
  ``limit_semantics="raw_rad"`` (see ``readers/heteroscedastic.py``).

Both are pure tensor functions (no learned parameters), so they live
alongside the heads rather than as methods on any one of them -- every head
family (:mod:`~kinescore.heads.attentive`, :mod:`~kinescore.heads.mlp`,
:mod:`~kinescore.heads.heteroscedastic`, :mod:`~kinescore.heads.disentangled`)
can be paired with either.
"""
from __future__ import annotations

from typing import Tuple

import torch

__all__ = ["squash_to_limits", "clamp_for_fk"]


def _broadcast_limits(t: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
    """Reshape a ``(n_joints,)`` limit vector to broadcast against ``t``."""
    limit = limit.to(device=t.device, dtype=t.dtype)
    return limit.view(*([1] * (t.dim() - 1)), -1)


def squash_to_limits(raw: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor
                     ) -> torch.Tensor:
    """Sigmoid-squash raw head output into ``[lo, hi]``.

    ``raw``: ``(..., n_joints)`` unconstrained. ``lo``, ``hi``: ``(n_joints,)``
    radians. Returns ``(..., n_joints)`` inside ``[lo, hi]`` -- strictly
    inside ``(lo, hi)`` for any finite ``sigmoid(raw)`` in ``(0,1)``, landing
    exactly on ``lo``/``hi`` only where ``|raw|`` is large enough that fp32
    ``sigmoid`` itself saturates to ``0.0``/``1.0`` (empirically ``|raw| >~
    20``) -- a floating-point property of the squash, not a violation of it.

    See the module docstring -- this makes joint-limit violation structurally
    unobservable (defect D7). Do not use this to *report* limit compliance;
    use it only to produce a pose that is safe to feed to FK.
    """
    lo_b = _broadcast_limits(raw, lo)
    hi_b = _broadcast_limits(raw, hi)
    return lo_b + (hi_b - lo_b) * torch.sigmoid(raw)


def clamp_for_fk(q_raw: torch.Tensor, q_lo: torch.Tensor, q_hi: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
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
