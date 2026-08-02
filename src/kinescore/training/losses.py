"""Training losses for the heteroscedastic pose head.

Ported verbatim (numerics unchanged) from
``models.posendf.readout_v2.beta_nll_loss`` and
``models.kinematics.readout.loss_limit``. Live here, not in
:mod:`kinescore.heads.heteroscedastic`, because that module ships only
:class:`~kinescore.heads.heteroscedastic.ReadoutV2Head`'s *inference* pass --
training losses live with the training code that consumes them. See
``legacy_docs/DECISIONS.md`` D-B for why ``loss_limit`` is a soft hinge and not
another squash (defect D7), and for ``beta_nll_loss``'s re-weighting rationale.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["beta_nll_loss", "loss_limit"]


def beta_nll_loss(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor,
                  beta: float = 0.5, weight: torch.Tensor | None = None
                  ) -> torch.Tensor:
    """Seitzer (2022) beta-NLL for a diagonal Gaussian head.

    Per-element Gaussian NLL, re-weighted by ``detach(var)**beta`` -- see
    ``legacy_docs/DECISIONS.md`` D-B for why (stops the variance head starving the
    mean head of gradient on high-variance targets).

    Parameters
    ----------
    mu, logvar:
        Predicted mean / log-variance, same shape, typically ``(B, T, n_joints)``.
    target:
        Ground truth, same shape as ``mu`` (radians for a joint-angle head).
    beta:
        In ``[0, 1]`` (default ``0.5``): ``0`` is plain NLL, ``1`` is MSE-like.
    weight:
        Optional non-negative weight broadcastable to ``mu``, for per-joint or
        per-frame masking; zero-weight elements don't contribute to the mean.

    Returns
    -------
    torch.Tensor
        Scalar: mean over the (weighted) elements.
    """
    mu = mu.float()
    logvar = logvar.float()
    target = target.float()
    var = torch.exp(logvar)
    nll = 0.5 * (logvar + (target - mu) ** 2 * torch.exp(-logvar))
    loss = nll * var.detach() ** beta  # beta-NLL re-weight
    if weight is not None:
        w = weight.to(loss.dtype)
        loss = loss * w
        denom = w.expand_as(loss).sum().clamp_min(1e-8)
        return loss.sum() / denom
    return loss.mean()


def loss_limit(q_hat: torch.Tensor, joint_limits: torch.Tensor) -> torch.Tensor:
    """Soft joint-limit penalty (hinge on lower / upper bounds).

    Ported verbatim from ``models.kinematics.readout.loss_limit``: zero inside
    ``[lo, hi]``, growing linearly outside it -- the same expression
    :func:`kinescore.heads.ranges.clamp_for_fk` uses to *report* the
    violation at inference time, here used to *train against* it instead. See
    ``legacy_docs/DECISIONS.md`` D-B for why this is a soft penalty, not a squash.

    Parameters
    ----------
    q_hat:
        Predicted joint angles, any shape ``(..., n_joints)``, raw radians
        (unclamped -- this is meant to be called on a head's direct output,
        e.g. ``ReadoutV2Head``'s ``mu``, not on ``clamp_for_fk``'s output).
    joint_limits:
        Per-joint limits ``(n_joints, 2)`` as ``[lo, hi]`` (radians), e.g.
        ``torch.stack([robot.q_lo, robot.q_hi], dim=-1)``.

    Returns
    -------
    torch.Tensor
        Scalar penalty, averaged over every element of ``q_hat``.
    """
    lo = joint_limits[:, 0]  # (n_joints,)
    hi = joint_limits[:, 1]  # (n_joints,)
    # Broadcasts (n_joints,) limits against q_hat's trailing dimension.
    over = F.relu(q_hat - hi)
    under = F.relu(lo - q_hat)
    return (over + under).mean()
