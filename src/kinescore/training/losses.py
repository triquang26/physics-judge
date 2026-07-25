"""Training loss for the heteroscedastic pose head.

Ported verbatim (numerics unchanged) from
``models.posendf.readout_v2.beta_nll_loss``. It lives here rather than in
:mod:`kinescore.heads.heteroscedastic` because that module deliberately ships
only :class:`~kinescore.heads.heteroscedastic.ReadoutV2Head`'s *inference*
forward pass -- see its docstring: "the training-time losses ... are not part
of the pose-reader contract this package owns and live with the training code
that consumes them." This is that training code.
"""
from __future__ import annotations

from typing import Optional

import torch

__all__ = ["beta_nll_loss"]


def beta_nll_loss(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor,
                  beta: float = 0.5, weight: Optional[torch.Tensor] = None
                  ) -> torch.Tensor:
    """Seitzer (2022) beta-NLL for a diagonal Gaussian head.

    Per-element Gaussian NLL is ``0.5 * (logvar + (target-mu)**2 * exp(-logvar))``.
    Each element is re-weighted by ``detach(var)**beta`` (``var = exp(logvar)``):
    this interpolates the gradient scaling between plain NLL (``beta=0``) and
    a pure MSE-like weighting (``beta=1``), which is what stops the variance
    head from starving the mean head of gradient on high-variance targets --
    a known failure mode of the un-weighted Gaussian NLL, where the model
    learns to inflate ``sigma`` on hard examples rather than fit ``mu``
    better.

    Parameters
    ----------
    mu:
        Predicted mean, any shape, typically ``(B, T, n_joints)``.
    logvar:
        Predicted log-variance, same shape as ``mu``.
    target:
        Ground truth, same shape as ``mu`` (radians for a joint-angle head).
    beta:
        Beta in ``[0, 1]`` (default ``0.5``, the source's default -- a
        midpoint between the two failure modes ``beta=0``/``beta=1`` trade
        off).
    weight:
        Optional non-negative weight broadcastable to ``mu``'s shape, for
        per-joint or per-frame masking (e.g. gating out low-confidence
        frames before they contribute a gradient). Masked (zero-weight)
        elements do not contribute to the mean.

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
