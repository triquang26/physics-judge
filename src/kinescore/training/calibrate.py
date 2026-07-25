"""Post-hoc sigma calibration for the heteroscedastic pose head.

Ported (numerics unchanged) from
``models.posendf.readout_v2.fit_sigma_temperature``, with a small,
non-numeric wrapper (:class:`CalibrationResult` /
:func:`calibrate_sigma`/:func:`apply_temperature`) added so a caller doesn't
have to hand-assemble the standardized residual itself and can persist the
fitted temperature alongside a checkpoint.

Why this is a separate, cheap post-hoc step and not folded into training
--------------------------------------------------------------------------
beta-NLL training (:func:`kinescore.training.losses.beta_nll_loss`) recovers
the correct *ranking* of predicted sigma (which frames/joints are noisier
than others) but is routinely mis-*scaled*: the standardized residual
``z = (mu - target) / sigma`` on a clean validation split typically does not
have unit std (``std(z) > 1`` means the head is overconfident). Fitting a
single scalar (or per-joint) temperature ``T`` after training, then reporting
``sigma_cal = T * sigma``, fixes the absolute scale without touching the
trained weights or perturbing the sigma *ranking* every percentile-based
:mod:`kinescore.readers.heteroscedastic`/ensemble gate depends on.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "CalibrationResult", "standardized_residual", "fit_sigma_temperature",
    "calibrate_sigma", "apply_temperature",
]


def standardized_residual(mu: torch.Tensor, target: torch.Tensor,
                          sigma: torch.Tensor) -> torch.Tensor:
    """``(mu - target) / sigma`` -- the quantity :func:`fit_sigma_temperature` fits."""
    return (mu - target) / sigma


def fit_sigma_temperature(z: torch.Tensor, per_joint: bool = False,
                          eps: float = 1e-8) -> torch.Tensor:
    """Fit a post-hoc sigma **temperature** ``T`` so calibrated ``z / T`` has unit std.

    The moment-matching / Gaussian-MLE optimum that forces ``std(z / T) == 1``
    is simply ``T = std(z)`` (a well-trained mean head has ``mean(z) ~ 0``, so
    the centered and raw std coincide). This is a *scale-only* recalibration.

    Parameters
    ----------
    z:
        Standardized residuals ``(mu - target) / sigma``, any shape. When
        ``per_joint`` the LAST axis is the joint axis and one ``T`` is fit
        per joint; otherwise a single global scalar spans all elements.
    per_joint:
        Fit a ``(n_joints,)`` vector instead of a scalar.
    eps:
        Floor keeping ``T`` strictly positive (guards a degenerate
        all-identical-``z`` validation split, where ``std`` would be exactly
        ``0`` and dividing by it downstream would produce ``inf``).

    Returns
    -------
    torch.Tensor
        fp32, scalar ``()`` (global) or ``(n_joints,)`` (per-joint), ``>= eps``.
    """
    z = z.float()
    if per_joint:
        T = z.reshape(-1, z.shape[-1]).std(dim=0, unbiased=True)
    else:
        T = z.reshape(-1).std(unbiased=True)
    return T.clamp_min(eps)


@dataclass(frozen=True)
class CalibrationResult:
    """A fitted sigma temperature plus enough context to apply it correctly.

    Parameters
    ----------
    temperature:
        Scalar ``()`` or ``(n_joints,)``, see :func:`fit_sigma_temperature`.
    per_joint:
        Whether ``temperature`` is per-joint (must match when
        :func:`apply_temperature` is later called on a new ``sigma``).
    n_samples:
        How many ``(frame, joint)`` residuals the fit was computed over --
        recorded so a temperature fit on a handful of validation frames
        (unreliable) is distinguishable from one fit on thousands.
    """

    temperature: torch.Tensor
    per_joint: bool
    n_samples: int


def calibrate_sigma(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor,
                    *, per_joint: bool = False) -> CalibrationResult:
    """Fit a :class:`CalibrationResult` from one validation batch.

    Parameters
    ----------
    mu, logvar:
        A heteroscedastic head's raw output on a held-out validation split
        (e.g. :class:`~kinescore.heads.heteroscedastic.ReadoutV2Head`'s
        ``forward()`` dict entries), any matching shape.
    target:
        Ground-truth joint angles, same shape as ``mu``.
    per_joint:
        See :func:`fit_sigma_temperature`.
    """
    sigma = torch.exp(0.5 * logvar.float())
    z = standardized_residual(mu.float(), target.float(), sigma)
    temperature = fit_sigma_temperature(z, per_joint=per_joint)
    return CalibrationResult(temperature=temperature, per_joint=per_joint,
                             n_samples=int(z.numel()))


def apply_temperature(sigma: torch.Tensor, calibration: CalibrationResult
                      ) -> torch.Tensor:
    """``sigma * calibration.temperature``, broadcasting a per-joint fit correctly."""
    return sigma * calibration.temperature.to(device=sigma.device, dtype=sigma.dtype)
