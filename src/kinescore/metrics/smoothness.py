"""Dimensionless, scale-invariant movement-smoothness metrics on an EE speed profile.

The physics referee's ``mean_jerk_mps3`` (``temporal.MeanJerk``) is a *raw*
jerk magnitude in m/s^3: it depends on the movement's amplitude, duration and
sampling rate, so it is not comparable across clips of different speed or
length. This module ports the two citable, dimensionless motor-control
smoothness standards from Marionette-fkjepa's ``models/physics/smoothness.py``
**verbatim** (pure numpy, fp64 internally, no torch, no FK dependency):

* :func:`sparc` -- Spectral Arc Length (Balasubramanian, Melendez-Calderon,
  Roby-Brami & Burdet, *"On the analysis of movement smoothness"*, J.
  NeuroEngineering and Rehabilitation 12:112, 2015). A negative dimensionless
  value; smoother -> closer to 0.
* :func:`dimensionless_jerk` / :func:`log_dimensionless_jerk` -- the
  normalised jerk cost of Hogan & Sternad, *"Sensitivity of smoothness
  measures to movement duration, amplitude, and arrests"*, J. Motor Behavior
  41(6):529-534 (2009), log form from Balasubramanian et al. 2015. Higher
  LDLJ = smoother (sign matches SPARC).

``dt_exponent`` -- these are not both dt-invariant for the same reason
--------------------------------------------------------------------------
``log_dimensionless_jerk`` earns its name literally: ``dimensionless_jerk``
multiplies ``duration**5`` (which scales as ``dt**1`` per unit of ``duration``
-> contributes ``dt_exponent -5``) by an integral of squared jerk (jerk scales
as ``dt**-3``, squared is ``dt**-6``, integrated with one more factor of
``dt`` from the ``* dt`` quadrature weight -> contributes ``dt_exponent +5``)
divided by squared path length (path length's own ``dt`` cancels exactly
against the ``/dt`` inside the speed used to compute it -- see the docstring
on :func:`dimensionless_jerk`). Net: **exactly** ``dt**0``, algebraically, not
approximately -- verified by ``tests/test_metric_registry_conformance.py`` to
the same ``rtol=1e-6`` as every other declared metric. ``LogDimensionlessJerk``
therefore declares ``dt_exponent=0``.

:func:`sparc` has no such cancellation. It builds its frequency axis from
``fs = 1/dt`` and keeps only bins below a **fixed absolute** cutoff
``fc=10 Hz``. Changing ``dt`` changes which physical frequencies the
``nfft``-sized bin grid represents, so it changes *which portion of the same
normalised spectrum* the arc-length sum is taken over -- not a clean
multiplicative scaling of the result, and not the kind of thing a single
exponent can capture (the same failure mode as an absolute threshold
elsewhere in this package: ``fc`` is a constant in Hz, not a quantity that
moves with ``dt``). ``Sparc`` therefore declares ``dt_exponent=None``.

Aggregation over embodiments with a variable end-effector count
-----------------------------------------------------------------
The source computed these per-arm for a fixed bimanual (left/right) layout.
Both metric classes here instead aggregate (nan-aware mean) over
``robot.ee_sites()``, so the same code scores a 1-EE Franka or an N-EE
humanoid without a robot-specific branch.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from kinescore.core.metric import MetricContext, MetricSpec, MetricValue, register
from kinescore.metrics._base import SafeMetric

__all__ = [
    "sparc", "dimensionless_jerk", "log_dimensionless_jerk",
    "Sparc", "LogDimensionlessJerk",
]

#: Minimum frames for a stable spectrum / 3rd derivative. Source constant
#: ``_MIN_T``; surfaced here as each metric's declared ``min_frames`` so a
#: too-short clip yields an explicit ``too_few_frames`` reason instead of the
#: pure function's internal (silent) ``nan``.
MIN_T = 8
_EPS = 1e-9


def _speed_profile(pos_traj: np.ndarray, dt: float) -> np.ndarray:
    """``(T,3) | (T,) -> (T-1,)`` speed magnitude of the position trajectory (m/s).

    Verbatim port of the source's ``_speed_profile``.
    """
    p = np.asarray(pos_traj, dtype=np.float64)
    if p.ndim == 1:
        p = p[:, None]
    v = np.diff(p, axis=0) / float(dt)
    return np.linalg.norm(v, axis=-1)


# ── SPARC -- spectral arc length (Balasubramanian et al. 2012/2015) ─────────
def sparc(speed_profile, dt: float, fc: float = 10.0, amp_th: float = 0.05,
          padlevel: int = 4) -> float:
    """Spectral arc length of a movement speed profile (dimensionless, <= 0).

    Verbatim port of the source's ``sparc`` (itself a faithful replication of
    ``spectral_arclength`` in https://github.com/siva82kb/smoothness). See the
    module docstring for the algorithm summary and why it is not dt-invariant.
    """
    v = np.asarray(speed_profile, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size < MIN_T or not np.any(np.abs(v) > _EPS):
        return float("nan")
    fs = 1.0 / float(dt)
    try:
        nfft = int(pow(2, np.ceil(np.log2(v.size)) + padlevel))
        f = np.arange(0, fs, fs / nfft)
        Mf = np.abs(np.fft.fft(v, nfft))
        mx = Mf.max()
        if mx <= _EPS:
            return float("nan")
        Mf = Mf / mx
        fc_inx = np.nonzero(f <= fc)[0]
        f_sel, Mf_sel = f[fc_inx], Mf[fc_inx]
        inx = np.nonzero(Mf_sel >= amp_th)[0]
        if inx.size == 0:
            return float("nan")
        band = np.arange(inx[0], inx[-1] + 1)
        f_sel, Mf_sel = f_sel[band], Mf_sel[band]
        span = f_sel[-1] - f_sel[0]
        if span <= _EPS or f_sel.size < 2:
            return float("nan")
        arc = -np.sum(np.sqrt(np.square(np.diff(f_sel) / span)
                              + np.square(np.diff(Mf_sel))))
        return float(arc)
    except Exception:                                              # noqa: BLE001
        return float("nan")


# ── Dimensionless jerk (Hogan & Sternad 2009, position form) ────────────────
def dimensionless_jerk(pos_traj, dt: float) -> float:
    """Dimensionless jerk cost of an EE position trajectory (higher = jerkier).

    Verbatim port of the source's ``dimensionless_jerk``. ``nan`` for
    ``T < MIN_T`` or a stationary trajectory (zero path length).
    """
    p = np.asarray(pos_traj, dtype=np.float64)
    if p.ndim == 1:
        p = p[:, None]
    T = p.shape[0]
    if T < MIN_T:
        return float("nan")
    dt = float(dt)
    speed = np.linalg.norm(np.diff(p, axis=0) / dt, axis=-1)       # (T-1,)
    path_len = float(np.sum(speed) * dt)                           # dt cancels exactly
    if path_len <= _EPS:
        return float("nan")
    duration = (T - 1) * dt
    jerk = np.diff(p, n=3, axis=0) / (dt ** 3)                     # (T-3,3) m/s^3
    integral = float(np.sum(np.square(jerk).sum(axis=-1)) * dt)
    dlj = (duration ** 5 / (path_len ** 2)) * integral
    return float(dlj)


def log_dimensionless_jerk(pos_traj, dt: float) -> float:
    """``LDLJ = -ln|DLJ|`` -- higher = smoother. Verbatim port of the source."""
    dlj = dimensionless_jerk(pos_traj, dt)
    if not np.isfinite(dlj) or abs(dlj) <= _EPS:
        return float("nan")
    return float(-np.log(abs(dlj)))


def _ee_trajectories(P: torch.Tensor, ee_sites: Sequence[int]) -> list:
    """``(B,T,K,3)`` + EE indices -> list of ``(T,3)`` float64 numpy arrays.

    One entry per ``(batch element, EE site)`` pair, flattened -- this is the
    population the two metric classes below nan-aware-average over.
    """
    out = []
    for b in range(P.shape[0]):
        for site in ee_sites:
            out.append(P[b, :, site, :].detach().cpu().double().numpy())
    return out


def _nanmean_or_nan(values: list) -> float:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


class Sparc(SafeMetric):
    """Spectral arc length, aggregated (nan-aware) over ``robot.ee_sites()``.

    See the module docstring for why this declares ``dt_exponent=None``.
    """

    spec = MetricSpec(
        key="sparc", units="dimensionless", dt_exponent=None,
        direction="higher_better", requires=frozenset({"P"}), min_frames=MIN_T,
        description=(
            "Spectral arc length of each end-effector's speed profile "
            "(Balasubramanian 2015), nan-aware mean over robot.ee_sites(). "
            "Closer to 0 = smoother. dt_exponent=None: built from an "
            "absolute Hz cutoff (fc=10), not a power-law-preserving "
            "constant divisor."))

    def _compute(self, ctx: MetricContext) -> MetricValue:
        if ctx.robot is None:
            return MetricValue.unavailable(self.spec.key, "missing_input:robot")
        ee_sites = ctx.robot.ee_sites()
        vals = [sparc(_speed_profile(traj, ctx.dt), ctx.dt)
                for traj in _ee_trajectories(ctx.P, ee_sites)]
        return self._ok(_nanmean_or_nan(vals))


class LogDimensionlessJerk(SafeMetric):
    """Log dimensionless jerk, aggregated (nan-aware) over ``robot.ee_sites()``.

    ``dt_exponent=0``: see the module docstring for the exact algebraic
    cancellation that makes this metric dt-invariant by construction, unlike
    :class:`Sparc`.
    """

    spec = MetricSpec(
        key="log_dimensionless_jerk", units="dimensionless", dt_exponent=0,
        direction="higher_better", requires=frozenset({"P"}), min_frames=MIN_T,
        description=(
            "LDLJ = -ln|DLJ| (Hogan & Sternad 2009 / Balasubramanian 2015) "
            "of each end-effector's position trajectory, nan-aware mean "
            "over robot.ee_sites(). Higher = smoother. Exactly dt-invariant "
            "by algebraic construction (duration^5 * integral(jerk^2) "
            "cancels dt exactly against path_len^2)."))

    def _compute(self, ctx: MetricContext) -> MetricValue:
        if ctx.robot is None:
            return MetricValue.unavailable(self.spec.key, "missing_input:robot")
        ee_sites = ctx.robot.ee_sites()
        vals = [log_dimensionless_jerk(traj, ctx.dt)
                for traj in _ee_trajectories(ctx.P, ee_sites)]
        return self._ok(_nanmean_or_nan(vals))


register(Sparc())
register(LogDimensionlessJerk())
