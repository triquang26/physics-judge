"""Two different questions about ``dt``, two different tests.

``test_metric_registry_conformance.py`` asks "does the formula divide by
``dt`` exactly as many times as declared" -- checkable to float32 precision
by holding the sample arrays fixed and only changing the declared ``dt``.

This file asks the *physical* question instead: if a clip is actually
subsampled (``P[::k]``, ``dt*k``), does a properly-declared, homogeneous
metric still recover approximately the same physical answer? This is not
purely algebraic -- resampling changes which frames get differenced, so real
finite-difference truncation error enters, and the two versions can only be
expected to agree within a few percent, not to float32 precision. The
tolerances below (``rtol=0.05`` at k=2, ``0.10`` at k=4) leave roughly 3x
headroom over the actual measured source errors (0.2-5.7%, per
``legacy_docs/PROVENANCE.md``'s D1 analysis) -- tight enough to catch a real
regression, loose enough not to flake on legitimate FD truncation error.

The **negative control** (:func:`test_wrong_dt_is_detected`) is the other
half of the story: it does NOT resample. It keeps the exact same ``P`` array
and scores it once with the correct ``dt`` and once with a ``dt`` that is
wrong by 2x (the literal D1 scenario -- see ``kinescore.core.clip``'s module
docstring: stride-2-decimated frames scored as if they were stride-1). Since
nothing is resampled, the resulting ratios for homogeneous metrics are
*exact* powers of 2 (to float32 precision, ``rtol=1e-5``) -- this is the same
"same array, different declared dt" trick the registry conformance test
uses, specialised to the one scenario the whole benchmark exists to catch.
``accel_violation_frac`` (``dt_exponent=None``) is included specifically
because it does *not* scale by a clean power -- it is a threshold crossing --
which is exactly why it is the most dt-error-sensitive metric in the
package: a 2x dt error does not gently nudge it, it can snap it from "no
violations" to "a third of all frames violating".
"""
from __future__ import annotations

import math

import torch

import kinescore.metrics  # noqa: F401  (side effect: populates the registry)
from kinescore.core.metric import MetricContext
from kinescore.metrics.temporal import (
    AccelViolationFrac,
    MeanAccel,
    MeanJerk,
    MeanSpeed,
)

# ===========================================================================
# test_dt_invariance: real resampling, loose tolerance
# ===========================================================================

def _smooth_trajectory(t: torch.Tensor) -> torch.Tensor:
    """``(N,) times -> (N,K=2,3) positions``: a smooth, non-polynomial path.

    A sum of a few incommensurate sinusoids plus a linear drift -- smooth
    enough that finite-difference truncation error shrinks predictably with
    the sampling step, but with genuinely nonzero jerk everywhere (unlike a
    pure polynomial motion, which would make some of these derivatives exactly
    zero and under-exercise the comparison).
    """
    x1 = 0.3 * torch.sin(1.3 * t) + 0.1 * torch.cos(2.7 * t) + 0.05 * t
    y1 = 0.2 * torch.cos(0.9 * t) + 0.05 * torch.sin(3.1 * t)
    z1 = 0.1 * torch.sin(1.7 * t + 0.4)
    x2 = 0.25 * torch.sin(1.1 * t + 0.2) - 0.05 * t
    y2 = 0.15 * torch.cos(1.9 * t)
    z2 = 0.08 * torch.cos(2.3 * t)
    kp1 = torch.stack([x1, y1, z1], dim=-1)
    kp2 = torch.stack([x2, y2, z2], dim=-1)
    return torch.stack([kp1, kp2], dim=1)                          # (N,2,3)


def _fine_clip(fine_dt: float, n_fine: int) -> torch.Tensor:
    t = torch.arange(n_fine, dtype=torch.float64) * fine_dt
    P = _smooth_trajectory(t.double()).float()
    return P.unsqueeze(0)                                          # (1,N,2,3)


def _score(metric, P: torch.Tensor, dt: float) -> float:
    ctx = MetricContext(dt=dt, P=P)
    v = metric.compute(ctx)
    assert v.available, f"expected available, got reason={v.reason}"
    return v.value


def _assert_subsample_agrees(metric, rtol_k2: float, rtol_k4: float,
                             fine_dt: float = 0.01, n_fine: int = 400) -> None:
    P_fine = _fine_clip(fine_dt, n_fine)
    baseline = _score(metric, P_fine, fine_dt)

    for k, rtol in ((2, rtol_k2), (4, rtol_k4)):
        P_sub = P_fine[:, ::k]
        dt_sub = fine_dt * k
        v_sub = _score(metric, P_sub, dt_sub)
        rel = abs(v_sub - baseline) / (abs(baseline) + 1e-9)
        assert rel < rtol, (
            f"{metric.spec.key}: k={k} subsample disagrees with the fine "
            f"baseline by {rel:.4f} (rtol={rtol}); baseline={baseline!r} "
            f"subsampled={v_sub!r}")


def test_mean_speed_dt_invariant():
    _assert_subsample_agrees(MeanSpeed(), rtol_k2=0.05, rtol_k4=0.10)


def test_mean_accel_dt_invariant():
    _assert_subsample_agrees(MeanAccel(), rtol_k2=0.05, rtol_k4=0.10)


def test_mean_jerk_dt_invariant():
    _assert_subsample_agrees(MeanJerk(), rtol_k2=0.05, rtol_k4=0.10)


# ===========================================================================
# negative control: WRONG dt is exactly detected (same array, no resampling)
# ===========================================================================

def _euler_integrate_from_accel(a: torch.Tensor, dt: float) -> torch.Tensor:
    """``a: (B,T-2,K,3) -> P: (B,T,K,3)`` by double Euler integration at ``dt``.

    Constructs positions such that the finite-difference acceleration
    RECOVERED from P at exactly this ``dt`` equals ``a`` exactly (this is
    the discrete inverse of ``fd(fd(P,dt),dt)``): starting from rest,
    ``v[t+1] = v[t] + a[t]*dt``, ``P[t+1] = P[t] + v[t]*dt``.
    """
    b, tm2, k, _ = a.shape
    v = torch.zeros(b, tm2 + 1, k, 3)
    for t in range(tm2):
        v[:, t + 1] = v[:, t] + a[:, t] * dt
    P = torch.zeros(b, tm2 + 2, k, 3)
    for t in range(tm2 + 1):
        P[:, t + 1] = P[:, t] + v[:, t] * dt
    return P


def _accel_violation_fixture() -> tuple[torch.Tensor, float, float]:
    """A deterministic ``P`` where ``accel_violation_frac`` goes exactly
    0.000 (true dt) -> 0.3875 (dt wrong by 2x), reproducing the measured D1
    figure from the design doc.

    Construction: 80 = (T-2=10) x (K=8) per-(frame,keypoint) accelerations,
    magnitude either 0.5 or 2.0 m/s^2 (both comfortably under the default
    accel_bound=5.0), with exactly 31 of the 80 set to 2.0 -- a fixed,
    reproducible permutation (seed 0). Positions are built by double Euler
    integration at ``true_dt`` (see :func:`_euler_integrate_from_accel`), so
    the finite-difference acceleration recovered from P at ``true_dt`` is
    EXACTLY the constructed array (Euler integration is the discrete inverse
    of the difference-based acceleration used everywhere in this package).

    Scoring the SAME P at ``wrong_dt = true_dt/2`` (the D1 scenario: frames
    were really true_dt apart -- e.g. stride-2-decimated -- but scored as if
    stride-1) inflates every recovered acceleration by exactly
    ``(true_dt/wrong_dt)**2 = 4x``: magnitude 0.5 -> 2.0 (still < 5.0, no
    violation), magnitude 2.0 -> 8.0 (> 5.0, violation). So the violating
    fraction goes from 0/80 to exactly 31/80 = 0.3875.
    """
    true_dt = 0.2
    wrong_dt = 0.1
    b, tm2, k = 1, 10, 8
    total = tm2 * k
    n_high = 31

    mag = torch.full((total,), 0.5)
    mag[:n_high] = 2.0
    g = torch.Generator().manual_seed(0)
    mag = mag[torch.randperm(total, generator=g)].view(b, tm2, k)

    a = torch.zeros(b, tm2, k, 3)
    a[..., 0] = mag                                                 # all along x
    P = _euler_integrate_from_accel(a, true_dt)
    return P, true_dt, wrong_dt


def test_wrong_dt_is_detected():
    """The D1 negative control: a 2x dt error is not silently absorbed.

    Same ``P`` array scored at the correct ``dt`` and at a ``dt`` wrong by
    2x (no resampling -- see the module docstring for why that makes the
    ratios exact rather than approximate). Checks:

    * speed/accel/jerk ratios are EXACTLY 2/4/8 (rtol=1e-5) -- the direct
      consequence of ``dt_exponent`` 1/2/3 applied to the same underlying
      arrays, per ``ops.fd``'s "divide by dt, nothing else" contract.
    * ``accel_violation_frac`` goes from 0.000 to 0.3875 -- the metric most
      sensitive to a wrong dt of any in the package (see
      ``kinescore.metrics.temporal``'s module docstring), because it
      thresholds against a fixed physical constant rather than scaling
      smoothly.
    """
    P, true_dt, wrong_dt = _accel_violation_fixture()

    speed_true = _score(MeanSpeed(), P, true_dt)
    speed_wrong = _score(MeanSpeed(), P, wrong_dt)
    accel_true = _score(MeanAccel(), P, true_dt)
    accel_wrong = _score(MeanAccel(), P, wrong_dt)
    jerk_true = _score(MeanJerk(), P, true_dt)
    jerk_wrong = _score(MeanJerk(), P, wrong_dt)

    assert math.isclose(speed_wrong / speed_true, 2.0, rel_tol=1e-5)
    assert math.isclose(accel_wrong / accel_true, 4.0, rel_tol=1e-5)
    assert math.isclose(jerk_wrong / jerk_true, 8.0, rel_tol=1e-5)

    frac_true = _score(AccelViolationFrac(), P, true_dt)
    frac_wrong = _score(AccelViolationFrac(), P, wrong_dt)
    assert frac_true == 0.0, f"expected 0.000 at the true dt, got {frac_true}"
    assert math.isclose(frac_wrong, 0.3875, rel_tol=1e-5), (
        f"expected accel_violation_frac == 0.3875 at the wrong dt, got {frac_wrong}")
