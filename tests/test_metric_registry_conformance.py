"""Auto-parametrised structural check: every registered metric's declared
``dt_exponent`` is verified numerically, not just documented.

This is the guarantee that closes defect D1 (a wrong/hidden ``dt`` silently
corrupting a benchmark) at the registry level: a metric cannot be added to
``kinescore.metrics`` without a :class:`~kinescore.core.metric.MetricSpec`,
and every spec's ``dt_exponent`` is checked here against the metric's actual
arithmetic, for every metric currently registered -- so a future metric that
declares the wrong exponent fails CI immediately, on the metric it was added
in, rather than surfacing three releases later as "why does this number look
different at 10fps vs 30fps".

Why this can use ``rtol=1e-6`` (not the looser tolerance used elsewhere)
--------------------------------------------------------------------------
See ``kinescore/metrics/ops.py``'s module docstring for the full argument;
briefly: this test holds the sample arrays (``P``/``R``/``q``/``q_raw``)
completely fixed and only changes the ``dt`` **value** passed in
``MetricContext``. Every derivative in this package is built by dividing by
``dt`` some fixed number of times (:func:`kinescore.metrics.ops.fd`), so
scaling ``dt`` by ``k`` scales such a derivative by a power of ``k`` with no
other source of numerical error -- unlike *resampling* a trajectory
(``P[::k]``, ``dt*k``), which is what ``test_dt_invariance.py`` checks
instead, at a much looser tolerance, because resampling changes which
samples are differenced and introduces real finite-difference truncation
error. This test is not "does the metric behave sensibly under
subsampling" (that's the other file); it is "does the formula divide by
``dt`` exactly as many times as its spec claims", which is checkable to
float32 precision.

Metrics that are structurally excluded from this check
----------------------------------------------------------
* ``dt_exponent=None`` metrics (``accel_violation_frac``,
  ``vel_violation_frac``, ``no_teleport_frac``, ``total_energy_tstd``,
  ``sparc``) declare exactly that they are *not* homogeneous in ``dt`` --
  see each one's own docstring for why. Nothing to verify as a power law.
* A metric whose value is unavailable on the synthetic fixture used here
  (e.g. it needs a robot capability -- ``colliders``, ``support_polygon`` --
  that the fixture robot does not have, or an optional URDF field like
  ``vel_limits``/``effort_limits`` the fixture leaves ``None``) is skipped
  with an explicit reason rather than silently passing: an unavailable
  metric trivially satisfies any numeric equality (``nan == nan/k**n`` is
  never true, so it would actually *fail* if not skipped), and skip-with-
  reason keeps that failure mode from masquerading as either a pass or a
  silent gap.
"""
from __future__ import annotations

import math

import pytest
import torch
from _fake_robot import FakeRobot

import kinescore.metrics  # noqa: F401  (side effect: populates the registry)
from kinescore.core.metric import MetricContext, all_metrics

# ── fixture trajectory ────────────────────────────────────────────────────
#: A single, shared "clip" -- arbitrary but non-degenerate values so every
#: quantity built from it (speed, accel, jerk, angular rate, joint rate...)
#: is comfortably non-zero and finite. T=10 clears every metric's
#: min_frames (max requirement in this package is 8, smoothness's MIN_T).
_B, _T, _K, _NQ = 1, 10, 3, 2
_DT0 = 0.1


def _make_ctx(dt: float, robot) -> MetricContext:
    g = torch.Generator().manual_seed(0)
    P = torch.cumsum(torch.randn(_B, _T, _K, 3, generator=g) * 0.05, dim=1)
    # R: a smoothly varying, valid rotation sequence (small per-frame twists
    # composed), not just random matrices -- angular metrics need genuine
    # SO(3) elements to extract a sensible vee(skew(...)) rate from.
    angles = torch.cumsum(torch.rand(_B, _T, _K, generator=g) * 0.05, dim=1)
    c, s = torch.cos(angles), torch.sin(angles)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    R = torch.stack([
        torch.stack([c, -s, zero], dim=-1),
        torch.stack([s, c, zero], dim=-1),
        torch.stack([zero, zero, one], dim=-1),
    ], dim=-2)
    q = torch.cumsum(torch.randn(_B, _T, _NQ, generator=g) * 0.05, dim=1)
    q_raw = q.clone()
    return MetricContext(dt=dt, P=P, R=R, q=q, q_raw=q_raw, robot=robot)


@pytest.fixture(scope="module")
def fixture_robot() -> FakeRobot:
    """A robot with every optional field populated except vel/effort limits.

    ``vel_limits``/``effort_limits`` are left ``None`` on purpose: those two
    metrics (``vel_violation_frac``, ``effort_proxy``) are exercised for
    their *gating* behaviour in ``test_joint_dynamics``-style coverage
    elsewhere; here they simply get skipped as "unavailable on this fixture"
    like the capability-gated feasibility metrics.
    """
    return FakeRobot(
        bone_pairs=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        bone_lengths=torch.tensor([1.0, 1.0], dtype=torch.float32),
        q_lo=torch.tensor([-10.0, -10.0]), q_hi=torch.tensor([10.0, 10.0]),
    )


def _metric_cases():
    return sorted(all_metrics().items())


@pytest.mark.parametrize("key,metric", _metric_cases(), ids=lambda x: x if isinstance(x, str) else "")
def test_dt_exponent_conformance(key, metric, fixture_robot):
    spec = metric.spec
    if spec.dt_exponent is None:
        pytest.skip(f"{key}: dt_exponent=None, excluded from the power-law check "
                    f"by design (see its module docstring)")

    ctx1 = _make_ctx(_DT0, fixture_robot)
    ctx2 = _make_ctx(_DT0 * 3, fixture_robot)   # k=3: distinguishes 1/2/3 cleanly

    v1 = metric.compute(ctx1)
    v2 = metric.compute(ctx2)

    if not v1.available or not v2.available:
        pytest.skip(f"{key}: unavailable on the shared fixture "
                    f"(v1.reason={v1.reason!r}, v2.reason={v2.reason!r}) -- "
                    f"needs a capability/optional field this fixture doesn't provide")

    assert math.isfinite(v1.value) and math.isfinite(v2.value), (
        f"{key}: available MetricValue must be finite (v1={v1.value}, v2={v2.value})")

    k = 3.0
    expected = v1.value / (k ** spec.dt_exponent)
    if expected == 0.0:
        assert abs(v2.value) < 1e-6, (
            f"{key}: dt_exponent={spec.dt_exponent} predicts 0 at k*dt but got {v2.value}")
    else:
        rel = abs(v2.value - expected) / abs(expected)
        assert rel < 1e-6, (
            f"{key}: dt_exponent={spec.dt_exponent} declared but "
            f"metric(k*dt)={v2.value!r} != metric(dt)/k**dt_exponent={expected!r} "
            f"(rel error {rel:.3e}, k={k})")


def test_every_metric_has_a_spec_and_unique_key():
    metrics = all_metrics()
    assert len(metrics) > 0, "expected kinescore.metrics to have registered something"
    for key, metric in metrics.items():
        assert metric.spec.key == key


def test_all_metrics_importable_without_suites_module():
    """Importing kinescore.metrics alone (not .suites) must populate the registry.

    Guards the layering documented in metrics/__init__.py and suites.py: the
    registry must not depend on the suite module having been imported.
    """
    import subprocess
    import sys as _sys
    result = subprocess.run(
        [_sys.executable, "-c",
         "import kinescore.metrics; from kinescore.core.metric import all_metrics; "
         "assert len(all_metrics()) >= 20, all_metrics(); print('ok')"],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
