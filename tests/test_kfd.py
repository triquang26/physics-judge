"""W1 profile distance and KFD (approx + exact).

Covers:
* ``profile_w1(x, x) == 0``, ``profile_w1(x, x + c) == |c|``.
* ``kfd_approx`` of a distribution against itself is ~0.
* the exact ``kfd`` agrees with ``kfd_approx`` when the two covariances
  commute, and differs when they do not -- exactly the gap ``kfd_approx``'s
  docstring documents (symmetrise-then-eigh equals the true matrix sqrt only
  under commutativity).
"""
from __future__ import annotations

import numpy as np
import pytest

from kinescore.reference.distances import kfd, kfd_approx, profile_w1

pytest.importorskip("scipy", reason="exact kfd() needs scipy (kinescore[bench])")


def test_profile_w1_self_is_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)  # well below the subsample cap: deterministic
    assert profile_w1(x, x) == pytest.approx(0.0, abs=1e-12)


def test_profile_w1_of_a_constant_shift_is_the_shift():
    rng = np.random.default_rng(1)
    x = rng.normal(size=1000)
    for c in (2.5, -1.0, 0.001):
        assert profile_w1(x, x + c) == pytest.approx(abs(c), abs=1e-9)


def test_profile_w1_subsampling_is_seeded_and_deterministic():
    rng = np.random.default_rng(2)
    x = rng.normal(size=5_000_000)
    y = x + 1.0
    a = profile_w1(x, y, subsample_cap=10_000, seed=42)
    b = profile_w1(x, y, subsample_cap=10_000, seed=42)
    assert a == b  # same seed -> identical subsample -> identical result
    # x and y are subsampled with independent seeds (seed, seed+1), so a
    # constant shift is only approximately recovered (two different 10k
    # subsamples of a continuous distribution have quantile-estimation
    # noise) -- unlike the exact case below the cap, where both sides use
    # the *entire* array and the shift is exact.
    assert a == pytest.approx(1.0, abs=0.05)


def test_kfd_approx_self_vs_self_is_near_zero():
    rng = np.random.default_rng(3)
    m = rng.normal(size=(200, 3))
    cov = np.cov(m.T)
    mu = m.mean(axis=0)
    d = kfd_approx(mu, cov, mu, cov)
    assert d == pytest.approx(0.0, abs=1e-8)


def test_kfd_exact_agrees_with_approx_on_commuting_covariances():
    # Diagonal matrices always commute.
    mu1, mu2 = np.zeros(3), np.array([0.1, -0.2, 0.3])
    cov1 = np.diag([1.0, 2.0, 3.0])
    cov2 = np.diag([4.0, 5.0, 6.0])
    assert np.allclose(cov1 @ cov2, cov2 @ cov1)

    approx = kfd_approx(mu1, cov1, mu2, cov2)
    exact = kfd(mu1, cov1, mu2, cov2)
    assert exact == pytest.approx(approx, abs=1e-9)


def test_kfd_exact_differs_from_approx_on_non_commuting_covariances():
    mu1, mu2 = np.zeros(2), np.zeros(2)
    cov1 = np.array([[4.0, 2.0], [2.0, 3.0]])
    cov2 = np.array([[3.0, 1.0], [1.0, 5.0]])
    assert not np.allclose(cov1 @ cov2, cov2 @ cov1)

    approx = kfd_approx(mu1, cov1, mu2, cov2)
    exact = kfd(mu1, cov1, mu2, cov2)
    # The gap is exactly what the docstring predicts: eig-symmetrise !=
    # sqrtm(prod) once the covariances don't commute. Empirically ~0.08 for
    # this pair; assert a generous but non-trivial threshold so the test
    # doesn't depend on the exact numerical coincidence.
    assert abs(exact - approx) > 1e-3


def test_kfd_without_scipy_raises_helpful_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "scipy.linalg" or name.startswith("scipy"):
            raise ImportError("simulated missing scipy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="kinescore\\[bench\\]"):
        kfd(np.zeros(2), np.eye(2), np.zeros(2), np.eye(2))
