"""Group 3: distribution realism -- W1 profile distance and Kinematic Frechet Distance.

Both compare a *set* of generated rollouts' motion-quantity samples against the
frozen :class:`~kinescore.reference.fingerprint.RealMotionReference`. Absolute
speed/accel/jerk are task-dependent (a fast wipe and a slow insertion are both
"real"), so only the *shape* of the distribution against real data is
meaningful -- this is FVD's idea applied to kinematics instead of pixels.

Ported from the source's ``RealMotionReference.w1`` / ``_frechet`` with three
behavioural changes, each a defect fix documented at its call site:

* ``torch.quantile`` -> ``np.quantile`` with seeded subsampling above a cap
  (``profile_w1`` docstring below) -- ``torch.quantile`` raises outright above
  ~1.6e7 elements, which a long clip or a large pooled reference set can
  exceed; ``np.quantile`` has no such ceiling, and the cap makes the remaining
  behaviour deliberate rather than accidental.
* ``_frechet`` is renamed :func:`kfd_approx` and its docstring says plainly
  that it is an eigendecomposition *approximation* of the Frechet distance,
  not the exact FID formula it was presented as -- see that function's
  docstring for why the two coincide only when the covariances commute.
* An exact path, :func:`kfd`, is added behind a lazy ``scipy`` import (the
  ``bench`` extra) for callers that need the true value rather than the cheap
  approximation.

KFD (either variant) is a Frechet distance over the raw ``[mean, std]``
feature vector of each motion quantity -- it is **not scale-invariant**: a
speed feature computed at half the frame rate is (up to noise) exactly
2x the "same" motion's feature at full rate, since a speed's ``dt_exponent``
is ``1`` (see ``core/metric.py``). Comparing a KFD computed against rollouts
at one ``dt`` with one at another is therefore not "noisier", it is
*meaningless* -- the two feature spaces are different linear rescalings of
each other. This is exactly why :class:`RealMotionReference` pins ``dt`` and
refuses (``RateMismatchError``) to score a mismatched rate (defect D2).
"""
from __future__ import annotations

import warnings
from typing import Dict, Sequence, Union

import numpy as np
import torch

__all__ = ["profile_w1", "kfd_approx", "kfd"]

#: Above this many elements, quantile estimation subsamples (seeded, so
#: repeated calls on the same input are reproducible) rather than handing the
#: full array to ``np.quantile``. ``np.quantile`` itself has no hard element
#: -count ceiling the way ``torch.quantile`` does, but sorting an unbounded
#: array is still O(n log n) wall-clock and memory the caller did not ask for;
#: this cap keeps that cost bounded and explicit.
DEFAULT_SUBSAMPLE_CAP = 2_000_000

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float]]


def _as_numpy(x: ArrayLike) -> np.ndarray:
    """Coerce a torch tensor / array-like into a flat float64 numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).ravel()


def _subsample(x: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if cap is None or x.size <= cap:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.size, size=cap, replace=False)
    return x[idx]


def _quantiles(x: ArrayLike, n_q: int, *, subsample_cap: int = DEFAULT_SUBSAMPLE_CAP,
               seed: int = 0) -> np.ndarray:
    """``n_q`` evenly-spaced quantiles of ``x``, seeded-subsampled above the cap."""
    arr = _subsample(_as_numpy(x), subsample_cap, seed)
    if arr.size == 0:
        raise ValueError("cannot take quantiles of an empty sample array")
    qs = np.linspace(0.0, 1.0, n_q)
    return np.quantile(arr, qs)


def profile_w1(a: ArrayLike, b: ArrayLike, n_q: int = 100, *,
               subsample_cap: int = DEFAULT_SUBSAMPLE_CAP, seed: int = 0) -> float:
    """1-Wasserstein distance between two 1-D sample sets, via an ``n_q``-point
    quantile-function approximation: ``mean(|quantile(a) - quantile(b)|)``.

    This is the source's ``RealMotionReference.w1`` inner computation, ported
    verbatim except ``torch.quantile`` -> ``np.quantile``. The source called
    ``torch.quantile(x.float().flatten().cpu(), qs)`` directly on whatever was
    passed; for an input above ~1.6e7 elements that raises
    ``RuntimeError: quantile() input tensor is too large`` with no recovery
    path, which is reachable in practice from a long clip's per-frame samples
    or a reference built by pooling many real rollouts. ``np.quantile`` has no
    such limit; the ``subsample_cap``/``seed`` pair reproduces the *spirit* of
    a size limit (bounded cost) without the source's hard failure, and is
    deterministic for a fixed seed.

    ``a`` and ``b`` are subsampled with seeds ``seed`` and ``seed + 1``
    respectively so that comparing a sample set against itself
    (``profile_w1(x, x)``) still draws two independent subsamples rather than
    trivially cancelling -- see ``tests/test_kfd.py`` for why that matters
    (with the full array, i.e. below the cap, subsampling is a no-op and the
    result is exactly ``0.0``).
    """
    qa = _quantiles(a, n_q, subsample_cap=subsample_cap, seed=seed)
    qb = _quantiles(b, n_q, subsample_cap=subsample_cap, seed=seed + 1)
    return float(np.mean(np.abs(qa - qb)))


def _feat_vector(samples: Dict[str, ArrayLike], keys: Sequence[str]) -> np.ndarray:
    """Per-rollout feature = ``[mean, std]`` of each quantity, SUITE key order.

    ``keys`` must come from ``MetricSuite.quantity_keys`` -- never from
    ``samples.keys()``. The source took ``qkeys = [k for k in
    samples_list[0]]``: whichever quantities the *first* real rollout happened
    to have (e.g. ``angspeed``/``qdot`` present only when that particular
    rollout had rotation or joint-angle data) silently became the reference's
    permanent key set, and a later rollout missing one of those keys raised a
    bare ``KeyError`` deep inside ``torch.stack`` (defect D4b). Pinning
    ``keys`` to the suite makes the key set a declared property of the suite,
    not an accident of iteration order; callers
    (``RealMotionReference.build``) additionally validate every rollout has
    every declared key *before* calling this, and raise a message listing
    exactly which rollouts are missing which keys, instead of a bare
    ``KeyError`` on whichever key happened to be checked first.
    """
    parts = []
    for k in keys:
        s = _as_numpy(samples[k])
        parts.append(float(s.mean()))
        parts.append(float(s.std()))
    return np.array(parts, dtype=np.float64)


def _as_np_mat(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def kfd_approx(mu1: ArrayLike, cov1: ArrayLike, mu2: ArrayLike, cov2: ArrayLike) -> float:
    """Frechet-style distance between two Gaussians -- an eig-based
    APPROXIMATION of the true Frechet distance, not the exact FID formula.

    The true (population) Frechet/FID distance between :math:`N(\\mu_1,
    \\Sigma_1)` and :math:`N(\\mu_2, \\Sigma_2)` is::

        d^2 = |mu1 - mu2|^2 + tr(Sigma1 + Sigma2 - 2*sqrtm(Sigma1 @ Sigma2))

    where ``sqrtm`` is the *matrix* square root of the (generally
    non-symmetric) product ``Sigma1 @ Sigma2``. The source computed that
    ``sqrtm`` as ``eigh(0.5*(prod + prod.T))``, i.e. it symmetrised the
    product first and then took the symmetric-eigendecomposition square root
    of *that*. Symmetrising and then ``eigh``-ing is only equal to
    ``sqrtm(Sigma1 @ Sigma2)`` when ``Sigma1`` and ``Sigma2`` commute
    (``Sigma1 @ Sigma2 == Sigma2 @ Sigma1``), which holds e.g. when either
    covariance is a multiple of the identity or the two share an eigenbasis,
    but not in general for two arbitrary real reference/candidate motion
    covariances. The source named this function ``_frechet`` and called it
    "the FID formula" in a comment; it is kept here, math untouched, under an
    honest name. Use :func:`kfd` for the exact value.

    Cheap (no scipy dependency): an eigh of a small (``2 * n_quantities``)
    symmetric matrix.
    """
    mu1 = _as_np_mat(mu1)
    mu2 = _as_np_mat(mu2)
    cov1 = _as_np_mat(cov1)
    cov2 = _as_np_mat(cov2)
    diff = mu1 - mu2
    prod = cov1 @ cov2
    sym = 0.5 * (prod + prod.T)
    vals, vecs = np.linalg.eigh(sym)
    sqrt_prod = (vecs * np.clip(vals, 0.0, None) ** 0.5) @ vecs.T
    return float(diff @ diff + np.trace(cov1 + cov2 - 2.0 * sqrt_prod))


def kfd(mu1: ArrayLike, cov1: ArrayLike, mu2: ArrayLike, cov2: ArrayLike) -> float:
    """Exact Kinematic Frechet Distance via ``scipy.linalg.sqrtm``.

    Same formula as :func:`kfd_approx`'s docstring, but ``sqrtm(Sigma1 @
    Sigma2)`` is computed exactly (Schur-based algorithm) instead of via the
    symmetrise-then-``eigh`` approximation, so this agrees with
    :func:`kfd_approx` whenever the two covariances commute and can differ
    when they do not (see ``tests/test_kfd.py``).

    Requires ``scipy`` (the ``bench`` extra: ``pip install kinescore[bench]``)
    -- imported lazily so the core package stays scipy-free, matching the
    project's "core stays small" policy (see ``pyproject.toml``).
    """
    try:
        from scipy.linalg import sqrtm
    except ImportError as exc:  # pragma: no cover - exercised only without scipy
        raise ImportError(
            "kfd() requires scipy; install with `pip install kinescore[bench]`, "
            "or use kfd_approx() for the dependency-free eigendecomposition "
            "approximation") from exc

    mu1 = _as_np_mat(mu1)
    mu2 = _as_np_mat(mu2)
    cov1 = _as_np_mat(cov1)
    cov2 = _as_np_mat(cov2)
    diff = mu1 - mu2
    prod = cov1 @ cov2
    sqrt_prod = sqrtm(prod)
    if np.iscomplexobj(sqrt_prod):
        # A numerically indefinite or non-commuting product can leave a tiny
        # imaginary residue even though the true Frechet distance is real;
        # warn only if that residue is not negligible relative to the real
        # part, then take the real part either way.
        imag = float(np.max(np.abs(sqrt_prod.imag)))
        real_scale = 1.0 + float(np.max(np.abs(sqrt_prod.real)))
        if imag > 1e-6 * real_scale:
            warnings.warn(
                f"kfd(): sqrtm(cov1 @ cov2) has a non-negligible imaginary "
                f"part (max |imag|={imag:.3g}); taking the real part",
                stacklevel=2)
        sqrt_prod = sqrt_prod.real
    return float(diff @ diff + np.trace(cov1 + cov2 - 2.0 * sqrt_prod))
