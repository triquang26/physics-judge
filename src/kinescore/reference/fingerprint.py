"""The frozen real-data fingerprint: :class:`RealMotionReference`.

Everything a rollout is scored *against* -- invariant-residual baselines,
per-quantity reference distributions, and the Gaussian used for KFD -- lives
here, pinned to the rate and the suite it was built at. This module fixes two
of the four defects this subpackage exists for; the other two (varying PIS
term count, zero-baseline blowup) are fixed in :mod:`kinescore.reference.normalize`
using the ``term_keys``/``floors`` this module computes and stores.

D2 -- the reference did not serialize ``dt``
---------------------------------------------
The source's ``save()``/``load()`` persisted ``inv_baseline``, ``quantiles``,
``feat_mu``, ``feat_cov``, ``quantity_keys`` and ``n_q`` -- never the frame
interval the reference was built at. Every one of those fields scales as some
power of ``1/dt`` (a jerk baseline as ``1/dt**3``, see ``core/metric.py``'s
``dt_exponent``), so loading a reference built at 5 Hz and scoring a 2.5 Hz
run against it silently rescales every metric by up to 8x -- undetectable
from the file, or the code, or the numbers alone. Here ``dt`` is a required,
``validate_dt``'d constructor argument, it is serialized (schema 2), and
:meth:`RealMotionReference.check_rate` refuses (:class:`RateMismatchError`)
to score a clip whose ``dt`` disagrees, unless the caller explicitly opts in
via ``allow_rate_mismatch=True`` (which downgrades to a warning).

D3 (term-set half) -- ``suite_id``/``term_keys`` come from the suite, not a rollout
------------------------------------------------------------------------------------
The source derived which residual keys mattered implicitly, from whatever was
present. Here ``term_keys`` is ``suite.invariant_keys`` -- a property of the
:class:`~kinescore.core.suite.MetricSuite` the reference was built against,
fixed before any real data is seen. ``normalize.invariance_score`` then
averages over exactly this fixed set for every clip, present or not (see that
module for how a missing/NaN term is handled instead of silently dropped).

D3b -- floors, so a near-zero baseline cannot blow up the PIS denominator
---------------------------------------------------------------------------
A key whose real-data baseline is ~0 (e.g. ``rigidity_wobble_mm`` on a clean
reader) makes the source's ``denom = base*(tol-1)+1e-8`` collapse to ``1e-8``,
which saturates the per-key score to ``1.0`` on any tiny excess and then
dominates the unweighted PIS mean regardless of how physical the rest of the
rollout is. ``floors[k]`` is fit here, at build time, as ``max(median absolute
deviation of k's real values, a unit-aware default)`` -- e.g. half a
millimetre for the two ``*_mm`` rigidity residuals -- and ``normalize.py``
uses ``max(base, floors[k])`` in place of the bare ``base``.

D4b -- the reference's quantity-key set does not depend on iteration order
------------------------------------------------------------------------------
The source took ``qkeys = [k for k in samples_list[0]]``: whichever motion
quantities the *first* rollout happened to carry became the reference's
permanent quantity set, and a later rollout missing one raised a bare
``KeyError`` from deep inside ``torch.stack``. Here the quantity set is
``suite.quantity_keys``, and :meth:`RealMotionReference.build` validates every
rollout against it up front, raising a single message listing exactly which
rollouts are missing which keys -- see ``distances._feat_vector``.
"""
from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from kinescore.core.clip import validate_dt
from kinescore.core.suite import MetricSuite
from kinescore.reference.distances import ArrayLike, _as_numpy, _feat_vector, _quantiles

__all__ = ["RealMotionReference", "RateMismatchError", "UNIT_FLOORS", "SCHEMA_VERSION"]

#: On-disk schema version. Bump when the persisted field set changes; ``load``
#: dispatches on this (absent entirely => the legacy, pre-D2 format).
SCHEMA_VERSION = 2

#: Unit-aware default floors for the PIS denominator (defect D3b), keyed by
#: metric key. These are *defaults* only: :meth:`RealMotionReference.build`
#: additionally fits a per-key median-absolute-deviation floor from the real
#: data itself and stores ``max(mad, UNIT_FLOORS.get(key, 0.0))``, so a key
#: not listed here still gets a sane data-driven floor rather than none.
UNIT_FLOORS: Dict[str, float] = {
    "rigidity_wobble_mm": 0.5,
    "rigidity_residual_mm": 0.5,
    "mean_jerk_mps3": 1e-3,
    "total_energy_tstd": 1e-3,
    "momentum_dp_mean": 1e-3,
}


class RateMismatchError(ValueError):
    """A clip's ``dt`` does not match the ``dt`` a reference was built at.

    Every baseline and quantile in a :class:`RealMotionReference` scales as
    some power of ``1/dt``. Scoring against a mismatched reference does not
    fail on its own -- it silently rescales every metric, which is exactly
    how ``core/clip.py``'s defect D1 went undetected upstream. This is the
    reference-side half of that fix: D1 pins ``dt`` on the clip, this pins it
    on the reference and refuses to compare the two when they disagree.
    """


def _mad(values: Sequence[float]) -> float:
    """Median absolute deviation: a robust spread estimate (no Gaussian assumption)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


class RealMotionReference:
    """Frozen fingerprint of real robot motion, pinned to a rate and a suite.

    Construct via :meth:`build` from real rollouts, not directly, except when
    reconstructing from :meth:`load`. All fields are copied on construction so
    the reference does not alias caller-owned arrays.

    Parameters
    ----------
    dt:
        Frame interval (seconds) the reference was built at. Required,
        ``validate_dt``'d (defect D2).
    suite_id:
        ``MetricSuite.suite_id`` of the suite this reference was built
        against. ``"legacy:v1"`` for references recovered from a schema-less
        (pre-D2) file, which pinned no suite at all.
    term_keys:
        The fixed PIS term set (``suite.invariant_keys`` at build time).
    inv_baseline:
        Per-key median real-data residual, one entry per ``term_keys``.
    floors:
        Per-key PIS-denominator floor (defect D3b), one entry per
        ``term_keys``.
    quantiles:
        Per-quantity ``n_q``-point reference quantile function, pooled over
        real rollouts. One entry per ``quantity_keys``.
    feat_mu, feat_cov:
        Gaussian (mean, covariance) over per-rollout ``[mean, std]`` feature
        vectors, for KFD (see ``distances.py``).
    quantity_keys:
        The fixed distributional-comparison key set (``suite.quantity_keys``
        at build time; defect D4b).
    n_q:
        Number of quantile points per quantity.
    schema:
        On-disk schema version this instance corresponds to.
    """

    def __init__(self, *, dt: float, suite_id: str, term_keys: Sequence[str],
                 inv_baseline: Dict[str, float], floors: Dict[str, float],
                 quantiles: Dict[str, ArrayLike], feat_mu: ArrayLike,
                 feat_cov: ArrayLike, quantity_keys: Sequence[str],
                 n_q: int = 100, schema: int = SCHEMA_VERSION) -> None:
        self.dt = validate_dt(dt)
        self.suite_id = str(suite_id)
        self.term_keys: Tuple[str, ...] = tuple(term_keys)
        self.inv_baseline: Dict[str, float] = {k: float(v) for k, v in inv_baseline.items()}
        self.floors: Dict[str, float] = {k: float(v) for k, v in floors.items()}
        self.quantiles: Dict[str, np.ndarray] = {
            k: np.array(_as_numpy(v), copy=True) for k, v in quantiles.items()}
        self.feat_mu: np.ndarray = np.array(_as_numpy(feat_mu), copy=True)
        feat_cov_arr = np.array(_as_numpy(feat_cov), copy=True)
        n = self.feat_mu.shape[0]
        self.feat_cov: np.ndarray = feat_cov_arr.reshape(n, n) if n else feat_cov_arr.reshape(0, 0)
        self.quantity_keys: Tuple[str, ...] = tuple(quantity_keys)
        self.n_q = int(n_q)
        self.schema = int(schema)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, suite: MetricSuite, residual_list: List[Dict[str, float]],
              samples_list: List[Dict[str, ArrayLike]], dt: float,
              n_q: int = 100) -> "RealMotionReference":
        """Fit a reference from real rollouts' residuals and motion samples.

        ``suite`` -- never ``residual_list[0]``/``samples_list[0]`` -- is the
        sole source of which keys end up in the reference (defects D3, D4b).
        Every rollout must supply every key ``suite`` declares; a rollout
        missing one raises here, at build time, with a message listing every
        offending rollout and key, rather than an accidental subset silently
        becoming "the reference" or a bare ``KeyError`` surfacing later at
        score time.

        Parameters
        ----------
        suite:
            Declares ``term_keys`` (``suite.invariant_keys``) and
            ``quantity_keys`` (``suite.quantity_keys``).
        residual_list:
            One ``{key: value}`` dict of invariant residuals per real rollout.
        samples_list:
            One ``{key: array}`` dict of per-frame motion-quantity samples per
            real rollout, same order as ``residual_list``.
        dt:
            Frame interval the real rollouts were scored at.
        n_q:
            Quantile points per quantity.
        """
        dt = validate_dt(dt)
        if not residual_list:
            raise ValueError("cannot build a reference from an empty residual_list")
        if not samples_list:
            raise ValueError("cannot build a reference from an empty samples_list")

        term_keys = suite.invariant_keys
        quantity_keys = suite.quantity_keys

        missing_qty = {i: sorted(k for k in quantity_keys if k not in s)
                       for i, s in enumerate(samples_list)}
        missing_qty = {i: ks for i, ks in missing_qty.items() if ks}
        if missing_qty:
            detail = "; ".join(f"rollout[{i}] missing {ks}"
                                for i, ks in sorted(missing_qty.items()))
            raise ValueError(
                f"cannot build reference: {len(missing_qty)}/{len(samples_list)} "
                f"real rollouts are missing quantity keys declared by suite "
                f"{suite.name!r} ({list(quantity_keys)}) -- {detail}")

        inv_baseline: Dict[str, float] = {}
        floors: Dict[str, float] = {}
        missing_terms = []
        for k in term_keys:
            vals = [float(r[k]) for r in residual_list
                    if k in r and math.isfinite(r[k])]
            if not vals:
                missing_terms.append(k)
                continue
            inv_baseline[k] = float(np.median(vals))
            floors[k] = max(_mad(vals), UNIT_FLOORS.get(k, 0.0))
        if missing_terms:
            raise ValueError(
                f"cannot build reference: invariant key(s) {missing_terms} "
                f"(declared by suite {suite.name!r}) are present in ZERO of "
                f"{len(residual_list)} real rollouts' residuals -- a PIS term "
                f"with no real-data baseline cannot be normalised")

        quantiles = {
            k: _quantiles(np.concatenate([_as_numpy(s[k]) for s in samples_list]), n_q)
            for k in quantity_keys
        }

        if quantity_keys:
            feats = np.stack([_feat_vector(s, quantity_keys) for s in samples_list])
            feat_mu = feats.mean(axis=0)
            feat_cov = np.cov(feats.T) if feats.shape[0] > 1 else np.eye(feats.shape[1])
        else:
            feat_mu = np.zeros(0)
            feat_cov = np.zeros((0, 0))

        return cls(dt=dt, suite_id=suite.suite_id, term_keys=term_keys,
                   inv_baseline=inv_baseline, floors=floors, quantiles=quantiles,
                   feat_mu=feat_mu, feat_cov=feat_cov, quantity_keys=quantity_keys,
                   n_q=n_q, schema=SCHEMA_VERSION)

    # ------------------------------------------------------------------
    # D2: the rate check
    # ------------------------------------------------------------------

    def check_rate(self, dt: float, *, allow_rate_mismatch: bool = False) -> bool:
        """Verify ``dt`` (the rate a clip/rollout was scored at) matches this
        reference's rate. Returns ``True`` iff there was a mismatch (only
        reachable when ``allow_rate_mismatch=True``, since otherwise this
        raises).

        Raises
        ------
        RateMismatchError
            If ``dt`` disagrees by more than ``1e-6`` relative and
            ``allow_rate_mismatch`` is ``False`` (the default).
        """
        dt = validate_dt(dt)
        mismatch = abs(dt - self.dt) > 1e-6 * self.dt
        if mismatch:
            ratio = self.dt / dt
            msg = (f"reference was built at dt={self.dt}s but is being scored "
                   f"against dt={dt}s ({ratio:.4g}x); baselines and quantiles "
                   f"scale as 1/dt**n, so this silently rescales every metric "
                   f"unless allow_rate_mismatch=True is passed knowingly")
            if not allow_rate_mismatch:
                raise RateMismatchError(msg)
            warnings.warn(msg, stacklevel=2)
        return mismatch

    # ------------------------------------------------------------------
    # Group 3: distribution realism (see distances.py for the math)
    # ------------------------------------------------------------------

    def w1(self, samples: Dict[str, ArrayLike], dt: Optional[float] = None,
           allow_rate_mismatch: bool = False) -> Dict[str, float]:
        """Per-quantity 1-Wasserstein distance of ``samples`` to the reference."""
        if dt is not None:
            self.check_rate(dt, allow_rate_mismatch=allow_rate_mismatch)
        out: Dict[str, float] = {}
        for k, ref_q in self.quantiles.items():
            if k not in samples:
                continue
            q = _quantiles(samples[k], self.n_q)
            out[f"w1_{k}"] = float(np.mean(np.abs(q - ref_q)))
        return out

    def _feat_gaussian(self, samples_list: List[Dict[str, ArrayLike]]):
        feats = np.stack([_feat_vector(s, self.quantity_keys) for s in samples_list])
        mu = feats.mean(axis=0)
        cov = np.cov(feats.T) if feats.shape[0] > 1 else np.zeros_like(self.feat_cov)
        return mu, cov

    def kfd_approx(self, samples_list: List[Dict[str, ArrayLike]],
                   dt: Optional[float] = None,
                   allow_rate_mismatch: bool = False) -> float:
        """Kinematic Frechet Distance (eig-approximation) of a SET of rollouts vs real.

        See :func:`kinescore.reference.distances.kfd_approx` for what
        "approximation" means here.
        """
        if dt is not None:
            self.check_rate(dt, allow_rate_mismatch=allow_rate_mismatch)
        from kinescore.reference.distances import kfd_approx as _kfd_approx
        mu, cov = self._feat_gaussian(samples_list)
        return _kfd_approx(mu, cov, self.feat_mu, self.feat_cov)

    def kfd(self, samples_list: List[Dict[str, ArrayLike]],
            dt: Optional[float] = None, allow_rate_mismatch: bool = False) -> float:
        """Exact Kinematic Frechet Distance (``scipy.linalg.sqrtm``; needs ``kinescore[bench]``)."""
        if dt is not None:
            self.check_rate(dt, allow_rate_mismatch=allow_rate_mismatch)
        from kinescore.reference.distances import kfd as _kfd
        mu, cov = self._feat_gaussian(samples_list)
        return _kfd(mu, cov, self.feat_mu, self.feat_cov)

    # ------------------------------------------------------------------
    # persistence (D2)
    # ------------------------------------------------------------------
    #
    # torch.save/torch.load (rather than a numpy-only format) is used
    # deliberately here, and only here: it is what the source's save()/load()
    # used, so it is the only way to read an actual legacy artifact for the
    # backward-compatibility path below. The numerics everywhere else in this
    # subpackage are numpy-only.

    def save(self, path: str) -> None:
        """Serialize (schema 2): includes ``dt``, ``suite_id``, ``term_keys``, ``floors``."""
        payload = {
            "schema": SCHEMA_VERSION,
            "dt": self.dt,
            "suite_id": self.suite_id,
            "term_keys": list(self.term_keys),
            "inv_baseline": dict(self.inv_baseline),
            "floors": dict(self.floors),
            "quantiles": {k: np.asarray(v) for k, v in self.quantiles.items()},
            "feat_mu": self.feat_mu,
            "feat_cov": self.feat_cov,
            "quantity_keys": list(self.quantity_keys),
            "n_q": self.n_q,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, dt: Optional[float] = None) -> "RealMotionReference":
        """Deserialize.

        A schema-2 file records its own ``dt`` and is loaded as-is (passing
        ``dt=`` for a schema-2 file is only checked, not needed -- see below).

        A legacy (schema-less, pre-D2) file has no ``dt`` field at all, and
        this **raises** unless ``dt=`` is supplied explicitly:
        the file predates the fix, so nothing in it can tell you the rate it
        was built at, and guessing silently reintroduces defect D2. Passing
        ``dt=`` is an explicit acknowledgement of what rate the caller
        believes the file was built at (e.g. ``load(path, dt=0.2)``).
        """
        d = torch.load(path, map_location="cpu", weights_only=False)
        schema = d.get("schema") if isinstance(d, dict) else None
        if schema is None:
            return cls._load_legacy(d, dt)
        if schema != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported reference schema {schema!r}; this build of "
                f"kinescore only reads schema {SCHEMA_VERSION}")
        if dt is not None:
            dt = validate_dt(dt)
            if abs(dt - d["dt"]) > 1e-6 * d["dt"]:
                raise ValueError(
                    f"load(path, dt={dt}) disagrees with the rate recorded in "
                    f"the file (dt={d['dt']}); omit dt= to use the recorded "
                    f"rate, it does not need to be re-supplied for schema 2")
        return cls(dt=d["dt"], suite_id=d["suite_id"], term_keys=d["term_keys"],
                   inv_baseline=d["inv_baseline"], floors=d["floors"],
                   quantiles=d["quantiles"], feat_mu=d["feat_mu"], feat_cov=d["feat_cov"],
                   quantity_keys=d["quantity_keys"], n_q=d["n_q"], schema=schema)

    @classmethod
    def _load_legacy(cls, d: dict, dt: Optional[float]) -> "RealMotionReference":
        required = {"inv_baseline", "quantiles", "feat_mu", "feat_cov", "quantity_keys", "n_q"}
        missing = required - set(d)
        if missing:
            raise ValueError(f"not a recognisable reference file: missing {sorted(missing)}")
        if dt is None:
            raise ValueError(
                "legacy reference (v1) records no dt; baselines scale as "
                "1/dt^n -- re-pass the rate it was built at: load(path, dt=0.2)")
        dt = validate_dt(dt)
        # The legacy format also never pinned a fixed term/quantity set (D3,
        # D4b): whatever keys happen to be in the file are whatever happened
        # to survive the source's iteration-order-dependent construction. We
        # surface that honestly (suite_id="legacy:v1") rather than pretend
        # this reference is comparable to a suite-built one.
        term_keys = tuple(sorted(d["inv_baseline"]))
        floors = {k: UNIT_FLOORS.get(k, 0.0) for k in term_keys}
        warnings.warn(
            "loaded a legacy (schema-1) reference: suite_id='legacy:v1', and "
            "floors fall back to unit defaults only (no per-rollout residuals "
            "were persisted in this format to fit a median-absolute-deviation "
            "floor from); its term_keys is exactly whatever keys happened to "
            "be in the file -- the iteration-order defect (D3/D4b) this "
            "schema exists to fix. Rebuild the reference from real rollouts "
            "to get a suite-pinned, floor-fitted one.",
            stacklevel=3)
        return cls(dt=dt, suite_id="legacy:v1", term_keys=term_keys,
                   inv_baseline=d["inv_baseline"], floors=floors,
                   quantiles=d["quantiles"], feat_mu=d["feat_mu"], feat_cov=d["feat_cov"],
                   quantity_keys=tuple(d["quantity_keys"]), n_q=int(d["n_q"]), schema=1)

    def __repr__(self) -> str:
        return (f"RealMotionReference(schema={self.schema}, dt={self.dt}, "
                f"suite_id={self.suite_id!r}, n_terms={len(self.term_keys)}, "
                f"n_quantities={len(self.quantity_keys)})")
