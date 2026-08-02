"""The Physical Invariance Score (PIS): invariant residuals -> one comparable number.

D3 -- a fixed term set, not "whatever was present"
----------------------------------------------------
The source's ``invariance_score`` iterated over ``residuals.items()`` and
``continue``'d past any key missing from ``self.inv_baseline``::

    for k, v in residuals.items():
        base = self.inv_baseline.get(k)
        if base is None:
            continue
        ...
    pis = sum(scores.values()) / max(len(scores), 1)

``PhysicsConsistency.invariant_residuals()`` upstream emits 6 keys
unconditionally, +1 with rotations, +3 with joint angles -- so a clip scored
with joint angles produced a mean over 9-10 terms and one scored from
keypoints alone produced a mean over 6, and *both were reported as the same
number and directly compared*. ``INVARIANT_KEYS`` was declared in the source
but never actually used to pin anything.

Here the term set is :attr:`~kinescore.reference.fingerprint.RealMotionReference.term_keys`
-- fixed once, at reference-build time, from ``MetricSuite.invariant_keys``
(see ``fingerprint.py``) -- and every call to :func:`invariance_score`
evaluates exactly that set, present or not. A declared term that is missing
or NaN is not silently skipped: ``policy="strict"`` (the default) makes the
*whole* PIS ``NaN`` with a reason instead of quietly averaging over fewer
terms; ``policy="available"`` opts back into an average-over-what's-present,
but *records* ``n_terms`` against ``n_terms_declared`` so the two clips from
the motivating example are visibly different rather than identically
reported; ``policy="nan"`` lets a missing term's ``NaN`` propagate through an
ordinary arithmetic mean rather than being special-cased. ``n_terms ==
n_terms_declared`` for every clip under ``"strict"``/``"nan"`` -- see
``tests/test_pis_term_set.py``, which is exactly the invariant the source
broke.

D3b -- a floor under the denominator
--------------------------------------
The source's denominator was ``base * (tol - 1.0) + 1e-8``. When a key's
real-data baseline is ~0 (e.g. ``rigidity_wobble_mm`` on a clean reader), that
denominator collapses to ``1e-8``: the per-key score saturates to ``1.0`` on
any tiny excess and then dominates the unweighted mean regardless of how
physical everything else is. Here the denominator is ``max(base,
floors[k]) * (tol - 1.0)``, where ``floors[k]`` was fit at reference-build
time (``fingerprint.RealMotionReference.build``) from the spread of real
values for that key. **This changes nothing for a well-behaved key**: when
``base > floors[k]``, ``max(base, floors[k]) == base``, so the new
denominator equals the old one to within ``1e-8`` (the source's fixed
epsilon, now absent, is the only difference) -- see
``tests/test_pis_zero_baseline.py`` for the no-op guarantee this is tested
against, and its saturating-vs-floored contrast on a near-zero baseline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from kinescore.reference.fingerprint import RealMotionReference

__all__ = ["InvarianceResult", "invariance_score", "assert_comparable",
           "ComparabilityError", "DEFAULT_TOL", "DEFAULT_FRAC_TOL"]

DEFAULT_TOL = 3.0
DEFAULT_FRAC_TOL = 0.1


class ComparabilityError(ValueError):
    """Raised by :func:`assert_comparable` when two PIS results are not comparable."""


@dataclass(frozen=True)
class InvarianceResult:
    """One clip's Physical Invariance Score, plus enough provenance to trust it.

    Parameters
    ----------
    pis:
        The score in ``[0, 1]``, lower is more physical -- ``NaN`` when a
        declared term was missing under ``policy="strict"``/``"nan"``.
    per_key:
        Every declared term's individual score (``NaN`` where missing) --
        the static-schema guarantee from ``core/suite.py`` applied to PIS.
    n_terms:
        Terms actually averaged over. Equals ``n_terms_declared`` under
        ``"strict"``/``"nan"``; may be smaller under ``"available"``.
    n_terms_declared:
        ``len(reference.term_keys)`` -- the fixed size of the term set.
    term_set_id:
        ``reference.suite_id``. Two :class:`InvarianceResult` are comparable
        iff their ``term_set_id`` match -- see :func:`assert_comparable`.
    reason:
        Why ``pis`` is ``NaN``, or ``None`` when it isn't.
    rate_mismatch:
        ``True`` iff ``dt`` was supplied, disagreed with the reference's
        ``dt``, and ``allow_rate_mismatch=True`` downgraded that from a raise
        to a flag (defect D2).
    """

    pis: float
    per_key: dict[str, float]
    n_terms: int
    n_terms_declared: int
    term_set_id: str
    reason: str | None = None
    rate_mismatch: bool = False


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def _is_missing(v: float | None) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _key_score(k: str, v: float, base: float, floor: float, tol: float, frac_tol: float) -> float:
    if k.endswith("_frac"):
        # Fraction-valued keys (units="fraction") are already in [0,1] and
        # use an absolute tolerance -- untouched by the floor fix, which only
        # concerns the relative-tolerance path below.
        return _clip01((v - base) / frac_tol)
    denom = max(base, floor) * (tol - 1.0)
    if denom <= 0.0:
        # base and floor are both exactly 0 (only possible for a key with no
        # UNIT_FLOORS default whose real data is degenerate): fall back to a
        # tiny denominator so the result is still a saturating [0,1] score
        # rather than a ZeroDivisionError. This is deliberately far smaller
        # than the legacy 1e-8, and only ever reached when max(base,floor)==0,
        # so it never perturbs the well-conditioned no-op guarantee (that
        # guarantee only claims equality when base > floor > 0).
        denom = 1e-12
    return _clip01((v - base) / denom)


def invariance_score(residuals: dict[str, float], reference: RealMotionReference,
                      policy: str = "strict", *, tol: float = DEFAULT_TOL,
                      frac_tol: float = DEFAULT_FRAC_TOL, dt: float | None = None,
                      allow_rate_mismatch: bool = False,
                      legacy: bool = False) -> InvarianceResult:
    """Normalise invariant residuals against the real baseline -> PIS in ``[0,1]``.

    0 = as physical as real data; 1 = severe violation.

    Parameters
    ----------
    residuals:
        One clip's invariant residuals, as produced by the suite's metrics.
    reference:
        The fingerprint to normalise against; supplies ``term_keys``,
        ``inv_baseline`` and ``floors``.
    policy:
        How to handle a declared term that is missing from ``residuals`` or
        ``NaN`` in it (see module docstring): ``"strict"`` (default) makes
        the whole ``pis`` ``NaN`` with a reason; ``"available"`` averages over
        whichever declared terms are present, recording ``n_terms`` against
        ``n_terms_declared``; ``"nan"`` lets the missing term's ``NaN``
        propagate through an ordinary mean.
    tol, frac_tol:
        Continuous keys map ``[base, tol*base] -> [0, 1]``; ``*_frac`` keys
        use the absolute tolerance ``frac_tol`` instead (source defaults).
    dt:
        The clip's frame interval, if the caller wants the D2 rate check run
        here rather than (or in addition to) separately via
        ``reference.check_rate``. ``None`` (default) skips the check.
    allow_rate_mismatch:
        Only meaningful when ``dt`` is given: downgrade a rate mismatch from
        ``RateMismatchError`` to a warning, and record it as
        ``InvarianceResult.rate_mismatch``.
    legacy:
        Reproduce the source's ``invariance_score`` verbatim, defects D3/D3b
        included -- for golden-file regression tests ONLY, see
        ``tests/test_reference_golden.py``. Never use this for real scoring;
        it is the exact behaviour this module exists to replace.
    """
    if legacy:
        return _invariance_score_legacy(residuals, reference, tol, frac_tol)

    if policy not in ("strict", "available", "nan"):
        raise ValueError(f"unknown policy {policy!r}; expected 'strict', 'available' or 'nan'")

    rate_mismatch = False
    if dt is not None:
        rate_mismatch = reference.check_rate(dt, allow_rate_mismatch=allow_rate_mismatch)

    term_keys = reference.term_keys
    n_declared = len(term_keys)
    per_key: dict[str, float] = {}
    missing: list[str] = []
    for k in term_keys:
        v = residuals.get(k)
        if _is_missing(v):
            missing.append(k)
            per_key[k] = float("nan")
            continue
        base = reference.inv_baseline[k]
        floor = reference.floors.get(k, 0.0)
        per_key[k] = _key_score(k, float(v), base, floor, tol, frac_tol)

    if policy == "available":
        present_vals = [per_key[k] for k in term_keys if k not in missing]
        n_terms = len(present_vals)
        pis = float(np.mean(present_vals)) if present_vals else float("nan")
        reason = (f"missing_terms:{sorted(missing)}; averaged over "
                  f"{n_terms}/{n_declared} available terms") if missing else None
        return InvarianceResult(pis, per_key, n_terms, n_declared,
                                reference.suite_id, reason, rate_mismatch)

    if policy == "strict":
        if missing:
            pis = float("nan")
            reason = (f"missing_terms:{sorted(missing)} out of {n_declared} "
                      f"declared -- not silently averaged over the rest (D3)")
        else:
            pis = float(np.mean(list(per_key.values())))
            reason = None
        return InvarianceResult(pis, per_key, n_declared, n_declared,
                                reference.suite_id, reason, rate_mismatch)

    # policy == "nan": an ordinary arithmetic mean over the fixed term set,
    # letting a missing term's NaN propagate via IEEE 754 rather than being
    # special-cased into a diagnostic reason string.
    pis = float(np.mean(list(per_key.values())))
    return InvarianceResult(pis, per_key, n_declared, n_declared,
                            reference.suite_id, None, rate_mismatch)


def _invariance_score_legacy(residuals: dict[str, float], reference: RealMotionReference,
                             tol: float, frac_tol: float) -> InvarianceResult:
    """Reproduce the source's ``RealMotionReference.invariance_score`` verbatim.

    Term set = whatever keys ``residuals`` happens to have that are also in
    ``reference.inv_baseline`` (defect D3); denominator =
    ``base*(tol-1)+1e-8`` with no floor (defect D3b). For golden regression
    tests only -- see the module docstring's ``legacy`` parameter note.
    """
    scores: dict[str, float] = {}
    for k, v in residuals.items():
        base = reference.inv_baseline.get(k)
        if base is None:
            continue
        if k.endswith("_frac"):
            scores[k] = _clip01((v - base) / frac_tol)
        else:
            denom = base * (tol - 1.0) + 1e-8
            scores[k] = _clip01((v - base) / denom)
    n = len(scores)
    pis = float(sum(scores.values()) / max(n, 1))
    return InvarianceResult(pis, scores, n, n, reference.suite_id,
                            reason="legacy=True: term set = whatever residuals/baseline "
                                   "keys intersected (D3), not the suite's fixed term_keys")


def assert_comparable(a: InvarianceResult, b: InvarianceResult) -> None:
    """Raise unless two PIS results were produced against the same term set.

    ``term_set_id`` is ``reference.suite_id``: two results are comparable
    exactly when they came from references built against the same
    :class:`~kinescore.core.suite.MetricSuite` (same declared metrics, same
    declared invariant subset). This is the PIS-level enforcement of the
    guarantee ``MetricSuite.suite_id`` documents -- see ``core/suite.py``.
    """
    if a.term_set_id != b.term_set_id:
        raise ComparabilityError(
            f"PIS results are not comparable: term_set_id {a.term_set_id!r} != "
            f"{b.term_set_id!r}. They were produced against references built "
            f"from different MetricSuites (or one predates suite-pinning, e.g. "
            f"a legacy reference's 'legacy:v1'), so their term sets are not "
            f"known to be the same -- averaging or diffing them is exactly "
            f"defect D3.")
