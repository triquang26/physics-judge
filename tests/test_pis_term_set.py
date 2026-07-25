"""D3: PIS averages over a FIXED term set, not whatever residual keys showed up.

Covers:
* a residual dict WITH a term and one WITHOUT it produce the SAME ``n_terms``
  under the fix (the missing term makes the whole PIS ``NaN`` under
  ``policy="strict"`` instead of silently shrinking the averaged set).
* the ``legacy=True`` replica reproduces the source's asymmetry: the two
  residual dicts DO get different ``n_terms`` there, which is exactly the bug.
* ``assert_comparable`` raises across two references with differing term sets.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from kinescore.core.metric import MetricSpec
from kinescore.core.suite import MetricSuite
from kinescore.reference.fingerprint import RealMotionReference
from kinescore.reference.normalize import (assert_comparable,
                                            ComparabilityError,
                                            invariance_score)


class _M:
    def __init__(self, key, units="mm"):
        self.spec = MetricSpec(key=key, units=units, dt_exponent=0,
                               direction="lower_better", requires=frozenset())

    def compute(self, ctx):  # pragma: no cover
        raise NotImplementedError


def _build_reference(keys=("k1", "k2", "k3"), name="term_set_suite"):
    suite = MetricSuite(name, [_M(k) for k in keys], invariant_keys=list(keys))
    rng = np.random.default_rng(1)
    residual_list = [{k: float(1.0 + 0.1 * rng.normal()) for k in keys} for _ in range(10)]
    samples_list = [{} for _ in range(10)]
    return RealMotionReference.build(suite, residual_list, samples_list, dt=0.2)


@pytest.fixture
def reference():
    return _build_reference()


def test_fixed_term_set_gives_same_n_terms_with_and_without_a_key(reference):
    residuals_with_all = {"k1": 1.0, "k2": 1.0, "k3": 1.0}
    residuals_missing_one = {"k1": 1.0, "k2": 1.0}  # k3 absent, e.g. no joint angles

    r_with = invariance_score(residuals_with_all, reference, policy="strict")
    r_without = invariance_score(residuals_missing_one, reference, policy="strict")

    # The headline fix: both report the SAME n_terms (the suite's fixed
    # declared count), regardless of what happened to be present.
    assert r_with.n_terms == r_without.n_terms == 3 == r_with.n_terms_declared

    # But they are not silently equated as numbers: the incomplete one is NaN
    # with a reason, not a mean over fewer terms.
    assert math.isfinite(r_with.pis)
    assert math.isnan(r_without.pis)
    assert r_without.reason is not None and "missing_terms" in r_without.reason


def test_policy_available_records_the_present_vs_declared_split(reference):
    residuals_missing_one = {"k1": 1.0, "k2": 1.0}
    r = invariance_score(residuals_missing_one, reference, policy="available")
    assert r.n_terms == 2
    assert r.n_terms_declared == 3
    assert math.isfinite(r.pis)
    assert r.reason is not None and "2/3" in r.reason


def test_legacy_replica_reproduces_the_varying_term_count(reference):
    residuals_with_all = {"k1": 1.0, "k2": 1.0, "k3": 1.0}
    residuals_missing_one = {"k1": 1.0, "k2": 1.0}

    r_with = invariance_score(residuals_with_all, reference, legacy=True)
    r_without = invariance_score(residuals_missing_one, reference, legacy=True)

    # This is the documented defect: under legacy=True, n_terms DOES vary
    # (3 vs 2) and both numbers were nonetheless reported as "the PIS".
    assert r_with.n_terms == 3
    assert r_without.n_terms == 2
    assert r_with.n_terms != r_without.n_terms
    assert math.isfinite(r_with.pis)
    assert math.isfinite(r_without.pis)


def test_missing_key_entirely_absent_from_baseline_is_ignored_by_legacy():
    """Sanity check that the legacy replica really is the source's continue-past-unknown-key behaviour."""
    reference = _build_reference(keys=("k1", "k2"))
    residuals = {"k1": 1.0, "k2": 1.0, "unknown_extra_key": 999.0}
    r = invariance_score(residuals, reference, legacy=True)
    assert r.n_terms == 2  # unknown_extra_key silently skipped, as in the source


def test_assert_comparable_raises_across_differing_term_sets():
    ref_a = _build_reference(keys=("k1", "k2", "k3"), name="suite_a")
    ref_b = _build_reference(keys=("k1", "k2"), name="suite_b")

    r_a = invariance_score({"k1": 1.0, "k2": 1.0, "k3": 1.0}, ref_a, policy="strict")
    r_b = invariance_score({"k1": 1.0, "k2": 1.0}, ref_b, policy="strict")

    assert r_a.term_set_id != r_b.term_set_id
    with pytest.raises(ComparabilityError):
        assert_comparable(r_a, r_b)


def test_assert_comparable_accepts_same_term_set():
    reference = _build_reference()
    r1 = invariance_score({"k1": 1.0, "k2": 1.0, "k3": 1.0}, reference, policy="strict")
    r2 = invariance_score({"k1": 1.1, "k2": 0.9, "k3": 1.0}, reference, policy="strict")
    assert_comparable(r1, r2)  # must not raise
