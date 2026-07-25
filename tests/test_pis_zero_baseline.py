"""D3b: a near-zero baseline must not blow up the PIS denominator.

The source's denominator was ``base * (tol - 1.0) + 1e-8``. For a key whose
real-data baseline is ~0, that denominator collapses to ``~1e-8`` and the
per-key score saturates to 1.0 on the tiniest excess, regardless of how
physical the rest of the rollout is.

Covers:
* ``base=1e-9, v=1e-6``: the legacy denominator saturates to 1.0; the floored
  one gives a small value.
* the well-conditioned case (``base`` comfortably above its floor) is
  unchanged by the fix to within the legacy formula's own ``1e-8`` epsilon --
  the numerical no-op guarantee that the fix touches ONLY pathological keys.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from kinescore.reference.fingerprint import RealMotionReference
from kinescore.reference.normalize import (DEFAULT_TOL, _key_score,
                                            invariance_score)


def _reference(inv_baseline, floors, term_keys=None):
    term_keys = term_keys or tuple(inv_baseline)
    return RealMotionReference(
        dt=0.2, suite_id="test:zero_baseline", term_keys=term_keys,
        inv_baseline=inv_baseline, floors=floors, quantiles={},
        feat_mu=np.zeros(0), feat_cov=np.zeros((0, 0)), quantity_keys=())


def test_legacy_denominator_saturates_near_zero_baseline():
    base, floor, v = 1e-9, 0.5, 1e-6
    ref = _reference({"z": base}, {"z": floor})

    legacy = invariance_score({"z": v}, ref, legacy=True)
    fixed = invariance_score({"z": v}, ref, policy="strict")

    # Legacy: denom = 1e-9*2 + 1e-8 ~= 1.2e-8 -> (v-base)/denom saturates to 1.
    assert legacy.pis == pytest.approx(1.0)
    # Floored: denom = max(1e-9, 0.5)*2 = 1.0 -> a small, informative value.
    assert fixed.pis < 1e-3
    assert fixed.pis > 0.0
    assert fixed.pis < legacy.pis


def test_floor_key_score_matches_hand_computation():
    base, floor, v, tol = 1e-9, 0.5, 1e-6, DEFAULT_TOL
    expected_denom = max(base, floor) * (tol - 1.0)
    expected_score = min(max((v - base) / expected_denom, 0.0), 1.0)
    assert _key_score("z", v, base, floor, tol, frac_tol=0.1) == pytest.approx(expected_score)


def test_well_conditioned_case_is_a_numerical_no_op():
    """When base > floor, the fix must touch NOTHING but the pathological keys.

    The legacy denominator is ``base*(tol-1)+1e-8``; the fixed one is
    ``max(base,floor)*(tol-1)``, which equals ``base*(tol-1)`` exactly when
    ``base > floor``. The two denominators therefore differ by EXACTLY the
    legacy epsilon (1e-8), which is the guaranteed-tiny gap the task asks to
    test for ("equal to within 1e-8") -- not literal bit-identity, since the
    fix's entire point is that the arbitrary epsilon is gone.
    """
    base, floor, tol = 1.0, 0.1, DEFAULT_TOL  # base well above its floor
    legacy_denom = base * (tol - 1.0) + 1e-8
    fixed_denom = max(base, floor) * (tol - 1.0)
    assert abs(fixed_denom - legacy_denom) <= 1e-8

    ref = _reference({"z": base}, {"z": floor})
    for v in (1.05, 1.5, 2.0, 2.9):
        legacy = invariance_score({"z": v}, ref, legacy=True)
        fixed = invariance_score({"z": v}, ref, policy="strict")
        # The resulting PIS differs only by the propagated 1e-8-scale epsilon
        # removal -- effectively a no-op for any well-conditioned key.
        assert math.isclose(legacy.pis, fixed.pis, rel_tol=0.0, abs_tol=1e-6)


def test_frac_keys_are_untouched_by_the_floor(monkeypatch=None):
    """`*_frac` keys keep the absolute frac_tol path -- floors never apply to them."""
    base, floor, v, frac_tol = 0.0, 0.0, 0.05, 0.1
    ref = _reference({"joint_limit_frac": base}, {"joint_limit_frac": floor})
    result = invariance_score({"joint_limit_frac": v}, ref, policy="strict", frac_tol=frac_tol)
    assert result.pis == pytest.approx(min(max((v - base) / frac_tol, 0.0), 1.0))
    assert result.pis == pytest.approx(0.5)
