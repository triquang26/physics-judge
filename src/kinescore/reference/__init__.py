"""Real-data fingerprint and normalised aggregate scores (PIS, W1, KFD).

See ``fingerprint.py`` and ``normalize.py`` for the defects this subpackage
fixes relative to the source (``motion_reference.py``): a reference that
never recorded its frame rate (D2), a Physical Invariance Score averaged over
a varying number of terms (D3) with a denominator that could blow up near a
zero baseline (D3b), and a distributional-comparison key set that depended on
iteration order (D4b).
"""
from __future__ import annotations

from kinescore.reference.distances import kfd, kfd_approx, profile_w1
from kinescore.reference.fingerprint import (RateMismatchError,
                                              RealMotionReference,
                                              SCHEMA_VERSION, UNIT_FLOORS)
from kinescore.reference.normalize import (ComparabilityError,
                                            DEFAULT_FRAC_TOL, DEFAULT_TOL,
                                            InvarianceResult,
                                            assert_comparable,
                                            invariance_score)

__all__ = [
    "RealMotionReference", "RateMismatchError", "UNIT_FLOORS", "SCHEMA_VERSION",
    "InvarianceResult", "invariance_score", "assert_comparable", "ComparabilityError",
    "DEFAULT_TOL", "DEFAULT_FRAC_TOL",
    "profile_w1", "kfd_approx", "kfd",
]
