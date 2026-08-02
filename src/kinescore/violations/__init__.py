"""Per-frame violation scoring: one error type = one GT-calibrated ``Detector``.

Promotes a validated prototype into a package matching
:mod:`kinescore.core.metric`'s style. See
:mod:`kinescore.violations.detectors` for the ``Detector`` contract (and why
each detector takes a :class:`~kinescore.core.metric.MetricContext` rather
than a bespoke clip type), and :mod:`kinescore.violations.scorer` for
:class:`ViolationScorer`, which calibrates every detector on GT clips and
scores a new clip into one report per error type.
"""
from __future__ import annotations

from kinescore.violations.detectors import (
    Detector,
    JerkDetector,
    JointLimitDetector,
    RigidityDetector,
    SelfCollisionDetector,
    TeleportDetector,
)
from kinescore.violations.scorer import DETECTORS, ViolationScorer

__all__ = [
    "Detector",
    "RigidityDetector",
    "JerkDetector",
    "TeleportDetector",
    "JointLimitDetector",
    "SelfCollisionDetector",
    "ViolationScorer",
    "DETECTORS",
]
