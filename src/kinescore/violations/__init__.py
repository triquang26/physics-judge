"""Physics-violation detectors and the scorer that calibrates them."""
from kinescore.violations import segments
from kinescore.violations.detectors import (
    Detector,
    JerkDetector,
    JointLimitDetector,
    RigidityDetector,
    SelfCollisionDetector,
    TeleportDetector,
)
from kinescore.violations.scorer import DETECTORS, HEADLINE, ViolationScorer

__all__ = [
    "Detector", "RigidityDetector", "JerkDetector", "TeleportDetector",
    "JointLimitDetector", "SelfCollisionDetector", "ViolationScorer", "DETECTORS", "HEADLINE", "segments",
]
