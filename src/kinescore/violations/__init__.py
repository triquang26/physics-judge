"""Physics-violation detectors and the scorer that calibrates them.

Each detector turns one clip's predicted keypoints into a per-frame score;
thresholds are calibrated on real motion so a clip cannot set its own bar.
"""
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
