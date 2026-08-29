"""Fourier GR-1: bimanual humanoid, two arms + feet."""
from kinescore.robots.gr1.colliders import RobotColliders
from kinescore.robots.gr1.fk import GR1FK
from kinescore.robots.gr1.spec import GR1Spec

__all__ = ["GR1FK", "RobotColliders", "GR1Spec"]
