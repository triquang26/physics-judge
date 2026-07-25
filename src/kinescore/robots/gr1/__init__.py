"""Fourier GR-1: bimanual humanoid, two arms + feet.

Importing this subpackage is cheap -- ``GR1FK`` only imports
``pytorch_kinematics`` inside its own ``__init__`` (see ``fk.py``), so
constructing :class:`GR1Spec` (which also needs ``KINESCORE_ASSETS`` to be set
and the URDF to exist) is what actually pulls in the kinematics stack and
touches the filesystem, not this import.
"""
from kinescore.robots.gr1.colliders import RobotColliders
from kinescore.robots.gr1.fk import GR1FK
from kinescore.robots.gr1.spec import GR1Spec

__all__ = ["GR1FK", "RobotColliders", "GR1Spec"]
