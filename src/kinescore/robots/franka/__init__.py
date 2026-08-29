"""Franka Panda: 7-DOF arm + 2-finger gripper, bolted to a table.

Importing this subpackage is cheap -- ``FrankaFK`` only imports
``pytorch_kinematics`` / ``robot_descriptions`` inside its own ``__init__``
(see ``fk.py``), so constructing :class:`FrankaSpec` is what actually pulls
those in, not this import.
"""
from kinescore.robots.franka.fk import FrankaFK
from kinescore.robots.franka.spec import FrankaSpec

__all__ = ["FrankaFK", "FrankaSpec"]
