"""ALOHA bimanual: two 6-DOF arms + grippers, table-mounted (see spec.py).

Importing this subpackage is cheap -- ``AlohaFK`` only imports
``pytorch_kinematics`` inside its own ``__init__`` (see ``fk.py``), so
constructing :class:`AlohaSpec` (which also needs ``KINESCORE_ASSETS`` to be
set and the composite URDF to exist) is what actually pulls in the
kinematics stack and touches the filesystem, not this import.
"""
from kinescore.robots.aloha.fk import AlohaFK
from kinescore.robots.aloha.spec import AlohaSpec

__all__ = ["AlohaFK", "AlohaSpec"]
