"""Airbot MMK2: bimanual 6-DOF arms (no hand, no legs -- see spec.py).

Importing this subpackage is cheap -- ``AirbotMMK2FK`` only imports
``pytorch_kinematics`` inside its own ``__init__`` (see ``fk.py``), so
constructing :class:`AirbotMMK2Spec` (which also needs ``KINESCORE_ASSETS``
to be set and the composite URDF to exist) is what actually pulls in the
kinematics stack and touches the filesystem, not this import.
"""
from kinescore.robots.airbot_mmk2.fk import AirbotMMK2FK
from kinescore.robots.airbot_mmk2.spec import AirbotMMK2Spec

__all__ = ["AirbotMMK2FK", "AirbotMMK2Spec"]
