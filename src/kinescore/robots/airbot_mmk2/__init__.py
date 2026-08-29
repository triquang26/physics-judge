"""Airbot MMK2: bimanual 6-DOF arms (no hand, no legs -- see spec.py)."""
from kinescore.robots.airbot_mmk2.fk import AirbotMMK2FK
from kinescore.robots.airbot_mmk2.spec import AirbotMMK2Spec

__all__ = ["AirbotMMK2FK", "AirbotMMK2Spec"]
