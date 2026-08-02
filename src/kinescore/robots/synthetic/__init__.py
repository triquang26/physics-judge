"""Synthetic2R: closed-form 2-link planar arm, no URDF, no optional deps.

Importing this subpackage is free -- ``Synthetic2R`` (see ``spec.py``) has no
URDF, no ``pytorch_kinematics``, no network -- which is why
``robots/__init__.py`` imports it eagerly rather than through the lazy
``Registry``-backed factory pattern the other robots use. See ``spec.py``'s module
docstring for the full rationale and why this is a package (mirroring
``franka/``, ``gr1/``, ``airbot_mmk2/``) despite having no ``fk.py``.
"""
from kinescore.robots.synthetic.spec import Synthetic2R

__all__ = ["Synthetic2R"]
