"""Robot registry: ``get_robot(name) -> RobotSpec``.

Importing this module never requires ``pytorch_kinematics`` or
``robot_descriptions``. :class:`~kinescore.robots.synthetic.Synthetic2R` is
imported eagerly -- it has no such dependency and is the CPU-only test path --
while the URDF-backed robots are resolved lazily inside :func:`get_robot`, so
asking for ``"synthetic_2r"`` never pulls in the kinematics stack.
"""
from __future__ import annotations

from kinescore.core.registry import Registry
from kinescore.core.robot import RobotSpec
from kinescore.robots.synthetic import Synthetic2R

__all__ = ["get_robot", "available_robots"]


def _build_franka(**kwargs) -> RobotSpec:
    from kinescore.robots.franka.spec import FrankaSpec
    return FrankaSpec(**kwargs)


def _build_gr1(**kwargs) -> RobotSpec:
    from kinescore.robots.gr1.spec import GR1Spec
    return GR1Spec(**kwargs)


def _build_airbot_mmk2(**kwargs) -> RobotSpec:
    from kinescore.robots.airbot_mmk2.spec import AirbotMMK2Spec
    return AirbotMMK2Spec(**kwargs)


def _build_aloha_bimanual(**kwargs) -> RobotSpec:
    from kinescore.robots.aloha.spec import AlohaSpec
    return AlohaSpec(**kwargs)


def _build_synthetic(**kwargs) -> RobotSpec:
    return Synthetic2R(**kwargs)


#: Registry key -> zero-import-cost factory. Each factory does its own heavy
#: imports lazily (see module docstring); constructing the registry below
#: does not import pytorch_kinematics/robot_descriptions itself.
_REGISTRY: Registry[RobotSpec] = Registry(kind="robot")
_REGISTRY.register("franka_panda", _build_franka)
_REGISTRY.register("fourier_gr1", _build_gr1)
_REGISTRY.register("airbot_mmk2", _build_airbot_mmk2)
_REGISTRY.register("aloha_bimanual", _build_aloha_bimanual)
_REGISTRY.register("synthetic_2r", _build_synthetic)


def get_robot(name: str, **kwargs) -> RobotSpec:
    """Construct the named :class:`~kinescore.core.robot.RobotSpec`.

    Parameters
    ----------
    name:
        One of :func:`available_robots`.
    **kwargs:
        Forwarded to the robot's constructor (e.g. ``device``, ``dtype`` for
        Franka/GR-1; ``link1_m``/``link2_m`` for the synthetic arm).

    Raises
    ------
    ValueError
        If ``name`` is not registered -- lists the valid names, so a typo'd
        robot name fails at the call site instead of surfacing as a confusing
        ``AttributeError`` deep in a metric.
    kinescore.paths.MissingPathError
        Propagated unmodified from ``GR1Spec`` when ``KINESCORE_ASSETS`` is
        unset or the GR-1 URDF is not checked out under it.
    """
    return _REGISTRY.get(name, **kwargs)


def available_robots() -> tuple[str, ...]:
    """Registered robot names, in a stable (sorted) order."""
    return _REGISTRY.available()
