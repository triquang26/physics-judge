"""Robot registry: ``get_robot(name) -> RobotSpec``."""
from __future__ import annotations

from kinescore.core.registry import Registry
from kinescore.core.robot import RobotSpec
from kinescore.robots.synthetic import Synthetic2R

__all__ = ["get_robot", "available_robots"]


def _build_gr1(**kwargs) -> RobotSpec:
    from kinescore.robots.gr1.spec import GR1Spec
    return GR1Spec(**kwargs)


def _build_airbot_mmk2(**kwargs) -> RobotSpec:
    from kinescore.robots.airbot_mmk2.spec import AirbotMMK2Spec
    return AirbotMMK2Spec(**kwargs)


def _build_synthetic(**kwargs) -> RobotSpec:
    return Synthetic2R(**kwargs)


def _build_a1x_ee(**kwargs) -> RobotSpec:
    from kinescore.robots.a1x_ee.spec import A1XEESpec
    return A1XEESpec(**kwargs)


def _build_aloha_bimanual(**kwargs) -> RobotSpec:
    from kinescore.robots.aloha.spec import AlohaSpec
    return AlohaSpec(**kwargs)


#: Registry key -> zero-import-cost factory. Each factory does its own heavy
#: imports lazily (see module docstring); constructing the registry below
#: does not import pytorch_kinematics/robot_descriptions itself.
_REGISTRY: Registry[RobotSpec] = Registry(kind="robot")
_REGISTRY.register("fourier_gr1", _build_gr1)
_REGISTRY.register("airbot_mmk2", _build_airbot_mmk2)
_REGISTRY.register("synthetic_2r", _build_synthetic)
_REGISTRY.register("a1x_ee", _build_a1x_ee)
_REGISTRY.register("aloha_bimanual", _build_aloha_bimanual)


def get_robot(name: str, **kwargs) -> RobotSpec:
    """Construct the named :class:`~kinescore.core.robot.RobotSpec`.

    Parameters
    ----------
    name:
        One of :func:`available_robots`.
    **kwargs:
        Forwarded to the robot's constructor (e.g. ``device``, ``dtype`` for
        GR-1/Airbot; ``link1_m``/``link2_m`` for the synthetic arm).

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
