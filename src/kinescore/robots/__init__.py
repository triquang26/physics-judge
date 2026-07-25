"""Robot registry: ``get_robot(name) -> RobotSpec``.

Importing this module must never require ``pytorch_kinematics`` or
``robot_descriptions``. ``kinescore.robots.synthetic.Synthetic2R`` is imported
eagerly (it has no such dependency and is the whole point of the CPU-only test
path -- see its module docstring), but Franka and GR-1 are resolved lazily
inside :func:`get_robot`, so a caller who only ever asks for
``"synthetic_2r"`` -- the metric-layer test suite, mainly -- never triggers the
``import pytorch_kinematics`` in ``franka/fk.py`` / ``gr1/fk.py``, which can
otherwise fail on a machine that has not installed the (heavier, optional-in-
spirit-even-though-listed-as-a-core-dependency) kinematics stack.
"""
from __future__ import annotations

from collections.abc import Callable

from kinescore.core.robot import RobotSpec
from kinescore.robots.synthetic import Synthetic2R

__all__ = ["get_robot", "available_robots"]


def _build_franka(**kwargs) -> RobotSpec:
    from kinescore.robots.franka.spec import FrankaSpec
    return FrankaSpec(**kwargs)


def _build_gr1(**kwargs) -> RobotSpec:
    from kinescore.robots.gr1.spec import GR1Spec
    return GR1Spec(**kwargs)


def _build_synthetic(**kwargs) -> RobotSpec:
    return Synthetic2R(**kwargs)


#: Registry key -> zero-import-cost factory. Each factory does its own heavy
#: imports lazily (see module docstring); constructing the entries dict below
#: does not import pytorch_kinematics/robot_descriptions itself.
_FACTORIES: dict[str, Callable[..., RobotSpec]] = {
    "franka_panda": _build_franka,
    "fourier_gr1": _build_gr1,
    "synthetic_2r": _build_synthetic,
}


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
    KeyError
        If ``name`` is not registered -- lists the valid names, so a typo'd
        robot name fails at the call site instead of surfacing as a confusing
        ``AttributeError`` deep in a metric.
    kinescore.paths.MissingPathError
        Propagated unmodified from ``GR1Spec`` when ``KINESCORE_ASSETS`` is
        unset or the GR-1 URDF is not checked out under it.
    """
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise KeyError(
            f"unknown robot {name!r}; available: {sorted(_FACTORIES)}") from None
    return factory(**kwargs)


def available_robots() -> tuple[str, ...]:
    """Registered robot names, in a stable (sorted) order."""
    return tuple(sorted(_FACTORIES))
