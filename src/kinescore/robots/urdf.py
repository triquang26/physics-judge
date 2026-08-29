"""Shared URDF loading: resolve, hash, and parse joint limits."""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from kinescore.paths import MissingPathError, env_path

__all__ = [
    "JointLimits",
    "sha256_file",
    "resolve_asset_urdf",
    "resolve_robot_description_urdf",
    "parse_joint_limits",
]


@dataclass(frozen=True)
class JointLimits:
    """Per-joint limits parsed from a URDF ``<joint><limit .../></joint>``."""

    lower: float
    upper: float
    velocity: float | None = None
    effort: float | None = None


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's exact bytes, for ``RobotSpec.urdf_sha256``."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_asset_urdf(relative_path: str) -> Path:
    """Resolve a URDF under ``KINESCORE_ASSETS``.

    Parameters
    ----------
    relative_path:
        Path of the URDF relative to ``KINESCORE_ASSETS``, e.g.
        ``"grx/GRX/GR1/gr1t1/urdf/gr1t1_fourier_hand_6dof.urdf"``.

    Raises
    ------
    MissingPathError
        If ``KINESCORE_ASSETS`` is unset, or the resolved file does not exist.
        Both are operator configuration problems, not code bugs -- the message
        names the exact thing to fix, per ``kinescore.paths.env_path``'s
        contract. Callers building a robot whose assets are optional (GR-1)
        should let this propagate rather than catching it and fabricating a
        spec, so callers of *that* code get the same clear failure.
    """
    base = env_path("KINESCORE_ASSETS")
    path = (base / relative_path).resolve()
    if not path.is_file():
        raise MissingPathError(
            f"KINESCORE_ASSETS is set to {base}, but {relative_path!r} does "
            f"not exist under it (looked for {path}). The asset tree may not "
            f"be checked out at this location, or the relative path is stale."
        )
    return path


def resolve_robot_description_urdf(module_attr: str) -> Path:
    """Resolve a URDF shipped by the ``robot_descriptions`` package.

    Parameters
    ----------
    module_attr:
        Submodule name under ``robot_descriptions``, e.g.
        ``"panda_description"``. ``robot_descriptions`` exposes each robot as
        its own lazily-importable submodule rather than a plain package
        attribute -- ``getattr(robot_descriptions, module_attr)`` raises
        ``AttributeError`` even after the package itself is imported, so this
        goes through :func:`importlib.import_module` instead.

    Imported lazily: importing this function must not require the
    ``robot_descriptions`` package to be installed, so that
    ``kinescore.robots`` (the registry) stays importable without the full
    kinematics stack -- see ``robots/__init__.py``'s module docstring.
    """
    import importlib

    module = importlib.import_module(f"robot_descriptions.{module_attr}")
    return Path(module.URDF_PATH)


def parse_joint_limits(urdf_path: Path,
                        joint_names: Sequence[str]) -> dict[str, JointLimits]:
    """Parse ``<limit>`` tags for the named joints from a URDF file.

    Raises
    ------
    ValueError
        If any requested joint has no ``<limit>`` in the file -- this
        surfaces a typo'd joint name (or a URDF that dropped a joint) at
        construction time instead of as a downstream ``KeyError`` far from
        the cause.
    """
    root = ET.parse(urdf_path).getroot()
    wanted = set(joint_names)
    out: dict[str, JointLimits] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in wanted:
            continue
        if joint.get("type") not in ("revolute", "prismatic"):
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        velocity = limit.get("velocity")
        effort = limit.get("effort")
        out[name] = JointLimits(
            lower=float(limit.get("lower", "0")),
            upper=float(limit.get("upper", "0")),
            velocity=float(velocity) if velocity is not None else None,
            effort=float(effort) if effort is not None else None,
        )
    missing = wanted - out.keys()
    if missing:
        raise ValueError(
            f"joint(s) {sorted(missing)} have no revolute/prismatic <limit> "
            f"in {urdf_path}")
    return out
