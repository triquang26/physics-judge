"""Shared URDF loading: resolve, hash, and parse joint limits.

Two robots, two resolution strategies
--------------------------------------
Franka ships inside the ``robot_descriptions`` package (a pinned pip
dependency; its Panda URDF + meshes are a few MB and get cached under
``~/.cache/robot_descriptions`` on first use), so nothing about it needs
``KINESCORE_ASSETS``. GR-1 does not: the fkjepa asset tree that carries its
URDF is ~285 MB (meshes for a 44-DoF humanoid, several body variants), so
either vendoring it into this repo or piggy-backing it onto
``robot_descriptions`` would blow up clone size for the common case of a
Franka-only run. It is instead resolved from ``KINESCORE_ASSETS``, an
operator-owned directory that is never committed
(:func:`kinescore.paths.env_path`); construction fails loudly with
:class:`~kinescore.paths.MissingPathError` when the variable is unset or the
file is missing, rather than silently falling back to a path that happens to
exist on the machine this was written on (see the ``kinescore.paths`` module
docstring for the class of bug that default-fallback pattern causes).

Why every ``RobotSpec`` records a URDF hash
--------------------------------------------
Both ``robot_descriptions`` and an asset-tree checkout are mutable across
time -- a version bump, a corrected mesh, a different Panda variant with an
extra flange. None of that raises an error; it just quietly changes a few
keypoint positions by millimetres, which is exactly the kind of drift that is
invisible until an unrelated metric report looks slightly off weeks later.
Hashing the exact URDF bytes that FK was built from and threading it onto
``RobotSpec.urdf_sha256`` turns that into something a `diff` between two run
records catches immediately.
"""
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
    """Per-joint limits parsed from a URDF ``<joint><limit .../></joint>``.

    ``velocity`` / ``effort`` are ``None`` when the URDF omits them -- URDF
    only requires ``lower``/``upper`` on revolute/prismatic joints, the rest
    is optional, and a fabricated cap would misrepresent the source file.
    """

    lower: float
    upper: float
    velocity: float | None = None
    effort: float | None = None


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's exact bytes, for ``RobotSpec.urdf_sha256``.

    Streamed in 1 MiB chunks so a large URDF (GR-1's is a few hundred KB, but
    nothing here assumes that) never gets read into memory as one blob.
    """
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
        goes through :func:`importlib.import_module` instead (verified
        against the installed ``robot_descriptions`` package; see
        ``legacy_docs/PROVENANCE.md``).

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

    Only ``revolute``/``prismatic`` joints carry a ``<limit>`` element in a
    valid URDF; a fixed or continuous joint does not, and asking for one is a
    caller bug rather than a missing-data case, so it raises.

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
