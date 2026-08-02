"""URDF ``<inertial>`` parsing, and per-chain dynamics assembly for torque.

Ported from ``Marionette-fkjepa/scripts/gr1/54_torque_feasibility.py``'s
``_inertial()`` (source lines 73-87) and ``build_dynamics()`` (source lines
89-115) -- see ``legacy_docs/PROVENANCE.md`` for the full source->destination
record. That script computes hand-rolled Newton-Euler inverse dynamics over
GR-1's two arm chains; this module is the URDF-reading half of it, made
robot-agnostic (any single open kinematic chain, on any URDF) and reusable
across robots, the way :mod:`kinescore.robots.urdf` already is for joint
*limits*.

**Do not duplicate what ``robots/urdf.py`` already does.** That module owns
``<joint><limit .../>`` parsing (:func:`kinescore.robots.urdf.parse_joint_limits`,
which already extracts ``effort`` per joint) and URDF resolution/hashing.
This module owns exactly what that one does not: ``<link><inertial>`` (mass,
centre of mass, inertia tensor) and the small amount of extra ``<joint>``
bookkeeping (child link, rotation axis) a Newton-Euler pass needs that a pure
joint-*limit* reader has no reason to carry.

Two defects found and fixed while porting ``build_dynamics()``
-----------------------------------------------------------------
The source's per-joint loop has two silent fallbacks, both of which convert
"the URDF simply never declared this" into a *fabricated, benign-looking*
number rather than an honest "unknown":

* **Missing/zero joint effort -> treated as ~unconstrained** (source:
  ``e = float(lim.get("effort")) if lim is not None else 1e9; e = e if e > 0
  else 1e9``). A joint with no declared rating silently reads as "practically
  infinite headroom" in every downstream percent-of-rated calculation --
  the exact shape of bug this package already refuses for
  ``effort_proxy``/``vel_violation_frac`` (see
  ``kinescore/metrics/joint_dynamics.py``'s module docstring): an unmeasured
  quantity must not silently look like a clean bill of health.
* **Missing ``<inertial>`` -> treated as exactly massless/inertialess**
  (source: ``info = _inertial(L[child]) or (0.0, np.zeros(3), np.zeros((3,
  3)))``). A link with no declared inertial block contributes *zero* force
  and moment to every joint's torque sum, not "unknown" -- an even sharper
  instance of the ``NaN`` vs ``0.0`` distinction ``core/robot.py`` and
  ``core/metric.py`` document everywhere else in this package.

:func:`build_chain_dynamics` therefore represents a missing effort/inertial
as ``NaN`` in :class:`ChainDynamics`'s arrays instead of ``inf``/``0.0``, and
:mod:`kinescore.metrics.torque` refuses to compute a torque number at all
(``NaN`` with a named reason) when any joint/link it needs is ``NaN`` here --
see that module's docstring and ``legacy_docs/PROVENANCE.md`` (defect recorded
there). On the real GR-1 URDF used to produce the recorded
``torque_summary.json`` numbers, every one of the 14 arm joints and their
child links *does* declare both fields (verified while porting), so this
fix changes nothing about any reproduced number -- it only changes what
happens on a URDF that omits one, from "silently plausible" to "loudly
NaN".
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "LinkInertial",
    "ChainDynamics",
    "rpy_to_matrix",
    "parse_link_inertials",
    "build_chain_dynamics",
]


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw -> rotation matrix, ``Rz(y) Ry(p) Rx(r)``.

    Verbatim arithmetic port of ``54_torque_feasibility.py``'s ``_rpy()``
    (source lines 66-70), expressed as the ``Rz @ Ry @ Rx`` product it
    computes. This is the same convention (and, independently, the same
    result) as ``kinescore.robots.gr1.colliders``'s private
    ``_rpy_to_matrix`` -- kept as its own small copy here rather than an
    import, since this module must stay usable without ``colliders.py``
    (GR-1-specific, out of this port's file scope) and the function is a
    five-line trig identity, not URDF-loading logic worth centralising.
    """
    r, p, y = (float(rpy[0]), float(rpy[1]), float(rpy[2]))
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


@dataclass(frozen=True)
class LinkInertial:
    """One ``<inertial>`` block, resolved into the owning LINK's frame.

    Parameters
    ----------
    mass:
        Kilograms, > 0 (a link with mass <= 0 or no ``<inertial>`` at all is
        simply absent from :func:`parse_link_inertials`'s return value, not
        represented as a zero -- see module docstring).
    com:
        ``(3,)`` metres, the ``<inertial><origin xyz>`` translation in the
        link frame. A point has no orientation to rotate, so this is the raw
        URDF value.
    inertia:
        ``(3,3)`` kg*m^2 about the CoM, rotated from the ``<inertial>``
        block's own local frame into the *link* frame by the
        ``<inertial><origin rpy>`` rotation (``R @ I_local @ R.T``). This is
        what lets a caller later compute ``I_world = R_link_world @ inertia
        @ R_link_world.T`` without needing to know anything about
        ``<inertial>``'s own local rotation.
    """

    mass: float
    com: np.ndarray
    inertia: np.ndarray


def parse_link_inertials(urdf_path: str | Path,
                          link_names: set[str] | None = None
                          ) -> dict[str, LinkInertial]:
    """Parse ``<inertial>`` for links in a URDF.

    Parameters
    ----------
    urdf_path:
        Path to the URDF file.
    link_names:
        Restrict parsing to these link names; ``None`` (default) parses
        every link in the file that carries a positive-mass ``<inertial>``.

    Returns
    -------
    dict[str, LinkInertial]
        Keyed by link name. Links with no ``<inertial>`` element, no
        ``<mass>``, no ``<inertia>``, or ``mass <= 0`` are **omitted**, not
        zero-filled -- a caller asking "does this link carry mass" should
        get a ``KeyError``/``dict.get()``-returns-``None``, never a
        fabricated ``LinkInertial(mass=0, ...)`` that would silently vanish
        from a sum while looking like "measured and found massless". Mirrors
        ``54_torque_feasibility.py``'s ``_inertial()`` (source lines 73-87)
        function body verbatim, generalised from "one link element" to
        "every link in a URDF, or a named subset".
    """
    root = ET.parse(str(urdf_path)).getroot()
    wanted = set(link_names) if link_names is not None else None
    out: dict[str, LinkInertial] = {}
    for link in root.findall("link"):
        name = link.get("name")
        if wanted is not None and name not in wanted:
            continue
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_el = inertial.find("mass")
        if mass_el is None:
            continue
        mass = float(mass_el.get("value"))
        if mass <= 0:
            continue
        it = inertial.find("inertia")
        if it is None:
            continue
        origin = inertial.find("origin")
        com = (np.array([float(x) for x in origin.get("xyz", "0 0 0").split()])
               if origin is not None else np.zeros(3))
        rpy = (np.array([float(x) for x in origin.get("rpy", "0 0 0").split()])
               if origin is not None else np.zeros(3))
        i_local = np.array([
            [float(it.get("ixx")), float(it.get("ixy")), float(it.get("ixz"))],
            [float(it.get("ixy")), float(it.get("iyy")), float(it.get("iyz"))],
            [float(it.get("ixz")), float(it.get("iyz")), float(it.get("izz"))],
        ])
        R = rpy_to_matrix(rpy)
        out[name] = LinkInertial(mass=mass, com=com, inertia=R @ i_local @ R.T)
    return out


@dataclass(frozen=True)
class ChainDynamics:
    """Per-joint Newton-Euler dynamics parameters for one open kinematic chain.

    Attributes
    ----------
    joint_names:
        ``(n_joint,)`` the ordered joint names this chain was built from.
    links:
        ``(n_joint,)`` -- or ``(n_joint+1,)`` if a massive ``ee_link`` was
        appended (see :func:`build_chain_dynamics`) -- child-link names, one
        per joint, in joint order.
    axis:
        ``(len(links), 3)`` unit joint axis, in the frame FK reports for the
        SAME-INDEXED link in :attr:`links` (this is what
        ``kinescore.robots.gr1.fk.GR1FK.link_frames`` returns per link, and
        exactly what the source's ``joint_torques()`` assumes at ``d["axis"][j]``
        paired with ``R[:, j]``). Zero for an appended ``ee_link`` entry,
        which is a massive body, not an actuated joint.
    effort:
        ``(n_joint,)`` rated effort (N.m), one per joint. ``NaN`` where the
        URDF's ``<limit>`` omits ``effort`` or the joint has no ``<limit>``
        at all -- see module docstring's "defects found and fixed".
    mass, com, inertia:
        ``(len(links),)`` / ``(len(links),3)`` / ``(len(links),3,3)``
        per-LINK dynamics data, aligned with :attr:`links`. ``mass`` is
        ``NaN`` (not ``0.0``) for a link with no ``<inertial>`` block --
        same reasoning as ``effort``.
    n_joint:
        ``len(joint_names)`` (``== len(effort)``); ``links``/``axis`` may be
        one longer than this if a massive ``ee_link`` was appended.
    """

    joint_names: tuple[str, ...]
    links: tuple[str, ...]
    axis: np.ndarray
    effort: np.ndarray
    mass: np.ndarray
    com: np.ndarray
    inertia: np.ndarray
    n_joint: int


def build_chain_dynamics(urdf_path: str | Path, joint_names: tuple[str, ...],
                          ee_link: str | None = None) -> ChainDynamics:
    """Assemble per-joint Newton-Euler dynamics parameters for one open chain.

    Generalises ``54_torque_feasibility.py``'s ``build_dynamics()`` (source
    lines 89-115) from "GR-1's two hardcoded arms in an ``ARM = {"left":
    ..., "right": ...}`` dict" to any ordered sequence of joint names on any
    URDF -- the per-arm loop body's arithmetic is unchanged, only decoupled
    from the GR-1-specific dict so a future single-chain robot (or a second
    humanoid) reads the same way. Called once per arm/chain by
    :mod:`kinescore.metrics.torque`.

    Parameters
    ----------
    urdf_path:
        Path to the URDF file.
    joint_names:
        Ordered revolute/prismatic joint names forming one open chain
        (base-to-tip order matters: it is the order :attr:`ChainDynamics
        .effort`/``.joint_names`` come back in, and the torque recursion in
        :mod:`kinescore.metrics.torque` sums *distal* links for a proximal
        joint by index, exactly as the source does).
    ee_link:
        An optional further-distal link (beyond the last joint's child) to
        include in the mass/force sum if -- and only if -- it has its own
        positive-mass ``<inertial>`` block not already covered by a joint's
        child link. Matches the source's end-effector-link handling
        (source lines 107-111); on the real GR-1 URDF this is a no-op (the
        EE link has no ``<inertial>`` block at all -- verified while
        porting), kept for fidelity to the source and for any future robot
        whose EE link does carry mass.

    Raises
    ------
    ValueError
        If a joint name is not present in the URDF at all (a caller bug --
        a typo'd joint name -- not a "missing declared data" case, so it
        raises rather than propagating a ``NaN``; contrast with a joint that
        exists but omits ``effort``, which is represented as ``NaN``, not an
        exception).
    """
    root = ET.parse(str(urdf_path)).getroot()
    joints_by_name = {j.get("name"): j for j in root.findall("joint")}
    inertials = parse_link_inertials(urdf_path)

    links: list[str] = []
    axes: list[np.ndarray] = []
    effort: list[float] = []
    for jn in joint_names:
        j = joints_by_name.get(jn)
        if j is None:
            raise ValueError(f"joint {jn!r} not found in {urdf_path}")
        child_el = j.find("child")
        if child_el is None or child_el.get("link") is None:
            raise ValueError(f"joint {jn!r} in {urdf_path} has no <child link=.../>")
        child = child_el.get("link")

        axis_el = j.find("axis")
        if axis_el is not None and axis_el.get("xyz"):
            ax = np.array([float(x) for x in axis_el.get("xyz").split()])
            norm = float(np.linalg.norm(ax))
            ax = ax / norm if norm > 1e-12 else ax
        else:
            ax = np.array([0.0, 0.0, 1.0])  # URDF default axis when omitted

        limit = j.find("limit")
        raw_effort = limit.get("effort") if limit is not None else None
        if not raw_effort:  # None (no <limit>/no effort=) or "" (declared but empty)
            e = float("nan")
        else:
            e = float(raw_effort)
            if not (e > 0):
                e = float("nan")

        links.append(child)
        axes.append(ax)
        effort.append(e)

    n_joint = len(joint_names)
    if ee_link is not None and ee_link in inertials and ee_link not in links:
        links.append(ee_link)
        axes.append(np.zeros(3))

    mass = np.array([inertials[lk].mass if lk in inertials else float("nan")
                     for lk in links])
    com = np.array([inertials[lk].com if lk in inertials else
                    np.full(3, float("nan")) for lk in links])
    inertia = np.array([inertials[lk].inertia if lk in inertials else
                        np.full((3, 3), float("nan")) for lk in links])

    return ChainDynamics(
        joint_names=tuple(joint_names), links=tuple(links),
        axis=np.stack(axes), effort=np.array(effort),
        mass=mass, com=com, inertia=inertia, n_joint=n_joint)
