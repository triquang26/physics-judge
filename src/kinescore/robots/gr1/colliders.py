"""URDF -> mechanical collision/inertial spec (analytic, no mesh / no trimesh).

Parses the URDF once (``xml.etree``, exactly like ``GR1FK._read_urdf_limits``) and
holds, as ``register_buffer`` tensors that follow ``.to(device)``:

* **Body-core collision spheres** — the torso / base / head that the arms must not
  penetrate. Each ``<collision>`` primitive is turned into a set of spheres in its
  *link frame*: a ``<cylinder>`` becomes spheres sampled along its local +Z axis
  (posed by the collision ``<origin xyz rpy>``); a ``<sphere>`` becomes one sphere.
  Meshes (fingers/hands) are skipped — arm↔torso / arm↔arm is the hallucination
  mode we target, not finger self-collision.
* **Per-link mass + inertial CoM** — every link with ``<inertial>``, for the
  centre-of-mass balance check.

Arm geometry is NOT baked in here: the arms are represented at runtime as capsules
along the FK keypoint chain (see :class:`MechanicalFeasibility`), so they move with
``q`` differentiably.

Provenance
----------
Ported **verbatim** from ``models/physics/robot_colliders.py`` (Marionette-fkjepa)
-- no line of the class body below is changed; the module docstring above is also
unchanged. See ``kinescore/robots/gr1/spec.py`` for how ``GR1Spec`` poses these
buffers using ``GR1FK.link_frames`` and exposes them for the ``COLLIDERS`` /
``SUPPORT_POLYGON`` capabilities, which is new code, not part of this port.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

__all__ = ["RobotColliders"]

# GR-1 defaults; override via ctor for another robot.
_BODY_CORE_LINKS: Tuple[str, ...] = ("torso_link", "base_link", "head_pitch_link")
_FOOT_LINKS: Tuple[str, ...] = ("left_foot_roll_link", "right_foot_roll_link")


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw -> rotation matrix ``Rz(y) Ry(p) Rx(r)``."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p),
                              np.sin(p), np.cos(y), np.sin(y))
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _origin_T(elem) -> np.ndarray:
    """4x4 transform from a URDF ``<origin xyz rpy>`` element (identity if absent)."""
    T = np.eye(4)
    if elem is not None:
        T[:3, 3] = np.fromstring(elem.get("xyz", "0 0 0"), sep=" ")
        T[:3, :3] = _rpy_to_matrix(np.fromstring(elem.get("rpy", "0 0 0"), sep=" "))
    return T


class RobotColliders(nn.Module):
    """Analytic collision spheres + inertial CoM parsed from a URDF.

    Parameters
    ----------
    urdf_path: path to the robot URDF.
    body_core_links: links whose ``<collision>`` primitives become the obstacle
        spheres the arms are checked against (default: GR-1 torso/base/head).
    foot_links: contact links whose rest positions define the support polygon.
    n_cyl_spheres: spheres sampled along each cylinder's axis (default 3).

    Buffers
    -------
    sphere_centers ``(S,3)``, sphere_radii ``(S,)``, sphere_link ``(S,)`` long —
        body-core spheres in their link frames (index into ``self.body_links``).
    link_mass ``(L,)``, link_com ``(L,3)`` — per-mass-link inertial data (index
        into ``self.mass_links``).
    """

    def __init__(self, urdf_path: str,
                 body_core_links: Sequence[str] = _BODY_CORE_LINKS,
                 foot_links: Sequence[str] = _FOOT_LINKS,
                 n_cyl_spheres: int = 3) -> None:
        super().__init__()
        self.urdf_path = str(urdf_path)
        self.body_links: List[str] = list(body_core_links)
        self.foot_links: List[str] = list(foot_links)
        root = ET.parse(self.urdf_path).getroot()
        links = {l.get("name"): l for l in root.findall("link")}

        # ── body-core collision spheres (link frame) ─────────────────────────
        centers, radii, sph_link = [], [], []
        for li, name in enumerate(self.body_links):
            if name not in links:
                raise ValueError(f"body-core link '{name}' not in URDF")
            col = links[name].find("collision")
            if col is None:
                raise ValueError(f"link '{name}' has no <collision>")
            T = _origin_T(col.find("origin"))
            geom = col.find("geometry")
            cyl, sph = geom.find("cylinder"), geom.find("sphere")
            if cyl is not None:
                r = float(cyl.get("radius"))
                L = float(cyl.get("length"))
                zs = np.linspace(-L / 2, L / 2, n_cyl_spheres)
                for z in zs:
                    c = (T @ np.array([0, 0, z, 1.0]))[:3]
                    centers.append(c); radii.append(r); sph_link.append(li)
            elif sph is not None:
                centers.append(T[:3, 3]); radii.append(float(sph.get("radius")))
                sph_link.append(li)
            else:
                raise ValueError(f"link '{name}' collision is not cylinder/sphere")

        self.register_buffer("sphere_centers", torch.tensor(np.array(centers), dtype=torch.float32))
        self.register_buffer("sphere_radii", torch.tensor(np.array(radii), dtype=torch.float32))
        self.register_buffer("sphere_link", torch.tensor(sph_link, dtype=torch.long))

        # ── per-link mass + inertial CoM (for balance) ───────────────────────
        avail = set(name for name in links)
        mass_links, masses, coms = [], [], []
        for name, l in links.items():
            inert = l.find("inertial")
            if inert is None:
                continue
            m = inert.find("mass")
            if m is None or float(m.get("value")) <= 0:
                continue
            com = _origin_T(inert.find("origin"))[:3, 3]
            mass_links.append(name); masses.append(float(m.get("value"))); coms.append(com)
        self.mass_links: List[str] = mass_links
        self.register_buffer("link_mass", torch.tensor(masses, dtype=torch.float32))
        self.register_buffer("link_com", torch.tensor(np.array(coms), dtype=torch.float32))

    # ── posing (given per-link SE(3) frames from FK) ─────────────────────────

    def posed_body_spheres(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Body-core spheres in base frame.

        ``frames``: ``(B,T,len(body_links),4,4)`` SE(3) of ``self.body_links``.
        Returns ``(centers (B,T,S,3), radii (S,))``.
        """
        R = frames[..., :3, :3]                                  # (B,T,Lb,3,3)
        p = frames[..., :3, 3]                                   # (B,T,Lb,3)
        Rc = R[:, :, self.sphere_link]                           # (B,T,S,3,3)
        pc = p[:, :, self.sphere_link]                           # (B,T,S,3)
        c = self.sphere_centers.to(Rc.device, Rc.dtype)          # (S,3)
        centers = torch.einsum("btsij,sj->btsi", Rc, c) + pc     # (B,T,S,3)
        return centers, self.sphere_radii.to(centers.device, centers.dtype)

    def world_com(self, frames: torch.Tensor) -> torch.Tensor:
        """Whole-body CoM in base frame. ``frames``: ``(B,T,len(mass_links),4,4)`` -> ``(B,T,3)``."""
        R = frames[..., :3, :3]
        p = frames[..., :3, 3]
        com = self.link_com.to(R.device, R.dtype)                # (L,3)
        com_world = torch.einsum("btlij,lj->btli", R, com) + p   # (B,T,L,3)
        m = self.link_mass.to(R.device, R.dtype)                 # (L,)
        return (com_world * m[None, None, :, None]).sum(2) / m.sum()
