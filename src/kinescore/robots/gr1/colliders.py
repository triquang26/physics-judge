"""URDF -> mechanical collision/inertial spec (analytic, no mesh / no trimesh).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

__all__ = ["RobotColliders"]

# GR-1 defaults; override via ctor for another robot.
_BODY_CORE_LINKS: tuple[str, ...] = ("torso_link", "base_link", "head_pitch_link")
_FOOT_LINKS: tuple[str, ...] = ("left_foot_roll_link", "right_foot_roll_link")


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
        self.body_links: list[str] = list(body_core_links)
        self.foot_links: list[str] = list(foot_links)
        root = ET.parse(self.urdf_path).getroot()
        links = {link_el.get("name"): link_el for link_el in root.findall("link")}

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
                    centers.append(c)
                    radii.append(r)
                    sph_link.append(li)
            elif sph is not None:
                centers.append(T[:3, 3])
                radii.append(float(sph.get("radius")))
                sph_link.append(li)
            else:
                raise ValueError(f"link '{name}' collision is not cylinder/sphere")

        self.register_buffer("sphere_centers", torch.tensor(np.array(centers), dtype=torch.float32))
        self.register_buffer("sphere_radii", torch.tensor(np.array(radii), dtype=torch.float32))
        self.register_buffer("sphere_link", torch.tensor(sph_link, dtype=torch.long))

        # ── per-link mass + inertial CoM (for balance) ───────────────────────
        mass_links, masses, coms = [], [], []
        for name, link_el in links.items():
            inert = link_el.find("inertial")
            if inert is None:
                continue
            m = inert.find("mass")
            if m is None or float(m.get("value")) <= 0:
                continue
            com = _origin_T(inert.find("origin"))[:3, 3]
            mass_links.append(name)
            masses.append(float(m.get("value")))
            coms.append(com)
        self.mass_links: list[str] = mass_links
        self.register_buffer("link_mass", torch.tensor(masses, dtype=torch.float32))
        self.register_buffer("link_com", torch.tensor(np.array(coms), dtype=torch.float32))

    # ── posing (given per-link SE(3) frames from FK) ─────────────────────────

    def posed_body_spheres(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Body-core spheres in base frame."""
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
