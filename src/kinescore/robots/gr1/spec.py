"""``RobotSpec`` for the Fourier GR-1: bimanual humanoid, two arms + feet."""
from __future__ import annotations

from typing import Any

import torch

from kinescore.core.robot import Capability, rigid_bone_mask
from kinescore.robots.gr1.colliders import RobotColliders
from kinescore.robots.gr1.fk import (
    EE_LINK,
    FINGERTIPS_LEFT,
    FINGERTIPS_RIGHT,
    GR1FK,
    KEYPOINTS_LEFT,
    KEYPOINTS_RIGHT,
)
from kinescore.robots.urdf import resolve_asset_urdf, sha256_file

__all__ = ["GR1Spec", "GR1_URDF_RELPATH"]

#: Location of the GR-1 URDF relative to ``KINESCORE_ASSETS``. An operator
#: populating ``KINESCORE_ASSETS`` mirrors this subtree.
GR1_URDF_RELPATH = "grx/GRX/GR1/gr1t1/urdf/gr1t1_fourier_hand_6dof.urdf"


class GR1Spec:
    """``RobotSpec`` for the Fourier GR-1 bimanual humanoid.

    Parameters
    ----------
    device, dtype:
        Passed straight through to the underlying :class:`GR1FK` /
        :class:`RobotColliders`.
    urdf_path:
        Override the URDF location instead of resolving it from
        ``KINESCORE_ASSETS`` / :data:`GR1_URDF_RELPATH`. Mainly for tests that
        point at a specific fixture URDF.

    Raises
    ------
    kinescore.paths.MissingPathError
        If ``urdf_path`` is not given and ``KINESCORE_ASSETS`` is unset, or
        the GR-1 URDF is not checked out under it. Construction fails here
        rather than lazily on first FK call, so a missing asset tree is
        diagnosable at the point a ``GR1Spec`` is requested, not three metrics
        deep into a scoring run.

    Attributes
    ----------
    All ``RobotSpec`` protocol attributes, plus (see module docstring):

    colliders:
        The wrapped :class:`RobotColliders`.
    fk:
        The wrapped :class:`GR1FK`, for callers needing the lower-level
        per-side ``ee_pose`` / ``fingers_fk`` helpers that have no
        ``RobotSpec``-protocol equivalent.

    ``keypoint_links`` layout
    --------------------------
    ``KEYPOINTS_LEFT + KEYPOINTS_RIGHT`` (``K = 12``): indices ``0..5`` are the
    left arm (shoulder -> ... -> end effector), ``6..11`` the right arm, in the
    same per-arm order ``GR1FK.keypoints_fk`` returns. :meth:`ee_sites` names
    indices 5 and 11 (the two ``*_end_effector_link`` keypoints).
    """

    #: Registry key (kinescore.robots.get_robot).
    name = "fourier_gr1"
    #: Predicted DOF: left arm (7) + right arm (7) + waist (3) + six actuators per hand.
    n_joints = GR1FK.N_Q

    def __init__(self, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32,
                 urdf_path: str | None = None) -> None:
        if urdf_path is None:
            urdf_path = str(resolve_asset_urdf(GR1_URDF_RELPATH))
        self.urdf_sha256: str | None = sha256_file(urdf_path)

        self.fk = GR1FK(urdf_path, device=device, dtype=dtype)
        self.colliders = RobotColliders(urdf_path)

        left = KEYPOINTS_LEFT + FINGERTIPS_LEFT
        right = KEYPOINTS_RIGHT + FINGERTIPS_RIGHT
        self.keypoint_links: tuple[str, ...] = left + right
        n_left = len(left)

        # ---- joint limits (29,); GR1FK has no effort data -> no EFFORT_LIMITS
        self.q_lo = self.fk.q_lo.clone()
        self.q_hi = self.fk.q_hi.clone()
        self.vel_limits: torch.Tensor | None = self.fk.q_vel_max.clone()
        self.effort_limits: torch.Tensor | None = None

        # ---- bones: concatenate the two per-arm chains, right offset by n_left
        self.bone_pairs = torch.cat(
            [self.fk.bone_pairs_left, self.fk.bone_pairs_right + n_left], dim=0)
        self.bone_lengths = torch.cat(
            [self.fk.bone_lengths_left, self.fk.bone_lengths_right], dim=0)
        # Every remaining bone is an arm segment: GR1FK builds none that end on a
        # fingertip, and the shortest arm bone measures 0.023 m.
        mask = rigid_bone_mask(self.bone_lengths)
        self.rigid_bone_pairs = self.bone_pairs[mask]
        self.rigid_bone_lengths = self.bone_lengths[mask]

        self.capabilities: frozenset[str] = frozenset(
            {Capability.ROTATIONS, Capability.COLLIDERS, Capability.SUPPORT_POLYGON})

    # ------------------------------------------------------------------ #
    # RobotSpec protocol
    # ------------------------------------------------------------------ #
    def forward_kinematics(self, q: torch.Tensor,
                           aux: Any | None = None) -> torch.Tensor:
        """``(B,T,29) -> (B,T,22,3)``. ``aux`` is unused (reserved for hand DoF;
        """
        left = self.fk.keypoints_fk(q, "left")
        right = self.fk.keypoints_fk(q, "right")
        return torch.cat([left, right], dim=2)

    def forward_transforms(self, q: torch.Tensor, aux: Any | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B,T,29) -> P (B,T,22,3), R (B,T,22,3,3)``. ``aux`` unused, see above."""
        p_l, r_l = self.fk.forward_transforms(q, "left")
        p_r, r_r = self.fk.forward_transforms(q, "right")
        return torch.cat([p_l, p_r], dim=2), torch.cat([r_l, r_r], dim=2)

    def ee_sites(self) -> tuple[int, ...]:
        """Two end-effector sites: left, then right ``*_end_effector_link``."""
        return (self.keypoint_links.index(EE_LINK["left"]),
                self.keypoint_links.index(EE_LINK["right"]))

    # ------------------------------------------------------------------ #
    # COLLIDERS / SUPPORT_POLYGON extensions -- see module docstring
    # ------------------------------------------------------------------ #
    def body_collider_spheres(self, q: torch.Tensor
                              ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B,T,17) -> (centers (B,T,S,3), radii (S,))`` torso/base/head spheres.
        """
        frames = self.fk.link_frames(q, self.colliders.body_links)
        return self.colliders.posed_body_spheres(frames)

    def world_com(self, q: torch.Tensor) -> torch.Tensor:
        """``(B,T,17) -> (B,T,3)`` whole-body centre of mass in base frame."""
        frames = self.fk.link_frames(q, self.colliders.mass_links)
        return self.colliders.world_com(frames)

    def support_polygon(self, q: torch.Tensor) -> torch.Tensor:
        """``(B,T,17) -> (B,T,2,3)`` foot positions (left, right) in base frame.
        """
        frames = self.fk.link_frames(q, self.colliders.foot_links)
        return frames[..., :3, 3]
