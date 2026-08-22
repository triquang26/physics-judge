"""``RobotSpec`` for ALOHA bimanual: two 6-DOF arms + 2-finger grippers, table-mounted.

:class:`AlohaSpec` wraps :class:`~kinescore.robots.aloha.fk.AlohaFK` and
adapts it to the :class:`~kinescore.core.robot.RobotSpec` protocol: bimanual
per-side keypoint concatenation, plus a real gripper handled through ``aux``
with a structural bone exclusion, doubled for two arms.

Registered under the key ``"aloha_bimanual"``.

Why no ``COLLIDERS`` / ``SUPPORT_POLYGON``
---------------------------------------------
``aloha_bimanual.urdf`` is kinematics-only (joint origins, axes, limits,
inertials -- no mesh geometry), so there is no collision primitive to back
``COLLIDERS`` with, and a declared-but-unbacked capability is worse than an
undeclared one. ALOHA is two arms bolted to a table: no legs, no mobile base,
so ``SUPPORT_POLYGON`` has no balance margin to report either.
"""
from __future__ import annotations

from typing import Any

import torch

from kinescore.core.robot import Capability
from kinescore.robots.aloha.constants import (
    ACTUATED_LINKS,
    EE_LINK,
    KEYPOINTS_LEFT,
    KEYPOINTS_RIGHT,
)
from kinescore.robots.aloha.fk import AlohaFK
from kinescore.robots.base import structural_rigid_bone_mask, warn_dropped_bones
from kinescore.robots.urdf import resolve_asset_urdf, sha256_file

__all__ = ["AlohaSpec", "ALOHA_URDF_RELPATH"]

#: Location of the composite bimanual URDF relative to ``KINESCORE_ASSETS``:
#: two Interbotix vx300s arms, xacro-expanded and merged under a synthetic
#: ``world`` root with the Menagerie ALOHA mount transforms.
ALOHA_URDF_RELPATH = "aloha/urdf/aloha_bimanual.urdf"


class AlohaSpec:
    """``RobotSpec`` for ALOHA bimanual's two 6-DOF arms + grippers.

    Parameters
    ----------
    device, dtype:
        Passed straight through to the underlying :class:`AlohaFK`.
    urdf_path:
        Override the URDF location instead of resolving it from
        ``KINESCORE_ASSETS`` / :data:`ALOHA_URDF_RELPATH`. Mainly for tests
        that point at a specific fixture URDF.

    Raises
    ------
    kinescore.paths.MissingPathError
        If ``urdf_path`` is not given and ``KINESCORE_ASSETS`` is unset, or
        the composite URDF is not checked out under it.

    ``keypoint_links`` layout
    --------------------------
    ``KEYPOINTS_LEFT + KEYPOINTS_RIGHT`` (``K = 18``): indices ``0..8`` are
    the left arm (shoulder -> ... -> gripper -> fingers -> TCP), ``9..17``
    the right arm, in the same per-arm order ``AlohaFK.keypoints_fk`` returns.

    ``aux`` contract
    -----------------
    ``forward_kinematics(q, aux)`` / ``forward_transforms(q, aux)`` accept
    ``aux`` as the two grippers' opening in ``[0, 1]``, shape ``(B, T, 2)`` =
    ``[left, right]``, or ``None`` for both fully closed (see
    :class:`~kinescore.robots.aloha.fk.AlohaFK`'s module docstring).
    """

    #: Registry key (kinescore.robots.get_robot).
    name = "aloha_bimanual"
    #: Predicted DOF: left arm (6) + right arm (6); grippers travel through
    #: `aux`, not `q` -- see fk.py's module docstring.
    n_joints = AlohaFK.N_Q

    def __init__(self, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32,
                 urdf_path: str | None = None) -> None:
        if urdf_path is None:
            urdf_path = str(resolve_asset_urdf(ALOHA_URDF_RELPATH))
        self.urdf_sha256: str | None = sha256_file(urdf_path)

        self.fk = AlohaFK(urdf_path, device=device, dtype=dtype)

        self.keypoint_links: tuple[str, ...] = KEYPOINTS_LEFT + KEYPOINTS_RIGHT
        n_left = len(KEYPOINTS_LEFT)

        # ---- joint limits (12,) -- see AlohaFK docstring: no margin
        self.q_lo = self.fk.q_lo.clone()
        self.q_hi = self.fk.q_hi.clone()
        self.vel_limits: torch.Tensor | None = self.fk.q_vel_max.clone()
        self.effort_limits: torch.Tensor | None = self.fk.q_effort_max.clone()

        # ---- bones: concatenate the two per-arm chains, right offset by n_left
        self.bone_pairs = torch.cat(
            [self.fk.bone_pairs_left, self.fk.bone_pairs_right + n_left], dim=0)
        self.bone_lengths = torch.cat(
            [self.fk.bone_lengths_left, self.fk.bone_lengths_right], dim=0)
        # D9: 3 of the 8 per-arm bones end on a gripper-actuated link
        # (gripper_link->left_finger_link, left_finger_link->right_finger_link,
        # right_finger_link->ee_gripper_link) -- exactly the Franka pattern,
        # doubled for two arms. See constants.py's ACTUATED_LINKS docstring.
        mask = self._rigid_bone_mask()
        warn_dropped_bones("AlohaSpec", self.keypoint_links, self.bone_pairs,
                           self.bone_lengths, mask)
        self.rigid_bone_pairs = self.bone_pairs[mask]
        self.rigid_bone_lengths = self.bone_lengths[mask]

        self.capabilities: frozenset[str] = frozenset(
            {Capability.ROTATIONS, Capability.EFFORT_LIMITS})

    def _rigid_bone_mask(self) -> torch.Tensor:
        """Structural (D9) + degenerate-length rigid-bone selection.

        Delegates to :func:`kinescore.robots.base.structural_rigid_bone_mask`
        -- see that function's docstring, and
        :class:`~kinescore.robots.franka.spec.FrankaSpec._rigid_bone_mask`
        for the two-rule pattern this mirrors.
        """
        return structural_rigid_bone_mask(
            self.keypoint_links, self.bone_pairs, self.bone_lengths,
            ACTUATED_LINKS)

    # ------------------------------------------------------------------ #
    # RobotSpec protocol
    # ------------------------------------------------------------------ #
    def _gripper_bt2(self, q: torch.Tensor, aux: Any | None) -> torch.Tensor:
        """Normalise ``aux`` to a ``(B, T, 2)`` = ``[left, right]`` gripper tensor."""
        b, t = q.shape[0], q.shape[1]
        if aux is None:
            return q.new_zeros(b, t, 2)
        gripper = aux
        if gripper.shape[:2] != (b, t) or gripper.shape[-1] != 2:
            raise ValueError(
                f"aux (gripper) must broadcast to (B,T,2) = ({b},{t},2); got "
                f"{tuple(gripper.shape)}")
        return gripper

    def forward_kinematics(self, q: torch.Tensor,
                           aux: Any | None = None) -> torch.Tensor:
        """``(B,T,12) -> (B,T,18,3)``; see ``aux`` contract above."""
        gripper2 = self._gripper_bt2(q, aux)
        left = self.fk.keypoints_fk(q, gripper2, "left")
        right = self.fk.keypoints_fk(q, gripper2, "right")
        return torch.cat([left, right], dim=2)

    def forward_transforms(self, q: torch.Tensor, aux: Any | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B,T,12) -> P (B,T,18,3), R (B,T,18,3,3)``; see ``aux`` contract above."""
        gripper2 = self._gripper_bt2(q, aux)
        p_l, r_l = self.fk.forward_transforms(q, gripper2, "left")
        p_r, r_r = self.fk.forward_transforms(q, gripper2, "right")
        return torch.cat([p_l, p_r], dim=2), torch.cat([r_l, r_r], dim=2)

    def ee_sites(self) -> tuple[int, ...]:
        """Two end-effector sites: left, then right ``ee_gripper_link``."""
        return (self.keypoint_links.index(EE_LINK["left"]),
                self.keypoint_links.index(EE_LINK["right"]))
