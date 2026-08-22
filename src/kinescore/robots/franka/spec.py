"""``RobotSpec`` for the Franka Panda: 7-DOF arm + 2-finger gripper on a table.

:class:`FrankaSpec` adapts :class:`~kinescore.robots.franka.fk.FrankaFK` to
the :class:`~kinescore.core.robot.RobotSpec` protocol. Three decisions it
makes:

1. **Degenerate/actuation-only bones are dropped from ``rigid_bone_pairs``.**
   See :data:`kinescore.robots.franka.constants.RIGID_BONE_MIN_M`: three of
   the seven consecutive-keypoint bones touch a finger link, whose position
   tracks the gripper's prismatic joint rather than rigid arm structure, so
   including them makes "the gripper opened" read as "the arm deformed".
2. **No ``SUPPORT_POLYGON``, no ``COLLIDERS`` capability.** A Panda is bolted
   to a table: it has no balance margin to report, and this spec carries no
   collision geometry. Declaring the capabilities anyway would produce a
   number indistinguishable from a real measurement -- a balance detector
   that always reads "stable" for a robot that cannot fall over measures
   nothing.
3. **Forward kinematics does no linear search per call.**
   :meth:`FrankaSpec.__init__` resolves each Panda joint's index into the
   ``pytorch_kinematics`` chain once and stores it as a ``LongTensor``
   buffer, so scoring assembles the chain input with a single vectorised
   ``th[:, idx] = q`` scatter.
"""
from __future__ import annotations

from typing import Any

import torch

from kinescore.core.robot import Capability
from kinescore.robots.base import structural_rigid_bone_mask, warn_dropped_bones
from kinescore.robots.franka.constants import (
    ACTUATED_LINKS,
    PANDA_ARM_JOINTS,
    PANDA_FINGER_JOINTS,
    PANDA_FINGER_MAX,
    RIGID_BONE_MIN_M,
)
from kinescore.robots.franka.fk import FrankaFK
from kinescore.robots.urdf import resolve_robot_description_urdf, sha256_file

__all__ = ["FrankaSpec"]


class FrankaSpec:
    """``RobotSpec`` for the Franka Panda.

    Parameters
    ----------
    device, dtype:
        Passed straight through to the underlying :class:`FrankaFK`.

    Attributes
    ----------
    All ``RobotSpec`` protocol attributes (see ``kinescore.core.robot``), plus:

    fk:
        The wrapped :class:`FrankaFK` instance. Exposed for callers
        that need the lower-level ``ee_pose`` helper (end-effector pose with
        the gripper held closed) that has no ``RobotSpec``-protocol
        equivalent; not used by :meth:`forward_kinematics` /
        :meth:`forward_transforms` themselves (see class docstring point 3).

    ``aux`` contract
    -----------------
    ``forward_kinematics(q, aux)`` / ``forward_transforms(q, aux)`` accept
    ``aux`` as the gripper opening in ``[0, 1]``: either a tensor broadcastable
    to ``(B, T, 1)`` (``(B, T)`` and ``(B, T, 1)`` both accepted), or ``None``
    for a fully closed gripper (matching ``FrankaFK.ee_pose``'s convention,
    since a closed gripper does not affect any arm-link pose).
    """

    #: Registry key (kinescore.robots.get_robot).
    name = "franka_panda"
    #: Arm DOF that ``q`` carries; the gripper travels through ``aux``, not ``q``.
    n_joints = 7

    def __init__(self, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32) -> None:
        self.fk = FrankaFK(device=device, dtype=dtype)
        self.keypoint_links: tuple[str, ...] = self.fk.keypoint_links

        # ---- precomputed chain-index buffers (see class docstring, point 3) --
        chain_names = self.fk.chain.get_joint_parameter_names()
        self._arm_chain_idx = torch.tensor(
            [chain_names.index(n) for n in PANDA_ARM_JOINTS], dtype=torch.long)
        self._finger_chain_idx = torch.tensor(
            [chain_names.index(n) for n in PANDA_FINGER_JOINTS
             if n in chain_names], dtype=torch.long)

        # ---- joint limits (arm only; the gripper has no RobotSpec limit slot) -
        self.q_lo = self.fk.joint_limits[:, 0].clone()
        self.q_hi = self.fk.joint_limits[:, 1].clone()
        self.vel_limits: torch.Tensor | None = self.fk.joint_vel_limits.clone()
        self.effort_limits: torch.Tensor | None = self.fk.joint_effort_limits.clone()

        # ---- bones: full set + rigid-only subset ----------------------------
        self.bone_pairs = self.fk.bone_pairs.clone()
        self.bone_lengths = self.fk.bone_lengths.clone()
        mask = self._rigid_bone_mask()
        warn_dropped_bones("FrankaSpec", self.keypoint_links, self.bone_pairs,
                           self.bone_lengths, mask)
        self.rigid_bone_pairs = self.bone_pairs[mask]
        self.rigid_bone_lengths = self.bone_lengths[mask]

        self.capabilities: frozenset[str] = frozenset(
            {Capability.ROTATIONS, Capability.EFFORT_LIMITS})

        urdf_path = resolve_robot_description_urdf("panda_description")
        self.urdf_sha256: str | None = sha256_file(urdf_path)

    def _rigid_bone_mask(self) -> torch.Tensor:
        """Select bones that actually measure arm rigidity.

        Delegates to :func:`kinescore.robots.base.structural_rigid_bone_mask`
        -- see that function's docstring for the two-rule pattern (D9). On the
        Panda, rule 1 alone drops bones 4, 5 and 6 and rule 2 alone would drop
        only bone 5 -- the intersection is what makes the exclusion principled
        rather than a threshold tuned to one robot.
        """
        return structural_rigid_bone_mask(
            self.keypoint_links, self.bone_pairs, self.bone_lengths,
            ACTUATED_LINKS, min_length_m=RIGID_BONE_MIN_M)

    # ------------------------------------------------------------------ #
    # RobotSpec protocol
    # ------------------------------------------------------------------ #
    def _gripper_bt1(self, q: torch.Tensor, aux: Any | None) -> torch.Tensor:
        """Normalise ``aux`` to a ``(B, T, 1)`` gripper-opening tensor."""
        b, t = q.shape[0], q.shape[1]
        if aux is None:
            return q.new_zeros(b, t, 1)
        gripper = aux
        if gripper.ndim == 2:
            gripper = gripper.unsqueeze(-1)
        if gripper.shape[:2] != (b, t) or gripper.shape[-1] != 1:
            raise ValueError(
                f"aux (gripper) must broadcast to (B,T,1) = ({b},{t},1); got "
                f"{tuple(gripper.shape)}")
        return gripper

    def _theta(self, q: torch.Tensor, gripper: torch.Tensor) -> torch.Tensor:
        """Assemble the ``(B*T, n_chain_joints)`` FK input via index-buffer scatter.

        Numerically equivalent to ``FrankaFK._joint_tensor`` (same values land
        in the same chain-parameter slots); the difference is purely how the
        slots are found -- a one-time-computed index buffer here versus a
        ``list.index`` call per joint per invocation there. See the class
        docstring's point 3.
        """
        b, t = q.shape[0], q.shape[1]
        if q.shape[-1] != self.n_joints:
            raise ValueError(
                f"Expected q of shape (B,T,{self.n_joints}), got {tuple(q.shape)}")

        # Device-following: `self.fk.chain` is a plain attribute, not an
        # nn.Module registered submodule or buffer, so `FrankaFK.to(device)`
        # never relocates it (see `FrankaFK._ensure_device`'s docstring). We
        # call that exact method -- rather than reimplement the same
        # device-move logic here -- so FrankaSpec and the wrapped FrankaFK can
        # never disagree about which device the chain lives on.
        self.fk._ensure_device(q.device)
        q = q.to(device=self.fk.device, dtype=self.fk.dtype)
        gripper = gripper.to(device=self.fk.device, dtype=self.fk.dtype)

        q_flat = q.reshape(b * t, self.n_joints)
        finger = gripper.reshape(b * t, 1) * PANDA_FINGER_MAX

        arm_idx = self._arm_chain_idx.to(q.device)
        finger_idx = self._finger_chain_idx.to(q.device)

        th = q_flat.new_zeros(b * t, self.fk.chain.n_joints)
        th[:, arm_idx] = q_flat
        if finger_idx.numel():
            th[:, finger_idx] = finger.expand(-1, finger_idx.numel())
        return th

    def forward_kinematics(self, q: torch.Tensor,
                           aux: Any | None = None) -> torch.Tensor:
        """``(B,T,7) -> (B,T,K,3)`` keypoint positions; see ``aux`` contract above."""
        b, t = q.shape[0], q.shape[1]
        gripper = self._gripper_bt1(q, aux)
        with torch.autocast(device_type=q.device.type, enabled=False):
            th = self._theta(q, gripper)
            transforms = self.fk.chain.forward_kinematics(th)
            cols = [transforms[name].get_matrix()[:, :3, 3]
                    for name in self.keypoint_links]
            P = torch.stack(cols, dim=1)
        return P.reshape(b, t, len(self.keypoint_links), 3)

    def forward_transforms(self, q: torch.Tensor, aux: Any | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B,T,7) -> P (B,T,K,3), R (B,T,K,3,3)``; see ``aux`` contract above."""
        b, t = q.shape[0], q.shape[1]
        gripper = self._gripper_bt1(q, aux)
        with torch.autocast(device_type=q.device.type, enabled=False):
            th = self._theta(q, gripper)
            transforms = self.fk.chain.forward_kinematics(th)
            mats = [transforms[name].get_matrix() for name in self.keypoint_links]
            M = torch.stack(mats, dim=1)
        k = len(self.keypoint_links)
        P = M[..., :3, 3].reshape(b, t, k, 3)
        R = M[..., :3, :3].reshape(b, t, k, 3, 3)
        return P, R

    def ee_sites(self) -> tuple[int, ...]:
        """Single end-effector site: ``panda_hand_tcp``'s index in ``keypoint_links``."""
        return (self.keypoint_links.index("panda_hand_tcp"),)
