"""Galaxea A1X end-effector rigid body, driven by the logged EE pose.

The A1X singleview corpus logs ``observation.state`` as
``(x, y, z, roll, pitch, yaw, gripper_width)`` -- an EE pose, not joint
angles (the values break ``a1x.urdf``'s one-signed limits on joints 2 and 3).
``q`` is the 6-DOF pose; keypoints are fixed offsets in the EE frame taken
from ``a1x.urdf`` (userguide-galaxea/URDF @ 2e5d31e), rigid by construction.
"""
from __future__ import annotations

from typing import Any

import torch

from kinescore.core.robot import Capability

__all__ = ["A1XEESpec", "EE_OFFSETS_M"]

EE_OFFSETS_M: tuple[tuple[float, float, float], ...] = (
    (-0.08165, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.03689, 0.013453, 0.0),
    (0.03689, -0.013453, 0.0),
)


class A1XEESpec:
    """``RobotSpec`` over ``q = (x, y, z, roll, pitch, yaw)``; ``aux`` ignored."""

    name = "a1x_ee"
    n_joints = 6
    keypoint_links: tuple[str, ...] = (
        "arm_link6", "gripper_link", "gripper_finger_link1",
        "gripper_finger_link2")

    def __init__(self, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32) -> None:
        self._offsets = torch.tensor(EE_OFFSETS_M, dtype=dtype, device=device)

        self.bone_pairs = torch.tensor(
            [[0, 1], [1, 2], [1, 3], [2, 3]], dtype=torch.long)
        diffs = (self._offsets[self.bone_pairs[:, 0]]
                 - self._offsets[self.bone_pairs[:, 1]])
        self.bone_lengths = diffs.norm(dim=-1).cpu()
        self.rigid_bone_pairs = self.bone_pairs.clone()
        self.rigid_bone_lengths = self.bone_lengths.clone()

        self.q_lo = torch.tensor(
            [-2.0, -2.0, -2.0, -torch.pi, -torch.pi, -torch.pi], dtype=dtype)
        self.q_hi = torch.tensor(
            [2.0, 2.0, 2.0, torch.pi, torch.pi, torch.pi], dtype=dtype)
        self.vel_limits: torch.Tensor | None = None
        self.effort_limits: torch.Tensor | None = None

        self.capabilities: frozenset[str] = frozenset({Capability.ROTATIONS})
        self.urdf_sha256: str | None = None

    def forward_kinematics(self, q: torch.Tensor,
                           aux: Any | None = None) -> torch.Tensor:
        P, _ = self.forward_transforms(q, aux)
        return P

    def forward_transforms(self, q: torch.Tensor, aux: Any | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.ndim != 3 or q.shape[-1] != self.n_joints:
            raise ValueError(
                f"Expected q of shape (B,T,{self.n_joints}), got {tuple(q.shape)}")
        pos = q[..., :3]
        rot = _euler_zyx(q[..., 3], q[..., 4], q[..., 5])
        offsets = self._offsets.to(device=q.device, dtype=q.dtype)
        P = pos.unsqueeze(2) + torch.einsum("btij,kj->btki", rot, offsets)
        R = rot.unsqueeze(2).expand(-1, -1, offsets.shape[0], -1, -1)
        return P, R.contiguous()

    def ee_sites(self) -> tuple[int, ...]:
        return (1,)


def _euler_zyx(roll: torch.Tensor, pitch: torch.Tensor,
               yaw: torch.Tensor) -> torch.Tensor:
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    row0 = torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                       dim=-1)
    row1 = torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                       dim=-1)
    row2 = torch.stack([-sp, cp * sr, cp * cr], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)
