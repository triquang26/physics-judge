"""Differentiable Airbot MMK2 forward kinematics: two 6-DOF arm chains.

Wraps a ``pytorch_kinematics`` chain built from
``airbot_mmk2_bimanual_arms.urdf`` (see that file's own header comment, or
``KINESCORE_ASSETS/airbot_mmk2/urdf/MANIFEST.json``, for how it was composed
from two upstream DISCOVERSE sources -- the arm's own URDF plus the MMK2
torso's left/right mount transforms). Structurally this mirrors
:class:`~kinescore.robots.gr1.fk.GR1FK` (two arm chains, one predicted-joint
vector, per-side keypoint extraction) but is simpler: no shared waist root
(the two arms mount independently on a fixed ``torso`` link), and no
finger/hand FK at all -- see ``constants.py``'s module docstring for why the
hand is out of scope entirely, not merely unpredicted.

Predicted state is the full URDF DOF
-------------------------------------
Unlike GR1FK (17 of the URDF's 70 joints), every joint this chain has
(``left_arm_joint_1..6``, ``right_arm_joint_1..6`` -- 12 total) is a
predicted joint. There are no unpredicted branches sitting at rest pose here,
because the composite URDF itself only contains the two arm chains (see
``build_airbot_mmk2_urdf.py``); the ``torso`` root is a fixed anchor, not a
joint.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from kinescore.robots.airbot_mmk2.constants import (
    EE_LINK,
    KEYPOINTS_LEFT,
    KEYPOINTS_RIGHT,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
)
from kinescore.robots.base import (
    assert_keypoints_in_urdf,
    build_pred_chain_index,
    consecutive_bone_pairs,
    read_joint_limit_arrays,
    rest_pose_bone_lengths,
    scatter_predicted,
)

__all__ = ["AirbotMMK2FK"]


class AirbotMMK2FK(nn.Module):
    """Differentiable bimanual Airbot MMK2 arm forward kinematics.

    Parameters
    ----------
    urdf_path:
        Path to ``airbot_mmk2_bimanual_arms.urdf``.
    device, dtype:
        Where the chain / buffers live and the float dtype.

    Buffers
    -------
    q_lo, q_hi: ``(12,)``
        Per predicted-joint position limits (radians), in canonical order
        ``[left_arm(6), right_arm(6)]``, read straight from the URDF
        ``<limit>`` tags -- **no margin widening** (unlike GR1's
        ``_LIMIT_MARGIN``): there is no teleop-exceeds-URDF-limit evidence for
        this robot yet, so widening here would be an unjustified guess.
    q_vel_max, q_effort_max: ``(12,)``
        Per predicted-joint rated velocity (rad/s) / effort (N.m) from the
        same ``<limit>`` tags. Unlike GR1 (no effort data at all) and Franka
        (effort ported from a different upstream), this URDF's ``<limit
        effort="...">`` values come from the same DISCOVERSE
        ``airbot_play_v3_gripper_fixed.urdf`` source as the position limits,
        so :class:`~kinescore.robots.airbot_mmk2.spec.AirbotMMK2Spec` can
        honestly declare ``Capability.EFFORT_LIMITS``.
    bone_pairs_left / _right: ``(5, 2)`` long; bone_lengths_*: ``(5,)``
        Rest-pose consecutive-keypoint geometry per arm.

    Shapes
    ------
    Input  ``q12``: ``(B, T, 12)`` = ``[left_arm 0:6, right_arm 6:12]``.
    Output ``P``  : ``(B, T, K, 3)`` per-arm keypoint xyz in the ``torso`` frame.
    """

    N_LEFT, N_RIGHT = 6, 6
    N_Q = 12

    def __init__(self, urdf_path: str, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        import pytorch_kinematics as pk  # local import -- see robots/__init__.py

        self.device = torch.device(device)
        self.dtype = dtype
        self.urdf_path = str(urdf_path)

        with open(urdf_path, "rb") as f:
            chain = pk.build_chain_from_urdf(f.read())
        self.chain = chain.to(dtype=self.dtype, device=self.device)
        self._chain_joint_names: list[str] = list(self.chain.get_joint_parameter_names())
        self.n_joints = self.chain.n_joints
        if self.n_joints != self.N_Q:
            raise ValueError(
                f"expected {self.N_Q} joints in {urdf_path!r}, found "
                f"{self.n_joints}: {self._chain_joint_names}")

        avail = set(self.chain.get_frame_names(exclude_fixed=False))
        self.keypoints = {"left": KEYPOINTS_LEFT, "right": KEYPOINTS_RIGHT}
        assert_keypoints_in_urdf(avail, self.keypoints, ee_link=EE_LINK)
        self.num_keypoints = len(KEYPOINTS_LEFT)

        # canonical predicted-joint order and its chain-index positions
        self.pred_joints: tuple[str, ...] = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
        self._pred_chain_idx = build_pred_chain_index(
            self._chain_joint_names, self.pred_joints)
        self.register_buffer("pred_chain_idx", self._pred_chain_idx)

        lo, hi, vel, eff = read_joint_limit_arrays(urdf_path, self.pred_joints)
        self.register_buffer("q_lo", torch.tensor(lo, dtype=self.dtype))
        self.register_buffer("q_hi", torch.tensor(hi, dtype=self.dtype))
        self.register_buffer("q_vel_max", torch.tensor(vel, dtype=self.dtype))
        self.register_buffer("q_effort_max", torch.tensor(eff, dtype=self.dtype))

        for side in ("left", "right"):
            bp, bl = self._compute_rest_bones(side)
            self.register_buffer(f"bone_pairs_{side}", bp)
            self.register_buffer(f"bone_lengths_{side}", bl)

    # ── construction helpers ─────────────────────────────────────────────────
    def _full_theta(self, q12: torch.Tensor) -> torch.Tensor:
        """Scatter the 12 predicted joints into the ``(N, n_joints)`` chain input.

        Trivial here (``n_joints == N_Q == 12``, no unpredicted branches -- see
        module docstring) but kept as an explicit scatter rather than a bare
        reshape so a future URDF revision that adds a branch (e.g. a verified
        hand chain) does not silently break joint-order assumptions.
        """
        if q12.ndim != 3 or q12.shape[-1] != self.N_Q:
            raise ValueError(f"expected q of shape (B,T,{self.N_Q}), got {tuple(q12.shape)}")
        self._ensure_device(q12.device)
        q = q12.to(device=self.device, dtype=self.dtype)
        b, t = q.shape[0], q.shape[1]
        q_flat = q.reshape(b * t, self.N_Q)
        return scatter_predicted(q_flat, self.pred_chain_idx, self.n_joints)

    def _ensure_device(self, device: torch.device) -> None:
        device = torch.device(device)
        if device != self.device:
            self.chain = self.chain.to(device=device)
            self.device = device

    def _compute_rest_bones(self, side: str) -> tuple[torch.Tensor, torch.Tensor]:
        pairs = consecutive_bone_pairs(self.num_keypoints)
        with torch.no_grad():
            q0 = torch.zeros(1, 1, self.N_Q, dtype=self.dtype, device=self.device)
            p0 = self.keypoints_fk(q0, side)[0, 0]  # (K,3)
        return pairs, rest_pose_bone_lengths(p0, pairs)

    # ── forward kinematics ───────────────────────────────────────────────────
    def _stack(self, transforms: dict[str, object], links: Sequence[str], n: int,
               want_rot: bool = False):
        mats = [transforms[name].get_matrix() for name in links]  # K×(n,4,4)
        M = torch.stack(mats, dim=1)                              # (n,K,4,4)
        P = M[..., :3, 3]
        return (P, M[..., :3, :3]) if want_rot else P

    def keypoints_fk(self, q12: torch.Tensor, side: str) -> torch.Tensor:
        """``(B,T,12) -> (B,T,K,3)`` per-arm keypoint positions in the torso frame."""
        b, t = q12.shape[0], q12.shape[1]
        with torch.autocast(device_type=q12.device.type, enabled=False):
            th = self._full_theta(q12)
            tf = self.chain.forward_kinematics(th)
            P = self._stack(tf, self.keypoints[side], b * t)
        return P.reshape(b, t, self.num_keypoints, 3)

    def forward_transforms(self, q12: torch.Tensor, side: str
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B,T,12) -> P (B,T,K,3), R (B,T,K,3,3)`` (positions + per-link rotations)."""
        b, t = q12.shape[0], q12.shape[1]
        with torch.autocast(device_type=q12.device.type, enabled=False):
            th = self._full_theta(q12)
            tf = self.chain.forward_kinematics(th)
            P, R = self._stack(tf, self.keypoints[side], b * t, want_rot=True)
        P = P.reshape(b, t, self.num_keypoints, 3)
        R = R.reshape(b, t, self.num_keypoints, 3, 3)
        return P, R

    @property
    def joint_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q_lo, self.q_hi
