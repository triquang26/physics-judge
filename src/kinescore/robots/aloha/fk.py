"""Differentiable ALOHA bimanual forward kinematics: two 6-DOF arms + grippers.

Wraps a ``pytorch_kinematics`` chain built from ``aloha_bimanual.urdf`` (two
Interbotix vx300s arms merged under a synthetic ``world`` root -- see
``KINESCORE_ASSETS/aloha/urdf/`` and ``legacy_docs/ADDING_ALOHA_NOTES.md`` for the
asset provenance). Structurally this mirrors
:class:`~kinescore.robots.airbot_mmk2.fk.AirbotMMK2FK` (two independent arm
chains, one predicted-joint vector, per-side keypoint extraction via
:mod:`kinescore.robots.base`) with one addition Airbot MMK2 has no use for:
a real 2-finger gripper per arm, handled through an ``aux`` channel exactly
the way :class:`~kinescore.robots.franka.fk.FrankaFK` handles the Panda's,
just doubled for two arms.

Predicted state: 12 arm joints; the gripper travels through ``aux``
-------------------------------------------------------------------------
``n_joints = 12`` = ``[left_arm(6), right_arm(6)]`` -- the six revolute
joints per side (``waist, shoulder, elbow, forearm_roll, wrist_angle,
wrist_rotate``; see ``constants.py``'s module docstring for the real-data
verification of this order). The 7th per-arm actuator (the gripper) is
**not** part of ``q``: like the Panda's two finger joints, it drives keypoint
links (``{side}/left_finger_link``, ``{side}/right_finger_link``) whose
position is not "arm rigidity" in the sense ``rigid_bone_pairs`` measures,
and folding a fundamentally different actuation channel (a linear gripper
open/close) into the same 12-vector as six rotational arm joints would make
``n_joints`` a worse description of "the pose reader's rotational-DOF output"
for no benefit -- exactly Franka's precedent (``FrankaSpec.n_joints = 7``,
gripper via ``aux``), doubled here for two arms rather than one.

``aux`` contract: ``gripper2`` is a per-side opening in ``[0, 1]``, shape
broadcastable to ``(B, T, 2)`` = ``[left, right]``. It is scattered onto BOTH
prismatic finger joints per side: ``{side}/left_finger`` (URDF range
``[0.021, 0.057]`` m, closed->open) and ``{side}/right_finger`` (the
elementwise negation, ``[-0.057, -0.021]`` m -- confirmed by reading the URDF,
not assumed: the two fingers slide as mirror images of each other, not
independent DoF). ``aux=None`` means both grippers fully closed
(``gripper2 = 0``), matching :class:`FrankaFK`'s closed-gripper default and
used as this class's own "rest pose" convention for
:func:`~kinescore.robots.base.rest_pose_bone_lengths`.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from kinescore.robots.aloha.constants import (
    EE_LINK,
    FINGER_HI_M,
    FINGER_LO_M,
    KEYPOINTS_LEFT,
    KEYPOINTS_RIGHT,
    LEFT_ARM_JOINTS,
    LEFT_FINGER_JOINT,
    RIGHT_ARM_JOINTS,
    RIGHT_FINGER_JOINT,
)
from kinescore.robots.base import (
    assert_keypoints_in_urdf,
    build_pred_chain_index,
    consecutive_bone_pairs,
    read_joint_limit_arrays,
    rest_pose_bone_lengths,
    scatter_predicted,
)

__all__ = ["AlohaFK"]


class AlohaFK(nn.Module):
    """Differentiable bimanual ALOHA arm+gripper forward kinematics.

    Parameters
    ----------
    urdf_path:
        Path to ``aloha_bimanual.urdf``.
    device, dtype:
        Where the chain / buffers live and the float dtype.

    Buffers
    -------
    q_lo, q_hi: ``(12,)``
        Per predicted-joint position limits (radians), canonical order
        ``[left_arm(6), right_arm(6)]``, read straight from the URDF
        ``<limit>`` tags -- no margin widening (no teleop-exceeds-URDF-limit
        evidence for this robot, same reasoning as
        :class:`~kinescore.robots.airbot_mmk2.fk.AirbotMMK2FK`).
    q_vel_max, q_effort_max: ``(12,)``
        Per predicted-joint rated velocity (rad/s) / effort (N.m), same
        ``<limit>`` tags -- every one of the 12 arm joints declares both in
        this URDF (verified), so
        :class:`~kinescore.robots.aloha.spec.AlohaSpec` can honestly declare
        ``Capability.EFFORT_LIMITS``.
    bone_pairs_left / _right: ``(8, 2)`` long; bone_lengths_*: ``(8,)``
        Rest-pose (both grippers closed) consecutive-keypoint geometry per arm.

    Shapes
    ------
    Input  ``q12``     : ``(B, T, 12)`` = ``[left_arm 0:6, right_arm 6:12]``.
    Input  ``gripper2``: ``(B, T, 2)``  = ``[left, right]`` opening in ``[0,1]``.
    Output ``P``       : ``(B, T, K, 3)`` per-arm keypoint xyz in the ``world`` frame.
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

        avail = set(self.chain.get_frame_names(exclude_fixed=False))
        self.keypoints = {"left": KEYPOINTS_LEFT, "right": KEYPOINTS_RIGHT}
        assert_keypoints_in_urdf(avail, self.keypoints, ee_link=EE_LINK)
        self.num_keypoints = len(KEYPOINTS_LEFT)

        # canonical predicted-joint order and its chain-index positions
        self.pred_joints: tuple[str, ...] = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
        self._pred_chain_idx = build_pred_chain_index(
            self._chain_joint_names, self.pred_joints)
        self.register_buffer("pred_chain_idx", self._pred_chain_idx)

        # gripper aux -> the 4 prismatic finger joints (2 per side)
        finger_joints = (
            LEFT_FINGER_JOINT["left"], RIGHT_FINGER_JOINT["left"],
            LEFT_FINGER_JOINT["right"], RIGHT_FINGER_JOINT["right"],
        )
        self._finger_chain_idx = build_pred_chain_index(
            self._chain_joint_names, finger_joints)
        self.register_buffer("finger_chain_idx", self._finger_chain_idx)

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
    def _full_theta(self, q12: torch.Tensor, gripper2: torch.Tensor) -> torch.Tensor:
        """Scatter the 12 predicted arm joints AND both grippers' fingers.

        ``q12`` lands at :attr:`pred_chain_idx` (arm joints); ``gripper2``
        (``[left, right]`` opening in ``[0,1]``) is affine-mapped to
        ``[FINGER_LO_M, FINGER_HI_M]`` and scattered onto each side's
        ``left_finger`` joint, with ``right_finger`` set to its negation (see
        module docstring). Every other chain joint (the cosmetic ``gripper``
        servo-horn continuous joint, the two fixed ``mount`` joints) is left
        at ``0.0`` -- unobserved and irrelevant to every keypoint this class
        exposes.
        """
        if q12.ndim != 3 or q12.shape[-1] != self.N_Q:
            raise ValueError(f"expected q of shape (B,T,{self.N_Q}), got {tuple(q12.shape)}")
        if gripper2.ndim != 3 or gripper2.shape[-1] != 2:
            raise ValueError(
                f"expected gripper of shape (B,T,2), got {tuple(gripper2.shape)}")
        self._ensure_device(q12.device)
        q = q12.to(device=self.device, dtype=self.dtype)
        grip = gripper2.to(device=self.device, dtype=self.dtype)
        b, t = q.shape[0], q.shape[1]
        q_flat = q.reshape(b * t, self.N_Q)
        th = scatter_predicted(q_flat, self.pred_chain_idx, self.n_joints)

        grip_flat = grip.reshape(b * t, 2)
        left_val = FINGER_LO_M + grip_flat[:, 0] * (FINGER_HI_M - FINGER_LO_M)
        right_val = FINGER_LO_M + grip_flat[:, 1] * (FINGER_HI_M - FINGER_LO_M)
        idx = self.finger_chain_idx.to(th.device)
        # order matches finger_joints above: (L.left_finger, L.right_finger,
        # R.left_finger, R.right_finger); *_right_finger mirrors *_left_finger.
        th[:, idx[0]] = left_val
        th[:, idx[1]] = -left_val
        th[:, idx[2]] = right_val
        th[:, idx[3]] = -right_val
        return th

    def _ensure_device(self, device: torch.device) -> None:
        device = torch.device(device)
        if device != self.device:
            self.chain = self.chain.to(device=device)
            self.device = device

    def _compute_rest_bones(self, side: str) -> tuple[torch.Tensor, torch.Tensor]:
        pairs = consecutive_bone_pairs(self.num_keypoints)
        with torch.no_grad():
            q0 = torch.zeros(1, 1, self.N_Q, dtype=self.dtype, device=self.device)
            g0 = torch.zeros(1, 1, 2, dtype=self.dtype, device=self.device)  # closed
            p0 = self.keypoints_fk(q0, g0, side)[0, 0]  # (K,3)
        return pairs, rest_pose_bone_lengths(p0, pairs)

    # ── forward kinematics ───────────────────────────────────────────────────
    def _stack(self, transforms: dict[str, object], links: Sequence[str], n: int,
               want_rot: bool = False):
        mats = [transforms[name].get_matrix() for name in links]  # K×(n,4,4)
        M = torch.stack(mats, dim=1)                              # (n,K,4,4)
        P = M[..., :3, 3]
        return (P, M[..., :3, :3]) if want_rot else P

    def keypoints_fk(self, q12: torch.Tensor, gripper2: torch.Tensor,
                     side: str) -> torch.Tensor:
        """``(B,T,12), (B,T,2) -> (B,T,K,3)`` per-arm keypoint positions."""
        b, t = q12.shape[0], q12.shape[1]
        with torch.autocast(device_type=q12.device.type, enabled=False):
            th = self._full_theta(q12, gripper2)
            tf = self.chain.forward_kinematics(th)
            P = self._stack(tf, self.keypoints[side], b * t)
        return P.reshape(b, t, self.num_keypoints, 3)

    def forward_transforms(self, q12: torch.Tensor, gripper2: torch.Tensor,
                           side: str) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B,T,12), (B,T,2) -> P (B,T,K,3), R (B,T,K,3,3)``."""
        b, t = q12.shape[0], q12.shape[1]
        with torch.autocast(device_type=q12.device.type, enabled=False):
            th = self._full_theta(q12, gripper2)
            tf = self.chain.forward_kinematics(th)
            P, R = self._stack(tf, self.keypoints[side], b * t, want_rot=True)
        P = P.reshape(b, t, self.num_keypoints, 3)
        R = R.reshape(b, t, self.num_keypoints, 3, 3)
        return P, R

    @property
    def joint_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q_lo, self.q_hi
