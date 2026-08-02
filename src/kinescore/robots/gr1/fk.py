"""Differentiable Fourier **GR-1** forward kinematics for the bimanual pixel judge.

Mirrors :class:`models.kinematics.fk.FrankaFK`, but for the GR-1 humanoid: two arm
chains sharing the **waist** as their common root, so the per-frame readout pose
``q`` is turned into 3-D keypoint positions in the robot **base frame** (the head
camera moves — neck ``[19:22]`` — so we never work in camera frame for the
physics/readout path; see the judge docs).

Joint layout (verified in Phase 0 against ``meta/modality.json`` and the URDF
joint limits, ``scripts/gr1/00_eda.py`` / ``01_validate_fk.py``)::

    left_arm   state[0:7]    right_arm  state[22:29]    waist  state[41:44]

The canonical **predicted** vector this module consumes is the 17-D concatenation
``q17 = [left_arm(7), right_arm(7), waist(3)]`` plus a 2-D gripper ``[left,right]``
in ``[0,1]`` (the Fourier 6-DoF hand is treated as an open/close proxy — no finger
physics; the arm chains end at the wrist/hand). Joint limits are read straight
from the URDF and published as ``q_lo``/``q_hi`` for the reader's raw (unsquashed)
prediction to be checked against post-hoc -- see ``limit_semantics="raw_rad"``
in ``kinescore.core.reader`` for why no squash sits in the prediction path itself.

Frozen, analytic, fully differentiable (``pytorch_kinematics``). No trainable
weights.

Provenance
----------
Ported from ``models/kinematics/gr1_fk.py`` (Marionette-fkjepa). Every method
body below is unchanged from the source **except**
:meth:`GR1FK.hand_flexion_mean`, which is the source's ``state_to_gripper``
renamed (identical implementation, zero numeric change) -- see that method's
docstring for why the rename was required rather than merely nice-to-have.
``kinescore.robots.gr1.spec.GR1Spec`` is the new code that wraps this class
(and :class:`~kinescore.robots.gr1.colliders.RobotColliders`) into a
``RobotSpec``; nothing in *this* file references ``kinescore`` at all, which is
what "ported verbatim" is meant to guarantee -- an independent reader can diff
this file against the source and see nothing but the rename and this docstring.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

__all__ = ["GR1FK", "LEFT_ARM_JOINTS", "RIGHT_ARM_JOINTS", "WAIST_JOINTS"]


# ── URDF joint names, in the dataset's 44-dim sub-block order (Phase-0 verified) ──
LEFT_ARM_JOINTS: tuple[str, ...] = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_wrist_yaw_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
)
RIGHT_ARM_JOINTS: tuple[str, ...] = (
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_wrist_yaw_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
)
WAIST_JOINTS: tuple[str, ...] = ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")

# Per-arm keypoint links (shoulder -> upper arm -> elbow -> wrist -> hand -> EE), K=6.
KEYPOINTS_LEFT: tuple[str, ...] = (
    "left_upper_arm_pitch_link", "left_upper_arm_yaw_link", "left_lower_arm_pitch_link",
    "left_hand_yaw_link", "left_hand_pitch_link", "left_end_effector_link",
)
KEYPOINTS_RIGHT: tuple[str, ...] = (
    "right_upper_arm_pitch_link", "right_upper_arm_yaw_link", "right_lower_arm_pitch_link",
    "right_hand_yaw_link", "right_hand_pitch_link", "right_end_effector_link",
)
EE_LINK = {"left": "left_end_effector_link", "right": "right_end_effector_link"}

# Approximate Fourier 6-DoF hand coupling: each (urdf_joint, hand_dim, sign) sets
# joint = clip(sign * hand6[dim], URDF limits). hand6 in [0,~1.5] = flexion amount.
# 6 motors -> 11 finger joints: dim0-3 = index/middle/ring/pinky (drive proximal +
# intermediate of that finger), dim4 = thumb yaw/opposition, dim5 = thumb pitch+distal.
# NOT the true (unpublished) coupling — a visual approximation only.
_FINGERS = (("thumb_proximal_yaw", 4, -1.0), ("thumb_proximal_pitch", 5, +1.0),
            ("thumb_distal", 5, +1.0), ("index_proximal", 0, -1.0),
            ("index_intermediate", 0, -1.0), ("middle_proximal", 1, -1.0),
            ("middle_intermediate", 1, -1.0), ("ring_proximal", 2, -1.0),
            ("ring_intermediate", 2, -1.0), ("pinky_proximal", 3, -1.0),
            ("pinky_intermediate", 3, -1.0))
FINGER_JOINTS = {
    "left":  tuple((f"L_{n}_joint", d, s) for n, d, s in _FINGERS),
    "right": tuple((f"R_{n}_joint", d, s) for n, d, s in _FINGERS),
}
# Render points per finger: (knuckle proximal link, fingertip link).
_FINGER_LINKS = (("thumb_proximal_yaw", "thumb_tip"), ("index_proximal", "index_tip"),
                 ("middle_proximal", "middle_tip"), ("ring_proximal", "ring_tip"),
                 ("pinky_proximal", "pinky_tip"))
FINGER_RENDER = {
    "left":  tuple((f"L_{a}_link", f"L_{b}_link") for a, b in _FINGER_LINKS),
    "right": tuple((f"R_{a}_link", f"R_{b}_link") for a, b in _FINGER_LINKS),
}
WRIST_LINK = {"left": "left_hand_pitch_link", "right": "right_hand_pitch_link"}

# Limits margin (rad): teleop occasionally exceeds the URDF limit slightly
# (e.g. right_wrist_pitch), so widen the published [q_lo, q_hi] clamp range a
# touch -- there is no squash to widen (raw_rad predictions are never forced
# inside limits); this only keeps clamp_for_fk/loss_limit from flagging a
# hair-over-limit teleop frame as a violation it isn't.
_LIMIT_MARGIN = 0.20


class GR1FK(nn.Module):
    """Differentiable GR-1 bimanual forward kinematics (two chains, shared waist root).

    Parameters
    ----------
    urdf_path:
        Path to the ``gr1t1_fourier_hand_6dof.urdf`` (or compatible) URDF.
    device, dtype:
        Where the chain / buffers live and the float dtype.

    Buffers
    -------
    q_lo, q_hi: ``(17,)``
        Per predicted-joint limits in the canonical order
        ``[left_arm(7), right_arm(7), waist(3)]`` (radians), read from the URDF
        and widened by :data:`_LIMIT_MARGIN`.
    q_vel_max: ``(17,)``
        Per predicted-joint max joint velocity (rad/s) from the URDF ``<limit
        velocity="…">`` attribute, same canonical order (no margin).
    bone_pairs_left / _right: ``(K-1, 2)`` long; bone_lengths_*: ``(K-1,)``
        Rest-pose rigid geometry per arm, for the rigidity residual.

    Shapes
    ------
    Input  ``q17``     : ``(B, T, 17)``  = [left_arm 0:7, right_arm 7:14, waist 14:17].
    Output ``P``       : ``(B, T, K, 3)`` per-arm keypoint xyz in the base frame.
    """

    N_LEFT, N_RIGHT, N_WAIST = 7, 7, 3
    N_Q = 17  # left_arm + right_arm + waist

    def __init__(self, urdf_path: str, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        import pytorch_kinematics as pk  # local import (heavy, like FrankaFK)

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
        for side, kps in self.keypoints.items():
            missing = [k for k in kps if k not in avail]
            if missing:
                raise ValueError(f"{side} keypoint links absent from URDF: {missing}")
            if EE_LINK[side] not in avail:
                raise ValueError(f"EE link {EE_LINK[side]} absent from URDF")
        self.num_keypoints = len(KEYPOINTS_LEFT)

        # canonical predicted-joint order and its chain-index positions
        self.pred_joints: tuple[str, ...] = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + WAIST_JOINTS
        self._pred_chain_idx = torch.tensor(
            [self._chain_joint_names.index(n) for n in self.pred_joints], dtype=torch.long)
        self.register_buffer("pred_chain_idx", self._pred_chain_idx)

        # joint limits (17,) from URDF, widened by margin; per-joint velocity caps (rad/s)
        lo, hi, vel = self._read_urdf_limits(urdf_path, self.pred_joints)
        self.register_buffer("q_lo", torch.tensor(lo, dtype=self.dtype) - _LIMIT_MARGIN)
        self.register_buffer("q_hi", torch.tensor(hi, dtype=self.dtype) + _LIMIT_MARGIN)
        self.register_buffer("q_vel_max", torch.tensor(vel, dtype=self.dtype))

        # rest-pose rigid bones per arm
        for side in ("left", "right"):
            bp, bl = self._compute_rest_bones(side)
            self.register_buffer(f"bone_pairs_{side}", bp)
            self.register_buffer(f"bone_lengths_{side}", bl)

    # ── construction helpers ─────────────────────────────────────────────────
    @staticmethod
    def _read_urdf_limits(urdf_path: str, joints: Sequence[str]) -> tuple[list, list, list]:
        """Return ``(lower, upper, velocity)`` per joint from the URDF ``<limit>`` tags.

        The ``velocity="…"`` attribute of the same ``<limit>`` element that carries
        the position limits gives the per-joint max joint velocity (rad/s); joints
        without one fall back to a large cap (never a false violation).
        """
        import xml.etree.ElementTree as ET
        root = ET.parse(urdf_path).getroot()
        lim, vlim = {}, {}
        for j in root.findall("joint"):
            limit_el = j.find("limit")
            if limit_el is not None and j.get("type") in ("revolute", "prismatic"):
                lim[j.get("name")] = (float(limit_el.get("lower")), float(limit_el.get("upper")))
                v = limit_el.get("velocity")
                vlim[j.get("name")] = float(v) if v is not None else 1.0e3
        lo = [lim[n][0] for n in joints]
        hi = [lim[n][1] for n in joints]
        vel = [vlim.get(n, 1.0e3) for n in joints]
        return lo, hi, vel

    def _full_theta(self, q17: torch.Tensor) -> torch.Tensor:
        """Scatter the 17 predicted joints into the full ``(N, n_joints)`` chain input.

        All non-predicted joints (legs, neck/head, fingers) stay at zero — they do
        not affect the arm keypoints (legs/head are separate branches; the arm
        chains end at the wrist before the fingers).
        """
        if q17.ndim != 3 or q17.shape[-1] != self.N_Q:
            raise ValueError(f"expected q of shape (B,T,{self.N_Q}), got {tuple(q17.shape)}")
        self._ensure_device(q17.device)
        q = q17.to(device=self.device, dtype=self.dtype)
        b, t = q.shape[0], q.shape[1]
        q_flat = q.reshape(b * t, self.N_Q)
        th = q_flat.new_zeros(b * t, self.n_joints)
        th[:, self.pred_chain_idx.to(th.device)] = q_flat
        return th

    def _ensure_device(self, device: torch.device) -> None:
        device = torch.device(device)
        if device != self.device:
            self.chain = self.chain.to(device=device)
            self.device = device

    def _compute_rest_bones(self, side: str) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.num_keypoints
        pairs = torch.tensor([[i, i + 1] for i in range(k - 1)], dtype=torch.long)
        with torch.no_grad():
            q0 = torch.zeros(1, 1, self.N_Q, dtype=self.dtype, device=self.device)
            p0 = self.keypoints_fk(q0, side)[0, 0]  # (K,3)
        diffs = p0[pairs[:, 1]] - p0[pairs[:, 0]]
        return pairs, diffs.norm(dim=-1).to(self.dtype).cpu()

    # ── forward kinematics ───────────────────────────────────────────────────
    def _stack(self, transforms: dict[str, object], links: Sequence[str], n: int,
               want_rot: bool = False):
        mats = [transforms[name].get_matrix() for name in links]  # K×(n,4,4)
        M = torch.stack(mats, dim=1)                              # (n,K,4,4)
        P = M[..., :3, 3]
        return (P, M[..., :3, :3]) if want_rot else P

    def keypoints_fk(self, q17: torch.Tensor, side: str) -> torch.Tensor:
        """``(B,T,17) -> (B,T,K,3)`` per-arm keypoint positions in base frame."""
        b, t = q17.shape[0], q17.shape[1]
        with torch.autocast(device_type=q17.device.type, enabled=False):
            th = self._full_theta(q17)
            tf = self.chain.forward_kinematics(th)
            P = self._stack(tf, self.keypoints[side], b * t)
        return P.reshape(b, t, self.num_keypoints, 3)

    # alias matching the plan's interface name
    def keypoints_of(self, q17: torch.Tensor, side: str) -> torch.Tensor:
        return self.keypoints_fk(q17, side)

    def link_frames(self, q17: torch.Tensor, link_names: Sequence[str]) -> torch.Tensor:
        """``(B,T,17) -> (B,T,L,4,4)`` full SE(3) of an arbitrary set of links.

        Base-frame homogeneous transforms for every name in ``link_names``, in that
        order. Used by the mechanical-feasibility checker to pose per-link collision
        spheres and inertial CoMs without touching the private ``_full_theta``.
        Differentiable, fp32 (autocast disabled), device-following — same contract as
        :meth:`keypoints_fk`.
        """
        b, t = q17.shape[0], q17.shape[1]
        with torch.autocast(device_type=q17.device.type, enabled=False):
            th = self._full_theta(q17)
            tf = self.chain.forward_kinematics(th)
            mats = [tf[name].get_matrix() for name in link_names]   # L×(B*T,4,4)
            M = torch.stack(mats, dim=1)                            # (B*T,L,4,4)
        return M.reshape(b, t, len(link_names), 4, 4)

    def forward_transforms(self, q17: torch.Tensor, side: str
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B,T,17) -> P (B,T,K,3), R (B,T,K,3,3)`` (positions + per-link rotations)."""
        b, t = q17.shape[0], q17.shape[1]
        with torch.autocast(device_type=q17.device.type, enabled=False):
            th = self._full_theta(q17)
            tf = self.chain.forward_kinematics(th)
            P, R = self._stack(tf, self.keypoints[side], b * t, want_rot=True)
        P = P.reshape(b, t, self.num_keypoints, 3)
        R = R.reshape(b, t, self.num_keypoints, 3, 3)
        return P, R

    def ee_pose(self, q17: torch.Tensor, side: str
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B,T,17) -> pos (B,T,3), rotvec (B,T,3)`` for the arm's EE link."""
        b, t = q17.shape[0], q17.shape[1]
        with torch.autocast(device_type=q17.device.type, enabled=False):
            th = self._full_theta(q17)
            tf = self.chain.forward_kinematics(th)
            mat = tf[EE_LINK[side]].get_matrix()                  # (B*T,4,4)
        pos = mat[:, :3, 3].reshape(b, t, 3)
        rotvec = self._matrix_to_rotvec(mat[:, :3, :3]).reshape(b, t, 3)
        return pos, rotvec

    @property
    def joint_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q_lo, self.q_hi

    # ── kinematic depth (for the observability-gate prior) ───────────────────
    def chain_depths(self):
        """Kinematic depth (# actuated joints from base) per predicted joint and keypoint.

        Distal links (wrist, hand) → large depth (near the ego camera, observable);
        proximal links (waist, shoulder) → small depth (occluded/global). Used to
        initialise the per-DoF observability gate γ.

        Returns ``(joint_depths[17], {side: kp_depths[K]})``.
        """
        import xml.etree.ElementTree as ET
        root = ET.parse(self.urdf_path).getroot()
        parent = {}                       # child_link -> (parent_link, actuated)
        child_of = {}                     # joint_name -> child_link
        for j in root.findall("joint"):
            cl = j.find("child").get("link")
            parent[cl] = (j.find("parent").get("link"),
                          j.get("type") in ("revolute", "prismatic"))
            child_of[j.get("name")] = cl

        def depth_of_link(link):
            d, cur = 0, link
            while cur in parent:
                par, act = parent[cur]
                d += int(act)
                cur = par
            return d

        jdepth = [depth_of_link(child_of[n]) for n in self.pred_joints]   # (17,)
        kpdepth = {s: [depth_of_link(kp) for kp in self.keypoints[s]]
                   for s in ("left", "right")}
        return jdepth, kpdepth

    # ── finger FK (approximate Fourier hand coupling, for visualisation) ──────
    def _theta_with_fingers(self, q17: torch.Tensor, hand6: torch.Tensor, side: str):
        """Full chain theta with arms+waist (q17) AND one side's fingers (hand6)."""
        th = self._full_theta(q17)                              # (B*T, n_joints)
        b, t = q17.shape[0], q17.shape[1]
        h = hand6.to(self.device, self.dtype).reshape(b * t, 6)
        names = self._chain_joint_names
        for jname, dim, sign in FINGER_JOINTS[side]:
            if jname in names:
                th[:, names.index(jname)] = sign * h[:, dim]
        return th

    def fingers_fk(self, q17: torch.Tensor, hand6: torch.Tensor, side: str):
        """Render points for one hand: wrist anchor + per-finger (knuckle, tip).

        Returns ``pts (B,T,1+2*5,3)`` ordered [wrist, (knuckle,tip)×5] and the bone
        list (wrist->knuckle, knuckle->tip per finger) as index pairs.
        """
        b, t = q17.shape[0], q17.shape[1]
        links = [WRIST_LINK[side]]
        for knuckle, tip in FINGER_RENDER[side]:
            links += [knuckle, tip]
        with torch.autocast(device_type=q17.device.type, enabled=False):
            th = self._theta_with_fingers(q17, hand6, side)
            tf = self.chain.forward_kinematics(th)
            P = self._stack(tf, links, b * t).reshape(b, t, len(links), 3)
        bones = []
        for f in range(5):
            k, tp = 1 + 2 * f, 2 + 2 * f
            bones += [(0, k), (k, tp)]                          # wrist->knuckle, knuckle->tip
        return P, bones

    # ── rotation utility (shared with FrankaFK) ──────────────────────────────
    @staticmethod
    def _matrix_to_rotvec(rot: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
        cos = ((trace - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps)
        angle = torch.acos(cos)
        rx = rot[:, 2, 1] - rot[:, 1, 2]
        ry = rot[:, 0, 2] - rot[:, 2, 0]
        rz = rot[:, 1, 0] - rot[:, 0, 1]
        axis2 = torch.stack([rx, ry, rz], dim=-1)
        sin = torch.sin(angle).clamp_min(eps)
        scale = angle / (2.0 * sin)
        return axis2 * scale.unsqueeze(-1)

    # ── helpers to slice the 44-dim state into the canonical q17 / grippers ───
    @staticmethod
    def state_to_q17(state44: torch.Tensor) -> torch.Tensor:
        """``(...,44) -> (...,17)`` = [left_arm 0:7, right_arm 22:29, waist 41:44]."""
        return torch.cat([state44[..., 0:7], state44[..., 22:29], state44[..., 41:44]], dim=-1)

    @staticmethod
    def hand_flexion_mean(state44: torch.Tensor) -> torch.Tensor:
        """``(...,44) -> (...,2)`` left/right mean raw hand-motor flexion.

        Renamed from the source's ``state_to_gripper``. Its docstring there
        claimed the output was "normalised to [0,1]"; the implementation below
        -- unchanged from the source -- is just ``.mean(-1)`` over the 6-dim
        Fourier hand-motor block, with **no normalisation applied**. The raw
        block is not bounded to ``[0,1]`` in general (see ``_FINGERS`` /
        ``FINGER_JOINTS`` in this module: ``hand6`` is documented there as
        living in ``[0, ~1.5]``), so a caller trusting the old docstring and
        the old name ("gripper", implying an open/close fraction like
        Franka's) would silently get values outside ``[0,1]``. The name now
        says exactly what is computed -- a mean flexion, in the hand's native
        units -- and callers that want an actual ``[0,1]`` proxy must
        normalise themselves against the block's known range.
        """
        lh = state44[..., 7:13].mean(-1, keepdim=True)
        rh = state44[..., 29:35].mean(-1, keepdim=True)
        g = torch.cat([lh, rh], dim=-1)
        return g
