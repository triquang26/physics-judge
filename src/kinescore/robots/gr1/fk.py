"""Differentiable Fourier **GR-1** forward kinematics for the bimanual pixel judge.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

__all__ = ["GR1FK", "LEFT_ARM_JOINTS", "RIGHT_ARM_JOINTS", "WAIST_JOINTS",
           "LEFT_HAND_DIMS", "RIGHT_HAND_DIMS"]


# ── URDF joint names, in the dataset's 44-dim sub-block order ──
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

#: Where each hand's six actuator values sit in the predicted joint vector.
LEFT_HAND_DIMS = tuple(range(17, 23))
RIGHT_HAND_DIMS = tuple(range(23, 29))

#: Fingertip links appended to each arm's keypoints, thumb first.
FINGERTIPS_LEFT: tuple[str, ...] = tuple(
    f"L_{f}_tip_link" for f in ("thumb", "index", "middle", "ring", "pinky"))
FINGERTIPS_RIGHT: tuple[str, ...] = tuple(
    f"R_{f}_tip_link" for f in ("thumb", "index", "middle", "ring", "pinky"))

# The corpus logs six actuator values per hand, each a non-negative flexion
# amount; the URDF has eleven finger joints per hand, so an actuator drives more
# than one. Each entry is ``(urdf_joint, actuator_dim, sign)`` and the joint takes
# ``sign * hand6[dim]``, clipped to the URDF limit.
#
# The dimension order is the one the logged values fit: assigning dim 5 to the
# thumb pitch puts 23.4% of 31k measured frames past that joint's 1.159 rad limit,
# while the order below leaves every dimension inside its joint's range.
_FINGERS = (("thumb_proximal_yaw", 0, -1.0), ("thumb_proximal_pitch", 1, +1.0),
            ("thumb_distal", 1, +1.0), ("index_proximal", 2, -1.0),
            ("index_intermediate", 2, -1.0), ("middle_proximal", 3, -1.0),
            ("middle_intermediate", 3, -1.0), ("ring_proximal", 4, -1.0),
            ("ring_intermediate", 4, -1.0), ("pinky_proximal", 5, -1.0),
            ("pinky_intermediate", 5, -1.0))
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
# (e.g. right_wrist_pitch); the margin widens the published [q_lo, q_hi] so a
# hair-over-limit teleop frame is not flagged.
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
    q_lo, q_hi: ``(29,)``
        Per predicted-joint limits in the canonical order
        ``[left_arm(7), right_arm(7), waist(3), left_hand(6), right_hand(6)]`` (radians), read from the URDF
        and widened by :data:`_LIMIT_MARGIN`.
    q_vel_max: ``(29,)``
        Per predicted-joint max joint velocity (rad/s) from the URDF ``<limit
        velocity="…">`` attribute, same canonical order (no margin).
    bone_pairs_left / _right: ``(K-1, 2)`` long; bone_lengths_*: ``(K-1,)``
        Rest-pose rigid geometry per arm, for the rigidity residual.

    Shapes
    ------
    Input  ``q``     : ``(B, T, 29)``  = [arms+waist 0:17, left_hand 17:23, right_hand 23:29].
    Output ``P``       : ``(B, T, K, 3)`` per-arm keypoint xyz in the base frame.
    """

    N_LEFT, N_RIGHT, N_WAIST, N_HAND = 7, 7, 3, 6
    N_Q = 29  # left_arm + right_arm + waist + both hands

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
        self.keypoints = {"left": KEYPOINTS_LEFT + FINGERTIPS_LEFT,
                          "right": KEYPOINTS_RIGHT + FINGERTIPS_RIGHT}
        for side, kps in self.keypoints.items():
            missing = [k for k in kps if k not in avail]
            if missing:
                raise ValueError(f"{side} keypoint links absent from URDF: {missing}")
            if EE_LINK[side] not in avail:
                raise ValueError(f"EE link {EE_LINK[side]} absent from URDF")
        self.num_keypoints = len(self.keypoints["left"])
        self.n_arm_keypoints = len(KEYPOINTS_LEFT)

        # canonical predicted-joint order and its chain-index positions
        self.arm_joints: tuple[str, ...] = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + WAIST_JOINTS
        hand_actuators = tuple(
            f"{p}_{n}_joint" for p, side in (("L", "left"), ("R", "right"))
            for n in ("thumb_proximal_yaw", "thumb_proximal_pitch", "index_proximal",
                      "middle_proximal", "ring_proximal", "pinky_proximal"))
        self.pred_joints: tuple[str, ...] = self.arm_joints + hand_actuators
        self._pred_chain_idx = torch.tensor(
            [self._chain_joint_names.index(n) for n in self.arm_joints], dtype=torch.long)
        self.register_buffer("pred_chain_idx", self._pred_chain_idx)
        self._finger_idx, self._finger_dim, self._finger_sign, finger_lo, finger_hi = \
            self._finger_tables(urdf_path)

        # joint limits (29,) from URDF, widened by margin; per-joint velocity caps (rad/s)
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
    def _finger_tables(self, urdf_path: str):
        """Index buffers turning the twelve actuator values into finger joint angles.

        Returns the chain positions of the eleven finger joints per hand, which
        actuator dimension drives each, the sign it enters with, and the URDF
        limits the result is clipped to. Building them once keeps the forward
        pass a scatter rather than a per-joint lookup.
        """
        import xml.etree.ElementTree as ET
        root = ET.parse(urdf_path).getroot()
        limits = {j.get("name"): (float(j.find("limit").get("lower")),
                                  float(j.find("limit").get("upper")))
                  for j in root.findall("joint")
                  if j.find("limit") is not None and j.get("type") == "revolute"}
        idx, dim, sign, lo, hi = [], [], [], [], []
        for base, dims in (("left", LEFT_HAND_DIMS), ("right", RIGHT_HAND_DIMS)):
            for name, d, s in FINGER_JOINTS[base]:
                idx.append(self._chain_joint_names.index(name))
                dim.append(dims[d]); sign.append(s)
                lo.append(limits[name][0]); hi.append(limits[name][1])
        as_long = lambda v: torch.tensor(v, dtype=torch.long)
        as_f = lambda v: torch.tensor(v, dtype=self.dtype)
        self.register_buffer("finger_lo", as_f(lo))
        self.register_buffer("finger_hi", as_f(hi))
        return as_long(idx), as_long(dim), as_f(sign), as_f(lo), as_f(hi)

    @staticmethod
    def _read_urdf_limits(urdf_path: str, joints: Sequence[str]) -> tuple[list, list, list]:
        """Return ``(lower, upper, velocity)`` per joint from the URDF ``<limit>`` tags.
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

    def _full_theta(self, q: torch.Tensor) -> torch.Tensor:
        """Scatter the 29 predicted joints into the full ``(N, n_joints)`` chain input.

        The seventeen arm and waist values go straight to their chain slots; the
        twelve actuator values fan out to the twenty-two finger joints through the
        coupling table, clipped to each joint's URDF range.
        """
        if q.ndim != 3 or q.shape[-1] != self.N_Q:
            raise ValueError(f"expected q of shape (B,T,{self.N_Q}), got {tuple(q.shape)}")
        self._ensure_device(q.device)
        q = q.to(device=self.device, dtype=self.dtype)
        b, t = q.shape[0], q.shape[1]
        q_flat = q.reshape(b * t, self.N_Q)
        th = q_flat.new_zeros(b * t, self.n_joints)
        th[:, self.pred_chain_idx.to(th.device)] = q_flat[:, :len(self.arm_joints)]
        dev = th.device
        driven = q_flat[:, self._finger_dim.to(dev)] * self._finger_sign.to(dev)
        th[:, self._finger_idx.to(dev)] = driven.clamp(
            self.finger_lo.to(dev), self.finger_hi.to(dev))
        return th

    def _ensure_device(self, device: torch.device) -> None:
        device = torch.device(device)
        if device != self.device:
            self.chain = self.chain.to(device=device)
            self.device = device

    def _compute_rest_bones(self, side: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Rest geometry of the arm chain only.

        Fingertips are keypoints so that speed and smoothness are measured on the
        fingers, but no bone ends on one: the distance from a fingertip to
        anything else changes as the hand opens, so such a bone would report
        actuation as deformation.
        """
        k = self.n_arm_keypoints
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

    def keypoints_fk(self, q: torch.Tensor, side: str) -> torch.Tensor:
        """``(B,T,29) -> (B,T,K,3)`` per-arm keypoint positions in base frame."""
        b, t = q.shape[0], q.shape[1]
        with torch.autocast(device_type=q.device.type, enabled=False):
            th = self._full_theta(q)
            tf = self.chain.forward_kinematics(th)
            P = self._stack(tf, self.keypoints[side], b * t)
        return P.reshape(b, t, self.num_keypoints, 3)

    def link_frames(self, q: torch.Tensor, link_names: Sequence[str]) -> torch.Tensor:
        """``(B,T,29) -> (B,T,L,4,4)`` full SE(3) of an arbitrary set of links.
        """
        b, t = q.shape[0], q.shape[1]
        with torch.autocast(device_type=q.device.type, enabled=False):
            th = self._full_theta(q)
            tf = self.chain.forward_kinematics(th)
            mats = [tf[name].get_matrix() for name in link_names]   # L×(B*T,4,4)
            M = torch.stack(mats, dim=1)                            # (B*T,L,4,4)
        return M.reshape(b, t, len(link_names), 4, 4)

    def forward_transforms(self, q: torch.Tensor, side: str
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B,T,29) -> P (B,T,K,3), R (B,T,K,3,3)`` (positions + per-link rotations)."""
        b, t = q.shape[0], q.shape[1]
        with torch.autocast(device_type=q.device.type, enabled=False):
            th = self._full_theta(q)
            tf = self.chain.forward_kinematics(th)
            P, R = self._stack(tf, self.keypoints[side], b * t, want_rot=True)
        P = P.reshape(b, t, self.num_keypoints, 3)
        R = R.reshape(b, t, self.num_keypoints, 3, 3)
        return P, R

    def ee_pose(self, q: torch.Tensor, side: str
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B,T,29) -> pos (B,T,3), rotvec (B,T,3)`` for the arm's EE link."""
        b, t = q.shape[0], q.shape[1]
        with torch.autocast(device_type=q.device.type, enabled=False):
            th = self._full_theta(q)
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
    def _theta_with_fingers(self, q: torch.Tensor, hand6: torch.Tensor, side: str):
        """Full chain theta with arms+waist (q) AND one side's fingers (hand6)."""
        th = self._full_theta(q)                              # (B*T, n_joints)
        b, t = q.shape[0], q.shape[1]
        h = hand6.to(self.device, self.dtype).reshape(b * t, 6)
        names = self._chain_joint_names
        for jname, dim, sign in FINGER_JOINTS[side]:
            if jname in names:
                th[:, names.index(jname)] = sign * h[:, dim]
        return th

    def fingers_fk(self, q: torch.Tensor, hand6: torch.Tensor, side: str):
        """Render points for one hand: wrist anchor + per-finger (knuckle, tip).
        """
        b, t = q.shape[0], q.shape[1]
        links = [WRIST_LINK[side]]
        for knuckle, tip in FINGER_RENDER[side]:
            links += [knuckle, tip]
        with torch.autocast(device_type=q.device.type, enabled=False):
            th = self._theta_with_fingers(q, hand6, side)
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

    # ── helpers to slice the 44-dim state into the canonical q / grippers ───
    @staticmethod
    def state_to_q17(state44: torch.Tensor) -> torch.Tensor:
        """``(...,44) -> (...,17)`` = [left_arm 0:7, right_arm 22:29, waist 41:44]."""
        return torch.cat([state44[..., 0:7], state44[..., 22:29], state44[..., 41:44]], dim=-1)

    @staticmethod
    def hand_flexion_mean(state44: torch.Tensor) -> torch.Tensor:
        """``(...,44) -> (...,2)`` left/right mean raw hand-motor flexion."""
        lh = state44[..., 7:13].mean(-1, keepdim=True)
        rh = state44[..., 29:35].mean(-1, keepdim=True)
        g = torch.cat([lh, rh], dim=-1)
        return g
