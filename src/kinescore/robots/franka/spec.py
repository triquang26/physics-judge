"""``RobotSpec`` for the Franka Panda: 7-DOF arm + 2-finger gripper on a table.

:class:`FrankaSpec` wraps the verbatim-ported :class:`~kinescore.robots.franka.fk.FrankaFK`
and adapts it to the frozen ``RobotSpec`` protocol (``kinescore.core.robot``).
It does not modify ``FrankaFK`` -- that module is a byte-for-byte port and stays
that way so it keeps producing the exact numbers the source benchmark did (see
its module docstring). Everything below is new code that sits *around* it.

Three design decisions this file makes, each fixing a defect the source had:

1. **Degenerate/actuation-only bones are dropped from ``rigid_bone_pairs``.**
   See :data:`kinescore.robots.franka.constants.RIGID_BONE_MIN_M` for the full
   reasoning -- in short, three of the seven consecutive-keypoint bones touch
   a finger link, whose position tracks the gripper's own prismatic joint
   rather than the rigid arm structure, so including them in a rigidity
   residual means "the gripper opened" reads as "the arm deformed".
2. **No ``SUPPORT_POLYGON``, no ``COLLIDERS`` capability.** A Panda is bolted to
   a table -- it has no feet and no balance margin to report, and this file
   ports no collision geometry. Declaring those capabilities anyway would let
   a metric compute a *number* for them, which is worse than ``NaN``: a
   balance metric that silently always reads "stable" for an arm that cannot
   fall over is not measuring anything, and nothing downstream can tell the
   difference between "measured as stable" and "structurally not
   applicable" (the exact ``0`` vs ``NaN`` distinction ``core/robot.py``
   documents for ``capabilities``).
3. **The forward-kinematics hot path does not do a linear search per call.**
   ``FrankaFK._joint_tensor`` (see ``fk.py``) calls ``names.index(name)``
   inside every ``forward``/``forward_transforms`` invocation -- fine for a
   handful of construction-time calls (that's all ``FrankaFK`` itself makes,
   to compute its rest-pose bone lengths), but ``FrankaSpec.forward_kinematics``
   is the per-clip, per-frame-batch scoring path. :meth:`FrankaSpec.__init__`
   resolves each Panda joint's index into the ``pytorch_kinematics`` chain's
   parameter list exactly once and stores it as a ``LongTensor`` buffer, so
   scoring assembles the chain's input with a single vectorised
   ``th[:, idx] = q`` scatter instead of a 7+2-iteration Python loop that
   calls ``list.index`` (itself ``O(n_joints)``) every time.
"""
from __future__ import annotations

import warnings
from typing import Any

import torch

from kinescore.core.robot import Capability, rigid_bone_mask
from kinescore.robots.franka.constants import (
    PANDA_ARM_JOINTS,
    PANDA_FINGER_JOINTS,
    PANDA_FINGER_MAX,
    ACTUATED_LINKS,
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
        The wrapped, verbatim :class:`FrankaFK` instance. Exposed for callers
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

        # ---- bones: full set (legacy-reproducible) + rigid-only subset ------
        self.bone_pairs = self.fk.bone_pairs.clone()
        self.bone_lengths = self.fk.bone_lengths.clone()
        mask = self._rigid_bone_mask()
        self._warn_dropped_bones(mask)
        self.rigid_bone_pairs = self.bone_pairs[mask]
        self.rigid_bone_lengths = self.bone_lengths[mask]

        self.capabilities: frozenset[str] = frozenset(
            {Capability.ROTATIONS, Capability.EFFORT_LIMITS})

        urdf_path = resolve_robot_description_urdf("panda_description")
        self.urdf_sha256: str | None = sha256_file(urdf_path)

    def _rigid_bone_mask(self) -> torch.Tensor:
        """Select bones that actually measure arm rigidity.

        Two independent rules, both of which a bone must pass:

        1. **Structural** -- neither endpoint is in
           :data:`~kinescore.robots.franka.constants.ACTUATED_LINKS`. A bone
           ending on a gripper finger changes length because the gripper opened,
           not because the arm deformed. This is read off the kinematic chain
           and transfers to any robot.
        2. **Degenerate-length** -- the rest length exceeds
           :data:`~kinescore.core.robot.DEGENERATE_BONE_M`. A bone whose
           endpoints coincide at rest carries no rigidity information whatever
           actuates it, and would divide by ~0.

        Rule 1 is primary. Rule 2 exists because it is cheap and catches a
        degenerate bone a future robot plugin might introduce without noticing.
        On the Panda, rule 1 alone drops bones 4, 5 and 6 and rule 2 alone would
        drop only bone 5 -- the intersection is what makes the exclusion
        principled rather than a threshold tuned to one robot.
        """
        actuated = torch.tensor(
            [self.keypoint_links[i] in ACTUATED_LINKS
             or self.keypoint_links[j] in ACTUATED_LINKS
             for i, j in self.bone_pairs.tolist()], dtype=torch.bool)
        long_enough = rigid_bone_mask(self.bone_lengths,
                                      min_length_m=RIGID_BONE_MIN_M)
        return (~actuated) & long_enough

    def _warn_dropped_bones(self, mask: torch.Tensor) -> None:
        """Warn once, by name, for every bone :attr:`rigid_bone_pairs` drops.

        Silent filtering here would be exactly the kind of thing that makes a
        downstream "why is my rigidity number different from the paper's"
        question take an afternoon instead of reading one warning.
        """
        dropped = [
            f"{self.keypoint_links[i]}->{self.keypoint_links[j]} "
            f"(rest length {self.bone_lengths[k].item():.4f} m)"
            for k, (i, j) in enumerate(self.bone_pairs.tolist())
            if not mask[k]
        ]
        if dropped:
            warnings.warn(
                "FrankaSpec: excluded from rigid_bone_pairs (endpoint is a "
                "gripper-actuated link, or the bone is degenerate at rest -- "
                "either way its length tracks actuation, not arm rigidity; see "
                "kinescore.robots.franka.constants.ACTUATED_LINKS): "
                + "; ".join(dropped),
                stacklevel=3,
            )

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
