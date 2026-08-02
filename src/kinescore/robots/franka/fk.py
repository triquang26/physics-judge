"""Differentiable Franka Panda forward-kinematics for Marionette.

This module turns a robot *joint-space* description ``(q, gripper)`` into 3-D
keypoint positions in the robot **base frame** (``panda_link0``), fully
differentiable w.r.t. ``q`` and ``gripper`` so the kinematics path can be used
both as a fixed grounding signal (encode observed states into keypoints) and as
a differentiable decoder for predicted joint angles (``q_hat -> P_hat`` feeding
:func:`models.kinematics.readout.loss_rigid`).

The chain is built once from the Franka Panda URDF shipped by
``robot_descriptions`` (the ``example-robot-data`` Panda). That URDF exposes:

* 7 revolute arm joints  : ``panda_joint1 .. panda_joint7``
* 2 prismatic finger joints: ``panda_finger_joint1`` / ``panda_finger_joint2``
* the flange ``panda_link8``, the hand ``panda_hand``, the TCP
  ``panda_hand_tcp`` and the two finger links ``panda_leftfinger`` /
  ``panda_rightfinger``.

Robotiq note
------------
The project conceptually targets a Robotiq 2F-85 gripper, but the Franka URDF
from ``robot_descriptions`` ships ``panda_hand`` plus the two Panda finger
joints. That is sufficient for v1 of the kinematics-grounding branch: the hand
pose and the two finger keypoints capture the gripper geometry we condition on.

Missing keypoint links
----------------------
The contract's :attr:`FrankaFK.KEYPOINT_LINKS` asks for ``panda_grasptarget``,
which is **not** present in this particular URDF. To keep the keypoint count
``K`` byte-stable for downstream modules (:class:`FKEncoder` defaults to
``num_keypoints=8``), any requested link that is absent is transparently
remapped to the closest physically-equivalent link that *does* exist
(``panda_grasptarget -> panda_hand_tcp``). The mapping is logged once at build
time and never changes the tensor shapes the rest of the pipeline sees.

Backward compatibility
----------------------
Importing or instantiating this module has no global side effects and does not
touch any existing training path; it only runs when an integration layer (gated
behind a config flag, default OFF) explicitly constructs a :class:`FrankaFK`.

Provenance
----------
Ported **verbatim** from ``judge/fk.py`` (Marionette-ciasc) /
``models/kinematics/fk.py`` (Marionette-fkjepa, byte-identical). Per
``kinescore``'s porting rule for this file, the FK math (every method body
below) is unchanged character-for-character; the only edits are the import
lines and the extraction of the module-level constant tables into
:mod:`kinescore.robots.franka.constants`, so the two ``_PANDA_*`` tuples this
file references are defined once and shared with
:mod:`kinescore.robots.franka.spec` rather than copy-pasted. See
``legacy_docs/PROVENANCE.md`` for the reproduced-defect writeup (D9,
``rigidity_residual_mm`` on a gripper open/close) that motivated auditing this
file in the first place.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import torch
import torch.nn as nn

from kinescore.robots.franka.constants import (
    KEYPOINT_LINKS,
    LINK_FALLBACKS as _LINK_FALLBACKS,
    PANDA_ARM_JOINTS as _PANDA_ARM_JOINTS,
    PANDA_EFFORT_LIMITS as _PANDA_EFFORT_LIMITS,
    PANDA_FINGER_JOINTS as _PANDA_FINGER_JOINTS,
    PANDA_FINGER_MAX as _PANDA_FINGER_MAX,
    PANDA_JOINT_LOWER as _PANDA_JOINT_LOWER,
    PANDA_JOINT_UPPER as _PANDA_JOINT_UPPER,
    PANDA_VEL_LIMITS as _PANDA_VEL_LIMITS,
)

__all__ = ["FrankaFK"]


class FrankaFK(nn.Module):
    """Differentiable Franka Panda forward kinematics.

    Wraps a ``pytorch_kinematics`` chain built from the Panda URDF and exposes a
    keypoint forward pass plus an end-effector pose helper. All tensor ops stay
    on the autograd graph (no ``.item()`` / numpy in the forward path), so
    gradients flow back to ``q`` and ``gripper``.

    Parameters
    ----------
    ee_link:
        Name of the link whose pose :meth:`ee_pose` returns. Defaults to
        ``"panda_hand"``.
    keypoint_links:
        Ordered names of the ``K`` links whose translations :meth:`forward`
        returns. Defaults to :attr:`KEYPOINT_LINKS` (``K = 8``). Names absent
        from the URDF are remapped via :data:`_LINK_FALLBACKS`.
    device:
        Device the FK buffers and joint tensors live on.
    dtype:
        Floating dtype for the chain and outputs.

    Buffers
    -------
    joint_limits: ``(7, 2)``
        Per arm-joint ``[lower, upper]`` limits in radians.
    bone_pairs: ``(P, 2)`` long
        Parent->child index pairs into the keypoint axis, for the rigid loss.
    bone_lengths: ``(P,)``
        Nominal (rest-pose) distances for each pair in :attr:`bone_pairs`.

    Shapes
    ------
    Input  ``q``       : ``(B, T, 7)``  arm joint angles, radians (UNNORMALIZED).
    Input  ``gripper`` : ``(B, T, 1)``  gripper openness in ``[0, 1]``.
    Output ``P``       : ``(B, T, K, 3)`` keypoint xyz in the base frame.
    """

    #: Default keypoint link names (``K = 8``). All present in the
    #: ``robot_descriptions`` example-robot-data Panda URDF.
    KEYPOINT_LINKS: tuple[str, ...] = KEYPOINT_LINKS

    def __init__(
        self,
        # ``panda_link8`` is the flange frame DROID logs as
        # ``observation.state.cartesian_position`` (verified: pos err 0.000 m,
        # rotation matches Euler-xyz to 0.01 deg). See tests/test_fk_alignment.py.
        ee_link: str = "panda_link8",
        keypoint_links: Sequence[str] = KEYPOINT_LINKS,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        # Local imports so merely importing this module does not hard-require the
        # (still-building) pytorch_kinematics / robot_descriptions stack.
        import pytorch_kinematics as pk  # noqa: WPS433 (intentional local import)
        from robot_descriptions import panda_description  # noqa: WPS433

        self.device = torch.device(device)
        self.dtype = dtype
        self.ee_link = str(ee_link)

        # ---- build the chain once on CPU, then move it to the target device ---
        with open(panda_description.URDF_PATH) as urdf_file:
            urdf_str = urdf_file.read()
        # A full (non-serial) chain retains *all* links (incl. the fingers and
        # TCP that a serial chain ending at panda_hand would prune), which we
        # need for the keypoint set.
        chain = pk.build_chain_from_urdf(urdf_str)
        chain = chain.to(dtype=self.dtype, device=self.device)
        self.chain = chain

        # Joint ordering reported by the chain (non-fixed joints only).
        self._chain_joint_names: list[str] = list(chain.get_joint_parameter_names())

        # ---- resolve & validate the keypoint links ---------------------------
        available = set(chain.get_frame_names(exclude_fixed=False))
        self._resolve_ee_link(available)
        self.keypoint_links: tuple[str, ...] = self._resolve_keypoint_links(
            keypoint_links, available
        )
        self.num_keypoints: int = len(self.keypoint_links)

        # ---- register buffers ------------------------------------------------
        joint_limits = torch.tensor(
            list(zip(_PANDA_JOINT_LOWER, _PANDA_JOINT_UPPER, strict=True)),
            dtype=self.dtype,
        )  # (7, 2)
        self.register_buffer("joint_limits", joint_limits)
        self.register_buffer("joint_vel_limits",
                             torch.tensor(_PANDA_VEL_LIMITS, dtype=self.dtype))      # (7,)
        self.register_buffer("joint_effort_limits",
                             torch.tensor(_PANDA_EFFORT_LIMITS, dtype=self.dtype))   # (7,)

        bone_pairs, bone_lengths = self._compute_rest_bones()
        self.register_buffer("bone_pairs", bone_pairs)      # (P, 2) long
        self.register_buffer("bone_lengths", bone_lengths)  # (P,)

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    def _resolve_ee_link(self, available: set) -> None:
        """Remap / validate :attr:`ee_link` against the URDF frames."""
        if self.ee_link in available:
            return
        for fallback in _LINK_FALLBACKS.get(self.ee_link, ()):
            if fallback in available:
                warnings.warn(
                    f"ee_link '{self.ee_link}' not in URDF; using fallback "
                    f"'{fallback}'.",
                    stacklevel=2,
                )
                self.ee_link = fallback
                return
        raise ValueError(
            f"ee_link '{self.ee_link}' is not a frame in the Panda URDF. "
            f"Available frames: {sorted(available)}"
        )

    def _resolve_keypoint_links(
        self, requested: Sequence[str], available: set
    ) -> tuple[str, ...]:
        """Map each requested keypoint link to one present in the URDF.

        Keeps the requested order and length (``K``) so downstream tensor shapes
        stay stable; substitutes via :data:`_LINK_FALLBACKS` for missing links.
        """
        resolved: list[str] = []
        for name in requested:
            if name in available:
                resolved.append(name)
                continue
            substitute = next(
                (fb for fb in _LINK_FALLBACKS.get(name, ()) if fb in available),
                None,
            )
            if substitute is None:
                raise ValueError(
                    f"Keypoint link '{name}' is absent from the Panda URDF and "
                    f"has no available fallback. Available frames: "
                    f"{sorted(available)}"
                )
            warnings.warn(
                f"Keypoint link '{name}' not in URDF; using fallback "
                f"'{substitute}'.",
                stacklevel=2,
            )
            resolved.append(substitute)
        return tuple(resolved)

    def _compute_rest_bones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``bone_pairs`` / ``bone_lengths`` from the rest (zero) pose.

        Consecutive keypoints are treated as a parent->child kinematic chain,
        giving ``K - 1`` bones whose rest-pose lengths anchor the rigid loss.

        Returns
        -------
        bone_pairs: ``(K - 1, 2)`` long
        bone_lengths: ``(K - 1,)``
        """
        k = self.num_keypoints
        pairs = torch.tensor(
            [[i, i + 1] for i in range(k - 1)], dtype=torch.long
        )  # (K-1, 2)

        # Rest pose: zero arm angles, closed gripper. Detached: these are static
        # nominal lengths, not part of any autograd graph.
        with torch.no_grad():
            q0 = torch.zeros(1, 1, 7, dtype=self.dtype, device=self.device)
            g0 = torch.zeros(1, 1, 1, dtype=self.dtype, device=self.device)
            p0 = self.forward(q0, g0)[0, 0]  # (K, 3)
        diffs = p0[pairs[:, 1]] - p0[pairs[:, 0]]  # (K-1, 3)
        lengths = diffs.norm(dim=-1).to(self.dtype).cpu()  # (K-1,)
        return pairs, lengths

    # ------------------------------------------------------------------ #
    # forward kinematics
    # ------------------------------------------------------------------ #
    def _ensure_device(self, device: torch.device) -> None:
        """Move the kinematic chain to ``device`` if needed (device-following).

        The pytorch_kinematics chain is a plain attribute, so ``module.to(...)``
        does NOT relocate it. We follow the input tensor's device so FK outputs
        match the caller (e.g. CUDA) and stay consistent with the registered
        buffers (which DO move with the module)."""
        device = torch.device(device)
        if device != self.device:
            self.chain = self.chain.to(device=device)
            self.device = device

    def _joint_tensor(self, q: torch.Tensor, gripper: torch.Tensor) -> torch.Tensor:
        """Assemble the full ``(B*T, n_joints)`` chain input from ``q, gripper``.

        ``q`` provides the 7 arm angles; ``gripper`` in ``[0, 1]`` is scaled to
        ``[0, _PANDA_FINGER_MAX]`` metres and broadcast to both finger joints.
        Remaining chain joints (none, for the Panda URDF) are left at zero.
        """
        if q.ndim != 3 or q.shape[-1] != 7:
            raise ValueError(f"Expected q of shape (B, T, 7), got {tuple(q.shape)}")
        if gripper.ndim != 3 or gripper.shape[-1] != 1:
            raise ValueError(
                f"Expected gripper of shape (B, T, 1), got {tuple(gripper.shape)}"
            )

        self._ensure_device(q.device)
        q = q.to(device=self.device, dtype=self.dtype)
        gripper = gripper.to(device=self.device, dtype=self.dtype)
        b, t = q.shape[0], q.shape[1]

        q_flat = q.reshape(b * t, 7)                    # (B*T, 7)
        finger = (gripper.reshape(b * t, 1) * _PANDA_FINGER_MAX)  # (B*T, 1)

        n_joints = self.chain.n_joints
        th = q_flat.new_zeros(b * t, n_joints)          # (B*T, n_joints)
        names = self._chain_joint_names
        for i, name in enumerate(_PANDA_ARM_JOINTS):
            th[:, names.index(name)] = q_flat[:, i]
        for name in _PANDA_FINGER_JOINTS:
            if name in names:
                th[:, names.index(name)] = finger[:, 0]
        return th

    def _stack_keypoints(
        self, transforms: dict[str, object], n: int
    ) -> torch.Tensor:
        """Stack keypoint-link translations into ``(n, K, 3)`` (differentiable)."""
        cols = [
            transforms[name].get_matrix()[:, :3, 3]  # (n, 3)
            for name in self.keypoint_links
        ]
        return torch.stack(cols, dim=1)  # (n, K, 3)

    def forward(self, q: torch.Tensor, gripper: torch.Tensor) -> torch.Tensor:
        """Compute keypoint positions in the robot base frame.

        Parameters
        ----------
        q:
            Arm joint angles, shape ``(B, T, 7)`` in radians (UNNORMALIZED).
        gripper:
            Gripper openness, shape ``(B, T, 1)`` in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Keypoint positions ``P`` of shape ``(B, T, K, 3)`` in the
            ``panda_link0`` base frame, differentiable w.r.t. ``q`` and
            ``gripper``.
        """
        b, t = q.shape[0], q.shape[1]
        # FK is geometric — run in the chain's (fp32) dtype, never under autocast,
        # to avoid fp16 index/dtype mismatches and keep metric accuracy.
        with torch.autocast(device_type=q.device.type, enabled=False):
            th = self._joint_tensor(q, gripper)             # (B*T, n_joints)
            transforms = self.chain.forward_kinematics(th)  # dict[name -> Transform3d]
            p = self._stack_keypoints(transforms, b * t)    # (B*T, K, 3)
        return p.reshape(b, t, self.num_keypoints, 3)   # (B, T, K, 3)

    def forward_transforms(
        self, q: torch.Tensor, gripper: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keypoint-link positions AND rotation matrices in the base frame.

        Same forward pass as :meth:`forward`, but also returns each keypoint
        link's 3×3 rotation, enabling angular-velocity/acceleration physics
        metrics. Single chain evaluation (no extra FK cost beyond stacking).

        Returns
        -------
        P: ``(B, T, K, 3)`` keypoint positions.
        R: ``(B, T, K, 3, 3)`` keypoint-link rotation matrices.
        """
        b, t = q.shape[0], q.shape[1]
        with torch.autocast(device_type=q.device.type, enabled=False):
            th = self._joint_tensor(q, gripper)
            transforms = self.chain.forward_kinematics(th)
            mats = [transforms[name].get_matrix() for name in self.keypoint_links]  # K×(B*T,4,4)
            M = torch.stack(mats, dim=1)                     # (B*T, K, 4, 4)
        P = M[..., :3, 3].reshape(b, t, self.num_keypoints, 3)
        R = M[..., :3, :3].reshape(b, t, self.num_keypoints, 3, 3)
        return P, R

    def ee_pose(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward kinematics of :attr:`ee_link` as ``(position, rotvec)``.

        The gripper is held closed (it does not affect arm-link poses).

        Parameters
        ----------
        q:
            Arm joint angles, shape ``(B, T, 7)`` in radians (UNNORMALIZED).

        Returns
        -------
        pos: ``(B, T, 3)``
            End-effector translation in the base frame.
        rotvec: ``(B, T, 3)``
            End-effector orientation as an axis-angle (rotation) vector.
        """
        b, t = q.shape[0], q.shape[1]
        gripper = q.new_zeros(b, t, 1)
        th = self._joint_tensor(q, gripper)             # (B*T, n_joints)
        transforms = self.chain.forward_kinematics(th)
        mat = transforms[self.ee_link].get_matrix()     # (B*T, 4, 4)
        pos = mat[:, :3, 3].reshape(b, t, 3)            # (B, T, 3)
        rotvec = self._matrix_to_rotvec(mat[:, :3, :3]) # (B*T, 3)
        return pos, rotvec.reshape(b, t, 3)

    # ------------------------------------------------------------------ #
    # rotation utilities (differentiable, batched)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _matrix_to_rotvec(rot: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Convert rotation matrices ``(N, 3, 3)`` to axis-angle ``(N, 3)``.

        Differentiable everywhere except the measure-zero set of exact pi
        rotations; numerically guarded near the ``angle -> 0`` limit.
        """
        # angle from trace: cos(theta) = (tr(R) - 1) / 2
        trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]  # (N,)
        cos = ((trace - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps)
        angle = torch.acos(cos)  # (N,)

        # axis * (2 sin theta) from the skew-symmetric part of R.
        rx = rot[:, 2, 1] - rot[:, 1, 2]
        ry = rot[:, 0, 2] - rot[:, 2, 0]
        rz = rot[:, 1, 0] - rot[:, 0, 1]
        axis2 = torch.stack([rx, ry, rz], dim=-1)  # (N, 3) == 2 sin(theta) * axis

        sin = torch.sin(angle).clamp_min(eps)      # (N,)
        scale = angle / (2.0 * sin)                # -> 0.5 as theta -> 0
        return axis2 * scale.unsqueeze(-1)         # (N, 3)
