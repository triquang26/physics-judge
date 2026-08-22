"""``Synthetic2R``: an analytic 2-link planar arm, no URDF, no network, no
``pytorch_kinematics``.

The URDF-backed robots need real assets, which makes them unsuitable for
testing the detector layer itself: whether a rigidity residual, a jerk or a
joint-limit term is computed correctly is a question about that math, not
about Panda geometry. ``Synthetic2R`` answers it in a CPU-only test with
closed-form FK, no I/O and no optional dependency.

Geometry
--------
Two revolute joints in the XY plane, base at the origin::

    base (0,0,0) --[link1: l1]--> elbow --[link2: l2]--> tip

``q = [theta1, theta2]`` in radians, ``theta1`` measured from the +X axis and
``theta2`` relative to link 1, so the elbow angle is ``theta1 + theta2``. This
is the simplest arm for which "bone length stays constant under rotation" and
"tip position is a nonlinear function of both joints" both hold -- the two
properties any FK-consuming detector must be exercised against.
"""
from __future__ import annotations

from typing import Any

import torch

from kinescore.core.robot import Capability

__all__ = ["Synthetic2R"]


class Synthetic2R:
    """Analytic 2-link planar arm implementing the ``RobotSpec`` protocol.

    Parameters
    ----------
    link1_m, link2_m:
        Lengths of the two links in metres (defaults are arbitrary but
        distinct, so a transposition bug between them is visible in tests).

    Keypoints (``K = 3``)
    -----------------------
    ``("base", "elbow", "tip")`` -- index 0 is the fixed origin, 1 the elbow,
    2 the tip (the sole end-effector site, see :meth:`ee_sites`).

    Capabilities
    -------------
    ``{ROTATIONS}`` only: this is a planar arm with no gripper, no feet and no
    ported collision geometry, but its FK still has a well-defined in-plane
    orientation at each keypoint (see :meth:`forward_transforms`), which is
    enough signal to exercise angular velocity and acceleration.
    """

    name = "synthetic_2r"
    n_joints = 2
    keypoint_links: tuple[str, ...] = ("base", "elbow", "tip")

    def __init__(self, link1_m: float = 0.4, link2_m: float = 0.3) -> None:
        self.link1_m = float(link1_m)
        self.link2_m = float(link2_m)

        # Full bone set: base->elbow, elbow->tip. Neither is degenerate (both
        # link lengths are > 0 by construction), so bone_pairs ==
        # rigid_bone_pairs -- there is nothing to drop, unlike the Franka
        # gripper (see kinescore.robots.franka.constants.RIGID_BONE_MIN_M).
        self.bone_pairs = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        self.bone_lengths = torch.tensor([self.link1_m, self.link2_m],
                                         dtype=torch.float32)
        self.rigid_bone_pairs = self.bone_pairs.clone()
        self.rigid_bone_lengths = self.bone_lengths.clone()

        # No physical joint-limit datasheet for a synthetic arm; a full
        # rotation range is the least assumption-laden default and keeps
        # joint-limit-violation metrics well-defined (never trivially zero,
        # never trivially saturated) when this robot feeds them.
        self.q_lo = torch.tensor([-torch.pi, -torch.pi], dtype=torch.float32)
        self.q_hi = torch.tensor([torch.pi, torch.pi], dtype=torch.float32)
        self.vel_limits: torch.Tensor | None = None
        self.effort_limits: torch.Tensor | None = None

        self.capabilities: frozenset[str] = frozenset({Capability.ROTATIONS})
        # No URDF was read to produce this robot -- recording a hash would be
        # a fabricated provenance claim, so this is None rather than e.g. "".
        self.urdf_sha256: str | None = None

    def forward_kinematics(self, q: torch.Tensor,
                           aux: Any | None = None) -> torch.Tensor:
        """``(B,T,2) -> (B,T,3,3)`` base/elbow/tip xy positions (z == 0).

        ``aux`` is accepted for protocol compliance and ignored -- this robot
        has no non-joint-angle state (no gripper, no hand DoF).
        """
        P, _ = self.forward_transforms(q, aux)
        return P

    def forward_transforms(self, q: torch.Tensor, aux: Any | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B,T,2) -> P (B,T,3,3), R (B,T,3,3,3)``.

        Closed form:
            base  = (0, 0, 0)
            elbow = (l1 cos(t1),        l1 sin(t1),        0)
            tip   = elbow + (l2 cos(t1+t2), l2 sin(t1+t2), 0)

        ``R`` is each keypoint's in-plane orientation as a z-axis rotation
        matrix -- identity at the base, ``Rz(t1)`` at the elbow (link 1's
        orientation), ``Rz(t1+t2)`` at the tip (link 2's orientation). This is
        the natural rigid-body frame for a revolute-jointed planar link (its
        own long axis rotates by the cumulative joint angle), not an arbitrary
        placeholder -- it is what makes ``ROTATIONS``-gated metrics (angular
        velocity, angular acceleration) exercise real, distinct-per-frame
        values on this robot instead of a constant identity.
        """
        if q.ndim != 3 or q.shape[-1] != self.n_joints:
            raise ValueError(
                f"Expected q of shape (B,T,{self.n_joints}), got {tuple(q.shape)}")
        b, t = q.shape[0], q.shape[1]
        t1 = q[..., 0]
        t2 = q[..., 1]
        t12 = t1 + t2

        zero = torch.zeros_like(t1)
        base = torch.stack([zero, zero, zero], dim=-1)
        elbow = torch.stack(
            [self.link1_m * torch.cos(t1), self.link1_m * torch.sin(t1), zero],
            dim=-1)
        tip = elbow + torch.stack(
            [self.link2_m * torch.cos(t12), self.link2_m * torch.sin(t12), zero],
            dim=-1)
        P = torch.stack([base, elbow, tip], dim=2)  # (B,T,3,3)

        R = torch.zeros(b, t, 3, 3, 3, dtype=q.dtype, device=q.device)
        R[:, :, 0] = torch.eye(3, dtype=q.dtype, device=q.device)
        R[:, :, 1] = _rotz(t1)
        R[:, :, 2] = _rotz(t12)
        return P, R

    def ee_sites(self) -> tuple[int, ...]:
        """Single end-effector site: the tip, index 2."""
        return (2,)


def _rotz(theta: torch.Tensor) -> torch.Tensor:
    """``(B,T) -> (B,T,3,3)`` batched rotation about the z-axis."""
    c, s = torch.cos(theta), torch.sin(theta)
    one = torch.ones_like(c)
    zero = torch.zeros_like(c)
    row0 = torch.stack([c, -s, zero], dim=-1)
    row1 = torch.stack([s, c, zero], dim=-1)
    row2 = torch.stack([zero, zero, one], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)
