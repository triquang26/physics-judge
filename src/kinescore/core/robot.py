"""Robot embodiment contract.

A ``RobotSpec`` is everything the detectors need to know about a robot:
forward kinematics, joint limits, which keypoints form rigid bones, and which
optional capabilities (collision geometry, a support polygon) it has.

Adding a robot means implementing this protocol -- see
``docs/ARCHITECTURE.md#adding-a-robot``.

The degenerate-bone problem
---------------------------
A bone between two keypoints that coincide carries no rigidity information.
The Franka's ``panda_leftfinger -> panda_rightfinger`` pair is the clearest
case: rest length 0.0 m, realised length equal to the gripper opening, so a
perfectly rigid arm that merely opens its gripper reads as a rigidity
violation.

Implementations therefore expose two bone sets: ``bone_pairs`` (everything)
and ``rigid_bone_pairs`` (degenerate bones dropped), and the rigidity detector
uses the latter.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch

__all__ = ["RobotSpec", "Capability", "DEGENERATE_BONE_M", "rigid_bone_mask"]

#: A bone shorter than this at rest carries no rigidity information -- its
#: measured "length" is actuation, not structure. 1 mm is far below any real
#: link and far above float noise.
DEGENERATE_BONE_M = 1e-3


class Capability:
    """Optional robot features that gate capability-dependent metrics."""

    COLLIDERS = "COLLIDERS"                # collision primitives -> self-collision
    SUPPORT_POLYGON = "SUPPORT_POLYGON"    # feet/base polygon -> CoM balance
    ROTATIONS = "ROTATIONS"                # FK returns link rotations -> angular
    EFFORT_LIMITS = "EFFORT_LIMITS"        # rated torques -> effort proxy


@runtime_checkable
class RobotSpec(Protocol):
    """Kinematic and geometric description of one robot embodiment.

    Attributes
    ----------
    name:
        Registry key, e.g. ``"franka_panda"``, ``"fourier_gr1"``.
    n_joints:
        Degrees of freedom the pose reader predicts and FK consumes.
    keypoint_links:
        Ordered link names whose origins become the keypoints ``P``.
    bone_pairs, bone_lengths:
        ``(n_bones,2)`` index pairs into the keypoint axis and their rest
        lengths in metres. The full set, degenerate bones included.
    rigid_bone_pairs, rigid_bone_lengths:
        The same with bones shorter than :data:`DEGENERATE_BONE_M` removed.
        **This is what the rigidity detector uses.**
    q_lo, q_hi:
        ``(n_joints,)`` joint position limits in radians.
    vel_limits, effort_limits:
        ``(n_joints,)`` rated joint velocity (rad/s) and effort (N.m), or
        ``None`` when the URDF does not declare them.
    capabilities:
        Frozen set of :class:`Capability` strings. Metrics requiring a
        capability the robot lacks return ``NaN`` with a reason rather than a
        fabricated number -- a table-mounted arm has no balance margin, and
        reporting ``0`` for it would be a lie.
    urdf_sha256:
        Hash of the URDF the kinematics were built from. Recorded in outputs so
        a silent upstream URDF change is diagnosable instead of appearing as a
        few-millimetre drift in every keypoint.
    """

    name: str
    n_joints: int
    keypoint_links: tuple[str, ...]
    bone_pairs: torch.Tensor
    bone_lengths: torch.Tensor
    rigid_bone_pairs: torch.Tensor
    rigid_bone_lengths: torch.Tensor
    q_lo: torch.Tensor
    q_hi: torch.Tensor
    vel_limits: torch.Tensor | None
    effort_limits: torch.Tensor | None
    capabilities: frozenset[str]
    urdf_sha256: str | None

    def forward_kinematics(self, q: torch.Tensor,
                           aux: Any | None = None) -> torch.Tensor:
        """``(B,T,n_joints) -> (B,T,K,3)`` keypoint positions in metres.

        ``aux`` carries robot-specific extras that are not joint angles -- the
        Franka gripper opening in ``[0,1]``, the GR-1 hand DoF. Positions are in
        the robot base frame, so they are independent of camera placement.
        """
        ...

    def forward_transforms(self, q: torch.Tensor, aux: Any | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """``-> (P (B,T,K,3), R (B,T,K,3,3))`` positions and link rotations."""
        ...

    def ee_sites(self) -> tuple[int, ...]:
        """Keypoint indices treated as end-effectors.

        One index for a single arm, two for a bimanual robot.
        """
        ...


def rigid_bone_mask(bone_lengths: torch.Tensor,
                    min_length_m: float = DEGENERATE_BONE_M) -> torch.Tensor:
    """Boolean mask selecting bones long enough to carry rigidity information.

    Parameters
    ----------
    bone_lengths:
        ``(n_bones,)`` rest lengths in metres.
    min_length_m:
        Bones at or below this are degenerate (coincident keypoints), so their
        measured length reflects actuation rather than structure.

    Returns
    -------
    torch.Tensor
        ``(n_bones,)`` bool mask, ``True`` for bones to keep.
    """
    return bone_lengths > min_length_m
