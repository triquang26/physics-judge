"""``RobotSpec`` for the Airbot MMK2: bimanual 6-DOF arms, no hand, no legs.

:class:`AirbotMMK2Spec` wraps :class:`~kinescore.robots.airbot_mmk2.fk.AirbotMMK2FK`
(two 6-DOF arm chains) and adapts it to the frozen ``RobotSpec`` protocol
(``kinescore.core.robot``).

Why this robot exists as a *separate* embodiment from GR-1
-------------------------------------------------------------
The ``dense/humanoid/.../multiview/ctrlworld/`` cell in this benchmark's data
tree is labelled ``humanoid`` but is **not** Fourier GR-1: every episode
directory is named ``AIRBOT_MMK2_*``/``Airbot_MMK2_*``, and the paired real
data's own ``meta/info.json`` declares ``robot_type: "Airbot_MMK2"`` /
``"discover_robotics_aitbot_mmk2"``. Scoring it with :class:`GR1Spec` would
run a reader that has never seen this robot and report plausible-looking
joint angles for the wrong kinematic chain -- worse than an empty row. This
class exists so that cell can be scored honestly, or (see below) so the
reasons it cannot be are precise rather than a guess.

Predicted state: arms only, no hand, no mobile base
-----------------------------------------------------
``n_joints = 12`` = ``[left_arm(6), right_arm(6)]`` -- see
``constants.py``'s module docstring for the full joint-order verification
against the real LeRobot parquet, and specifically for why the real robot's
12-DOF-per-side dexterous hand (``*_hand_joint_1..12_rad`` in the parquet) is
**excluded**: those columns' values (observed up to ~86) are not plausible
radians despite the column-name suffix, and no upstream source (RoboCOIN
repo, dataset README/info.yaml) documents their true unit or channel
order -- modelling them would mean fitting a reader against ground truth of
unverified meaning, exactly the failure mode this benchmark exists to avoid.
The mobile base (omni wheels) and head (yaw/pitch) are likewise absent: this
URDF (see ``build_airbot_mmk2_urdf.py``) contains *only* the two arm chains,
so unlike :class:`~kinescore.robots.gr1.spec.GR1Spec` (17 of 70 URDF joints
predicted, the rest at rest pose) there is no "sits at URDF rest pose" branch
here at all -- every joint in this URDF is a predicted joint.

Reader unit convention: ``limit_semantics = "raw_rad"``
-----------------------------------------------------------
The pose reader trained against this spec must report the raw predicted
angle even when it is outside ``[q_lo, q_hi]`` -- never squashed/clipped into
the legal range. A squashed head makes ``limit_violation_frac`` structurally
zero regardless of the video (the defect this benchmark's page calls out for
Franka's ``judge_v3l`` reader); this spec's ``q_lo``/``q_hi`` are meant to be
compared *after* a raw prediction, not used to clip the prediction itself.

Capabilities
-------------
``{ROTATIONS, EFFORT_LIMITS}``. No ``COLLIDERS`` (no mesh/collision geometry
was resolved -- this URDF is kinematics-only, matching the precedent already
in this asset tree for ``aloha_bimanual.urdf``). No ``SUPPORT_POLYGON`` (the
composite URDF has no legs/feet/mobile base -- it would be dishonest to
report a balance margin for links this class never poses). ``EFFORT_LIMITS``
*is* declared, unlike GR1 (no effort data in its URDF at all): the source
``airbot_play_v3_gripper_fixed.urdf`` carries real ``<limit effort="...">``
values alongside the position limits, so ``effort_proxy`` can compute a real
number for this robot rather than ``NaN``.
"""
from __future__ import annotations

from typing import Any

import torch

from kinescore.core.robot import Capability, rigid_bone_mask
from kinescore.robots.airbot_mmk2.constants import (
    EE_LINK,
    KEYPOINTS_LEFT,
    KEYPOINTS_RIGHT,
)
from kinescore.robots.airbot_mmk2.fk import AirbotMMK2FK
from kinescore.robots.urdf import resolve_asset_urdf, sha256_file

__all__ = ["AirbotMMK2Spec", "AIRBOT_MMK2_URDF_RELPATH"]

#: Location of the composite arms-only URDF relative to ``KINESCORE_ASSETS``.
#: See ``KINESCORE_ASSETS/airbot_mmk2/urdf/MANIFEST.json`` for provenance
#: (upstream repos, commits, license, sha256) of every source this URDF was
#: assembled from.
AIRBOT_MMK2_URDF_RELPATH = "airbot_mmk2/urdf/airbot_mmk2_bimanual_arms.urdf"


class AirbotMMK2Spec:
    """``RobotSpec`` for the Airbot MMK2's two 6-DOF arms.

    Parameters
    ----------
    device, dtype:
        Passed straight through to the underlying :class:`AirbotMMK2FK`.
    urdf_path:
        Override the URDF location instead of resolving it from
        ``KINESCORE_ASSETS`` / :data:`AIRBOT_MMK2_URDF_RELPATH`. Mainly for
        tests that point at a specific fixture URDF.

    Raises
    ------
    kinescore.paths.MissingPathError
        If ``urdf_path`` is not given and ``KINESCORE_ASSETS`` is unset, or
        the composite URDF is not checked out under it.

    ``keypoint_links`` layout
    --------------------------
    ``KEYPOINTS_LEFT + KEYPOINTS_RIGHT`` (``K = 12``): indices ``0..5`` are the
    left arm (``left_link1..left_link6``, shoulder -> flange), ``6..11`` the
    right arm, in the same per-arm order ``AirbotMMK2FK.keypoints_fk`` returns.
    """

    #: Registry key (kinescore.robots.get_robot).
    name = "airbot_mmk2"
    #: Predicted DOF: left arm (6) + right arm (6). See module docstring.
    n_joints = AirbotMMK2FK.N_Q

    def __init__(self, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32,
                 urdf_path: str | None = None) -> None:
        if urdf_path is None:
            urdf_path = str(resolve_asset_urdf(AIRBOT_MMK2_URDF_RELPATH))
        self.urdf_sha256: str | None = sha256_file(urdf_path)

        self.fk = AirbotMMK2FK(urdf_path, device=device, dtype=dtype)

        self.keypoint_links: tuple[str, ...] = KEYPOINTS_LEFT + KEYPOINTS_RIGHT
        n_left = len(KEYPOINTS_LEFT)

        # ---- joint limits (12,) -- see AirbotMMK2FK docstring: no margin
        self.q_lo = self.fk.q_lo.clone()
        self.q_hi = self.fk.q_hi.clone()
        self.vel_limits: torch.Tensor | None = self.fk.q_vel_max.clone()
        self.effort_limits: torch.Tensor | None = self.fk.q_effort_max.clone()

        # ---- bones: concatenate the two per-arm chains, right offset by n_left
        self.bone_pairs = torch.cat(
            [self.fk.bone_pairs_left, self.fk.bone_pairs_right + n_left], dim=0)
        self.bone_lengths = torch.cat(
            [self.fk.bone_lengths_left, self.fk.bone_lengths_right], dim=0)
        # Two of the five per-arm bones (link1->link2, link4->link5) coincide
        # at rest -- a real zero-offset joint pair on the physical AIRBOT Play
        # arm (see constants.py), not a non-predicted actuator. Both endpoints
        # of each ARE driven by predicted joints, so this is the "second,
        # independent safety net" case from docs/ARCHITECTURE.md#adding-a-robot's
        # checklist (D9), not the structural-exclusion case -- the library
        # default degenerate-length threshold is what does the work here,
        # same as GR1Spec, just with a nonzero result on this robot.
        mask = rigid_bone_mask(self.bone_lengths)
        self.rigid_bone_pairs = self.bone_pairs[mask]
        self.rigid_bone_lengths = self.bone_lengths[mask]

        self.capabilities: frozenset[str] = frozenset(
            {Capability.ROTATIONS, Capability.EFFORT_LIMITS})

    # ------------------------------------------------------------------ #
    # RobotSpec protocol
    # ------------------------------------------------------------------ #
    def forward_kinematics(self, q: torch.Tensor,
                           aux: Any | None = None) -> torch.Tensor:
        """``(B,T,12) -> (B,T,12,3)``. ``aux`` is unused (no hand/gripper DoF
        is modelled -- see module docstring)."""
        left = self.fk.keypoints_fk(q, "left")
        right = self.fk.keypoints_fk(q, "right")
        return torch.cat([left, right], dim=2)

    def forward_transforms(self, q: torch.Tensor, aux: Any | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B,T,12) -> P (B,T,12,3), R (B,T,12,3,3)``. ``aux`` unused, see above."""
        p_l, r_l = self.fk.forward_transforms(q, "left")
        p_r, r_r = self.fk.forward_transforms(q, "right")
        return torch.cat([p_l, p_r], dim=2), torch.cat([r_l, r_r], dim=2)

    def ee_sites(self) -> tuple[int, ...]:
        """Two end-effector sites: left, then right flange (``link6``)."""
        return (self.keypoint_links.index(EE_LINK["left"]),
                self.keypoint_links.index(EE_LINK["right"]))
