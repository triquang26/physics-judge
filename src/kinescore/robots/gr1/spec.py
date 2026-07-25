"""``RobotSpec`` for the Fourier GR-1: bimanual humanoid, two arms + feet.

:class:`GR1Spec` wraps the verbatim-ported :class:`~kinescore.robots.gr1.fk.GR1FK`
(two-chain arm kinematics) and :class:`~kinescore.robots.gr1.colliders.RobotColliders`
(URDF-derived collision spheres + inertial CoM) and adapts them to the frozen
``RobotSpec`` protocol (``kinescore.core.robot``). Neither wrapped class is
modified here -- see their own module docstrings for the porting note.

Predicted state excludes the legs
----------------------------------
``GR1FK``'s 17-D ``q17`` is ``[left_arm(7), right_arm(7), waist(3)]`` --
the pose reader this benchmark scores never predicts leg joint angles (the
recording setup this was built for is upper-body teleop; see ``gr1_fk.py``'s
module docstring). Every link in the URDF that is *not* reachable from those
17 joints -- both legs, the torso/base/head collider links, and the feet --
therefore sits at its URDF rest pose in every FK call this class makes.
:meth:`GR1Spec.support_polygon` and :meth:`GR1Spec.body_collider_spheres`
reflect that honestly: they return the same fixed foot/torso layout for every
frame. That is not a bug in this file -- it is what "17 predicted joints,
54 URDF joints" means -- but a metric consuming :attr:`GR1Spec.colliders` must
not expect the support polygon to track anything the reader cannot see.
:meth:`GR1Spec.world_com` *does* vary per frame, because the arms (which move)
carry mass too.

``COLLIDERS`` / ``SUPPORT_POLYGON`` are capability-gated extensions
---------------------------------------------------------------------
The frozen ``RobotSpec`` protocol (``kinescore/core/robot.py``) has no generic
field for collision geometry or a support polygon -- by design, since a
Franka has neither. ``kinescore.core.metric.MetricContext.available()``
already gates on ``"COLLIDERS" in robot.capabilities`` /
``"SUPPORT_POLYGON" in robot.capabilities`` rather than on a specific
attribute, so metrics that declare those requirements are expected to
``getattr`` the extra data they need directly off the concrete ``RobotSpec``
instance. This class is what backs that for GR-1: :attr:`colliders` (the
wrapped :class:`RobotColliders`) plus :meth:`body_collider_spheres`,
:meth:`world_com` and :meth:`support_polygon` (posed convenience wrappers,
since ``RobotColliders`` itself only knows how to pose *given* link frames --
it does not run FK).
"""
from __future__ import annotations

from typing import Any

import torch

from kinescore.core.robot import Capability, rigid_bone_mask
from kinescore.robots.gr1.colliders import RobotColliders
from kinescore.robots.gr1.fk import EE_LINK, GR1FK, KEYPOINTS_LEFT, KEYPOINTS_RIGHT
from kinescore.robots.urdf import resolve_asset_urdf, sha256_file

__all__ = ["GR1Spec", "GR1_URDF_RELPATH"]

#: Location of the GR-1 URDF relative to ``KINESCORE_ASSETS``. Matches the
#: layout of the fkjepa asset tree this was ported from
#: (``assets/grx/GRX/GR1/gr1t1/urdf/gr1t1_fourier_hand_6dof.urdf``); an
#: operator populating ``KINESCORE_ASSETS`` should mirror that subtree.
GR1_URDF_RELPATH = "grx/GRX/GR1/gr1t1/urdf/gr1t1_fourier_hand_6dof.urdf"


class GR1Spec:
    """``RobotSpec`` for the Fourier GR-1 bimanual humanoid.

    Parameters
    ----------
    device, dtype:
        Passed straight through to the underlying :class:`GR1FK` /
        :class:`RobotColliders`.
    urdf_path:
        Override the URDF location instead of resolving it from
        ``KINESCORE_ASSETS`` / :data:`GR1_URDF_RELPATH`. Mainly for tests that
        point at a specific fixture URDF.

    Raises
    ------
    kinescore.paths.MissingPathError
        If ``urdf_path`` is not given and ``KINESCORE_ASSETS`` is unset, or
        the GR-1 URDF is not checked out under it. Construction fails here
        rather than lazily on first FK call, so a missing asset tree is
        diagnosable at the point a ``GR1Spec`` is requested, not three metrics
        deep into a scoring run.

    Attributes
    ----------
    All ``RobotSpec`` protocol attributes, plus (see module docstring):

    colliders:
        The wrapped :class:`RobotColliders`.
    fk:
        The wrapped :class:`GR1FK`, for callers needing the lower-level
        per-side ``ee_pose`` / ``fingers_fk`` helpers that have no
        ``RobotSpec``-protocol equivalent.

    ``keypoint_links`` layout
    --------------------------
    ``KEYPOINTS_LEFT + KEYPOINTS_RIGHT`` (``K = 12``): indices ``0..5`` are the
    left arm (shoulder -> ... -> end effector), ``6..11`` the right arm, in the
    same per-arm order ``GR1FK.keypoints_fk`` returns. :meth:`ee_sites` names
    indices 5 and 11 (the two ``*_end_effector_link`` keypoints).
    """

    #: Registry key (kinescore.robots.get_robot).
    name = "fourier_gr1"
    #: Predicted DOF: left arm (7) + right arm (7) + waist (3). See module docstring.
    n_joints = GR1FK.N_Q

    def __init__(self, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32,
                 urdf_path: str | None = None) -> None:
        if urdf_path is None:
            urdf_path = str(resolve_asset_urdf(GR1_URDF_RELPATH))
        self.urdf_sha256: str | None = sha256_file(urdf_path)

        self.fk = GR1FK(urdf_path, device=device, dtype=dtype)
        self.colliders = RobotColliders(urdf_path)

        self.keypoint_links: tuple[str, ...] = KEYPOINTS_LEFT + KEYPOINTS_RIGHT
        n_left = len(KEYPOINTS_LEFT)

        # ---- joint limits (17,); GR1FK has no effort data -> no EFFORT_LIMITS
        self.q_lo = self.fk.q_lo.clone()
        self.q_hi = self.fk.q_hi.clone()
        self.vel_limits: torch.Tensor | None = self.fk.q_vel_max.clone()
        self.effort_limits: torch.Tensor | None = None

        # ---- bones: concatenate the two per-arm chains, right offset by n_left
        self.bone_pairs = torch.cat(
            [self.fk.bone_pairs_left, self.fk.bone_pairs_right + n_left], dim=0)
        self.bone_lengths = torch.cat(
            [self.fk.bone_lengths_left, self.fk.bone_lengths_right], dim=0)
        # No GR-1 arm bone is degenerate at rest (shortest measured 0.023 m,
        # far above DEGENERATE_BONE_M) -- unlike the Franka gripper, no arm
        # keypoint here is driven by a non-arm actuator, so the library
        # default threshold is the right one and nothing is dropped silently.
        mask = rigid_bone_mask(self.bone_lengths)
        self.rigid_bone_pairs = self.bone_pairs[mask]
        self.rigid_bone_lengths = self.bone_lengths[mask]

        self.capabilities: frozenset[str] = frozenset(
            {Capability.ROTATIONS, Capability.COLLIDERS, Capability.SUPPORT_POLYGON})

    # ------------------------------------------------------------------ #
    # RobotSpec protocol
    # ------------------------------------------------------------------ #
    def forward_kinematics(self, q: torch.Tensor,
                           aux: Any | None = None) -> torch.Tensor:
        """``(B,T,17) -> (B,T,12,3)``. ``aux`` is unused (reserved for hand DoF;
        see ``GR1FK.hand_flexion_mean`` -- the arm/EE keypoint chain this
        method reads ends before the fingers, so no hand data feeds it)."""
        left = self.fk.keypoints_fk(q, "left")
        right = self.fk.keypoints_fk(q, "right")
        return torch.cat([left, right], dim=2)

    def forward_transforms(self, q: torch.Tensor, aux: Any | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B,T,17) -> P (B,T,12,3), R (B,T,12,3,3)``. ``aux`` unused, see above."""
        p_l, r_l = self.fk.forward_transforms(q, "left")
        p_r, r_r = self.fk.forward_transforms(q, "right")
        return torch.cat([p_l, p_r], dim=2), torch.cat([r_l, r_r], dim=2)

    def ee_sites(self) -> tuple[int, ...]:
        """Two end-effector sites: left, then right ``*_end_effector_link``."""
        return (self.keypoint_links.index(EE_LINK["left"]),
                self.keypoint_links.index(EE_LINK["right"]))

    # ------------------------------------------------------------------ #
    # COLLIDERS / SUPPORT_POLYGON extensions -- see module docstring
    # ------------------------------------------------------------------ #
    def body_collider_spheres(self, q: torch.Tensor
                              ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B,T,17) -> (centers (B,T,S,3), radii (S,))`` torso/base/head spheres.

        Constant across ``q`` -- see the module docstring's "predicted state
        excludes the legs" note: none of :attr:`colliders`'s ``body_links``
        are reachable from the 17 predicted joints.
        """
        frames = self.fk.link_frames(q, self.colliders.body_links)
        return self.colliders.posed_body_spheres(frames)

    def world_com(self, q: torch.Tensor) -> torch.Tensor:
        """``(B,T,17) -> (B,T,3)`` whole-body centre of mass in base frame.

        Unlike :meth:`body_collider_spheres`, this *does* vary with ``q``: the
        arm links carry mass and are reachable from the predicted joints, so
        moving an arm shifts the computed CoM even though the legs do not move.
        """
        frames = self.fk.link_frames(q, self.colliders.mass_links)
        return self.colliders.world_com(frames)

    def support_polygon(self, q: torch.Tensor) -> torch.Tensor:
        """``(B,T,17) -> (B,T,2,3)`` foot positions (left, right) in base frame.

        Constant across ``q`` (feet are not reachable from the 17 predicted
        joints -- see module docstring). Still meaningful as the footprint a
        per-frame :meth:`world_com` can be checked against.
        """
        frames = self.fk.link_frames(q, self.colliders.foot_links)
        return frames[..., :3, 3]
