"""Franka Panda constants, extracted verbatim from the source ``fk.py``.

Every numeric literal here is byte-identical to
``judge/fk.py`` (Marionette-ciasc) / ``models/kinematics/fk.py`` (fkjepa) --
see ``robots/franka/fk.py``'s module docstring for the provenance note. This
module exists only so :class:`~kinescore.robots.franka.fk.FrankaFK` (the
ported class body) and :class:`~kinescore.robots.franka.spec.FrankaSpec` (the
``RobotSpec`` wrapper) can share one definition instead of two copies drifting
apart.
"""
from __future__ import annotations

__all__ = [
    "KEYPOINT_LINKS",
    "PANDA_JOINT_LOWER",
    "PANDA_JOINT_UPPER",
    "PANDA_VEL_LIMITS",
    "PANDA_EFFORT_LIMITS",
    "PANDA_ARM_JOINTS",
    "PANDA_FINGER_JOINTS",
    "PANDA_FINGER_MAX",
    "LINK_FALLBACKS",
    "RIGID_BONE_MIN_M",
]

#: Default keypoint link names (``K = 8``). All present in the
#: ``robot_descriptions`` example-robot-data Panda URDF.
KEYPOINT_LINKS: tuple[str, ...] = (
    "panda_link1",
    "panda_link3",
    "panda_link5",
    "panda_link7",
    "panda_hand",
    "panda_leftfinger",
    "panda_rightfinger",
    "panda_hand_tcp",
)

# Panda arm joint limits (radians), in joint1..joint7 order.
PANDA_JOINT_LOWER: tuple[float, ...] = (
    -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
)
PANDA_JOINT_UPPER: tuple[float, ...] = (
    2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973,
)
# Franka Panda datasheet limits (joint1..joint7) for physics feasibility checks.
PANDA_VEL_LIMITS: tuple[float, ...] = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)   # rad/s
PANDA_EFFORT_LIMITS: tuple[float, ...] = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)    # Nm

# Names of the 7 revolute arm joints, in canonical order.
PANDA_ARM_JOINTS: tuple[str, ...] = (
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
)
# Names of the 2 prismatic finger joints.
PANDA_FINGER_JOINTS: tuple[str, ...] = (
    "panda_finger_joint1", "panda_finger_joint2",
)

# Max opening (metres) of a single Panda finger joint (full open ~= 0.08 m span).
PANDA_FINGER_MAX: float = 0.04

# Fallback remapping for keypoint links that may be absent from some URDFs.
LINK_FALLBACKS: dict[str, tuple[str, ...]] = {
    # The example-robot-data Panda URDF has no explicit grasp target; the TCP
    # frame (panda_hand_tcp) is the natural stand-in.
    "panda_grasptarget": ("panda_hand_tcp", "panda_hand"),
}

# ---------------------------------------------------------------------------
# Rigid-bone threshold for FrankaSpec.rigid_bone_pairs (see spec.py).
# ---------------------------------------------------------------------------
#
# The consecutive-keypoint chain (see FrankaFK._compute_rest_bones) produces
# 7 bones from the 8 keypoints in KEYPOINT_LINKS, with rest-pose lengths
#
#   [0.316, 0.384, 0.088, 0.107, 0.0584, 0.0, 0.045]
#            link1  link3  link5  link7   hand  left  right
#             ->     ->     ->    -> hand  ->   finger-> hand
#            link3  link5  link7  -hand  leftfg  rightfg  tcp
#
# Bone 5 (panda_leftfinger -> panda_rightfinger) is the one with rest length
# EXACTLY 0.0 m -- the fingers coincide at the closed-gripper rest pose used to
# compute it -- and is the bone `kinescore.core.robot.rigid_bone_mask`'s
# default 1 mm threshold (DEGENERATE_BONE_M) is built to catch: a rigid,
# motionless arm that merely opens its gripper measures a nonzero
# "rigidity_residual_mm" purely because bone 5's realised length IS the
# gripper opening (see docs/PROVENANCE.md, defect D9).
#
# But bones 4 (panda_hand -> panda_leftfinger, 0.0584 m) and 6
# (panda_rightfinger -> panda_hand_tcp, 0.045 m) carry the same defect for a
# less obvious reason: panda_leftfinger / panda_rightfinger translate along
# the *prismatic* finger joints as the gripper opens and closes, so both
# bones' lengths also change with gripper actuation, not with any actual
# flex of the rigid 7-DOF arm structure. They are just as unsuitable as bone 5
# for a rigidity residual, they simply don't happen to start at 0.0 m.
#
# A rest-length threshold cannot distinguish "coincident" from "flexes with
# gripper actuation" -- a rest length says nothing about how a bone varies over
# time. A threshold that separates the two on THIS robot (anything between
# 0.0584 m and 0.088 m) only works by arithmetic coincidence: it would wrongly
# drop a short-but-genuinely-rigid link on a different robot, and wrongly keep a
# long actuated one. Distance is a proxy; the real criterion is *which joint
# moves the bone*.
#
# So the primary rule is structural: a bone is excluded if either endpoint is an
# ACTUATED link -- one whose pose depends on a joint outside the predicted arm
# state. For the Panda that is exactly the two prismatic fingers. This is
# derived from the kinematic chain, transfers to any robot, and needs no
# magic number.
ACTUATED_LINKS: frozenset[str] = frozenset({
    "panda_leftfinger",
    "panda_rightfinger",
})

# Retained only as a SECOND, independent safety net, applied on top of the
# structural rule: a bone whose endpoints are coincident at rest carries no
# rigidity information regardless of what actuates it, and would divide by ~0.
# This is the library default (kinescore.core.robot.DEGENERATE_BONE_M), stated
# here so the Franka's exclusion set is readable in one place.
RIGID_BONE_MIN_M: float = 1e-3
