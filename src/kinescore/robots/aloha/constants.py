"""URDF joint/link names for ALOHA bimanual, and the verified real-data mapping.

Joint semantics -- VERIFIED against real data, not assumed
-------------------------------------------------------------
``legacy_docs/ADDING_ALOHA_NOTES.md`` (the prior round's handoff) hypothesised the
14-dim ``action`` / 42-dim ``observation.state`` layout from the published
ALOHA/ACT convention alone, without loading a parquet. This round did:

* ``meta/modality.json`` (every one of the 16
  ``video_gen_physics_real_video/bimanual/multiview/makovian/<task>/`` trees
  carries an identical one) states explicitly::

      state:  qpos[0:14], qvel[14:28], effort[28:42]
      action: qpos[0:14]

  i.e. ``observation.state`` is ``[qpos(14), qvel(14), effort(14)]`` and
  ``action`` is a 14-dim commanded ``qpos`` -- confirming the prior round's
  hypothesis exactly, this time from the dataset's own metadata rather than
  the general ACT convention.
* ``meta/stats.json`` (``close_cardboard_box`` task) per-column min/max was
  cross-checked against ``aloha_bimanual.urdf``'s ``<limit>`` values:
  ``action`` index 1 ranges ``[-1.275, 0.253]`` against the URDF's
  ``left/shoulder`` limit ``[-1.850, 1.257]`` (upper bound nearly saturated);
  index 2 ranges ``[-0.112, 1.578]`` against ``left/elbow``'s
  ``[-1.763, 1.606]`` (upper bound nearly saturated); index 8 ranges
  ``[-1.229, -0.018]`` against ``right/shoulder``'s ``[-1.850, 1.257]``
  (entirely on the lower half, mirroring index 1's arm on the other side);
  index 9 ranges ``[0.808, 1.552]`` against ``right/elbow``'s
  ``[-1.763, 1.606]`` (entirely on the upper half, mirroring index 2). Indices
  6 and 13 (``[-0.618,-0.130]`` / ``[-0.598,-0.034]``) both sit in the real
  Interbotix vx300s gripper joint's practical range and are symmetric between
  the two arms -- the seventh ("gripper") slot per arm.
* A raw parquet read (``episode_000026.parquet``, ``close_cardboard_box``)
  confirms ``observation.state[:, :14]`` tracks ``action`` closely
  frame-to-frame for the six arm channels per side (sub-0.03 rad delta) and
  diverges more for the two gripper channels (~0.4-0.6 rad delta, consistent
  with ``action`` being a *commanded* target the slower gripper servo has not
  yet reached) -- exactly the "state tracks a commanded action" relationship
  expected of ``qpos``/target pairs, not two independent quantities.

This confirms the per-arm 7-slot order
``[waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper]``
(matching ``tonyzhaozh/aloha``'s ``JOINT_NAMES`` -- and, independently, this
URDF's own per-arm joint declaration order, see ``aloha_bimanual.urdf``),
left arm first then right (``action[0:7]`` = left, ``action[7:14]`` = right),
mirroring how GR-1's 44-dim state orders left-then-right.

Only the 12 arm joints (6 revolute per side, indices 0-5 and 7-12 of the
14-slot layout) are predicted here -- see ``fk.py``'s module docstring for
why the 7th slot per side (gripper) is carried through ``aux`` instead of
``q``, the same split :class:`~kinescore.robots.franka.spec.FrankaSpec` uses
for the Panda's finger joints.
"""
from __future__ import annotations

__all__ = [
    "LEFT_ARM_JOINTS", "RIGHT_ARM_JOINTS",
    "LEFT_FINGER_JOINT", "RIGHT_FINGER_JOINT",
    "KEYPOINTS_LEFT", "KEYPOINTS_RIGHT", "EE_LINK",
    "ACTUATED_LINKS", "FINGER_LO_M", "FINGER_HI_M",
]

#: URDF joint names (`aloha_bimanual.urdf`), matching the parquet's
#: `action`/`observation.state` qpos block order (see module docstring):
#: `[waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]` per arm
#: (the 7th, gripper, slot is `LEFT_FINGER_JOINT`/`RIGHT_FINGER_JOINT` below,
#: carried through `aux` rather than the predicted `q` -- see fk.py).
LEFT_ARM_JOINTS: tuple[str, ...] = (
    "left/waist", "left/shoulder", "left/elbow",
    "left/forearm_roll", "left/wrist_angle", "left/wrist_rotate",
)
RIGHT_ARM_JOINTS: tuple[str, ...] = (
    "right/waist", "right/shoulder", "right/elbow",
    "right/forearm_roll", "right/wrist_angle", "right/wrist_rotate",
)

#: Per-arm keypoint links (shoulder -> ... -> gripper -> fingers -> TCP), K=9.
#: `{side}/left_finger_link` / `{side}/right_finger_link` are driven by the
#: prismatic finger joints (gripper aux), not the 6 predicted arm joints --
#: the Franka `ACTUATED_LINKS` situation, mirrored here; see
#: `ACTUATED_LINKS` below and `docs/ARCHITECTURE.md#adding-a-robot`'s D9 checklist.
KEYPOINTS_LEFT: tuple[str, ...] = (
    "left/shoulder_link", "left/upper_arm_link", "left/upper_forearm_link",
    "left/lower_forearm_link", "left/wrist_link", "left/gripper_link",
    "left/left_finger_link", "left/right_finger_link", "left/ee_gripper_link",
)
KEYPOINTS_RIGHT: tuple[str, ...] = (
    "right/shoulder_link", "right/upper_arm_link", "right/upper_forearm_link",
    "right/lower_forearm_link", "right/wrist_link", "right/gripper_link",
    "right/left_finger_link", "right/right_finger_link", "right/ee_gripper_link",
)

#: TCP keypoint per side. Reachable from `{side}/gripper_link` through ONLY
#: fixed joints (`ee_arm` -> `gripper_bar` -> `ee_bar` -> `ee_gripper`, all
#: `type="fixed"` in the URDF) -- confirmed by tracing the URDF's kinematic
#: tree, not assumed. So its OWN pose never moves with gripper actuation; it
#: is included in ACTUATED_LINKS-based exclusion only because the CONSECUTIVE
#: bone `right_finger_link -> ee_gripper_link` has an actuated left endpoint
#: (see the D9 note on `ACTUATED_LINKS`), not because this link itself moves.
EE_LINK: dict[str, str] = {"left": "left/ee_gripper_link", "right": "right/ee_gripper_link"}

#: Prismatic finger joint names, one mimic pair per side (see fk.py's aux
#: scatter). `left_finger`'s URDF range is `[0.021, 0.057]` m; `right_finger`'s
#: is the elementwise negation `[-0.057, -0.021]` m (a mirror-image slide, not
#: an independent DoF -- both move together as the gripper opens/closes).
LEFT_FINGER_JOINT: dict[str, str] = {"left": "left/left_finger", "right": "right/left_finger"}
RIGHT_FINGER_JOINT: dict[str, str] = {"left": "left/right_finger", "right": "right/right_finger"}
FINGER_LO_M: float = 0.021   # closed (aux gripper = 0)
FINGER_HI_M: float = 0.057   # open   (aux gripper = 1)

#: Links whose pose is driven by the gripper `aux` channel, not by one of the
#: 12 predicted arm joints -- the Franka finger situation, doubled for two
#: arms. A `rigid_bone_pairs` bone with either endpoint in this set is
#: excluded (D9's structural rule); see `spec.py::AlohaSpec._rigid_bone_mask`.
ACTUATED_LINKS: frozenset[str] = frozenset({
    "left/left_finger_link", "left/right_finger_link",
    "right/left_finger_link", "right/right_finger_link",
})
