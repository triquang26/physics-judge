# ALOHA / bimanual handoff notes (descoped mid-implementation)

Bimanual support (`kinescore.robots.aloha`) was assigned, investigated, and
then explicitly descoped before any source file under `src/kinescore/` was
written -- the first benchmark run prioritizes humanoid (GR-1) and single_arm
(Franka), both of which already exist. **No `kinescore.robots.aloha` module
exists.** This note is the handoff so the next round does not repeat the
URDF-sourcing work.

## What exists already

A usable, kinematics-only bimanual URDF has been prepared and is sitting in
the asset tree at `$KINESCORE_ASSETS/aloha/urdf/aloha_bimanual.urdf` (29
links, 28 joints, sha256 in `$KINESCORE_ASSETS/MANIFEST.json`). It is **not**
referenced by any kinescore source file. Full provenance is in that
MANIFEST; summary:

- Per-arm kinematics: Interbotix's `vx300s.urdf.xacro`
  (`Interbotix/interbotix_ros_manipulators`, commit
  `0bb2b0e6d0e619bff02cf74dbd5af5681dcf80c9`), xacro-expanded twice
  (`robot_name=left` / `robot_name=right`) with the `xacro` PyPI package
  (installed to user site, not a kinescore dependency -- a one-time
  asset-prep tool, not a runtime dependency).
- Left/right base-mount transform (the two arms are ~0.94 m apart, the
  right arm rotated 180 deg about Z): read out of Mujoco Menagerie's
  `aloha/aloha.xml` (`google-deepmind/mujoco_menagerie`), body
  `left/base_link` at `pos="-0.469 -0.019 0.02"` identity rotation,
  `right/base_link` at `pos="0.469 -0.019 0.02" quat="0 0 0 1"`.
- Merge (renaming joints with a `left/`/`right/` prefix -- the Interbotix
  xacro namespaces link names via its `robot_name` arg but **not** joint
  names, so two raw expansions collide on `waist`, `shoulder`, etc. -- then
  adding a synthetic `world` root with two fixed mount joints) is a small
  hand-written script, kept alongside the pre-merge files at
  `$KINESCORE_ASSETS/aloha/urdf/vx300s_src/` for audit.
- **License: BSD-3-Clause** for both upstream sources (Trossen Robotics
  copyright on both the Interbotix xacro and the Menagerie ALOHA model).
- **No meshes fetched.** This is a kinematics-only URDF (joint
  origins/axes/limits/inertials). A `COLLIDERS` capability would need mesh
  geometry or hand-authored capsule primitives -- deferred along with
  everything else here.
- `robot_descriptions` (already a kinescore core dependency) *does* ship
  `aloha_mj_description`, but **only as MJCF**, and `pytorch_kinematics`'s
  MJCF path (`pytorch_kinematics.mjcf`) requires the `mujoco` package,
  which is not installed and not a kinescore dependency. That is why the
  URDF (xacro) route was used instead of parsing the MJCF directly -- worth
  revisiting only if `mujoco` becomes an accepted dependency.

## What was inferred but NOT verified against the real parquet

`$KINESCORE_DATA_ROOT`'s `video_gen_physics_real_video/` bimanual data had
not landed on disk at the time this was investigated (directory existed but
was empty). Everything below is inferred from the published ALOHA/ACT
convention, not read from an actual episode:

- **Joint order per arm**: `[waist, shoulder, elbow, forearm_roll,
  wrist_angle, wrist_rotate, gripper]` (6 revolute arm joints + 1 gripper),
  matching `tonyzhaozh/aloha` / mobile-ALOHA's `JOINT_NAMES` convention.
- **14-dim `action` layout**: `[left(7), right(7)]`, i.e. left arm's 7
  values first, then right arm's 7 -- the standard ordering across
  bimanual-arm datasets (mirrors how GR-1's own 44-dim state block orders
  left-then-right).
- **42-dim `observation.state`**: hypothesized to be `[qpos(14), qvel(14),
  effort(14)]`, matching the `sensor_msgs/JointState`-shaped
  position/velocity/effort triple ALOHA's own recording pipeline logs per
  arm. This is a strong prior (it is exactly what the ROS-side ALOHA
  teleop stack records) but **was not confirmed** by loading an actual
  `.parquet` file -- do that first before hardcoding a slice in
  `constants.py`.

**Action item for the next round**: load one episode's parquet from
`video_gen_physics_real_video/bimanual/...` and check both of the above
against the real `action` / `observation.state` columns before writing
`robots/aloha/constants.py`. Getting the joint order wrong silently
corrupts every FK output; this is exactly the class of mistake
`docs/ADDING_A_ROBOT.md` warns about.

## Keypoint / D9 plan (not implemented, but worked out)

For whoever picks this up: the natural per-arm keypoint chain (mirroring
Franka's 8-keypoint, D9-aware pattern rather than GR-1's finger-free one) is

```
shoulder_link -> upper_arm_link -> upper_forearm_link -> lower_forearm_link
  -> wrist_link -> gripper_link -> left_finger_link -> right_finger_link
  -> ee_gripper_link
```

(K=9 per arm, 18 total.) `left_finger_link` / `right_finger_link` are driven
by the prismatic finger joints (the gripper DoF), not the 6 arm joints --
exactly Franka's `ACTUATED_LINKS` situation, and it drops exactly 3 of the 8
per-arm bones from `rigid_bone_pairs` for the same reason (gripper_link ->
left_finger_link, left_finger_link -> right_finger_link, right_finger_link
-> ee_gripper_link). `ee_gripper_link` itself sits behind only **fixed**
joints from `gripper_link`, so it is a clean, gripper-independent TCP
keypoint -- confirmed by tracing the URDF's kinematic tree, not assumed.

## `metrics/feasibility.py:84 _arm_capsule_spheres` and `("left","right")`

Checked, not modified (out of scope -- owned by whoever maintains
`metrics/`). The hardcoded `_SIDES = ("left", "right")` iteration in that
module, and `_arm_capsule_spheres`'s call to `robot.fk.keypoints_fk(q,
side)`, both assume a `GR1FK`-shaped `robot.fk` object with a
`keypoints_fk(q, side)` method taking a *side string* and returning that
side's own keypoint chain. **This would work unmodified for an ALOHA
`RobotSpec`** built the way this note describes (an `AlohaFK` wrapping the
merged `aloha_bimanual.urdf` chain, exposing `keypoints_fk(q, "left"/
"right")` with the same signature) -- the string literals `"left"`/`"right"`
already match this URDF's own link-name prefixes, which is not a
coincidence: the prefix choice in the merge script was made specifically to
line up with this convention. The only real prerequisite is declaring
`Capability.COLLIDERS`, which was NOT attempted (no mesh geometry, no capsule
radii chosen, no body/torso spheres to check the arms against -- ALOHA has
no torso collider analogous to GR-1's, since it's two arms bolted to a
table, not a full body). If `COLLIDERS` were declared without addressing
that, `body_collider_spheres` would need a from-scratch definition (there is
no torso to collide with; the meaningful checks would be arm-vs-arm and
arm-vs-table, not arm-vs-body).

## `SUPPORT_POLYGON` / `com_margin_m`

Not applicable and must not be declared -- ALOHA is bolted to a table
(exactly like Franka). `com_margin_m` must be `NaN` with a
`missing_capability:SUPPORT_POLYGON`-style reason for this robot, never a
fabricated `0.0`. This matches `metrics/feasibility.py`'s own module
docstring and `core/robot.py`'s "a table-mounted arm has no balance margin"
argument.

## `limit_semantics`

Must be `"raw_rad"` from the start (this was the whole point of doing
ALOHA at all, per the plan) -- do not repeat Franka's `squashed`-head
mistake (defect D7, structurally-zero `limit_violation_frac`).
