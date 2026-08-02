# Adding a robot

A robot is one implementation of the `RobotSpec` protocol
(`src/kinescore/core/robot.py`) — everything the metric layer needs to know
about an embodiment: forward kinematics, joint limits, which keypoints form
rigid bones, and which optional capabilities (collision geometry, a support
polygon) it has. Two implementations ship:

- **`Synthetic2R`** (`src/kinescore/robots/synthetic.py`) — a 2-link planar
  arm, closed-form FK, no URDF, no network, no `pytorch_kinematics`. The
  simplest possible worked example; every field below has an obvious answer
  for it.
- **`GR1Spec`** (`src/kinescore/robots/gr1/spec.py`) — the Fourier GR-1
  bimanual humanoid: two arm chains, collision geometry, a support polygon,
  and a state vector that only partially reaches the URDF's full joint count.
  The hardest worked example; every field below has a non-obvious answer for
  it, worth reading in full.

Read `core/robot.py`'s module docstring before starting — it explains *why*
`bone_pairs`/`rigid_bone_pairs` are two separate fields (defect D9) and why
`capabilities` gates rather than a robot just implementing every method.

## The protocol, member by member

### `name: str`

Registry key. `Synthetic2R.name = "synthetic_2r"`; `GR1Spec.name =
"fourier_gr1"`. Register your factory in
`src/kinescore/robots/__init__.py::_FACTORIES` — **lazily**: the module
docstring there is explicit that importing `kinescore.robots` must never
require `pytorch_kinematics`/`robot_descriptions`, so both `_build_franka`
and `_build_gr1` do their heavy imports *inside* the factory function, not at
module scope. `Synthetic2R` is the one exception, imported eagerly, because
it has no such dependency and is the whole point of the CPU-only test path.
If your new robot needs a heavy optional dependency, follow the
`_build_franka`/`_build_gr1` pattern, not the `Synthetic2R` one.

### `n_joints: int`

Degrees of freedom the pose reader predicts and FK consumes.
`Synthetic2R.n_joints = 2`. `GR1Spec.n_joints = GR1FK.N_Q = 17` — **not** the
URDF's full joint count (44 for the GR-1); it is `[left_arm(7),
right_arm(7), waist(3)]`, the subset the pose reader this benchmark scores
was actually trained to predict. **This is the single most important
decision for a humanoid or any robot the reader doesn't fully observe**: pick
the predicted subset, not the mechanical total, and be explicit (in your
robot's module docstring, the way `robots/gr1/fk.py` and `robots/gr1/spec.py`
both do at length) about which links therefore sit at a fixed rest pose in
every FK call rather than tracking the video.

### `keypoint_links: tuple[str, ...]`

Ordered link names whose origins become the keypoints `P`.
`Synthetic2R.keypoint_links = ("base", "elbow", "tip")` (K=3).
`GR1Spec.keypoint_links = KEYPOINTS_LEFT + KEYPOINTS_RIGHT` (K=12, 6 per arm,
shoulder→...→end-effector). Order matters: it is baked into `bone_pairs`
(consecutive-index pairs) and into `ee_sites()`'s returned indices.

### `bone_pairs`, `bone_lengths` / `rigid_bone_pairs`, `rigid_bone_lengths`

`(n_bones,2)` index pairs into the keypoint axis and their rest lengths in
metres. **`bone_pairs`/`bone_lengths` is the full, legacy-reproducible set;
`rigid_bone_pairs`/`rigid_bone_lengths` is what rigidity metrics actually use
by default** — see the mandatory checklist item below (D9) for why these must
differ whenever any keypoint's position depends on something other than the
robot's own rigid-arm joints.

`Synthetic2R` builds both identically (`self.rigid_bone_pairs =
self.bone_pairs.clone()`) — its own docstring states why: "neither [bone] is
degenerate (both link lengths are > 0 by construction), so `bone_pairs ==
rigid_bone_pairs` — there is nothing to drop, unlike the Franka gripper."

`GR1Spec` concatenates the two arms' bones
(`self.fk.bone_pairs_left`/`_right`, offsetting the right arm's indices by
`n_left`) and applies `rigid_bone_mask(self.bone_lengths)` — the *library
default* threshold (`core.robot.DEGENERATE_BONE_M = 1e-3` m) — because "no
GR-1 arm bone is degenerate at rest (shortest measured 0.023 m, far above
`DEGENERATE_BONE_M`) ... no arm keypoint here is driven by a non-arm
actuator, so the library default threshold is the right one and nothing is
dropped silently" (`GR1Spec.__init__`'s comment). Note what this means:
**GR-1 needs no gripper-style exclusion because none of its arm keypoints are
actuated by anything outside the predicted 17 joints** — the degenerate-bone
problem is a Franka-specific consequence of the gripper, not a general
property every robot must special-case, but every robot must still *check*.

### `q_lo`, `q_hi: (n_joints,)`

Joint position limits in radians. `Synthetic2R` picks `[-pi, pi]` per joint —
"a full rotation range is the least assumption-laden default and keeps
joint-limit-violation metrics well-defined (never trivially zero, never
trivially saturated)" for a robot with no real datasheet. `GR1Spec` reads
these from `GR1FK.q_lo`/`q_hi`, which are the URDF's own `<limit>` values
widened by `_LIMIT_MARGIN = 0.20` rad — "teleop occasionally exceeds the URDF
limit slightly (e.g. right_wrist_pitch), so widen the squash range a touch to
stay valid" (`robots/gr1/fk.py`).

### `vel_limits`, `effort_limits: Optional[(n_joints,)]`

Rated joint velocity (rad/s) and effort (N·m), or `None` when the URDF
doesn't declare them. `Synthetic2R` sets both `None` — no real datasheet
exists for a synthetic arm. `GR1Spec.vel_limits = self.fk.q_vel_max.clone()`
(read from the URDF's `<limit velocity="...">`); `GR1Spec.effort_limits =
None` always — "`GR1FK` has no effort data at all" (`gr1/spec.py`'s comment).
This is the correct, honest answer, not a gap to fill in: `effort_proxy`
(`metrics/joint_dynamics.py`) is `NaN` with reason
`missing_input:effort_limits` for every GR-1 clip, and that is accurate — the
alternative (fabricating a number) is exactly what `core/robot.py`'s module
docstring warns against.

### `capabilities: frozenset[str]`

From `core.robot.Capability`: `COLLIDERS`, `SUPPORT_POLYGON`, `ROTATIONS`,
`EFFORT_LIMITS`. `Synthetic2R.capabilities = {ROTATIONS}` only (a planar arm
with a well-defined in-plane orientation at each keypoint, no gripper, no
feet, no ported collision geometry). `GR1Spec.capabilities = {ROTATIONS,
COLLIDERS, SUPPORT_POLYGON}` (backed by the wrapped `RobotColliders` +
`GR1FK.link_frames`). A capability you declare but don't actually back with
real data is worse than not declaring it — `core/robot.py`'s module docstring
is explicit: "a table-mounted arm has no balance margin, and reporting `0`
for it would be a lie." Declare a capability only when the concrete class
also implements the extension methods the corresponding metrics expect
(`body_collider_spheres`, `world_com`, `support_polygon` for `GR1Spec` — see
`metrics/feasibility.py`'s module docstring for the exact extension surface a
metric may rely on).

### `urdf_sha256: Optional[str]`

Hash of the URDF FK was built from — `None` only when there genuinely was no
URDF (`Synthetic2R`, closed-form FK). `GR1Spec.urdf_sha256 =
sha256_file(urdf_path)`, computed once at construction from
`robots/urdf.py::sha256_file`. This is not optional bookkeeping: a silent
upstream URDF version bump (a corrected mesh, a different variant) shifts
keypoints by millimetres in a way that is otherwise invisible until an
unrelated metric looks slightly off weeks later — see `robots/urdf.py`'s
module docstring for the full argument.

### `forward_kinematics(q, aux=None) -> (B,T,K,3)`

Keypoint positions in the robot base frame. `Synthetic2R`: two `sin`/`cos`
evaluations, closed form. `GR1Spec`: `torch.cat([self.fk.keypoints_fk(q,
"left"), self.fk.keypoints_fk(q, "right")], dim=2)` — delegates to the wrapped
(ported, verbatim) `GR1FK`. `aux` carries non-joint-angle state — the Franka
gripper opening in `[0,1]` for `FrankaSpec`, unused (reserved for hand DoF)
for `GR1Spec`. `Synthetic2R` accepts `aux` for protocol compliance and
ignores it (no such state).

### `forward_transforms(q, aux=None) -> (P, R)`

Same as above, plus per-keypoint `(B,T,K,3,3)` rotation matrices — required
whenever `Capability.ROTATIONS` is declared, since the angular-dynamics
metrics (`metrics/angular.py`) read `ctx.R`. `Synthetic2R`'s `R` is each
keypoint's in-plane orientation as a z-axis rotation — its own docstring is
explicit this is "the natural rigid-body frame for a revolute-jointed planar
link ... not an arbitrary placeholder — it is what makes `ROTATIONS`-gated
metrics ... exercise real, distinct-per-frame values on this robot instead of
a constant identity." A robot that declares `ROTATIONS` but returns a
constant identity `R` would make every angular metric silently report zero
motion regardless of the actual video — worth stress-testing explicitly (see
the checklist below).

### `ee_sites() -> tuple[int, ...]`

Keypoint indices treated as end-effectors, for the smoothness metrics
(`metrics/smoothness.py`'s `sparc`/`log_dimensionless_jerk`, which aggregate
over `robot.ee_sites()` rather than a hardcoded left/right pair). One index
for a single arm (`Synthetic2R.ee_sites() = (2,)`, the tip), two for a
bimanual robot (`GR1Spec.ee_sites()` looks up
`self.keypoint_links.index(EE_LINK["left"])` and `[...]["right"]` by name,
not a hardcoded `(5, 11)` — robust to a future reordering of
`keypoint_links`).

## Mandatory checklist

1. **Ship a `golden_fk_<robot>.npz` fixture.** Follow
   `tools/gen_golden.py::golden_fk`/`golden_gr1_fk`'s pattern: seed
   everything (`torch.manual_seed(0)`), FK a batch of random-but-in-limits
   poses plus a few handwritten poses (zeros, a "home" pose you define and
   document, a limit-corner pose), and freeze `P`/`R`/`bone_pairs`/
   `bone_lengths`/`q_lo`/`q_hi` to `.npz`. This is what lets a future refactor
   of your robot's FK be checked for regressions the same way
   `test_fk_parity_franka.py` checks Franka's.
2. **Assert literal rest-pose bone lengths.** Not "some positive number" —
   the *actual* millimetre value, the way
   `robots/franka/constants.py`'s comment block spells out all seven Panda
   bone lengths (`[0.316, 0.384, 0.088, 0.107, 0.0584, 0.0, 0.045]`) by name.
   A rest-pose bone length that silently drifts because someone edited a
   keypoint-link list is exactly the class of regression a symbolic "> 0"
   assertion would miss. See `tests/test_fk_rest_pose.py` for the pattern.
3. **Check for degenerate/actuated bones (D9).** Before shipping
   `rigid_bone_pairs`, ask: does any pair of *consecutive* keypoints in
   `keypoint_links` coincide at rest, or track a joint **outside** the robot's
   predicted `n_joints`? If yes, that bone must be excluded from
   `rigid_bone_pairs` by the **structural** rule first (an endpoint link is
   driven by a non-predicted actuator — see
   `robots/franka/spec.py::FrankaSpec._rigid_bone_mask`'s two-rule pattern,
   `ACTUATED_LINKS` for the Franka's concrete instance), with
   `core.robot.rigid_bone_mask`'s rest-length threshold
   (`DEGENERATE_BONE_M = 1e-3` m) as a **second, independent** safety net —
   not the only check, since a distance threshold alone cannot distinguish
   "coincident" from "flexes with actuation but has nonzero rest length" (see
   the Franka's bones 4 and 6, both non-degenerate by length yet still
   actuation-tracking — `robots/franka/constants.py`'s comment walks through
   exactly this). If your robot genuinely has no such bone (like both
   `Synthetic2R` and GR-1's arms), **say so explicitly** in the robot's module
   docstring — silence there reads as "nobody checked," not "verified clean."
   Pin whichever conclusion you reach with a test analogous to
   `tests/test_rigidity_gripper_contamination.py` /
   `tests/test_robot_degenerate_bones.py`.

## Asset policy

**Never vendor a URDF into this repository.** The GR-1 asset tree alone is
~285 MB (meshes for a 44-DoF humanoid, several body variants) — vendoring it,
or piggy-backing it onto a package like `robot_descriptions`, would blow up
clone size for the common case of a Franka-only run (Franka's Panda URDF
*does* ship via `robot_descriptions`, a pinned pip dependency cached under
`~/.cache/robot_descriptions` on first use — that is the one exception, and
it is a few MB, not hundreds).

For any robot whose URDF is not already covered by `robot_descriptions`:

1. Resolve it via `$KINESCORE_ASSETS`
   (`src/kinescore/robots/urdf.py::resolve_asset_urdf`) — an operator-owned
   directory, never committed, with **no hardcoded fallback path** (see
   `src/kinescore/paths.py`'s module docstring for the class of bug a baked-in
   default path causes: "a prior extraction baked another user's home
   directory into its defaults, so a fresh checkout resolved to directories
   that did not exist"). An unset `KINESCORE_ASSETS` or a missing file raises
   `MissingPathError` immediately, at construction, naming exactly what to
   fix — never a silent fallback.
2. Record the SHA-256 of the resolved URDF on `RobotSpec.urdf_sha256` at
   construction time (see the field description above) — this is what makes
   an upstream asset-tree drift diagnosable instead of a mysterious few-mm
   keypoint shift discovered weeks later.
3. Document the expected relative path under `KINESCORE_ASSETS` in your
   robot's module (see `robots/gr1/spec.py::GR1_URDF_RELPATH`'s docstring for
   the pattern: "an operator populating `KINESCORE_ASSETS` should mirror that
   subtree").
