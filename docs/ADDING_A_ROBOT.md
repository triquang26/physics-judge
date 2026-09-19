# Adding a robot

A robot enters kinescore as three declarations and one Python module. This
page opens with a concrete, real-file walkthrough (Franka Panda: URDF text to
keypoints to training target), then the general steps, then a second worked
example (ALOHA bimanual) that exercises every step including the ones Panda
skips.

## What you must supply

| thing | where it goes | required |
|---|---|---|
| a URDF | `$KINESCORE_ASSETS/<robot>/urdf/*.urdf` | only if you need forward kinematics |
| a corpus with logged robot state | `$KINESCORE_DATA_ROOT/corpus/...` | yes, it is the training signal |
| the video packing the generator emits | `configs/views.yaml` | yes, measured from real clips |

The URDF is never vendored into the repo. GR-1 alone is about 285 MB of
meshes, so `KINESCORE_ASSETS` points at a checkout outside the source tree.

## URDF to keypoints, concretely: Franka Panda

The rest of this page talks about "the URDF" and "the keypoints" in the
abstract. This section quotes the real files for one robot that has both: a
URDF on disk and a trained checkpoint. Use it as the concrete picture behind
every step below.

The URDF quoted here is a Franka Panda file found on local disk, not
byte-identical to the one `robot_descriptions.panda_description` resolves to
at runtime (that package is not installed in every environment). Panda's
joint layout, limits and finger travel are the same across distributions,
those numbers come from Franka's datasheet either way, but this particular
file has no `panda_hand_tcp` frame, while the one kinescore actually loads
does. That gap is exactly what `LINK_FALLBACKS` exists for (see step 5).

### 1. What a joint looks like inside the file

Three joints from the real file, trimmed of a `safety_controller` tag the
parser never reads:

```xml
<joint name="panda_joint1" type="revolute">
  <origin xyz="0 0 0.333" rpy="0 0 0" />
  <parent link="panda_link0" />
  <child link="panda_link1" />
  <axis xyz="0 0 1" />
  <limit lower="-2.8973" upper="2.8973" velocity="2.175" effort="87" />
</joint>

<joint name="panda_joint2" type="revolute">
  <origin xyz="0 0 0" rpy="-1.5708 0 0" />
  <parent link="panda_link1" />
  <child link="panda_link2" />
  <axis xyz="0 0 1" />
  <limit lower="-1.7628" upper="1.7628" velocity="2.175" effort="87" />
</joint>

<joint name="panda_finger_joint1" type="prismatic">
  <origin xyz="0 0 0.0584" rpy="0 0 0" />
  <parent link="panda_hand" />
  <child link="panda_leftfinger" />
  <axis xyz="0 1 0" />
  <limit lower="0.0" upper="0.04" velocity="0.2" effort="20" />
</joint>
```

`type` says how the joint moves: `revolute` turns about `axis`, `prismatic`
slides along it, `fixed` never moves. `origin` is the fixed placement of the
child link's frame relative to the parent's, applied before the joint's own
motion. `limit` bounds that one motion; `parse_joint_limits`
(`robots/urdf.py:94`) reads exactly `lower`, `upper`, `velocity`, `effort`
off it, for `revolute`/`prismatic` joints only, and raises naming the joint
if a joint you asked for has no `<limit>` at all.

The full arm, same shape, to the wrist:

| joint | type | parent → child | origin xyz | axis |
|---|---|---|---|---|
| `panda_joint1` | revolute | `panda_link0` → `panda_link1` | `0 0 0.333` | `0 0 1` |
| `panda_joint2` | revolute | `panda_link1` → `panda_link2` | `0 0 0` | `0 0 1` |
| `panda_joint3` | revolute | `panda_link2` → `panda_link3` | `0 -0.316 0` | `0 0 1` |
| `panda_joint4` | revolute | `panda_link3` → `panda_link4` | `0.0825 0 0` | `0 0 1` |
| `panda_joint5` | revolute | `panda_link4` → `panda_link5` | `-0.0825 0.384 0` | `0 0 1` |
| `panda_joint6` | revolute | `panda_link5` → `panda_link6` | `0 0 0` | `0 0 1` |
| `panda_joint7` | revolute | `panda_link6` → `panda_link7` | `0.088 0 0` | `0 0 1` |
| `panda_joint8` | fixed | `panda_link7` → `panda_link8` | `0 0 0.107` | — |
| `panda_hand_joint` | fixed | `panda_link8` → `panda_hand` | `0 0 0` | — |
| `panda_finger_joint1` | prismatic | `panda_hand` → `panda_leftfinger` | `0 0 0.0584` | `0 1 0` |
| `panda_finger_joint2` | prismatic | `panda_hand` → `panda_rightfinger` | `0 0 0.0584` | `0 -1 0` |

A URDF is nothing but this table: a tree of links joined by joints, each
joint a fixed placement plus at most one degree of freedom. No pose lives in
the file; every position below comes from walking this tree with an actual
joint value plugged in.

### 2. Building the chain in memory

```python
chain = pk.build_chain_from_urdf(urdf_bytes)         # keyed by every link/joint name above
chain.get_joint_parameter_names()
# -> ["panda_joint1", ..., "panda_joint7", "panda_finger_joint1", "panda_finger_joint2"]
```

`pytorch_kinematics` parses the same table and keeps only the non-fixed
joints as its degrees of freedom, in file order. `panda_joint8` and
`panda_hand_joint` are fixed, so they never appear here; their `origin` is
still applied on every forward pass, it is just never variable.

### 3. Picking which links become keypoints

`robots/franka/constants.py` names 8 of the 12 links, by hand, as the
predicted keypoints:

```python
KEYPOINT_LINKS: tuple[str, ...] = (
    "panda_link1", "panda_link3", "panda_link5", "panda_link7",
    "panda_hand", "panda_leftfinger", "panda_rightfinger", "panda_hand_tcp",
)
```

Nothing infers this list. Someone read the link names above and picked eight
that trace the arm end to end: shoulder, elbow, forearm, wrist, hand, both
fingers, tool point. The position in the tuple becomes the position on the
keypoint axis everywhere downstream, `panda_link1` is index 0, `panda_hand`
is index 4, `panda_hand_tcp` is index 7, and every tensor in this pipeline
that carries a `K` dimension is `K = 8` in this order.

### 4. Predicted values become a chain input

The corpus logs 7 arm angles plus a gripper opening. `_joint_tensor`
(`robots/franka/fk.py:237`) scatters them into the chain's own joint slots by
name:

```python
th = q_flat.new_zeros(b * t, n_joints)          # one column per chain joint
for i, name in enumerate(PANDA_ARM_JOINTS):     # 7 names, in corpus order
    th[:, names.index(name)] = q_flat[:, i]
for name in PANDA_FINGER_JOINTS:                # both finger joints
    th[:, names.index(name)] = gripper * PANDA_FINGER_MAX
```

`gripper` arrives in `[0, 1]` and is scaled to metres
(`PANDA_FINGER_MAX = 0.04`) before it lands in `th`; any chain joint nothing
scatters into stays at 0. On the Panda that never happens (every non-fixed
joint is named above), but on a robot with unpredicted joints this is the
line that leaves them at their URDF rest value instead of raising.

### 5. Reading positions back out

```python
transforms = chain.forward_kinematics(th)   # dict[link_name -> Transform3d], one 4x4 per link
M = torch.stack([transforms[name].get_matrix() for name in KEYPOINT_LINKS], dim=1)
P = M[..., :3, 3]     # (B*T, K, 3) position, panda_link0 base frame
R = M[..., :3, :3]    # (B*T, K, 3, 3) rotation, same frame
```

`forward_kinematics` walks the tree from `panda_link0` to every link, at each
joint composing `origin · joint_motion(th_i)` (a pure rotation for revolute,
a pure translation along `axis` for prismatic, nothing for fixed) onto the
matrix carried from its parent. The result is one 4×4 matrix per link name,
position in the last column, rotation in the upper-left 3×3. Indexing that
dict by the 8 names in `KEYPOINT_LINKS`, in order, and stacking is the entire
keypoint extraction, `robots/franka/spec.py:186` does exactly this in
`forward_kinematics(q, aux)`, the method the rest of kinescore calls. This is
also where `LINK_FALLBACKS` would substitute `panda_hand` for a missing
`panda_hand_tcp`, keeping `K = 8` stable across URDF variants.

### 6. The real numbers, at rest

Zero arm angles, closed gripper, the same pose `FrankaFK._compute_rest_bones`
uses to anchor the rigidity check:

| consecutive pair | distance |
|---|---|
| `panda_link1` → `panda_link3` | 0.316 m |
| `panda_link3` → `panda_link5` | 0.384 m |
| `panda_link5` → `panda_link7` | 0.088 m |
| `panda_link7` → `panda_hand` | 0.107 m |
| `panda_hand` → `panda_leftfinger` | 0.0584 m |
| `panda_leftfinger` → `panda_rightfinger` | 0.0 m |
| `panda_rightfinger` → `panda_hand_tcp` | 0.045 m |

The fifth pair is exactly 0.0 m: at the closed-gripper rest pose the two
finger links coincide, so that "bone" has no length to defend in the first
place.

### 7. Which of those seven bones the rigidity check actually keeps

Two separate cuts apply, not one.

`ACTUATED_LINKS = {panda_leftfinger, panda_rightfinger}` drops any bone
touching a finger, pairs 4, 5, and 6 above, because a finger's position
tracks the gripper's own prismatic joint, not the rigid arm structure; a
robot that opens its gripper would otherwise read as "the arm deformed"
every time. That leaves 4 of the 7 as `rigid_bone_pairs`
(`robots/franka/spec.py`, via `structural_rigid_bone_mask`).

`RigidityDetector` narrows once more: of those 4, the second pair
(`panda_link3` → `panda_link5`, the 0.384 m one) sits between two revolute
joints (`panda_joint4`, `panda_joint5`), so its 0.384 m distance is a
snapshot at this one pose, not a value the arm's structure guarantees at
every pose, the way a bone between two links joined only by fixed or
in-line joints would. `RigidityDetector(rigid_idx=(0, 2, 3))` reads only the
three pairs whose distance a rigid body actually fixes; the fourth stays in
`rigid_bone_pairs` for anything that wants the full structural set, just not
for this one detector.

### 8. Into the model

Training never runs FK live against a simulator. `KeypointTrainer.build_target`
(`training/trainer.py:238`) calls this same `forward_kinematics(q, aux)` once
per logged window, under `torch.no_grad()`:

```python
p = robot.forward_kinematics(q[:, None].to(device), aux)   # (1, T, K, 3) metres
target = p[:, 0].detach().cpu().float()                    # (T, K, 3)
```

That `(T, K, 3)` tensor, in the exact `KEYPOINT_LINKS` order from step 3, is
the label the diffusion head is fit against with masked smooth-L1
(`heads/diffusion.py`). At read time the head never touches the URDF: it
predicts the same `(B, T, K, 3)` shape straight from image tokens, and
`val_mm` is how far those predictions land from this same FK-computed target
on held-out clips. The URDF's only job is producing training labels; once a
reader is trained, forward kinematics runs again only to build the rigidity
threshold's reference lengths shown above.

## Check this first: what does your corpus log?

This decides how much work the robot is.

Open one parquet and look at `observation.state`. If it holds joint angles,
you write a full forward-kinematics module and you need the URDF. If it holds
an end-effector pose, you can skip `pytorch_kinematics` entirely.

Galaxea A1X took the second path. Its corpus logs
`(x, y, z, roll, pitch, yaw, gripper_width)`, and those values break
`a1x.urdf`'s limits on joints 2 and 3, so they are not joint angles at all.
`robots/a1x_ee/spec.py` is 117 lines, loads no URDF, and computes keypoints as
four fixed offsets in the end-effector frame:

```python
EE_OFFSETS_M = ((-0.08165, 0.0, 0.0), (0.0, 0.0, 0.0),
                (0.03689, 0.013453, 0.0), (0.03689, -0.013453, 0.0))
P = pos.unsqueeze(2) + torch.einsum("btij,kj->btki", rot, offsets)
```

Its `urdf_sha256` is `None` and `rigid_bone_pairs` equals `bone_pairs`, because
fixed offsets in one frame are rigid by construction.

The rest of this page assumes the first path, joint angles and a URDF.

## The eight steps

### 1. Put the URDF under `KINESCORE_ASSETS`

Expand xacro first. `aloha_bimanual.urdf` is two Interbotix vx300s arms,
xacro-expanded and merged under a synthetic `world` root, because
`pytorch_kinematics` reads one tree and ALOHA ships two.

### 2. Read the URDF by hand and write `constants.py`

Nothing here is discoverable at runtime, so all of it is declared:

- predicted joint names, in the order the corpus stores them
- `KEYPOINTS_*`, the link names whose origins become the keypoints
- `ACTUATED_LINKS`, links driven by an actuator you do not predict
- `EE_LINK`, the tool-centre point per side

Joint order comes from the corpus, not from the URDF's declaration order.
Read `meta/modality.json` in the published tree and record what it says.

### 3. Write `fk.py`

`robots/base.py` already has the eight helpers this needs, so the module is
mostly wiring:

```python
chain = pk.build_chain_from_urdf(open(urdf_path, "rb").read())
assert_keypoints_in_urdf(set(chain.get_frame_names(exclude_fixed=False)),
                         {"left": KEYPOINTS_LEFT, "right": KEYPOINTS_RIGHT},
                         ee_link=EE_LINK)
pred_chain_idx = build_pred_chain_index(chain.get_joint_parameter_names(),
                                        LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS)
lo, hi, vel, eff = read_joint_limit_arrays(urdf_path, pred_joints)
pairs = consecutive_bone_pairs(num_keypoints)
lengths = rest_pose_bone_lengths(keypoints_at_rest, pairs)
```

`scatter_predicted(q_flat, pred_chain_idx, n_chain_joints)` places the
predicted joints into the chain and leaves every other joint at zero.

### 4. Write `spec.py` and get the rigid-bone mask right

This is the one step where a mistake is silent. The rigidity detector measures
bone length against the URDF rest length, so a bone whose endpoint is dragged
by the gripper reads "the arm deformed" every time the gripper opens.

```python
mask = structural_rigid_bone_mask(self.keypoint_links, self.bone_pairs,
                                  self.bone_lengths, ACTUATED_LINKS)
warn_dropped_bones("AlohaSpec", self.keypoint_links, self.bone_pairs,
                   self.bone_lengths, mask)
self.rigid_bone_pairs = self.bone_pairs[mask]
```

Two rules apply. A bone with either endpoint in `ACTUATED_LINKS` is dropped,
and a bone shorter than `DEGENERATE_BONE_M` (1 mm) at rest is dropped, because
its measured length is float noise rather than structure.
`warn_dropped_bones` prints each dropped bone with its rest length, so run it
once and read the warning.

Declare only the capabilities the robot can back. `aloha_bimanual.urdf`
carries no mesh geometry, so `AlohaSpec` does not declare `COLLIDERS`. ALOHA is
bolted to a table, so it does not declare `SUPPORT_POLYGON`. A capability
declared without backing produces a number that looks measured and is not.

### 5. Register the factory

In `robots/__init__.py`, with the import inside the function so that importing
the registry does not pull in `pytorch_kinematics`:

```python
def _build_aloha_bimanual(**kwargs) -> RobotSpec:
    from kinescore.robots.aloha.spec import AlohaSpec
    return AlohaSpec(**kwargs)

_REGISTRY.register("aloha_bimanual", _build_aloha_bimanual)
```

### 6. Declare it in `configs/robots.yaml`

```yaml
aloha_bimanual:
  embodiment: bimanual
  keypoints: 18
```

`kinescore train` compares this count against what forward kinematics actually
returns and refuses the run if they disagree (`cli/cmd_train.py:122`).

### 7. Declare the corpus and the reader in `configs/cells.yaml`

```yaml
_corpus_bimanual_sv: &bimanual_sv
  corpus: bimanual_sv
  adapter: lerobot
  root: ${KINESCORE_DATA_ROOT}/corpus/bimanual/singleview
  cameras: [cam_high]
  joint_field: observation.state
  joint_columns: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
  gripper_columns: [6, 13]

readers:
  aloha_bimanual.bimanual_sv.sv1_16x9:
    robot: aloha_bimanual
    view: sv1_16x9
    train: *bimanual_sv
```

The reader id must read `<robot>.<corpus>.<view_id>` or the registry rejects
it. Note that `joint_columns` skips 6 and 13: those are the two grippers, and
they travel through `aux` rather than through `q`.

Then a cell, which is what you score:

```yaml
cells:
  bimanual.sv1_16x9.dreamgen:
    reader: aloha_bimanual.bimanual_sv.sv1_16x9
    select: {embodiment: bimanual, view: singleview, model: dreamgen}
```

### 8. Train, then calibrate

```bash
R=aloha_bimanual.bimanual_sv.sv1_16x9
kinescore data  --reader $R
kinescore cache --reader $R --device cuda
kinescore train --reader $R --device cuda --steps 8000
```

A new robot has no thresholds. `kinescore score` fits them itself, at the 95th
percentile of 24 real clips from the reader's own validation split, every time
it runs. There is no separate calibration command and no threshold is carried
between robots.

## Worked example: ALOHA bimanual

Two 6-DOF arms with 2-finger grippers, table-mounted, 12 predicted joints and
18 keypoints.

### The layout

```
src/kinescore/robots/aloha/
    constants.py   joint names, keypoint links, ACTUATED_LINKS, finger travel
    fk.py          AlohaFK: pytorch_kinematics chain, aux scatter, rest bones
    spec.py        AlohaSpec: the RobotSpec protocol, bimanual concatenation
```

### Predicted state

`n_joints = 12`, which is
`[waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]` per arm,
left then right. The 7th actuator per arm is the gripper, and it is not in
`q`. It arrives as `aux`, shaped `(B, T, 2)` and holding an opening in
`[0, 1]`, which is affine-mapped onto the prismatic finger joints:

```python
left_val = FINGER_LO_M + grip_flat[:, 0] * (FINGER_HI_M - FINGER_LO_M)
th[:, idx[0]] = left_val
th[:, idx[1]] = -left_val
```

`FINGER_LO_M` is 0.021 m closed and `FINGER_HI_M` is 0.057 m open. The right
finger is the negation of the left because the URDF declares it as a mirror
slide, which was confirmed by reading the file rather than assumed.
`aux=None` means both grippers closed, and that is also the rest pose used for
the bone lengths.

### Keypoints

Nine per arm, concatenated left then right, so indices 0 to 8 are the left arm
and 9 to 17 the right:

```
shoulder_link, upper_arm_link, upper_forearm_link, lower_forearm_link,
wrist_link, gripper_link, left_finger_link, right_finger_link, ee_gripper_link
```

### Which bones the rigidity detector keeps

Eight consecutive bones per arm, of which three are dropped:

| bone | kept | why |
|---|---|---|
| `gripper_link -> left_finger_link` | no | endpoint moves with the gripper |
| `left_finger_link -> right_finger_link` | no | both endpoints move with the gripper |
| `right_finger_link -> ee_gripper_link` | no | left endpoint moves with the gripper |
| the other five | yes | length is fixed by the arm structure |

`ee_gripper_link` itself never moves with actuation. It hangs off
`gripper_link` through four fixed joints. It loses its bone only because the
bone's other endpoint is a finger.

### What it produces

| field | value |
|---|---|
| `n_joints` | 12 |
| `keypoint_links` | 18 |
| `rigid_bone_pairs` | 10 of 16, which is what the two rules above give |
| `capabilities` | `ROTATIONS`, `EFFORT_LIMITS` |
| `urdf_sha256` | recorded in every output |

The bone count follows from 9 keypoints per arm, so 8 consecutive bones per
arm, minus the 3 the gripper rule drops. Read the actual list off
`warn_dropped_bones` on your first run rather than trusting the arithmetic.

`EFFORT_LIMITS` is declared because all 12 arm joints carry both `velocity`
and `effort` in this URDF, which was verified before declaring it.

## Where a mistake surfaces

Every one of these fails loudly at construction or at train time, not as a
wrong number later.

| mistake | what catches it |
|---|---|
| joint name typo | `read_joint_limit_arrays` raises and names the joint |
| predicted joint absent from the chain | `build_pred_chain_index` raises, rather than leaving a zero column |
| keypoint link absent from the URDF | `assert_keypoints_in_urdf` |
| keypoint count disagrees with `robots.yaml` | `cli/cmd_train.py:122` |
| head predicts a different keypoint count | `training/trainer.py:213` |
| reader id not `<robot>.<corpus>.<view>` | `registry/cells.py` |
| cell embodiment disagrees with the robot | `registry/cells.py:320` |
| registering a name twice | `core/registry.py`, which refuses to overwrite |
| the URDF changes upstream | `urdf_sha256` in `summary.json` |

The one mistake with no guard is a wrong `ACTUATED_LINKS`. Nothing can detect
that a bone you kept is actually gripper-driven, so read the
`warn_dropped_bones` output on your first run and check the list against the
URDF.

## Checklist

```
[ ] URDF under $KINESCORE_ASSETS, xacro expanded, one tree
[ ] joint order read from the corpus, not guessed from the URDF
[ ] constants.py: joints, keypoints, ACTUATED_LINKS, EE_LINK
[ ] fk.py using the helpers in robots/base.py
[ ] spec.py with structural_rigid_bone_mask, warn_dropped_bones read once
[ ] capabilities declared only where something backs them
[ ] factory registered in robots/__init__.py with a lazy import
[ ] configs/robots.yaml: embodiment + keypoint count
[ ] configs/cells.yaml: corpus anchor, reader, at least one cell
[ ] kinescore data / cache / train run clean
[ ] kinescore score fits thresholds on the robot's own real clips
```
