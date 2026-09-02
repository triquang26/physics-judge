# Single-arm, end to end

Every command from empty disk to bucket push, for the two single-arm readers.
General flag documentation lives in [QUICKSTART.md](QUICKSTART.md); this page
is the single-arm run book.

## The embodiments

Single-arm splits into two corpora that carry different robots AND different
state semantics — read `configs/cells.yaml` before assuming either.

### `a1x_sv` — Galaxea A1X, singleview

`single_arm/singleview` in `vgp_datasets`, one 640×480 `global` camera at
15 fps. Only the **makovian** splits are A1X (`put_cup_on_coaster`,
`turn_on_light_switch`, 120 episodes); non-makovian `press_three_buttons` is
a Franka Panda, so the corpus root is a filtered tree holding a `makovian`
symlink only:

```bash
mkdir -p $KINESCORE_DATA_ROOT/corpus/single_arm/singleview_a1x
ln -sfn ../singleview/makovian $KINESCORE_DATA_ROOT/corpus/single_arm/singleview_a1x/makovian
```

`observation.state` is **an EE pose, not joint angles**:
`(x, y, z, roll, pitch, yaw, gripper_width)`. The values break `a1x.urdf`'s
one-signed limits on joints 2 and 3, which is the proof. Hence the robot
`a1x_ee` (`src/kinescore/robots/a1x_ee/spec.py`): `q` is the 6-DOF pose,
4 keypoints ride the EE frame at offsets read from `a1x.urdf`
(bucket: `asset/a1x/`, upstream `userguide-galaxea/URDF` @ `2e5d31e`), and no
FK chain runs at train or score time.

### `single_arm_mv` — Franka Panda, multiview static

`single_arm/multiview` in `vgp_datasets`, DROID-style Franka at 15 fps with
three cameras. Two are static (`exterior_1_left`, `exterior_2_left`); the
wrist camera rides the arm and is dropped. The bench clips are a 768×432
2×2 grid whose top row is the two static views — the `mv4_grid_static` view
(`panels: [0, 1]`) reads exactly that row, and `data` packs the two exterior
cameras into the same cells with the bottom row black.

State is real joint angles here: `observation.state.joint_position` (7) plus
`observation.state.gripper_position`, consumed by the URDF-backed
`franka_panda` robot (8 keypoints, `robot_descriptions` Panda).

## Reading the config for a new robot

One robot = four declarations, all cross-checked at load time:

1. `src/kinescore/robots/<name>/spec.py` — the `RobotSpec`: `n_joints`,
   `keypoint_links`, FK, bone pairs. Register it in `robots/__init__.py`.
2. `configs/robots.yaml` — `<name>: {embodiment, keypoints}`; the keypoint
   count is checked against what FK actually produces.
3. `configs/cells.yaml` — a `_corpus_*` anchor (root, cameras, joint_field,
   joint_columns, gripper column/field, scene_key), a reader
   `<robot>.<corpus>.<view>`, and a cell `<embodiment>.<view>.<model>`.
4. `configs/views.yaml` — the packing, only if no existing view matches.
   Measure the panel size from the clips; never guess.

## 1. Download

```bash
kinescore pull --what train      # vgp_datasets: single_arm/{singleview,multiview}
kinescore pull --what radial     # radial/single_arm/... generated clips
kinescore pull --what dense      # dense reconstructions (clean references)
```

## 2. Train the readers

```bash
for R in a1x_ee.a1x_sv.sv1_4x3 franka_panda.single_arm_mv.mv4_grid_static; do
  kinescore data  --reader $R
  kinescore cache --reader $R --device cuda
  kinescore train --reader $R --device cuda --steps 3000
done
```

Reference: `a1x_ee.a1x_sv.sv1_4x3` reached val_mm ≈ 24.4 at 3000 steps
(108 train / 12 val episodes). EE-pose keypoints are a far easier target than a
full arm — compare within the embodiment only.

`franka_panda.single_arm_mv.mv4_grid_static` sits at val_mm ≈ 156 and is kept
out of the rating batch. Its corpus spans 22 buildings and 40 episodes whose
extrinsics vary by 1.55 m, so inferring a base frame from pixels is ill-posed
and the head cannot do better without a target expressed in camera frame.

## 3. Score

Radial clips sit outside the manifest — score through `--videos`:

```bash
kinescore score --cell single_arm.sv1_4x3.radial_dreamgen --device cuda \
    --videos $KINESCORE_DATA_ROOT/bench/radial/single_arm/output/singleview/dreamgen \
    --out out/radial.sv1_4x3.dreamgen

kinescore score --cell single_arm.mv4_grid_static.radial_dreamgen --device cuda \
    --videos $KINESCORE_DATA_ROOT/bench/radial/single_arm/output/multiview/dreamgen \
    --out out/radial.mv4_grid_static.dreamgen
```

Reference (a1x_sv, 12 val calibration clips at p95): rigidity 18.0 mm,
jerk 2.50e5 mm/s³. Two systematics to keep in mind when reading single-arm
jerk numbers:

- The generated clips run 16 fps against a 15 fps corpus; jerk scales with
  fps³ (×1.21).
- Dense reconstructions — near ground truth — carry the same jerk levels as
  radial (peak ratio ≈ 2.4 both), so single-arm jerk measures the renderer,
  not the generator. Judge single-arm hallucination by rigidity peaks and by
  the jerk violation *fraction* (real ≈ 0.04, generated ≈ 0.2), never by the
  jerk peak alone.

**One GPU per cell** — calibration drifts ~1% across GPUs; re-judge against
one canonical `summary.json` before merging split runs.

## 4. Export web bundles

```bash
kinescore export --results out/radial.sv1_4x3.dreamgen --name radial_single_arm_sv
kinescore export --results out/radial.mv4_grid_static.dreamgen --name radial_single_arm_mv
```

## 5. Push to the bucket

```bash
export HF_TOKEN=<write token>    # env only — never in a file, rotate after use
kinescore push --reader a1x_ee.a1x_sv.sv1_4x3 \
    --reader franka_panda.single_arm_mv.mv4_grid_static \
    --scores out/radial.sv1_4x3.dreamgen \
    --scores out/radial.mv4_grid_static.dreamgen \
    --web out/web/radial_single_arm_sv \
    --web out/web/radial_single_arm_mv
```

Targets under `hf://buckets/twanghcmut/hallucinate-bench`:
`train/<reader>/diffusion/`, `scores/<cell>/diffusion/`, `web/<bundle>/`.
