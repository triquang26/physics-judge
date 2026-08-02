# Data prep: where data goes, and the format each cell must satisfy

Two different things live under the `KINESCORE_*` env vars: data **to score**
(generated + real reference video, for benchmarking) and data **to train a
reader on** (real robot video + logged joints, for training). This doc covers
both — placement, the mandatory on-disk format, and the two commands
(`kinescore data ingest` / `kinescore data verify`) that check a scoring
dataset actually matches what it claims to be.

Every path resolves through a `KINESCORE_*` environment variable, with **no
fallback**: an unset variable raises and names itself
(`src/kinescore/paths.py`) rather than silently resolving to a path that
exists on someone else's machine. Copy `.env.example` to `.env` and fill it
in before anything else.

## The five roots

| variable | holds | regenerable? |
|---|---|---|
| `KINESCORE_DATA_ROOT` | video to score, and video/logs to train readers on | **no** — re-download or reconvert |
| `KINESCORE_ASSETS` | URDF + meshes per robot, `MANIFEST.json` | no — re-fetch from upstream |
| `KINESCORE_CKPT_DIR` | trained pose readers + `.provenance.json` beside each | no — retrain |
| `KINESCORE_CACHE_DIR` | precomputed frozen-backbone features (training) | **yes — delete freely**, `kinescore cache` rebuilds it |
| `KINESCORE_OUTPUT_DIR` | manifests, `results.jsonl`, CSVs, traces | yes — rerun |

`kinescore doctor` reports which of these resolve to a real, populated
directory before you run anything that needs them — run it after filling in
`.env`.

## Data to score: layout and the per-generator contract

```
$KINESCORE_DATA_ROOT/
├── video_gen_physics/                # HF doanh25032004/video_gen_physics -- dense/ ONLY, the
│   └── dense/                        #   other 8 top-level dirs are cache-accelerator variants,
│       ├── humanoid/                 #   hundreds of GB; --dry-run before pulling
│       └── single_arm/
├── video_gen_physics_real_video/     # HF doanh25032004/video_gen_physics_real_video
│   └── humanoid/                     #   NOTE: this is Airbot MMK2 real footage, not GR-1
├── cosmos_synthetic_data/            # HF doanh25032004/cosmos_synthetic_data -- construct-
│   ├── high/                         #   validity tiers for the score itself, not scoring input
│   └── low/
└── train/                            # reader training sets -- see "Data to train a reader" below
```

Directory names are exactly the HuggingFace repo names, so
`huggingface-cli download <repo> --local-dir $KINESCORE_DATA_ROOT/<repo-name>`
or `kinescore data pull --config configs/benchmark.yaml [--dry-run]` lands
data with nothing to rename. **Always `--dry-run` first** — the guard is not
decorative: `video_gen_physics` alone has nine top-level directories, of
which `dense/` is one, and pulling all nine is hundreds of GB. Set
`HF_HUB_DISABLE_XET=1` — measured on this host, the xet CDN throttled to
374 KB/s with repeated 429s; disabling it gave 2893 KB/s (7.7x). Directory
names inside the downloaded tree are not always what they say: `singleview`
and `single_view` both exist, with **different** episode sets, and
`dense/humanoid/.../multiview/ctrlworld/` is Airbot MMK2 despite sitting
under `humanoid/`. Treat every declared value (an `info.json` episode count, a
directory name) as a cross-check that can only fail, never a source of truth
— see [BENCHMARKING.md](BENCHMARKING.md)'s rate-policy section for the
concrete cost of trusting one instead.

### The per-generator file contract: `configs/data_spec.yaml`

Nothing about generator layout is hardcoded in Python — a resolution/fps/shape
change is a YAML edit, and `kinescore data verify` fails with the *specific*
clip path when a clip on disk disagrees with what's declared. Three
generators, three coexisting on-disk shapes (`kinescore.bench.layout.
RawHFLayout`'s module docstring has the full walk logic per shape):

| generator | shape | layout |
|---|---|---|
| `ctrlworld` | `episode_dir` | one directory per episode, fixed pred/gt filenames (`pred_all_views.mp4`/`gt_all_views.mp4`); 960x192, 5fps (2 anomalous probed episodes run 30fps — `fps_tolerant: true` is what stops those two from hard-failing `verify`), 3 views |
| `dreamgen` | `task_episode` | `<task>/episode_*.mp4` under a pinned `iter_*` dir, **no ground truth anywhere**; 768x432, 16fps, 1 view |
| `dreamdojo` | `flat_or_dir` | TWO coexisting shapes in one `iter_*` dir — flat (`NNNN_pred.mp4`/`NNNN_gt.mp4`) and dir (`episode_<id>/full_pred.mp4`+`full_gt.mp4`), neither a subset of the other; 640x480, fps **per robot** (10 for `fourier_gr1`, 15 for `franka_panda` — a flat fps table caused a real 480-episode silent-drop incident, see [BENCHMARKING.md](BENCHMARKING.md)), 1 view |

`horizon` values are the dataset's own spelling — `makovian`, not
"markovian" — reproduced verbatim; do not "fix" it, it would stop matching
real directory names.

### The robot table: `configs/robot_map.yaml`

`embodiment` (the on-disk directory: `humanoid`/`single_arm`/`bimanual`) is
**not** the robot. `humanoid` alone covers two physically different robots
depending on which generator wrote the clip — confirmed two ways: every
ctrlworld episode directory under `dense/humanoid/.../multiview/ctrlworld/`
is literally named `episode_AIRBOT_MMK2_*`, and dreamgen/dreamdojo's
`humanoid` clips are Fourier GR-1 (confirmed visually against real reference
footage) — a completely different kinematic tree. `robot_map.yaml` is what
lets `kinescore.bench.matrix` resolve the right robot per `(embodiment,
generator)` pair instead of per embodiment alone, and what makes a
`(robot, generator)` combination it doesn't list an automatic **N/A** cell
(reported as such, never silently "0 episodes found"):

```yaml
robots:
  fourier_gr1:    {embodiment: humanoid,   generators: [dreamgen, dreamdojo]}
  airbot_mmk2:    {embodiment: humanoid,   generators: [ctrlworld]}
  franka_panda:   {embodiment: single_arm, generators: [ctrlworld, dreamgen, dreamdojo]}
  aloha_bimanual: {embodiment: bimanual,   generators: [ctrlworld, dreamgen, dreamdojo]}
```

`configs/data_spec.yaml` repeats a copy of this table (`robots:` at its own
bottom) so `kinescore data verify` can run against that one file alone;
`verify` cross-checks the two agree.

### `kinescore data ingest` / `kinescore data verify`

```bash
kinescore data ingest --robot-map configs/robot_map.yaml \
  --data-spec configs/data_spec.yaml --out $KINESCORE_DATA_ROOT/bench

kinescore data verify --data-spec configs/data_spec.yaml \
  --canonical-root $KINESCORE_DATA_ROOT/bench
```

`ingest` symlinks (never copies, by default — `--copy` exists for
filesystems without symlink support) the raw HF tree into the canonical
`bench/<cache>/<robot>/<view>/<generator>/<horizon>/episode_XXXX/` shape,
writing one `cell_card.json` per cell (recording `n_episodes_declared` vs
`n_episodes_actual` — the declared count is recorded, never trusted) and
probing exactly one episode per cell for width/height/fps/codec.

`verify` then checks **every** clip in the canonical tree (not just the one
`ingest` probed) against `data_spec.yaml`: width/height always hard-fail on
mismatch, fps hard-fails unless the generator is `fps_tolerant`, a pred/gt
frame-count mismatch is a hard error, and a broken symlink is reported by
path. Nothing here needs a trained reader — `ingest`/`verify` operate purely
on file layout, which is why they're a separate step from `kinescore bench
run`/`kinescore score` (see [BENCHMARKING.md](BENCHMARKING.md)).

## Data to train a reader: the training input contract

A reader is trained per robot, against **real** logged joints — never
generated video (`training/cache.py::assert_real_joint_source` raises if an
annotation's `joint_source` is anything but `"real"`, specifically so a
synthetic-joint episode can never contaminate the training signal). The
contract every training input must satisfy, read by
`training/datasets.py::load_split`:

```
<video_root>/{train,val}/<episode_id>.mp4
<annotation_root>/{train,val}/<episode_id>.json
```

Each annotation JSON:

```json
{
  "joint_source": "real",
  "observation.state.joint_position": [[q_0...q_{n-1}], ...],
  "observation.state.gripper_position": [g_0, g_1, ...],
  "provenance": { "...": "free-form, for humans" }
}
```

`observation.state.joint_position` is `(T_logged, n_joints)` at the joint
log's **own native rate** — not resampled to the video's frame rate here.
`n_joints` must equal the target `RobotSpec.n_joints` (17 for `fourier_gr1`,
12 for `airbot_mmk2`, 12 for `aloha_bimanual`) — this is the part that
differs per robot and is where a wrong slice silently trains a reader against
mislabeled joints. `gripper_position` is optional (`load_split` loads fine
with `gripper=None` for a robot with no logged gripper, or one where a single
scalar can't represent it — see `aloha_bimanual` below).

**`--down-sample` is a property of the (video fps, joint-log rate) pair, and
it is required by every downstream command with no safe default.**
`kinescore cache`/`kinescore train-rawrad` read `down_sample` joint rows per
video frame (`idx = clip(arange(n_frames) * down_sample, 0, len(joints)-1)`);
getting it wrong trains against mis-paired labels, and nothing downstream
reports that it happened — the loss will simply converge to a worse number
with no error. If video and joint log share one rate, `down_sample=1`
(verify: parquet row count == decoded video frame count for a sample
episode). If the joint log runs faster than the video (e.g. 15Hz joints
against 5Hz video), `down_sample` is that ratio.

### The per-robot state mapping, verified against real data

| robot | source format | source state width | slice → `n_joints` | cams |
|---|---|---|---|---|
| `fourier_gr1` | LeRobot v2 (PhysicalAI GR-1 Teleop) | 44 | `state[0:7]+state[22:29]+state[41:44]` → 17 (`meta/modality.json`: `left_arm[0:7] left_hand[7:13] left_leg[13:19] neck[19:22] right_arm[22:29] right_hand[29:35] right_leg[35:41] waist[41:44]`; `GR1Spec.n_joints==17` is `[left_arm(7), right_arm(7), waist(3)]` — legs/hands/neck are logged but never predicted) | 1 (`ego_view_freq20`) |
| `airbot_mmk2` | LeRobot v2 | 36 | `state[0:12]` → 12 (`[left_arm_joint_1..6, right_arm_joint_1..6]`; the 24 hand dims `[12:36]` are dropped — their claimed `_rad` units don't check out numerically, unsafe to build a kinematic chain against) | 4 (`cam_high_rgb`, `cam_left_wrist_rgb`, `cam_right_wrist_rgb`, `cam_third_view` — naming varies per source-dataset batch, see below) |
| `aloha_bimanual` | LeRobot v1 (v1 and v2 share the identical `data/chunk-*/episode_*.parquet` + `videos/chunk-*/observation.images.<cam>/episode_*.mp4` layout — only the `codebase_version` string differs, so no v1-specific parsing is needed) | 42 | `qpos=state[0:14]`, drop indices 6 and 13 (the per-side gripper) → 12 (`meta/modality.json`: `qpos[0:14] qvel[14:28] effort[28:42]`; qpos is `[waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper]` per side, left then right) | 4 (`cam_high`, `cam_left_wrist`, `cam_low`, `cam_right_wrist`) |

`aloha_bimanual` writes **no** `gripper_position` key: `AlohaSpec`'s gripper
is a 2-value `aux` (`[left, right]` opening, `fk.py`), not the single scalar
`observation.state.gripper_position` supports — writing one placeholder value
for two independent grippers would repeat the exact "channel reads as
fabricated data for every frame" problem `airbot_mmk2_NOTICE.txt`'s CAVEAT 2
documents for a different reason (see [TRAINING.md](TRAINING.md)). Gripper is
optional in the loader's contract, so omitting the key is the honest choice,
not a shortcut.

### Multi-camera packing

A robot with more than one camera is packed into **one** mp4 per episode,
matching `core/clip.py::ViewLayout`'s default `packing="height"` (cameras
vstacked top-to-bottom in the order given) — the same packing `kinescore
cache --n-views N` (no `--view-order` override) assumes when it slices a
packed frame back into per-view crops. `scripts/convert_lerobot_to_train.py`
does this packing via `ffmpeg`'s `vstack` filter.

### Converting a real LeRobot dump: `scripts/convert_lerobot_to_train.py`

```bash
.venv/bin/python scripts/convert_lerobot_to_train.py \
  --robot airbot_mmk2 \
  --task-dir $KINESCORE_DATA_ROOT/.../AIRBOT_MMK2_boxs_storage \
  --task-dir $KINESCORE_DATA_ROOT/.../Airbot_MMK2_move_pan \
  --out-video      $KINESCORE_CACHE_DIR/airbot_mmk2_train_input/video \
  --out-annotation $KINESCORE_CACHE_DIR/airbot_mmk2_train_input/annotation
```

One converter, three robot mappings (the table above), `--task-dir` repeatable
for multi-task datasets. It does **not** take a `--down-sample` flag —
that's a property of the (video, joint-log) pair consumed later by `cache`/
`train-rawrad`, not something to bake in at conversion time. What it does do:
write the joint array at its own logged rate (one row per parquet row,
matching the precedent this generalises,
`cache/airbot_mmk2_train_input/convert_airbot_real_to_kinescore.py`), and
(unless `--no-probe`) print each episode's joint-row-count / video-frame-count
ratio via `ffprobe`, so the operator reads the right `--down-sample` off real
numbers instead of guessing. Split is deterministic: the last `--val-ratio`
(default 10%) of each task directory's episodes, by index — the same rule the
existing Airbot MMK2 converter used, not the scene-stratified split
`training/splits.py::stratified_episode_split` offers for a single pooled
directory (use that one instead if your task boundaries don't already give
you a safe val split — see [TRAINING.md](TRAINING.md#the-mm-acceptance-gate)
for why stratification matters).

### Why two Airbot MMK2 camera caches exist (`cache_airbot_mmk2_camhigh_subset{A,B}.sh`)

Not two different physical cameras. The accepted Airbot MMK2 reader was
trained on `cam_third_view`/`cam_front_rgb` (whichever shows the full arm
chain per task) — but the generated ctrlworld cells this reader is meant to
score only render `cam_high`, `cam_left_wrist`, `cam_right_wrist`, none of
which is the training camera. `scripts/cache_airbot_mmk2_camhigh_subsetA.sh`/
`subsetB.sh` are a retraining attempt on the camera the eval data actually
has — split into A/B only because the source LeRobot dump spells the "high"
camera two different ways across two task-naming conventions
(`cam_head_rgb` under mixed-case `Airbot_MMK2_*` task dirs vs `cam_high_rgb`
under all-caps `AIRBOT_MMK2_*` dirs). See each script's own header for status
(subset A completed a cache pass and a training run of unconfirmed outcome;
subset B's cache pass has not been run at all).
