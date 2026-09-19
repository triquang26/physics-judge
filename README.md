# kinescore

**Physics-plausibility benchmark for AI-generated robot video.**

A frozen DINOv3 backbone + a trained diffusion head read 3-D robot keypoints
out of the pixels; five analytic detectors measure physics violations on those
keypoints. No simulator, no VLM judge. Two properties make the score
trustworthy:

- **Non-circular**: the reader is frozen and the evaluated model never
  trained against it.
- **Calibrated on real motion**: every threshold is the 95th percentile of
  the same quantity measured on real teleop from the same robot and packing.

It is a **relative** referee: keypoint accuracy floors at tens of millimetres,
so compare systems through the same reader rather than reading absolutes.

## Pipeline

    HF ──pull──▶ bench clips + corpora
                    │ data     corpus → train tree
                    │ cache    frozen backbone → tokens
                    │ train    diffusion head ← forward-kinematics targets
                    │ score    generated clips → per-segment verdicts
                    │ report   one table over every cell
                    │ export   numbered clips + segments.json for a rating UI
                    └ push     hf sync to the bucket (train/ scores/ web/)

- **reader** = `<robot>.<corpus>.<view_id>`, one trained head.
- **cell** = `<embodiment>.<view_id>.<model>`, one scored unit.

Both are declared in `configs/cells.yaml`; panel geometry in
`configs/views.yaml` is measured, never inferred.

## Install

Three commands. No setup script.

```bash
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -e ".[dino,video,bench]"
cp .env.example .env
```

Install the extras named above, not `[dev]` alone. The core dependency list
holds only numpy, torch, pytorch-kinematics and robot-descriptions, so a bare
install has no `huggingface_hub` and `kinescore pull` fails on import.

`ffprobe` and `ffmpeg` must be on `PATH`. For a particular CUDA build, install
torch before the line above:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/<cuXXX>
```

### Fill in `.env`

No variable has a fallback, so an unset one raises immediately and names
itself.

```bash
KINESCORE_ASSETS=/shared/kinescore-store/assets      # URDF and mesh trees
KINESCORE_CKPT_DIR=/shared/kinescore-store/ckpt      # trained readers
KINESCORE_CACHE_DIR=/shared/kinescore-store/cache    # backbone token cache
KINESCORE_DATA_ROOT=/shared/kinescore-store/data     # clips and corpora
KINESCORE_OUTPUT_DIR=/shared/kinescore-store/out     # scores
```

Point these at ordinary disk. A RAM disk such as `/dev/shm` looks fast and
loses the tree on reboot, and the asset tree alone is hundreds of megabytes.

### Load `.env` before any command

The CLI reads the process environment and does not parse `.env` itself. Export
it yourself, once per shell:

```bash
set -a; . .env; set +a
```

## Score a folder of videos

Three steps from an empty machine to a table. Everything below assumes the
`.env` line above has been run.

### 1. Get the URDF assets and a trained reader

```bash
export KINESCORE_ASSETS=/shared/kinescore-store/assets
export KINESCORE_CKPT_DIR=/shared/kinescore-store/ckpt
export KINESCORE_OUTPUT_DIR=/shared/kinescore-store/out
export HF_TOKEN=<read token>

hf sync hf://buckets/twanghcmut/hallucinate-bench/asset $KINESCORE_ASSETS
hf buckets cp hf://buckets/twanghcmut/hallucinate-bench/train/fourier_gr1.humanoid_sv.sv1_16x9/diffusion/checkpoint.pt \
    $KINESCORE_CKPT_DIR/fourier_gr1.humanoid_sv.sv1_16x9.pt
```

Repeat the last two lines with a different reader id for another robot.

Readers that have been trained:

| reader | robot | reads |
|---|---|---|
| `fourier_gr1.humanoid_sv.sv1_16x9` | Fourier GR-1 | 768×432 |
| `fourier_gr1.humanoid_sv.sv1_4x3` | Fourier GR-1 | 640×480 |
| `aloha_bimanual.bimanual_sv.sv1_16x9` | ALOHA | 768×432 |
| `a1x_ee.a1x_sv.sv1_4x3` | Galaxea A1X | 640×480 |
| `franka_panda.single_arm_mv.mv4_grid_static` | Franka Panda | 2×2 of 384×216 |

`kinescore readers` prints all twelve declared readers and says which ones
have a checkpoint on disk.

### 2. Score

One line. Point `--videos` at any directory and it is searched recursively for
`*.mp4`.

```bash
kinescore score --cell humanoid.sv1_16x9.dreamgen --device cuda \
    --videos /path/to/your/clips \
    --out $KINESCORE_OUTPUT_DIR/myrun
```

The cell picks the reader, and the reader fixes what frame size is accepted.
A clip that is not 768×432 for this cell is rejected by name before the
backbone runs. `kinescore score --list` prints every cell.

Before it scores anything, `score` fits its own thresholds on 24 real clips
from that reader's validation split, at the 95th percentile. There is no
separate calibration step, and no threshold is shared between robots.

Useful flags: `--limit 40` caps clips, `--pattern full_pred.mp4` picks one
basename out of a tree that stores prediction and ground truth side by side,
and `--frame-chunk 16` lowers GPU memory on long clips.

### 3. Read the output

`--out` holds six things:

| file | one row per | holds |
|---|---|---|
| `summary.json` | run | cell, reader, checkpoint SHA-256, clip counts, the fitted thresholds |
| `segments.csv` | 16-frame segment | `<detector>_{reduce,value,threshold,violated}` |
| `metrics.csv` | clip | flagged fraction, worst segment, median severity per detector |
| `results.jsonl` | clip | the full per-frame series and flagged frame intervals |
| `run_manifest.json` | run | argv, git sha, host, config hashes |
| `render/` | clip | `<id>_clip.mp4`, keypoints and verdicts drawn on |

A real `summary.json`, trimmed:

```json
{"cell_id": "humanoid.sv1_16x9.dreamgen",
 "reader_id": "fourier_gr1.humanoid_sv.sv1_16x9",
 "checkpoint_sha256": "2686fe58...",
 "n_clips": 40, "n_failed": 0, "n_calibration_clips": 24, "percentile": 95.0,
 "thresholds": {"rigidity": {"units": "mm", "threshold": 46.98},
                "jerk": {"units": "mm/s^3", "threshold": 710178.55}}}
```

A real `segments.csv` row says a segment broke the joint limit and nothing
else:

```
segment,start_frame,end_frame,rigidity_value,rigidity_threshold,rigidity_violated,joint_limit_value,joint_limit_threshold,joint_limit_violated
0,0,15,34.2,46.98,False,9.4,8.59,True
```

Segments are 16 frames. Rigidity reduces a segment by its median, so 8 of 16
frames must be over the line. Every other detector reduces by the worst frame,
so one frame is enough. Five detectors always run and all five are written;
`rigidity` and `jerk` are the two a headline number is read off.

### Reporting across cells

```bash
kinescore report --by method
```

This only finds a cell's scores when they sit in that cell's own output
directory, which is `$KINESCORE_OUTPUT_DIR/<cell_id>`. A run written to a
custom `--out` shows as `not scored`. Either drop `--out` or name the
directory after the cell.

### Packaging for human rating

```bash
kinescore export --results $KINESCORE_OUTPUT_DIR/myrun --name myrun_bundle
```

Writes `$KINESCORE_OUTPUT_DIR/web/myrun_bundle/`: clips renumbered `1.mp4` to
`N.mp4` in scoring order, plus one `segments.json` carrying each segment's
`value`, its `ratio` against the threshold, and its verdict.

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for every flag, and
[docs/ADDING_A_ROBOT.md](docs/ADDING_A_ROBOT.md) to add an embodiment.

## Train a reader from scratch

Only needed for a robot or packing with no checkpoint on the bucket.

```bash
R=aloha_bimanual.bimanual_sv.sv1_16x9
kinescore pull  --what train
kinescore data  --reader $R                  # corpus to train tree, CPU
kinescore cache --reader $R --device cuda    # frozen backbone to token cache
kinescore train --reader $R --device cuda --steps 8000
```

`train` reports RMS keypoint error in millimetres on a scene-disjoint
validation split and writes `$KINESCORE_CKPT_DIR/$R.pt`. Quote `val_mm`, and
compare it only within one reader.

## Readers

| reader | robot | view | scores |
|---|---|---|---|
| `airbot_mmk2.humanoid_mv.mv4_row` | Airbot MMK2 | 4×(320×192) row | `humanoid.mv4_row.ctrlworld_4view_grid` |
| `airbot_mmk2.humanoid_mv.mv4_row_static` | Airbot MMK2 | panels 0–1 of `mv4_row` | `humanoid.mv4_row_static.ctrlworld_4view_grid` |
| `airbot_mmk2.humanoid_mv.mv4_grid` | Airbot MMK2 | 2×2 of 384×216 | `humanoid.mv4_grid.dreamgen` |
| `fourier_gr1.humanoid_sv.sv1_16x9` | Fourier GR-1 | 768×432 | `humanoid.sv1_16x9.dreamgen` |
| `fourier_gr1.humanoid_sv.sv1_4x3` | Fourier GR-1 | 640×480 | `humanoid.sv1_4x3.dreamdojo`, fastercache |
| `a1x_ee.a1x_sv.sv1_4x3` | Galaxea A1X (EE pose) | 640×480 | `single_arm.sv1_4x3.radial_dreamgen` (radial, via `--videos`) |
| `aloha_bimanual.bimanual_mv.mv4_row` | ALOHA 2×vx300s | 4×(320×192) row | `bimanual.mv4_row.ctrlworld_4view_grid` |
| `aloha_bimanual.bimanual_mv.mv4_grid` | ALOHA 2×vx300s | 2×2 of 384×216 | `bimanual.mv4_grid.dreamgen` |
| `aloha_bimanual.bimanual_sv.sv1_16x9` | ALOHA 2×vx300s | 768×432 | `bimanual.sv1_16x9.dreamgen`, `bimanual.sv1_16x9.radial_dreamgen` (radial, via `--videos`) |
| `aloha_bimanual.bimanual_sv.sv1_4x3` | ALOHA 2×vx300s | 640×480 | `bimanual.sv1_4x3.dreamdojo` |

`Synthetic2R` (closed-form, no URDF) is the CPU-only test fixture.

## Documentation

| | |
|---|---|
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | data, training, scoring, outputs |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | modules, the diffusion head, the registry |
| [`docs/ADDING_A_ROBOT.md`](docs/ADDING_A_ROBOT.md) | adding an embodiment, with the ALOHA walkthrough |
| [`docs/METRICS.md`](docs/METRICS.md) | detectors, segments, calibration |
| [`docs/BIMANUAL.md`](docs/BIMANUAL.md) | ALOHA bimanual end-to-end: every command from download to bucket push |
| [`docs/MULTIVIEW_STATIC.md`](docs/MULTIVIEW_STATIC.md) | Humanoid multiview on the two static cameras: run book + real-motion baseline |
