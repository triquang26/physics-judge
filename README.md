# kinescore

**Physics-plausibility benchmark for AI-generated robot video.**

A world model that draws a robot must draw one that *moves like a robot*.
kinescore reads 3-D robot keypoints out of the pixels with a frozen vision
backbone and a small trained head, then measures **analytic physics
violations** on those keypoints — no simulator, no VLM judge, no learned critic
downstream of the reader.

Two properties make the score trustworthy:

- **Non-circular.** The reader is a frozen DINOv3 backbone the evaluated model
  never trained against, so a model cannot score well by matching the judge's
  features.
- **Calibrated on real motion.** Every detector's threshold is a percentile of
  the same quantity measured on real teleop from the same robot and packing, so
  "violation" means "outside what this robot actually does", not a hand-picked
  constant.

## The pipeline

    HF ──kinescore pull──▶ bench clips + training corpora
                                   │
                            kinescore data   (corpus → train tree)
                                   │
                            kinescore cache  (frozen backbone → tokens)
                                   │
                            kinescore train  (head ← forward-kinematics targets)
                                   │
                            kinescore score  (bench clips → violations)
                                   │
                            kinescore report (one table over every cell)

Two ids run through every stage, both declared in `configs/cells.yaml`:

- **reader** = `<robot>.<corpus>.<view_id>` — one trained head. All three change
  the weights, so all three are in the name.
- **cell** = `<embodiment>.<view_id>.<model>` — one scored unit. `method`
  (`dense` / `augment` / `worldcache`), `role` (`dense` / `fast` /
  `weak_augment` / `strong_augment`) and `split` (`makovian` / `non_makovian`)
  partition clips *inside* a cell.

Packing is measured, never inferred from a name, and lives in
`configs/views.yaml`.

## Install

```bash
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env      # KINESCORE_ASSETS / CKPT_DIR / CACHE_DIR / DATA_ROOT / OUTPUT_DIR
```

`torch` is a base dependency. Install a CUDA build first
(`uv pip install torch --index-url https://download.pytorch.org/whl/<cuXXX>`)
if you need a specific CUDA version. `ffmpeg`/`ffprobe` must be on `PATH`.

## Data

`configs/sources.yaml` is the only thing `pull` reads, and every download is
pinned in `$KINESCORE_DATA_ROOT/REVISIONS.json`.

```bash
kinescore pull --list             # declared sources and what is on disk
kinescore pull --what bench       # 400 scored clips + manifest.json  (2.2 GB)
kinescore pull --what train       # the six training corpora          (15 GB)
```

```
$KINESCORE_DATA_ROOT/
    bench/      clips/<id>.mp4, manifest.json, augment|dense|worldcache/
    corpus/     {bimanual,humanoid,single_arm}/{multiview,singleview}/
    trees/      <reader_id>/{videos,annotation}/{train,val}/
    REVISIONS.json
```

Re-running `pull` fetches the revision already recorded, so a second machine
gets the same bytes. Pass `--revision` to move to a new one deliberately.

## Readers and what they score

```bash
kinescore readers                 # every head, its corpus, and its cells
kinescore models                  # every generator in the bench, and clip counts
```

| reader | robot | corpus | scores |
|---|---|---|---|
| `aloha_bimanual.bimanual_mv.mv4_row` | ALOHA | `bimanual_mv` | `bimanual.mv4_row.ctrlworld_4view_grid` |
| `aloha_bimanual.bimanual_mv.mv4_grid` | ALOHA | `bimanual_mv` | `bimanual.mv4_grid.dreamgen` |
| `aloha_bimanual.bimanual_sv.sv1_16x9` | ALOHA | `bimanual_sv` | `bimanual.sv1_16x9.dreamgen` |
| `aloha_bimanual.bimanual_sv.sv1_4x3` | ALOHA | `bimanual_sv` | `bimanual.sv1_4x3.dreamdojo` |
| `airbot_mmk2.humanoid_mv.mv4_row` | Airbot MMK2 | `humanoid_mv` | `humanoid.mv4_row.ctrlworld_4view_grid` |
| `airbot_mmk2.humanoid_mv.mv4_grid` | Airbot MMK2 | `humanoid_mv` | `humanoid.mv4_grid.dreamgen` |
| `fourier_gr1.humanoid_sv.sv1_16x9` | Fourier GR-1 | `humanoid_sv` | `humanoid.sv1_16x9.dreamgen` |
| `fourier_gr1.humanoid_sv.sv1_4x3` | Fourier GR-1 | `humanoid_sv` | `humanoid.sv1_4x3.dreamdojo` |
| `franka_panda.single_arm_mv.mv3_row` | Franka Panda | `single_arm_mv` | `single_arm.mv3_row.ctrlworld` |
| `franka_panda.single_arm_mv.mv4_grid_br_blank` | Franka Panda | `single_arm_mv` | `single_arm.mv4_grid_br_blank.dreamgen` |
| `franka_panda.single_arm_mv.sv1_4x3` | Franka Panda | `single_arm_mv` | `single_arm.sv1_4x3.dreamgen`, `single_arm.sv1_4x3.dreamdojo` |

11 heads cover the bench's 12 cells: the two single-arm singleview cells differ
only in generator, which does not change what a joint is.

## Train every reader

Three commands per reader, always in this order. Substitute any id from the
table above for `$R`.

```bash
R=franka_panda.single_arm_mv.mv3_row
kinescore data  --reader $R                 # corpus → train tree (ffmpeg, CPU)
kinescore cache --reader $R --device cuda   # frozen backbone → tokens
kinescore train --reader $R --device cuda   # → $KINESCORE_CKPT_DIR/$R.pt
```

All eleven, four at a time:

```bash
for R in $(kinescore readers --ids); do
    kinescore data --reader "$R" &
    [ $(jobs -r | wc -l) -ge 4 ] && wait
done; wait

for R in $(kinescore readers --ids); do
    kinescore cache --reader "$R" --device cuda
    kinescore train --reader "$R" --device cuda
done
```

`train` reports RMS keypoint error in millimetres on the scene-disjoint
validation split, and stamps it into the checkpoint.

## Score the bench

With no `--videos`, `score` takes exactly the clips `bench/manifest.json`
assigns to the cell, so the scored set is the published one.

```bash
kinescore score --list                                   # cells and their status
kinescore score --cell single_arm.mv3_row.ctrlworld --device cuda
```

Every cell, then the table:

```bash
for C in $(kinescore score --list | awk '{print $1}'); do
    kinescore score --cell "$C" --device cuda
done
kinescore report --by role --out out/report.json
```

`report` prints, per cell and per sub-partition, the fraction of clips with at
least one flagged interval per detector. `--by method` and `--by split` are the
other two axes.

Each stage writes a `run_manifest.json` next to its output: argv, git sha, host,
and the sha256 of every config it read.

## Robots

| Robot | Registry key | Joints predicted | Keypoints K |
|---|---|---|---|
| Franka Panda | `franka_panda` | 7 | 8 |
| ALOHA bimanual | `aloha_bimanual` | 12 | 18 |
| Airbot MMK2 | `airbot_mmk2` | 12 | 12 |
| Fourier GR-1 | `fourier_gr1` | 17 | 12 |
| `Synthetic2R` | `synthetic_2r` | 2 | 3 |

`Synthetic2R` is closed-form and needs no URDF; it is the CPU-only test
fixture. Adding a robot is implementing one protocol — see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#adding-a-robot).

The head predicts keypoints directly rather than joint angles passed through
forward kinematics. Under FK, limb lengths are fixed by construction, so a
rigidity residual would be identically zero and the reader could not see the
limb geometry a generator gets wrong. Forward kinematics still runs — once per
episode at data-prep time — to build the training target from logged joints.

## What each detector does *not* catch

| Detector | Measures | Does **not** detect |
|---|---|---|
| Rigidity | link lengths staying constant | anything about *speed*; bones with an endpoint on a gripper-driven link are excluded, since their length is actuation, not structure |
| Jerk | motion smoothness, per second | a smooth but kinematically impossible motion |
| Teleport | a keypoint jumping further than motion allows | a fast-but-continuous slide |
| Joint limit | a keypoint outside the robot's reachable envelope | a pose that is reachable but wrong |
| Self-collision | limbs interpenetrating | a robot for which no collision geometry is declared — reported as unavailable, never as a fabricated zero |

Jerk and teleport are normalised by `dt`, so the bench's 5–16 fps clips are
comparable. `ClipSpec` owns the timebase and `ffprobe` is cross-checked against
any declared rate; a disagreement is a hard error.

## Testing

```bash
pytest                # CPU-only, network-free, no checkpoints
pytest -m ""          # everything, including GPU / network / checkpoint tiers
```

The default tier needs no GPU, no downloads and no trained weights.

## Documentation

| | |
|---|---|
| [`docs/DATA_PREP.md`](docs/DATA_PREP.md) | sources, corpus layout, packing, the train tree |
| [`docs/TRAINING.md`](docs/TRAINING.md) | cache → train, per reader |
| [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) | scoring a cell, calibration, the report table |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | the module map; adding a robot, an adapter, a detector |
| [`docs/METRICS.md`](docs/METRICS.md) | the five detectors: formula, units, `dt` exponent, what each misses |

## Scope, honestly

This is an **evaluation methodology**, not a model. Absolute keypoint accuracy
has a floor of tens of millimetres, so kinescore is a **relative** referee: run
the same frozen judge on two systems and compare. Derivative-based quantities
cancel the reader's constant localisation bias, which is what makes that
comparison robust — but they do not cancel its *variance*, so a higher frame
rate amplifies reader noise into jerk. Score at a fixed, recorded frame rate.
