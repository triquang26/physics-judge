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

    corpus  ──kinescore data──▶  canonical train tree
                                        │
                                 kinescore cache  (frozen backbone → tokens)
                                        │
                                 kinescore train  (head ← forward-kinematics targets)
                                        │
                                 kinescore score  (generated clips → violations)

Two ids run through all four stages, and both are declared in
`configs/cells.yaml`:

- **reader** = `<robot>.<view_id>` — one trained head. Two corpora seen through
  the same packing of the same robot share it: a generator changes what the
  pixels look like, not what a joint is.
- **cell** = `<embodiment>.<view_id>.<model>` — one scored unit. `method`
  (dense / augment / worldcache) and `split` (makovian / non_makovian)
  partition clips *inside* a cell.

Packing is measured, never inferred from a name, and lives in
`configs/views.yaml`: `sv1`, `mv3_row`, `mv4_row`, `mv4_grid`,
`mv4_grid_br_blank`.

## Install

```bash
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env      # KINESCORE_ASSETS / CKPT_DIR / CACHE_DIR / DATA_ROOT / OUTPUT_DIR
```

`torch` is a base dependency. Install a CUDA build first
(`uv pip install torch --index-url https://download.pytorch.org/whl/<cuXXX>`)
if you need a specific CUDA version. `ffmpeg`/`ffprobe` must be on `PATH`.

## Run

```bash
kinescore data  --list                             # every reader and its status
kinescore data  --reader franka_panda.mv3_row      # corpus  → train tree
kinescore cache --reader franka_panda.mv3_row --device cuda
kinescore train --reader franka_panda.mv3_row      # → $KINESCORE_CKPT_DIR/<reader_id>.pt

kinescore score --list                             # every cell and its status
kinescore score --cell single_arm.mv3_row.ctrlworld --videos path/to/generated/
```

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
| Jerk | motion smoothness | a smooth but kinematically impossible motion |
| Teleport | a keypoint jumping further than motion allows | a fast-but-continuous slide |
| Joint limit | a keypoint outside the robot's reachable envelope | a pose that is reachable but wrong |
| Self-collision | limbs interpenetrating | a robot for which no collision geometry is declared — reported as unavailable, never as a fabricated zero |

All five threshold a frame-rate-dependent quantity, so a wrong `dt` moves every
number. `ClipSpec` owns the timebase and `ffprobe` is cross-checked against any
declared rate; a disagreement is a hard error.

## Testing

```bash
pytest                # CPU-only, network-free, no checkpoints
pytest -m ""          # everything, including GPU / network / checkpoint tiers
```

The default tier needs no GPU, no downloads and no trained weights.

## Documentation

| | |
|---|---|
| [`docs/DATA_PREP.md`](docs/DATA_PREP.md) | corpus layout, what an adapter yields, the canonical train tree |
| [`docs/TRAINING.md`](docs/TRAINING.md) | cache → train → the mm acceptance gate, per reader |
| [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) | scoring a cell, threshold calibration, reading the output |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | the module map; adding a robot, an adapter, a detector |
| [`docs/METRICS.md`](docs/METRICS.md) | the five detectors: formula, units, `dt` exponent, what each misses |

## Scope, honestly

This is an **evaluation methodology**, not a model. Absolute keypoint accuracy
has a floor of tens of millimetres, so kinescore is a **relative** referee: run
the same frozen judge on two systems and compare. Derivative-based quantities
cancel the reader's constant localisation bias, which is what makes that
comparison robust — but they do not cancel its *variance*, so a higher frame
rate amplifies reader noise into jerk. Score at a fixed, recorded frame rate.
