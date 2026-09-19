# Architecture

One learned component (a keypoint reader), five analytic detectors, and a
registry that decides which reader may score which video.

```
video frames ──▶ frozen DINOv3 ──▶ DiffusionHead ──▶ P (B,T,K,3) ──▶ detectors ──▶ report
                 (backbones/)      (heads/)          metres,          (violations/)
                                                     base frame
```

Everything downstream of `P` is closed-form geometry, so a detector cannot be
tuned to favour a generator, and the reader's competence is auditable
independently of any score: its error is reported in millimetres against
forward kinematics.

## Identities

| id | shape | one per | example |
|---|---|---|---|
| reader | `<robot>.<corpus>.<view_id>` | trained head | `fourier_gr1.humanoid_sv.sv1_16x9` |
| cell | `<embodiment>.<view_id>.<model>` | scored unit | `humanoid.sv1_4x3.dreamdojo` |

A cell names the reader it reads through; several cells can share one reader.
`method`, `role` and `split` partition clips *inside* a cell. Both ids are
declared in `configs/*.yaml` and resolved by `registry/cells.py`; a reader
with a non-empty `status` blocks itself and every cell that names it at
registry-resolution time.

## Modules

| package | responsibility |
|---|---|
| `adapters/` | corpus → `RawEpisode` (videos + joint arrays + scene key); `lerobot` is the only adapter |
| `registry/` | `configs/*.yaml` → readers/cells, view geometry, downloads, the bench manifest, the train tree, run manifests |
| `backbones/` | frozen DINOv3 ViT feature extraction, patch pooling, per-panel encoding |
| `heads/` | `DiffusionHead`: pooled tokens → keypoints by denoising |
| `readers/` | backbone + head → `Readout`; checkpoint save/load with identity checks |
| `robots/` | per-robot `RobotSpec`: FK, keypoint links, rigid bone pairs, joint limits, colliders |
| `training/` | token cache I/O, scene-disjoint splits, the trainer |
| `violations/` | the five detectors, percentile calibration, 16-frame segments |
| `video/` | ffprobe, frame decoding, overlay rendering, the export bundle |
| `core/` | shared contracts: `ClipSpec`/`ViewLayout`, `ClipContext`, `PoseReader`, `RobotSpec` |
| `cli/` | one `cmd_*.py` per subcommand, auto-discovered by `cli/main.py` |

## Backbone

`FeatureBackbone` (`backbones/dino.py`) wraps DINOv3 ViT-L/16
(`embed_dim=1024`, patch 16). Per frame it crops the packed frame into panels
via `ViewLayout`, resizes, runs the ViT under `torch.no_grad()`, strips
CLS/register tokens, pools patches, and returns `(N, V, P, D)` fp16.
`backbones/default.py` holds the single `BACKBONE_CFG` every stage builds
from, so cache, trainer and scorer agree about what produced a token.

## Diffusion head

`DiffusionHead` (`heads/diffusion.py`) takes `(B, T, V*P, D)` pooled tokens
and emits `(B, T, K, 3)` metres in the robot-base frame. Each query carries a
noised keypoint and its noise level; the head predicts the clean coordinate.
A read is DDIM sampling with `eta = 0`, averaging `n_samples` samples.
Coordinates are normalised to `[-1, 1]` by a workspace box measured off the
training targets at the start of `fit` and stored in the checkpoint; a head
asked to read before the box is fitted raises.

Training: masked smooth-L1 (`beta = 0.05` m) over `(B, T, K, 3)` windows of
16 frames, targets from forward kinematics on the logged joints. The head
predicts keypoints directly rather than joint angles through FK. Under FK,
limb lengths are fixed by construction, so rigidity would be identically
zero.

## Checkpoints

`readers/checkpoint.py` stores the head state dict plus `cfg` (robot,
view id, panel count, packing, head architecture) and `meta` (`train_mm`,
`val_mm`, `best_step`, episode counts). `load_reader` takes a
`ReaderExpectation` and refuses a checkpoint whose `cfg` disagrees, so a
mismatched head fails before the backbone is built.

## Robots

Keypoint counts are what `configs/robots.yaml` declares, and `kinescore train`
checks each one against what forward kinematics actually returns.

| robot | key | embodiment | keypoints | URDF |
|---|---|---|---|---|
| ALOHA 2×vx300s | `aloha_bimanual` | bimanual | 18 | `KINESCORE_ASSETS` |
| Fourier GR-1 | `fourier_gr1` | humanoid | 22 | `KINESCORE_ASSETS` |
| Airbot MMK2 | `airbot_mmk2` | humanoid | 12 | `KINESCORE_ASSETS` |
| Franka Panda | `franka_panda` | single_arm | 8 | `robot_descriptions` package |
| Galaxea A1X | `a1x_ee` | single_arm | 4 | none, EE pose |
| `Synthetic2R` | `synthetic_2r` | single_arm | 3 | none, closed form |

`a1x_ee` reads a logged end-effector pose rather than joint angles, so it
loads no URDF at runtime. `Synthetic2R` is the CPU-only test fixture.

### Adding a robot

A robot is one Python module under `robots/<name>/` plus three declarations.
The contract is the `RobotSpec` protocol in `core/robot.py`, and
`robots/base.py` holds the shared helpers a URDF-driven robot builds on.

See [ADDING_A_ROBOT.md](ADDING_A_ROBOT.md) for the steps, the ALOHA bimanual
walkthrough, and the list of what fails where.

## Detector interface

Each detector implements `fit` (optional), `per_frame(ctx) -> (T,)`,
`calibrate(scores, pct, floor)` and `report(ctx)`. `ViolationScorer` pools
per-frame scores over real clips, sets each threshold at a percentile, and
scores generated clips into reports with frame intervals. See
[METRICS.md](METRICS.md).
