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
predicts keypoints directly rather than joint angles through FK — under FK,
limb lengths are fixed by construction and rigidity would be identically zero.

## Checkpoints

`readers/checkpoint.py` stores the head state dict plus `cfg` (robot,
view id, panel count, packing, head architecture) and `meta` (`train_mm`,
`val_mm`, `best_step`, episode counts). `load_reader` takes a
`ReaderExpectation` and refuses a checkpoint whose `cfg` disagrees, so a
mismatched head fails before the backbone is built.

## Robots

| robot | key | joints | keypoints |
|---|---|---|---|
| Airbot MMK2 | `airbot_mmk2` | 12 | 12 |
| Fourier GR-1 | `fourier_gr1` | 17 | 12 |
| `Synthetic2R` | `synthetic_2r` | 2 | 3 |

`Synthetic2R` is closed-form, needs no URDF, and is the CPU-only test
fixture.

### Adding a robot

Implement `RobotSpec` (`robots/base.py`): joint names, `build_target`
(FK: `(T, n_joints)` → `(T, K, 3)` metres), `rigid_bone_pairs`, optional
colliders. Register it in `robots/__init__.py`, declare it in
`configs/robots.yaml` with its keypoint count — checked against FK output at
train time.

## Detector interface

Each detector implements `fit` (optional), `per_frame(ctx) -> (T,)`,
`calibrate(scores, pct, floor)` and `report(ctx)`. `ViolationScorer` pools
per-frame scores over real clips, sets each threshold at a percentile, and
scores generated clips into reports with frame intervals. See
[METRICS.md](METRICS.md).
