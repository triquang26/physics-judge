# Training a reader

A reader is one trained `KeypointHead`, identified as
`<robot>.<corpus>.<view_id>`. It is fitted in two commands: encode the canonical
tree into frozen backbone tokens, then fit the head against forward-kinematics
targets. `kinescore readers --ids` lists every id.

```bash
R=franka_panda.single_arm_mv.mv3_row
kinescore cache --reader $R --device cuda
kinescore train --reader $R --device cuda
```

## Stage 1 — cache

`kinescore cache` runs the frozen DINOv3 backbone once per episode and writes
pooled tokens to `$KINESCORE_CACHE_DIR/<reader_id>/{train,val}/<episode>.pt`.
The backbone is never trained, so encoding once and fitting many heads on the
same tokens is exact, not an approximation.

| flag | default | meaning |
|---|---|---|
| `--reader` | — | reader id |
| `--splits` | `train,val` | which splits to encode |
| `--device` | `cuda` | where the backbone runs |
| `--limit` | `0` | cap episodes per split |
| `--overwrite` | off | re-encode already-cached episodes |
| `--frame-chunk` | `32` | frames per backbone call; lower it if a long episode exhausts GPU memory |
| `--max-frames` | `0` | cap decoded frames per episode |

Each cache file is self-describing: it carries the reader id, the view layout
key, the backbone config, and the token shape. Loading refuses a file written
for a different reader, a different view, or a different backbone. A bare tensor
with no header is refused outright. The stage also asserts
`joint_source == "real"` in the matching annotation and skips episodes with no
annotation, reporting the count.

Re-running is cheap: already-cached episodes are counted and skipped unless
`--overwrite` is passed. Finishing an interrupted cache is therefore just the
same command again.

Training reads the cache, not the tree, so a cache that stopped early would
train on whatever episodes reached it and report an ordinary-looking
validation number. `kinescore train` compares the two per split and refuses
when the cache is short, naming both counts; `--allow-partial-cache` proceeds
anyway.

## Stage 2 — train

`kinescore train` builds the head from the robot's keypoint count and fits it.

| flag | default |
|---|---|
| `--head` | `keypoint` (see Heads) |
| `--steps` | `6000` |
| `--batch-size` | `32` windows per step |
| `--window-size` | `16` frames per window |
| `--lr` / `--lr-late` / `--lr-step-at` | `1e-3` / `5e-4` / `1500` |
| `--seed` | `0` |
| `--eval-every` / `--log-every` | `500` / `100` |
| `--device` | `cuda` |
| `--limit` | `0` (cap episodes per split) |
| `--read-workers` | `4` threads reading a batch's windows |
| `--allow-partial-cache` | off (see stage 1) |
| `--out` | the reader's own checkpoint path |

Before anything is built, the command checks `configs/robots.yaml`'s declared
keypoint count against what the robot's forward kinematics actually produces and
exits if they disagree — a mismatch would train a head whose outputs no detector
can interpret.

## Heads

`--head keypoint` regresses the coordinates: `K` queries cross-attend to the
frame's patch tokens, a bidirectional temporal encoder mixes each keypoint's
track, and a linear layer reads out metres.

`--head diffusion` denoises them instead. The query carries a noised keypoint
and the noise level it was drawn at, and the head predicts the clean
coordinate; a read is DDIM sampling with `eta = 0`, and `n_samples` samples are
averaged. Coordinates are normalised to `[-1, 1]` by a workspace box measured
off the training targets at the start of `fit` and stored in the checkpoint, so
a loaded head reads in the units it was trained on and a head asked to read
before the box is fitted raises rather than guessing one.

Both heads answer the same contract -- `(B, T, P, D)` tokens in, `(B, T, K, 3)`
metres out -- so the checkpoint records which one it holds and `load_reader`
rebuilds it.

## Memory

Tokens are the large half of the run: a three-panel episode is
`(T, 1728, 1024)` fp16, a few hundred megabytes, and a split runs to hundreds
of episodes. They stay on disk. `load_episodes` keeps the cache path and the
forward-kinematics target — kilobytes per episode — and each sampled window is
mapped, copied, and released at the point it is used. Resident token memory is
one batch of windows whatever the split size, so a split is bounded by disk
rather than by the allocation, and `--limit` is a speed knob rather than a way
to fit in memory.

`kinescore cache`, by contrast, holds one episode's frames at a time and is
bounded by `--frame-chunk` on the GPU side.

That trade puts a step's cost in the reads rather than the arithmetic: the
head's forward and backward pass is a few hundredths of a second at
`--batch-size 32`, while fetching that batch's windows is seconds. The reads
are issued concurrently for this reason, and `--read-workers` is the knob that
matters for throughput — past the core count it stops helping.

## Targets

Supervision is forward kinematics on the logged joint positions. `build_target`
maps `(T, n_joints)` joint angles to `(T, K, 3)` keypoint positions in the
robot-base frame, in metres. No manual annotation is involved: every target is
exactly reproducible from the joints already in the corpus.

| robot | joints | keypoints |
|---|---|---|
| `franka_panda` | 7 | 8 |
| `aloha_bimanual` | 12 | 18 |
| `airbot_mmk2` | 12 | 12 |
| `fourier_gr1` | 17 | 12 |
| `synthetic_2r` | 2 | 3 |

## Objective

Masked smooth-L1 with `beta = 0.05` (5 cm), over `(B, T, K, 3)`. Windows shorter
than `window_size` are padded and masked; padded frames contribute exactly zero
so a short episode does not bias the loss toward the origin.

`beta = 0.05` puts the quadratic-to-linear knee at 5 cm: sub-5-cm errors are
optimised as squared error, and a gross outlier — a frame the backbone could not
resolve — contributes linearly instead of dominating the batch.

## Reported numbers

`evaluate` reports RMS per-keypoint error in millimetres. `fit` returns a
`TrainResult` with `train_mm`, `val_mm`, `best_val_mm`, `best_step`, the loss
history, and the best state dict, selected on validation. `val_mm` is the number
to quote: the split is scene-disjoint, so it measures unseen scenes.

Both the checkpoint's `meta` and the stage's `run_manifest.json` record
`train_mm`, `val_mm`, `best_step`, and the per-split episode counts, so a
checkpoint carries the evidence for its own quality.

## Checkpoint identity

`save_reader` stamps the cell ids, robot, view id, view layout, and full head
architecture into `cfg`. At score time `ReaderExpectation` checks robot, view id,
panel count, packing and keypoint count before the backbone is even built, so a
mismatched head fails immediately instead of producing plausible-looking
nonsense.

## Next

[BENCHMARKING.md](BENCHMARKING.md) — score generated clips with the reader.
