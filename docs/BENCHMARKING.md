# Benchmarking

`kinescore score` reads generated clips through a trained reader and counts
physics violations against thresholds calibrated on real motion.

```bash
kinescore score --list                                    # cells and their status
kinescore score --cell single_arm.mv3_row.ctrlworld       # the bench's own clips
kinescore score --cell single_arm.mv3_row.ctrlworld \
    --videos /path/to/generated --out /path/to/report --percentile 95
kinescore report --by role --out out/report.json          # every scored cell
```

| flag | default | meaning |
|---|---|---|
| `--cell` | — | cell id |
| `--list` | — | print every declared cell and its status |
| `--videos` | the clips `bench/manifest.json` assigns to this cell | directory searched recursively for `*.mp4` |
| `--checkpoint` | the reader's own | reader checkpoint to score through |
| `--out` | the cell's own | output directory |
| `--calibration-clips` | `24` | real clips the thresholds are fitted on |
| `--percentile` | `95.0` | threshold percentile over real motion |
| `--max-frames` | `0` | cap frames per clip |
| `--limit` | `0` | cap clips scored |
| `--device` | `cuda` | where the backbone runs |

## What a cell is

`<embodiment>.<view_id>.<model>` — one scored unit. It names the robot, the view
packing, and the generator whose output is being judged. It resolves to exactly
one reader (`<robot>.<corpus>.<view_id>`), so several cells comparing different
generators on the same robot and packing are judged by the same head, and the
comparison is between generators rather than between readers.

`method` (`dense` / `augment` / `worldcache`), `role` (`dense` / `fast` /
`weak_augment` / `strong_augment`) and `split` (`makovian` / `non_makovian`) are
sub-partitions inside a cell. They come from `bench/manifest.json`, are carried
into every scored record, and are what `kinescore report --by` groups on.

A cell inherits its reader's `status`: a blocked reader blocks every cell that
names it, and `--list` shows why.

## Which reader judges which cell

Every cell resolves to one reader, and that reader is fitted on one corpus.
`kinescore score --list` prints this live; the assignment itself is
`configs/cells.yaml`.

| cell | reader | corpus | clips |
|---|---|---|---|
| `bimanual.mv4_row.ctrlworld_4view_grid` | `aloha_bimanual.bimanual_mv.mv4_row` | `bimanual_mv` | 34 |
| `bimanual.mv4_grid.dreamgen` | `aloha_bimanual.bimanual_mv.mv4_grid` | `bimanual_mv` | 34 |
| `bimanual.sv1_16x9.dreamgen` | `aloha_bimanual.bimanual_sv.sv1_16x9` | `bimanual_sv` | 34 |
| `bimanual.sv1_4x3.dreamdojo` | `aloha_bimanual.bimanual_sv.sv1_4x3` | `bimanual_sv` | 34 |
| `humanoid.mv4_row.ctrlworld_4view_grid` | `airbot_mmk2.humanoid_mv.mv4_row` | `humanoid_mv` | 34 |
| `humanoid.mv4_grid.dreamgen` | `airbot_mmk2.humanoid_mv.mv4_grid` | `humanoid_mv` | 34 |
| `humanoid.sv1_16x9.dreamgen` | `fourier_gr1.humanoid_sv.sv1_16x9` | `humanoid_sv` | 34 |
| `humanoid.sv1_4x3.dreamdojo` | `fourier_gr1.humanoid_sv.sv1_4x3` | `humanoid_sv` | 34 |
| `single_arm.mv3_row.ctrlworld` | `franka_panda.single_arm_mv.mv3_row` | `single_arm_mv` | 32 |
| `single_arm.mv4_grid_br_blank.dreamgen` | `franka_panda.single_arm_mv.mv4_grid_br_blank` | `single_arm_mv` | 32 |
| `single_arm.sv1_4x3.dreamgen` | `franka_panda.single_arm_mv.sv1_4x3` | `single_arm_mv` | 32 |
| `single_arm.sv1_4x3.dreamdojo` | `franka_panda.single_arm_mv.sv1_4x3` | `single_arm_mv` | 32 |

The twelve cells partition all 400 bench clips; every clip is judged exactly
once, by the reader trained on the same robot and view packing.

## The augment set is the negative control

55 of the 400 clips are real video deliberately corrupted to a known severity.
They exist to answer a question the generated clips cannot: **is the detector
alive?** A violation rate of zero on a generator means nothing until the same
detector fires on motion that is known to be wrong.

| role | n | `aug_tag` |
|---|---|---|
| `strong_augment` | 23 | `object_relocation` 13, `task_swap` 10 |
| `weak_augment` | 32 | `combo` 8, `edit_only` 8, `v04_purple` 4, `v05_orange` 3, `v06_pink` 3, `prompt_only` 2, `v00_red` / `v03_yellow` / `v07_cyan` / `v09_black` 1 each |

`strong_augment` moves the scene's semantics — the task is swapped or an object
is relocated — and every such clip in the source is included. `weak_augment` is
a small random sample of milder edits: colour variants (`v*`), prompt-only and
edit-only changes, and their combination.

Read them as a floor, not a ceiling: a detector that misses `strong_augment` is
broken, but one that catches it is only shown to catch corruption of that kind
and that size.

### Coverage is uneven, and most cells have none

The 23 `strong_augment` clips sit in three cells:

| cell | `strong_augment` | `weak_augment` |
|---|---|---|
| `bimanual.sv1_16x9.dreamgen` | 16 | 4 |
| `single_arm.mv4_grid_br_blank.dreamgen` | 4 | 4 |
| `bimanual.mv4_grid.dreamgen` | 3 | 6 |

The other nine cells have none, and five of those (`bimanual.sv1_4x3.dreamdojo`,
`humanoid.mv4_grid.dreamgen`, `humanoid.sv1_16x9.dreamgen`,
`humanoid.sv1_4x3.dreamdojo`, `single_arm.sv1_4x3.dreamdojo`) carry no augment
clips at all.

For those cells a violation rate stands on its own with nothing to calibrate it
against, so report it as a measurement of the generator and not as evidence that
the detector works. Extending coverage means sampling more `augment/` clips into
`bench/manifest.json` for the cells that lack them.

## Order of operations

1. Resolve the cell; refuse if it or its reader is blocked.
2. Load the checkpoint under a `ReaderExpectation` built from the cell — robot,
   view id, panel count, packing. A mismatch fails before the backbone is built.
3. **Calibrate.** Read up to `--calibration-clips` real clips from the reader's
   own `videos/val/` and fit every detector's threshold at `--percentile` of
   pooled real per-frame scores. If there are none, the run stops and tells you
   to materialise the reader's tree first.
4. Select the cell's clips from `bench/manifest.json`, unless `--videos`
   overrides, and score each into per-type reports.
5. Write `results.jsonl`, `summary.json`, `run_manifest.json`.

Calibration comes from the reader's *validation* split — real motion the head was
not fitted on. Thresholds therefore describe what this reader's error looks like
on real video it has not seen, not what it looks like on its training data.

## Why calibrate at all

A detector's raw score mixes real physics with reader error. A rigidity score of
12 mm might be a real bone-length violation or the head's own jitter. Fitting the
threshold on real motion, read through the same head, subtracts the reader's own
noise floor: whatever survives is motion the reader would not have produced on
real video. Two detectors also carry an absolute floor
(`rigidity` 18 mm, `joint_limit` 3°) so that on a corpus where real motion is
near-perfectly rigid, the threshold does not collapse to the reader's float
noise and flag everything.

The consequence to keep in mind: scores are comparable **within** a cell, and
across cells only when the readers have comparable `val_mm`. A cell whose reader
is worse will have looser thresholds.

## Output

`results.jsonl`, one line per clip:

```json
{"path": "...", "cell_id": "single_arm.mv3_row.ctrlworld", "id": "00317",
 "method": "dense", "role": "dense", "split": "makovian", "task": "...",
 "violations": {"rigidity": {...}, "jerk": {...}, "teleport": {...},
                "joint_limit": {...}, "self_collision": {...}}}
```

Each per-type report carries the count, the score, and the frame intervals where
the threshold was exceeded, so a flagged clip can be inspected at the frame that
caused it.

A clip that fails to read is written as `{"path": ..., "error": ...}` and
counted; the run continues and exits non-zero. Failures are recorded, never
silently dropped.

`summary.json` records the cell, the reader, the checkpoint **and its SHA-256**,
the video root, clip counts, the calibration percentile, and the fitted
thresholds. `run_manifest.json` adds the config sources. Two runs are comparable
only if these agree, and the files make that checkable rather than assumed.

## The report table

`kinescore report` reads every cell's `results.jsonl` and prints, per cell and
per sub-partition, the fraction of clips with at least one flagged interval per
detector. A cell that has not been scored is printed as `not scored` rather than
omitted. `--out` writes the same table as JSON.

```bash
kinescore report --by role      # dense / fast / weak_augment / strong_augment
kinescore report --by method    # dense / augment / worldcache
kinescore report --by split     # makovian / non_makovian
```

## Scope

kinescore judges physics plausibility of the arm's motion as read from pixels.
It does not judge task success, visual fidelity, prompt adherence, or object
dynamics — nothing outside the robot's own keypoints is modelled. See
[METRICS.md](METRICS.md) for what each detector does and does not detect.
