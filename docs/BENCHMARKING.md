# Benchmarking

`kinescore score` reads generated clips through a trained reader and counts
physics violations against thresholds calibrated on real motion.

```bash
kinescore score --list                                    # cells and their status
kinescore score --cell single_arm.mv3_row.ctrlworld
kinescore score --cell single_arm.mv3_row.ctrlworld \
    --videos /path/to/generated --out /path/to/report --percentile 95
```

| flag | default | meaning |
|---|---|---|
| `--cell` | — | cell id |
| `--list` | — | print every declared cell and its status |
| `--videos` | the cell's score tree | directory searched recursively for `*.mp4` |
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
one reader (`<robot>.<view_id>`), so several cells comparing different generators
on the same robot and packing are judged by the same head, and the comparison is
between generators rather than between readers.

`method` (`dense` / `augment` / `worldcache`) and `split` (`makovian` /
`non_makovian`) are sub-partitions inside a cell. They live in the clip paths, not
in separate cells.

A cell inherits its reader's `status`: a blocked reader blocks every cell that
names it, and `--list` shows why.

## Order of operations

1. Resolve the cell; refuse if it or its reader is blocked.
2. Load the checkpoint under a `ReaderExpectation` built from the cell — robot,
   view id, panel count, packing. A mismatch fails before the backbone is built.
3. **Calibrate.** Read up to `--calibration-clips` real clips from the reader's
   own `videos/val/` and fit every detector's threshold at `--percentile` of
   pooled real per-frame scores. If there are none, the run stops and tells you
   to materialise the reader's tree first.
4. Score each clip in `--videos` into per-type reports.
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
{"path": "...", "cell_id": "single_arm.mv3_row.ctrlworld",
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

## Scope

kinescore judges physics plausibility of the arm's motion as read from pixels.
It does not judge task success, visual fidelity, prompt adherence, or object
dynamics — nothing outside the robot's own keypoints is modelled. See
[METRICS.md](METRICS.md) for what each detector does and does not detect.
