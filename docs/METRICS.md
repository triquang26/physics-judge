# Metrics

Analytic functions of the predicted keypoints `P (T, K, 3)` in metres and the
timebase `dt`. No learned parameters, no per-generator tuning; the only
learned input is `P` itself.

| detector | units | flags | reads |
|---|---|---|---|
| `rigidity` | mm | above | bone lengths vs URDF rest lengths |
| `jerk` | mm/s³ | above | 3rd time difference of keypoints |

## Segments

Verdicts are per **16-frame segment**, not per frame
(`violations/segments.py`). A segment reduces the per-frame series and judges
the reduced value against the threshold:

| detector | reduce |
|---|---|
| `rigidity` | median (a violation must persist, one bad frame is reader noise) |
| `jerk` | worst |

`segments.csv`, the export `segments.json`, and the rendered overlays all
report at this granularity.

## Calibration

`ViolationScorer.calibrate(real_clips, pct=95.0)` pools each detector's
per-frame scores over real clips (the reader's own val split, motion the head
was not fitted on) and sets the threshold at the `pct`-th percentile. Scoring
before calibration raises; there is no default threshold.

Floor: `rigidity` 18 mm. Real motion is rigid up to float error, so without a
floor the rigidity threshold would collapse onto the reader's own jitter and
flag everything.

A threshold subtracts the reader's noise floor, so whatever survives is motion
the reader would not produce on real video. A cell whose reader has worse
`val_mm` has looser thresholds, so scores compare across cells only when the
readers are comparable.

### Segment-level baseline on real motion

The p95 is per **frame**; verdicts are per **segment** through the reduce, so
the fraction of *real* segments flagged is not 5%. `median` (rigidity) needs 8
of 16 frames over the threshold and lands below 5%; `worst` (jerk) needs one
frame and lands far above it. The exact baseline for a reader is measured by
scoring its own calibration clips through the same pipeline:

```bash
kinescore score --cell <any cell of the reader> \
    --videos $KINESCORE_DATA_ROOT/trees/<reader_id>/videos/val --limit 24 \
    --out out/baseline.<view>.real
```

Measured baselines (real val clips, in-sample with the thresholds):

| reader | segments | rigidity | jerk |
|---|---|---|---|
| `fourier_gr1.humanoid_sv.sv1_16x9` | 524 | 4.0% | 31.3% |
| `aloha_bimanual.bimanual_sv.sv1_16x9` | — | — | — |
| `a1x_ee.a1x_sv.sv1_4x3` | — | — | — |

Read a cell's rate as its **excess over the same reader's baseline**, not
against zero.

## rigidity

For each pair in `robot.rigid_bone_pairs`, `|P[:, a] - P[:, b]|` in mm; the
largest absolute deviation from the URDF rest length across bones, per frame.
`rigid_bone_pairs` excludes degenerate bones and bones whose endpoint sits on
a gripper-driven link; their length tracks actuation, not structure.

Does not detect an arm that is the right shape in the wrong place, or a warp
that preserves every bone length.

## jerk

Third difference `P[t] - 3P[t-1] + 3P[t-2] - P[t-3]`, norm per keypoint,
worst keypoint per frame, divided by `dt³` to get mm/s³. The first three
frames are zero; clips shorter than 4 frames score all-zero. Per second, not
per frame: thresholds calibrated at one frame rate apply to clips at another,
so a per-frame difference would rank the slowest-sampled generator as the
jerkiest.

Does not detect smooth but kinematically impossible motion. Reader variance
also scales with frame rate, so score at a fixed, recorded rate.

## Per-clip report fields

`report(ctx)` per detector: `units`, `threshold`, `fraction` (of frames
flagged), `n_flagged`, `severity_ratio_median`/`_p90`
(`per_frame / threshold`, so `> 1` means worse), `intervals` (`[start, end]`
frame pairs), `per_frame`.

## What they do not detect

Task success, visual fidelity, object dynamics, and contact all sit outside
the robot's own keypoint set. A clip that passes both detectors is not
physically correct; it is a clip in which the arm's own kinematics show no
violation this reader can see.
