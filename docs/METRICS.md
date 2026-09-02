# Metrics

Analytic functions of the predicted keypoints `P (T, K, 3)` in metres and the
timebase `dt`. No learned parameters, no per-generator tuning; the only
learned input is `P` itself. **Rigidity and jerk are the headline pair**; the
other three are reported alongside.

| detector | units | flags | reads |
|---|---|---|---|
| `rigidity` | mm | above | bone lengths vs URDF rest lengths |
| `jerk` | mm/s³ | above | 3rd time difference of keypoints |
| `teleport` | mm/s | above | 1st time difference of keypoints |
| `joint_limit` | deg | above | bend angles vs a GT-fitted envelope |
| `self_collision` | mm (min dist) | **below** | closest non-adjacent keypoint pair |

## Segments

Verdicts are per **16-frame segment**, not per frame
(`violations/segments.py`). A segment reduces the per-frame series and judges
the reduced value against the threshold:

| detector | reduce |
|---|---|
| `rigidity` | median — a violation must persist, one bad frame is reader noise |
| every other | worst |

`segments.csv`, the export `segments.json`, and the rendered overlays all
report at this granularity.

## Calibration

`ViolationScorer.calibrate(real_clips, pct=95.0)` pools each detector's
per-frame scores over real clips (the reader's own val split — motion the head
was not fitted on) and sets the threshold at the `pct`-th percentile
(`(100-pct)`-th for `self_collision`, where smaller is worse). Scoring before
calibration raises; there is no default threshold.

Floors: `rigidity` 18 mm, `joint_limit` 3°. Real motion is rigid up to float
error, so without a floor the rigidity threshold would collapse onto the
reader's own jitter and flag everything.

Consequences: a threshold subtracts the reader's noise floor, so whatever
survives is motion the reader would not produce on real video — and a cell
whose reader has worse `val_mm` has looser thresholds, so scores compare
across cells only when the readers are comparable.

### Segment-level baseline on real motion

The p95 is per **frame**; verdicts are per **segment** through the reduce, so
the fraction of *real* segments flagged is not 5%. `median` (rigidity) needs 8
of 16 frames over the threshold and lands below 5%; `worst` needs one frame
and lands far above it. The exact baseline for a reader is measured by scoring
its own calibration clips through the same pipeline:

```bash
kinescore score --cell <any cell of the reader> \
    --videos $KINESCORE_DATA_ROOT/trees/<reader_id>/videos/val --limit 24 \
    --out out/baseline.<view>.real
```

Measured baselines (real val clips, in-sample with the thresholds):

| reader | segments | rigidity | jerk | teleport | joint_limit | self_collision |
|---|---|---|---|---|---|---|
| `airbot_mmk2.humanoid_mv.mv4_grid_static` | 296 | 1.4% | 33.4% | 39.2% | 51.4% | 16.6% |
| `fourier_gr1.humanoid_sv.sv1_16x9` | 524 | 4.0% | 31.3% | 39.9% | 22.5% | 10.5% |
| `fourier_gr1.humanoid_sv.sv1_16x9` | 524 | 4.8% | 45.2% | 43.3% | 15.6% | 8.4% |

Read a cell's rate as its **excess over the same reader's baseline**, not
against zero. A generated set can also sit far *below* the baseline on the
dynamics detectors (jerk, teleport) — motion smoother than the real robot,
which those detectors do not reward or flag.

## rigidity

For each pair in `robot.rigid_bone_pairs`, `|P[:, a] - P[:, b]|` in mm; the
largest absolute deviation from the URDF rest length across bones, per frame.
`rigid_bone_pairs` excludes degenerate bones and bones whose endpoint sits on
a gripper-driven link — their length tracks actuation, not structure.

**Does not detect:** an arm that is the right shape in the wrong place, or a
warp that preserves every bone length.

## jerk

Third difference `P[t] - 3P[t-1] + 3P[t-2] - P[t-3]`, norm per keypoint,
worst keypoint per frame, divided by `dt³` → mm/s³. The first three frames are
zero; clips shorter than 4 frames score all-zero. Per second, not per frame:
thresholds calibrated at one frame rate apply to clips at another, so a
per-frame difference would rank the slowest-sampled generator as the jerkiest.

**Does not detect:** smooth but kinematically impossible motion. Reader
variance also scales with frame rate — score at a fixed, recorded rate.

## The other three

- `teleport` — first difference / `dt`, worst keypoint per frame. Separates a
  single discontinuity from sustained jitter (which jerk carries).
- `joint_limit` — bend angles vs a `[lo, hi]` envelope fitted on the 1%/99%
  quantiles of real motion; scores degrees outside it. The envelope is what
  the corpus shows, not the URDF's declared limits.
- `self_collision` — minimum distance across non-adjacent keypoint pairs,
  flagged **below** threshold. Only the robot's own keypoints; no environment.

## Per-clip report fields

`report(ctx)` per detector: `units`, `threshold`, `fraction` (of frames
flagged), `n_flagged`, `severity_ratio_median`/`_p90`
(`per_frame / threshold`, inverted for `self_collision` so `> 1` always means
worse), `intervals` (`[start, end]` frame pairs), `per_frame`.

## What none of them detect

Task success, visual fidelity, object dynamics, contact — anything outside
the robot's own keypoint set. A clip that passes all five is not physically
correct; it is a clip in which the arm's own kinematics show no violation
this reader can see.
