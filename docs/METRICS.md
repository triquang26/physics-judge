# Metrics

Five detectors, one error type each. All are analytic functions of the predicted
keypoints `P (T, K, 3)` in metres and the timebase `dt` — no learned parameters,
no per-generator tuning. The only learned input is `P` itself.

| detector | units | flags | reads |
|---|---|---|---|
| `rigidity` | mm | above | bone lengths vs URDF rest lengths |
| `jerk` | mm/frame³ | above | 3rd time difference of keypoints |
| `teleport` | mm/frame | above | 1st time difference of keypoints |
| `joint_limit` | deg | above | bend angles vs a GT-fitted envelope |
| `self_collision` | mm (min dist) | **below** | closest non-adjacent keypoint pair |

## The common contract

Each detector implements `per_frame(ctx) -> (T,)`, one score per frame. From
that, `report(ctx)` produces:

| field | meaning |
|---|---|
| `units` | this detector's unit string |
| `threshold` | calibrated boundary |
| `fraction` | fraction of frames flagged |
| `n_flagged` | number of frames flagged |
| `severity_ratio_median` / `_p90` | `per_frame / threshold`, reduced over the clip |
| `intervals` | `[start, end]` frame pairs where the flag ran |
| `per_frame` | the raw score series |

`fraction` saturates: once nearly every frame crosses the threshold it cannot
distinguish "barely over" from "catastrophically over". `severity_ratio` has no
ceiling, so two clips that both flag 100% of frames can still be ranked. It is
also threshold-relative rather than raw units, which makes it comparable across
detectors with different units — `max` over the five is a defensible single
per-clip "worst violation" scalar.

For `self_collision`, `higher_is_worse = False`: smaller distance is worse. The
severity ratio is inverted to `threshold / per_frame` so `> 1` always means
"worse than the calibration boundary" in both directions.

## Calibration

`ViolationScorer.calibrate(gt_contexts, pct=95.0)`:

1. `det.fit(gt_contexts)` — a no-op except for `joint_limit`.
2. pool `det.per_frame(c)` over every real clip.
3. `det.calibrate(scores, pct, floor)`.

`higher_is_worse` detectors take the `pct`-th percentile, floored at their entry
in `_CALIBRATION_FLOOR`. `lower_is_worse` detectors take the `(100 - pct)`-th
percentile and never apply a floor — a floor would push a "smaller is worse"
bound the wrong way.

Floors: `rigidity` 18 mm, `joint_limit` 3°. Real motion is exactly rigid up to
float error, so without a floor the rigidity threshold would collapse onto the
reader's own jitter and flag everything.

Scoring before calibration raises `RuntimeError` — there is no implicit default
threshold.

## rigidity

Rigid links must keep their URDF length. For each bone pair `(a, b)` in
`robot.rigid_bone_pairs`, compute `|P[:, a] - P[:, b]|` in mm and take the
largest absolute deviation from the rest length across bones.

`rigid_bone_pairs` already drops *degenerate* bones — near-zero rest length,
where a gripper opening would otherwise read as stretch. It does **not** drop a
bone that spans a moving joint: such a bone has a well-defined length at any
single pose, but not a constant one, so treating it as rigid manufactures
violations out of ordinary motion. `rigid_idx` narrows the set per robot. The
Franka needs `rigid_idx=(0, 2, 3)` — its bone index 1 spans a rotating joint.
This is a URDF-topology fact the detector cannot infer, so it is a constructor
argument, not a default.

**Does not detect:** a whole arm that is the right shape but in the wrong place,
or a warp that happens to preserve every bone length.

## jerk

Third difference `P[t] - 3P[t-1] + 3P[t-2] - P[t-3]`, norm per keypoint, worst
keypoint per frame, in mm/frame³. The first two and last frame are zero. Clips
shorter than 4 frames score all-zero.

**Does not detect:** smooth but impossible motion. A trajectory can be perfectly
smooth and still violate the robot's dynamics.

## teleport

First difference, norm per keypoint, worst keypoint per frame, in mm/frame.
Frame 0 is zero.

Overlaps with `jerk` by construction — a teleport also produces high jerk — but
separates a single discontinuity (one frame, high teleport) from sustained
jitter (many frames, high jerk). The intervals tell them apart.

**Does not detect:** a fast but continuous motion that is nonetheless faster than
the real robot can move; the threshold is calibrated on real motion, so
"fast for this corpus" is the only speed notion available.

## joint_limit

The only detector with per-joint state. `_bend` turns consecutive keypoint
triples into bend angles in degrees, `(T, K-2)`. `fit` learns a per-joint
`[lo, hi]` envelope from the `lo_q`/`hi_q` quantiles (default 1% / 99%) of real
bend angles; trimming the tails keeps one noisy real frame from blowing the
envelope open. `per_frame` scores degrees outside that envelope, worst joint per
frame. Only that excess is then percentile-calibrated.

Uncalibrated, it returns all-zero rather than guessing.

**Does not detect:** the URDF's declared joint limits. The envelope is what the
real corpus *shows*, which may be narrower (a limit the data never approaches) or
wider (reader error at the extremes) than the mechanical limit.

## self_collision

All keypoint pairs `(i, j)` with `j >= i + non_adjacent_gap` (default 2, which
excludes only bone-connected neighbours); minimum distance across pairs per
frame, in mm. Flagged **below** threshold.

A robot whose keypoint chain has short non-bone-connected branches close together
— two gripper fingertips one index apart — needs a larger gap, or those pairs,
close by construction rather than by fault, dominate the minimum.

**Does not detect:** collision with the environment, the table, or manipulated
objects. Only the robot's own keypoints are modelled.

## What none of them detect

- task success or intent
- visual fidelity, texture, lighting
- object dynamics, contact, or grasp physics
- anything outside the robot's own keypoint set

A clip that passes all five is *not* physically correct. It is a clip in which
the arm's own kinematics show no violation this reader can see.
