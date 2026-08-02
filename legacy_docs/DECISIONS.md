# Benchmark decisions, and the evidence behind each

Every choice below was contested by something in the data or the source
repositories, and each is recorded here with the measurement that settled it.
A decision without its evidence is an opinion; a decision with its evidence can
be re-argued when the data changes.

This is the "why" companion to [PROVENANCE.md](PROVENANCE.md) (what was
ported and what was fixed); the frame-rate rules live in
[../docs/BENCHMARKING.md](../docs/BENCHMARKING.md).

---

## D-A. Rigidity is scored with `rigidity_worst_bone_mm`, not `rigidity_residual_mm`

**The problem.** `rigidity_residual_mm` failed to separate generated from real
at all: median paired delta `+0.013 mm`, generated worse in **56.6%** of
episodes — a coin flip — on n=659 paired GR-1 dreamdojo episodes. Meanwhile two
independent smoothness rulers on the *same* episodes agreed strongly:

| ruler | median paired delta | generated worse in |
|---|---|---|
| `mean_jerk_mps3` | +4.201 | **89.8%** |
| `log_dimensionless_jerk` | −0.317 (higher_better) | **87.9%** |
| `rigidity_residual_mm` | +0.013 | **56.6%** |

**Three hypotheses were wrong.** It was not the frame rate (both members of
each pair are 10 fps), not the clip set, and not the reader. All three were
guesses, and all three were checked and discarded.

**The actual cause.** The published rigidity figure was never produced by the
function we ported. `rigidity_residual_mm` is a faithful port of the source
function *of that name*, but the page's number came from
`models/physics/feasibility.py::rigidity_field`, which differs twice:

| | published `rigidity_field` | ported `rigidity_residual_mm` |
|---|---|---|
| reference length | each bone's **own temporal median** | URDF **rest** length |
| reduction over bones | **`amax`** (worst link) | **`mean`** |

Both differences cost sensitivity, in the same direction:

* **Reference.** A reader with a constant per-bone localisation bias — it draws
  one link 2 mm too long in *every* frame — reads 2 mm of "rigidity error"
  against the URDF rest length while being perfectly rigid. Measured against
  that bone's own median, the bias cancels exactly and only real length
  *change* survives. Absolute reader accuracy has a floor of tens of mm (see
  README's "Scope, honestly"); a metric that does not cancel it is measuring
  the reader.
* **Reduction.** The failure mode this ruler exists to catch is **one** limb
  rubber-banding — the page's own gallery shows 20–34 mm on the worst clips.
  Averaging that against every well-behaved bone in the chain dilutes precisely
  the signal. A mean is not a conservative max; it is a blunter instrument.

**Decision.** `rigidity_worst_bone_mm` implements the published semantics and is
the rigidity number a report should quote. All three variants stay registered
and in the `full` suite: the two ports reproduce prior numbers, and showing
mean beside max is itself evidence about which reduction the signal lives in.

---

## D-B. No composite score, and no squashed readout

Two separate things that both reduce to "do not destroy information before the
reader sees it".

**No composite.** `reference/normalize.py::invariance_score` (PIS) maps each
residual through a hard double clip to `[0,1]` around a real-data median, with
`[base, 3·base] → [0,1]`, then takes an unweighted 10-term mean. Its saturation
is visible in the repo's own golden fixture: `case4_high_freq_jitter__pis = 0.6`
**exactly** — six terms pinned at 1.0, four floored at 0.0. It is legacy from
the source repositories, it is **not wired into any CLI command**, and it is not
part of the published method, which says so directly:

> "We keep each in its natural form rather than squashing everything into one
> 0–1 number — the physical units, read straight off the robot's own model, are
> the whole point."

Reports carry **magnitudes in physical units** (mm, m/s³, % of rated N·m) plus
per-ruler **separation**. These answer different questions and are not
interchangeable: a magnitude says *how large*, a separation says *how reliably*.

**No squash.** A squashed head computes `q = lo + (hi−lo)·sigmoid(raw)`, so the
prediction **cannot** leave `[lo, hi]` and `limit_violation_frac` is
structurally `0` regardless of the video — defect **D7**. kinescore reports it
as `null` + `unobservable:limit_semantics=squashed`, never `0`.

The GR-1 recipe does it correctly and is now the template for every embodiment:

```python
limits = stack([q_lo, q_hi])                       # straight from the URDF
lim    = loss_limit(mu, limits)                    # soft hinge: relu(q-hi) + relu(lo-q)
loss   = beta_nll_loss(mu, logvar, target, beta=0.5) + 0.05 * lim
```

The weight is the whole design. At `0.05` the penalty pulls the head toward the
feasible set when the image is ambiguous but is **far too weak to gag it** when
the image genuinely shows an impossible pose. A squash gags it unconditionally.
That single difference is what separates "violation measured and found zero"
from "violation unmeasurable". `loss_limit` (`training/losses.py`) never
squashes anything, only *penalizes* an out-of-range raw prediction; a hinge
strong enough to make that penalty hard would just re-derive D7 through a
back door, only more slowly than a sigmoid.

`beta_nll_loss` (Seitzer 2022) is the mean term: per-element Gaussian NLL
re-weighted by `detach(var)**beta`, interpolating between plain NLL
(`beta=0`) and MSE-like weighting (`beta=1`). The re-weighting exists to stop
the variance head from starving the mean head of gradient on high-variance
targets -- a known failure mode of un-weighted Gaussian NLL, where the model
learns to inflate `sigma` on hard examples instead of fitting `mu` better.
`beta=0.5` (the source's default) is the midpoint between those two failure
modes.

Available as `kinescore train-rawrad`.

**Update (2026-07-29, see PROVENANCE.md's D7 addendum).** The paragraph above
originally said the squashed path stayed in the codebase as the control that
proved removing the squash changed anything, and that `golden_predict_pose.npz`
recorded the sigmoid at ±20 saturation as documentation of D7. That decision
was reversed: the squashed path (`heads/ranges.py::squash_to_limits`,
`readers/squashed.py::SquashedPoseReader`, the squashed branch of
`readers/checkpoint.py::load_reader`, `training/trainer.py`, the `kinescore
train` subcommand, `golden_predict_pose.npz` and the test/generator that
produced it) has been **removed entirely**, once every real embodiment had
(or was actively getting) a `raw_rad` reader and the squashed checkpoints'
only remaining metric-relevant property was a structurally-fake `0`/`NaN`
anyway. It was on no production path, and once that stopped being useful as
a control, keeping it was just dead weight. See PROVENANCE.md's D7 addendum
for the full accounting of what moved.

---

## D-C. Scoring is absolute, not paired

The benchmark scores each clip on its own, in physical units, one CSV row per
clip.

Pairing was the original design and is still the statistically stronger
comparison where it is available — `dt` cancels inside a pair, which is why
`verify_manifest` now treats a `dt` mismatch within a pair as a hard error. It
was abandoned for the headline output because **dreamgen has no ground-truth
video anywhere**: `input/.../dreamgen/` holds first-frame conditioning PNGs and
batch-manifest JSON at every depth and every `view_dir` spelling, never a video.
The `gt_from: input` assumption in the original config was checked and found
wrong.

Absolute scoring also serves the actual use: a human sorting clips by a ruler
and going to watch the worst ones. That needs a per-clip number in a unit, not
a delta against a partner that may not exist.

---

## D-D. An embodiment is scored only when a robot is visible **and** a matching
reader exists

Reading joint angles from a frame that does not contain the arm returns
plausible numbers that sort, plot, and mean nothing — the most dangerous failure
mode available, because it does not look like one. Two cells are affected:

* **`dense/humanoid/.../multiview/ctrlworld/`** is labelled `humanoid` but is
  **Airbot MMK2**, not Fourier GR-1 — every episode directory is named
  `AIRBOT_MMK2_*` and the paired real data's `meta/info.json` declares
  `robot_type: "Airbot_MMK2"`. Its three rendered cameras are `cam_high`,
  `cam_left_wrist`, `cam_right_wrist`; extracted frames show `cam_high` on a
  bare table for one task subset and a partial forearm for the other, and both
  wrist views filled by a gripper.
* **`dense/single_arm/.../singleview/`** is genuinely DROID/Franka (its
  `metrics.json` names `exterior_1_left`, `exterior_2_left`, `wrist_left`), but
  at least some clips are the **wrist** camera: across a filmstrip the whole
  scene translates while the gripper stays fixed in frame.

Policy: probe what the camera shows, per cell, and report the proportion. A
cell that cannot be read is reported as such **with the evidence**, never with a
number. A well-evidenced "this cannot be scored, here is why" is a complete
result.

---

## D-E. One suite per output tree

`MetricSuite.suite_id` is a hash of the declared term set, and
`bench/stats.py::aggregate` refuses to pool rows from different suite ids. Two
folders scored under different suites have different column sets and are not
comparable, so every CSV and `SUMMARY.csv` carries `suite_name` and `suite_id`.

Three suites exist, for three different jobs:

| suite | n | for |
|---|---|---|
| `invariant_v1` | 26 | frozen — every golden fixture and prior published number was computed under this id; score with it only to line up with those |
| `all_metrics` (`metrics/suites.py::ALL_METRICS`, named `full` before a later rename — some earlier runs/docs still say `full`) | 28 | `invariant_v1` + `torque_frac_rated` + `rigidity_worst_bone_mm`. **The suite to score a fresh run with.** |
| `rate_free` (`metrics/suites.py::RATE_FREE`) | 9 | only `dt_exponent == 0` metrics — the sole valid basis for comparing clips at different frame rates |

`invariant_v1` was **not** extended when metrics were added, deliberately:
mutating it would have silently invalidated every golden fixture. Its id is
`sha256:cb01e10a9318c420` and is asserted unchanged.

`invariant_keys` (`INVARIANT_V1`'s ten-key PIS term set) mirrors the source's
`PhysicsConsistency.INVARIANT_KEYS` exactly, deliberately, for two reasons:
it was already the right list (every key is a task-*invariant* residual,
small/bounded regardless of what the robot is doing — unlike a task-dependent
magnitude like `mean_speed_mps`, where a fast wipe and a slow insertion are
both plausible at very different speeds), and keeping it identical makes a
PIS score here the direct, nameable successor of the source's own aggregate.
Every registered metric not in `invariant_keys` is still in `output_keys`
(every result row, for reporting/distributional comparison/debugging) — see
`docs/METRICS.md`'s legend.

No robot-specific suite exists yet: every metric already resolves to `NaN` +
reason when a robot lacks the capability it needs (per-metric, see
`docs/METRICS.md`), so a Franka-only or GR-1-only suite would today just be
the same term list under a different `suite_id`. Add one in
`metrics/suites.py` (not in `kinescore.robots`) once a robot needs a metric
promoted into its own `invariant_keys` — suite composition is this module's
job, robot capability is the robot spec's.

`RATE_FREE` filters `INVARIANT_V1`'s own `_ALL_METRIC_KEYS`, not the raw
metric registry — deliberately: `rigidity_residual_all_mm`/
`rigidity_wobble_all_mm` also have `dt_exponent=0` but are gripper-
contaminated legacy metrics excluded from `_ALL_METRIC_KEYS` for that reason
(see `docs/METRICS.md`); filtering through it keeps that exclusion instead of
re-admitting a known-bad metric into a new suite. See `docs/RATE_POLICY.md`
layer 3 for the rest of `RATE_FREE`'s rationale (why `sparc` is excluded
despite being "scale-free", why `log_dimensionless_jerk` is included).

`torque_frac_rated` matters beyond being one more column — it is the only ruler
in the package with a **real physical ceiling**. Every other magnitude is
unbounded and interpretable only by comparison, whereas "over 100% of rated
N·m" means genuinely impossible.

---

## D-F. Known cost: the reader smooths what the metric measures

`ReadoutV2Head` ends in a **bidirectional** temporal transformer (2 layers,
`t_max=64`, no causal mask — judging is offline, so restricting to the past
would discard half the available evidence for nothing). It earns its place: half
of training is deliberately degraded video (`p_degraded=0.5` against a separate
`deg640` cache), where a blurred frame is unreadable on its own but its
neighbours are not, and per-frame `sigma` is only meaningful *relative* to
neighbouring frames.

It is **not** integration — attention over a bounded window, never a state
accumulated from frame 0 — so nothing drifts, and the method's central claim
survives.

The cost is real and should be stated when quoting absolute jerk: a temporal
transformer is a low-pass filter over exactly the high-frequency content that
`mean_jerk_mps3` (a third derivative) measures. Measured jerk is therefore a
**lower bound**. The comparison stays fair because real and generated pass
through the same reader, but the absolute value is attenuated by an unmeasured
amount.

It is cheap to measure rather than caveat: `TemporalEncoder.forward` takes
`use_context`, and `use_context=False` returns `z` unchanged. Scoring the same
clips both ways makes the difference in `mean_jerk_mps3` the attenuation itself,
as a number. Not yet run.

---

## D-G. `ViewLayout` carries a packing mode, not just a height-stack assumption

**The problem.** `ViewLayout.view_height(H)` assumed every multiview clip
stacks its cameras on the image **height** axis: `H // n_views`. The corpus
has four packings, all measured:

| packing | example | dims |
|---|---|---|
| height stack | the original DROID caches | `H = n*view_h` |
| width stack | `ctrlworld` multiview, 771 files | `960x192` = 3 x `320x192` |
| 2x2 grid | `dreamgen` multiview, 257 files | `768x432` |
| separate files | `ctrlworld` also ships `view_0/1/2.mp4` | `320x192` each |

The height assumption is not merely incomplete on the width-stacked case, it
is **silently wrong**: `192 % 3 == 0`, so `view_height(192)` returned 64 and
sliced three meaningless horizontal bands with no error.
`backbones/dino.py`'s `encode` consumed that directly.

**The fix.** `ViewLayout` gained `packing` (`"height"` | `"width"` |
`"grid2x2"`), plus `n_panels`/`panels` to select a subset of the packed
frame's physical panels as this layout's exposed views. A single method,
`view_crops(frame_width, frame_height)`, computes every view's pixel box
`(top, bottom, left, right)` for any packing and subset -- callers (the DINO
backbone, `core/scorer.py`'s pre-flight check) never re-derive crop geometry
themselves.

**Refuse rather than guess.** Plain divisibility does not catch the
960x192 case (`192 % 3 == 0`), so `view_crops` additionally rejects a
resulting panel whose aspect ratio (`width/height`) falls outside
`[0.2, 5.0]` when there is more than one panel: a WIDTH-stacked 960x192
frame mis-sliced as 3 HEIGHT bands implies 960x64 panels, aspect 15.0,
against the real 320x192 panels' aspect 1.67. This is a plausibility
heuristic, not a certainty -- the primary fix is that `ctrlworld`'s and
`dreamgen`'s source plugins now declare their real packing explicitly
instead of relying on the (wrong) height default; the aspect guard is a
second line of defense for any future plugin that mis-declares packing on a
genuinely inconsistent frame.

**Key stability, verified.** `ViewLayout.key` is stored in checkpoints and
manifest rows and compared for equality. Every real row currently on disk
under `kinescore_runtime/out/**/bench_manifest.parquet` uses one of exactly
two keys -- `1x?:unnamed` (every singleview clip) and
`3x?:exterior_1+exterior_2+wrist` (ctrlworld's multiview clips, pre-fix,
mis-sliced as a height stack). Both are produced by the plain
`packing="height"`, no-subset case, whose `key` format is byte-for-byte
unchanged (`tests/test_clip_timebase.py::TestViewLayoutKeyStability`). Any
other packing or an explicit panel subset appends to the key instead of
colliding with it, since that describes genuinely different geometry.

**Wired.** `ctrlworld`'s plugin now declares `packing="width"`, `n_panels=3`,
`panels=(0, 1)` -- the two exterior panels (columns `0:320`/`320:640`), the
wrist panel dropped entirely. `dreamgen`'s plugin declares
`packing="grid2x2"`, `n_panels=4` for its multiview cell (currently N/A in
`configs/benchmark.yaml` pending its separate `gt_from` gap -- see
`kinescore.bench.sources.dreamgen`'s docstring -- so this is not yet
exercised by a real run, but is correct when that gap closes). `dreamdojo`
keeps the dataset-wide height-stack default, matching the original DROID
caches. Neither plugin's real per-quadrant camera identity for dreamgen's
grid is confirmed (one visual sample showed 3 populated quadrants and 1
solid black one) -- no panel subset is claimed there without more evidence.

**Not attempted.** The "separate files" packing (`ctrlworld`'s
`view_0/1/2.mp4`) is a different code shape entirely -- one `ClipSpec` per
file, `n_views=1` each, not a pixel-space crop -- and has no source plugin
reading it yet (see `kinescore.bench.sources.ctrlworld`'s docstring for the
verified fallback formula, not wired in because the packed
`gt_all_views.mp4`/`pred_all_views.mp4` pair has 100% coverage today).

---

## D-H. Airbot MMK2: joint order verified against real data; hand excluded (unknown units)

**Arm joints, verified.** `video_gen_physics_real_video/humanoid/multiview/
{makovian,non_makovian}/{AIRBOT_MMK2,Airbot_MMK2}_*/meta/info.json` declares
`observation.state`/`action` as a 36-D vector with explicit per-dimension
names: 6 left-arm + 6 right-arm + 12 left-hand + 12 right-hand, each `_rad`-
suffixed. The 12 arm dims (index 0-11) were spot-checked numerically against
ten episodes of `AIRBOT_MMK2_mobile_car`: every column's observed min/max
falls inside the corresponding joint's URDF `<limit>` (e.g. dim 0
`left_arm_joint_1_rad` observed in `[0.885, 1.428]` rad, inside
`airbot_play_v3_gripper_fixed.urdf`'s `joint1` limit `[-3.14, 2.09]`) --
consistent with genuine radians in the arm's own frame. Only these 12 are
modelled (`robots/airbot_mmk2/constants.py::LEFT_ARM_JOINTS`/`RIGHT_ARM_JOINTS`).

**Hand joints, excluded.** The 24 hand dims (index 12-35), despite the
`_rad` suffix, spot-checked up to ~86 (e.g. `left_hand_joint_2_rad` in
`[61.1, 85.8]`, `right_hand_joint_12_rad` up to `64.3`) -- far outside any
physically possible revolute range in radians (`2*pi ≈ 6.28`). Checked and
ruled out: the upstream RoboCOIN pipeline (`github.com/FlagOpen/RoboCOIN`)
has a per-robot `joint_units` config mechanism, but no Airbot/MMK2 entry
exists on any of its branches, and neither the dataset's HF README nor
`info.yaml` document a numeric unit for these columns (only
`end_effector_type: five_finger_hand`/`five_finger_gripper`). Fabricating a
hand kinematic chain against ground truth of unknown units would produce
exactly the "plausible-looking numbers that mean nothing" failure mode this
benchmark exists to avoid, so the hand is left out of `AirbotMMK2Spec`
entirely -- the same way `GR1Spec` excludes legs it was never trained to
predict. Add it once a real unit/ordering is obtained (Discover Robotics'
own driver, or a RoboCOIN maintainer), without touching the arm chain.

**Degenerate bones.** `left_link1`/`left_link2` coincide at rest (joint2's
URDF origin is `xyz="0 0 0"`, a real zero-offset joint pair on the physical
AIRBOT Play arm, not a modelling artefact), and so do `left_link4`/`left_link5`
(same reason, joint5) -- both bones are dropped from `rigid_bone_pairs` by
the degenerate-length safety net (`core.robot.rigid_bone_mask`), not a
structural actuated-link rule, since both endpoints are still driven by
predicted joints (unlike Franka's gripper case -- see `robots/franka/
constants.py`).

---

## D-I. CSV export: join design and path-derived grouping

`bench/csv_export.py` turns a scored `results.jsonl` into a mirrored tree of
per-clip CSVs. No composite score, no pairing (D-B/D-C apply the same way
here: absolute physical-unit values, one named sort metric, never a
formula), and an unavailable metric is an empty cell + `<metric>_reason`,
never `0` (`docs/SCHEMA.md` invariant 1) -- the two decisions unique to this
module are the join and the grouping key.

**Why the join.** A successfully-scored record's `"clip"` block
(`ScoredClip.to_record()`) only ever carries a *path* -- which episode/role
produced it is manifest identity, not scoring's job to know or get wrong
(same reasoning as `bench/stats.py::load_scores`'s identical join). A
**failed** record is the exception: its `"clip"` block IS the raw manifest
row (`bench/store.py::failed_record`), so the two record kinds disagree in
shape before any joining happens. `build_clip_rows` reconciles both:
manifest row first, overlaid by the record's own `"clip"` block (whose
post-rescore `dt`/`n_frames`/etc. win when present, since those are what was
actually scored).

**Why grouping is derived from the path, not `family`.** `family`/`method`
are whatever string a `bench.sources` plugin stamped, out of this module's
control. The output tree instead pattern-matches each clip's path against
the known on-disk layout (`group_key_for_path`) -- immune to whatever a
plugin's `family` convention does or doesn't encode, and it's the same
"mirror what's on disk" contract the CSV tree is specified against. A path
that doesn't match the known shape is grouped under `_unmatched` rather than
dropped, so an untaught family still produces a CSV instead of silently
vanishing.

**Suite name vs suite id, and the fps caveat.** A suite *name* can be
re-pointed at a new term set over its lifetime (renaming changes nothing
about which clips were scored -- see D-E's `full`->`all_metrics` rename) --
only `suite_id` (a hash of the declared term set) says whether two folders'
numbers are actually comparable, so both are their own columns. `SUMMARY.csv`
additionally warns when it mixes rows recorded at different native fps: a
real run showed a 16fps generator's `mean_jerk_mps3` median ~4x a 10fps
generator's on physically similar motion -- `(16/10)**3 == 4.096`, entirely
`dt_exponent`, not a real difference (`docs/RATE_POLICY.md`).
