# Rate policy: comparing clips recorded at different frame rates

The benchmark matrix spans clips at genuinely different native frame rates --
ctrlworld 5 fps, dreamdojo 10 fps, dreamgen 16 fps (probed; an earlier config
table said 10, which was wrong), real GR-1 teleop 20 fps, real ALOHA 30 fps:
**five** rates, not four. Worse, as the measured example below shows, even
"5 fps" is not a safe per-family constant -- two episodes inside one
ctrlworld cell probe at 30 fps, six times faster than the other 394 clips in
that same cell, which is a within-family anomaly, not merely a
between-generator one. Most physics metrics in this package are derivatives
(speed, acceleration, jerk, energy, momentum, ...), and a derivative computed
with a finite-difference step `dt` scales with `dt` -- see `core/metric.py`'s
`dt_exponent` and `docs/METRICS.md`. Comparing a 16 fps clip's raw jerk
against a 10 fps clip's raw jerk is not comparing two measurements of the
same quantity; it is comparing two different quantities that happen to share
a name -- and, per the example below, that is not even reliably true
*within* one nominal-rate family, let alone across generators.

The published project page states the policy in prose:

> "Comparing across different generators fairly requires **matching their
> frame rates first**; that matching is **ongoing**."
>
> "the raw jerk number depends on frame rate, so on clips recorded at
> different speeds it can look misleading -- the **scale-free** measures
> don't have that problem"
>
> Experiment 5's headline: "On **frame-rate-matched** footage -- the fair
> basis, since raw jerk depends on frame rate -- generated motion runs about
> 27% jerkier"

This document is what turns that prose into enforced code: four layers, in
the order you should reach for them, why `paired` is the default, which
metrics must never be compared across rates, and the honest caveats each
layer carries.

## Measured evidence, not a hypothetical

Two things were measured on this corpus rather than assumed. Both are
stronger arguments than the prose above, and both are the reason the
protections below are hard failures rather than warnings.

**(a) Ignoring the rate inverts the verdict.** Scoring real GR-1 teleop at
its native 20 fps against dreamdojo output at 10 fps gave real jerk ~117
m/s^3 against generated ~16-23 m/s^3 -- i.e. "real is 5x jerkier than
generated", and an AUROC of **0.00**. Not 0.5 (cannot separate): 0.00, the
ruler ranked every real clip as worse than every generated one. That is pure
arithmetic -- jerk carries `dt_exponent = 3`, so a 2x ratio in `dt` is 8x,
and 117 / 8 ~ 14.6. Re-encoding the real anchor to 10 fps gave real 15.7 vs
generated 21.3, i.e. generated jerkier by +5.5 [4.7, 7.0], p ~ 2e-20, n=116.
Same footage, opposite conclusion.

**(b) A per-family fps table cannot be correct, because a family is not
internally consistent.** Probing every clip under
`dense/bimanual/input/multiview/ctrlworld/makovian` -- one generator, one
cache, one view, one horizon -- found:

| probed rate | clips |
|---|---|
| 5 fps | 294 per-view (320x192) + 100 stacked (960x192) |
| **30 fps** | `episode_find_insert_small_gear_shaft__000018/view_{0,1,2}.mp4`, `episode_insert_lan_cable_into_the_hub__000014/view_{0,1,2}.mp4` |

There is no single correct table *value* to write for that family. This is
strictly worse than defect D3(b) (where a table overwrote a correct probe):
here the table concept itself is unsound. It is also 27x worse in magnitude
than defect D1, the project's usual cautionary tale -- D1 was a 2x `dt`
error giving 8x on jerk; 30-vs-5 fps is 6x giving **216x**. Two such clips
in a cell of ~100 would dominate any mean computed over it.

The consequence for `resolve_timebase`: the probe is the only source of
truth, and a table/CLI value is a **cross-check that can only ever fail,
never win**. A mismatch raises rather than being silently reconciled, so
those two episodes are dropped loudly instead of being rescaled by 216x in
silence.

**(c) The corpus has five native rates, and the cross-generator worst case
is over an order of magnitude.** The 30 fps anomaly above is why: ctrlworld
itself runs at 5 fps, not one of the four rates this document originally
assumed (dreamdojo 10, dreamgen 16, GR-1 teleop 20, ALOHA 30). Comparing
ctrlworld directly against dreamgen -- 5 fps vs 16 fps, a 3.2x ratio in
`dt` -- inflates raw jerk by **3.2^3 ~= 33x**. Cross-generator comparison on
any derivative metric is therefore not merely unfair in the abstract; on
this corpus it is off by more than an order of magnitude. That number is
the concrete reason layer 3 (`rate_free`, below) is a hard requirement for
any cross-generator claim on this data, not a nice-to-have.

**(d) A structural detail worth recording alongside this, measured rather
than inferred from a directory-naming convention.** A ctrlworld episode
directory contains `view_0.mp4`, `view_1.mp4`, `view_2.mp4` (320x192 each)
*and* `full_gt.mp4` (960x192, the three views stacked side by side) -- so a
3-panel **width** stack is confirmed for ctrlworld, not assumed (see
docs/DECISIONS.md D-G: it is NOT a height stack, `ViewLayout(n_views=3)`'s
old default -- the source plugin now declares `packing="width"` and exposes
only the 2 exterior panels as views, dropping the wrist one). One episode
was also observed with a frame-count mismatch between the stacked file and
its three per-view files (165 vs 167 frames). That is **flagged as
something to check, not asserted as general** -- it was seen in a single
episode so far; if it does generalise, the stacked and per-view forms are
not frame-aligned, which would need its own consistency check (in the
spirit of `verify_manifest`'s `dt`/`wh`/`codec` checks, though building that
check is out of scope for this change) before either form could be trusted
as ground truth for the other.

## The four layers

### 1. `paired` (default) -- score each clip against its own ground truth

Score every generated clip against **its own** ground truth: same scene,
same task, same episode, same native frame rate. `dt` cancels *within the
pair* because both clips share it -- this needs no suite restriction and no
resampling, and it is where the page's headline paired-delta figures (e.g.
"+5.5 [4.7, 7.0], p~2e-20, n=116") come from.

Because the whole argument depends on the two clips genuinely sharing a
timebase, `bench/manifest.py::verify_manifest` now checks `dt` for every
gt/pred pair, in the same shape as its existing width/height/codec checks
(`wh_ok`, `codec_ok` -> now also `dt_ok`, with `gt_dt`/`pred_dt` recorded in
the mismatch record). A pair whose two members were probed at different
frame rates is not a valid instance of "fps cancels in the pair" and is now a
**hard** mismatch (`ok=False`), not a warning -- exactly as a codec or
resolution mismatch already was, and by the same reasoning: nothing
downstream can tell the difference between "matched" and "silently
mismatched" once the numbers are computed, so it has to be caught before
scoring, not diagnosed after.

`paired` is the default in `core/scorer.py::Scorer` (`rate_policy="paired"`)
precisely because it requires no assumption beyond what pairing already
guarantees, and it is the only layer that supports the *full* metric suite
(`INVARIANT_V1`), not just the rate-free subset.

### 2. Anchor re-encode -- how the *published* "frame-rate-matched" numbers were actually produced

**This layer is not implemented by this change.** It is documented here so
the four layers form one coherent story, and because it is easy to confuse
with layer 4 (trajectory resampling) -- they solve the same problem
("compare a 10 fps clip against a 20 fps clip") in different, non-equivalent
ways.

What "frame-rate-matched footage" means on the published page is: the **real
reference footage** is re-encoded with `ffmpeg` to the generator's exact fps,
resolution, and (probed) compression quality -- e.g. real teleop re-encoded
to 640x480 @ 10 fps to match dreamdojo, or to 768x432 @ 16 fps to match
dreamgen. The generated clips themselves are never touched. The motivating
failure mode, measured directly: comparing raw 20 fps real jerk (~117)
against 10 fps generated jerk (~16-23) collapsed the real-vs-generated AUROC
to **0.00** -- a pure fps artifact, not a finding about motion quality.
Re-encoding the real anchor to the generator's own rate fixed it, and this is
where the page's 27% jerk / 37% rigidity cross-generator gaps come from.

Two things worth carrying into any future cross-generator report:

* Re-encoding **also** changes compression artifacts, not only frame rate --
  a caveat the original implementation handles by measuring a **re-encode
  null-delta** (a noise floor: re-encode real footage against itself and see
  how much the numbers move from compression alone), so a genuine gap can be
  told apart from an artifact of the re-encode step.
* Only the 10 fps anchor was ever scored end-to-end; the 16 fps anchor
  (`real_dm16`, built for dreamgen) was **prepared but never scored** -- the
  page's own open item. No cross-generator, rate-matched comparison against
  dreamgen has actually been produced yet.

This layer is being ported separately as `cli/cmd_anchor.py` (the re-encode
step) and `bench/noise_floor.py` (the null-delta measurement) by another
change; this document defers to those for the implementation and only
records the policy context here.

### 3. `rate_free` -- restrict to metrics that don't depend on `dt` at all

When no anchor re-encode is available, or a claim needs to hold for *any*
frame rate rather than one specific matched pair, restrict scoring to
`kinescore.metrics.suites.RATE_FREE`: the subset of the registry whose
`dt_exponent` is exactly `0`. These metrics are purely geometric or
static-threshold checks -- no time derivative anywhere in their formula --
so their value does not move when `dt` does, and comparing them across two
clips scored at different rates is legitimate.

`RATE_FREE`'s membership is **derived from the live metric registry**, not a
hand-copied list: `metrics/suites.py` filters `INVARIANT_V1`'s own declared
term set (`_ALL_METRIC_KEYS`) down to the keys whose `get_metric(key).spec.
dt_exponent == 0`. `tests/test_rate_policy.py` asserts this membership
against the registry directly, so a future metric added with the wrong
exponent (or a `RATE_FREE`-adjacent metric added by hand without checking)
cannot silently join a "rate-free" suite that isn't actually rate-free. As of
this writing that set is:

| key | why it's rate-free |
|---|---|
| `rigidity_residual_mm` | static bone-length deviation, no derivative |
| `rigidity_wobble_mm` | temporal std of a length, not of a derivative -- `dt_exponent=0` |
| `limit_violation_frac` | per-frame comparison against static URDF limits |
| `limit_excess_rad` | same static comparison, magnitude instead of a fraction |
| `limit_headroom_rad` | same static comparison, margin instead of a violation |
| `penetration_mm` | per-frame collision geometry, no derivative |
| `self_collision_frac` | thresholds a *spatial* penetration depth, not a `dt`-dependent quantity |
| `com_margin_m` | static support-polygon geometry |
| `log_dimensionless_jerk` | **proved** `dt_exponent=0` algebraically (see below) -- the one dimensionless smoothness metric that actually cancels `dt` |

`core/scorer.py::Scorer(..., rate_policy="rate_free")` enforces this
structurally at construction time: it checks every metric *in whatever suite
was actually passed in* for `dt_exponent == 0` and raises `ValueError` naming
the offending keys if any are rate-dependent, rather than silently swapping
in `RATE_FREE` behind the caller's back. This is the "suite đầy đủ mà so
chéo nhịp -> lỗi, trừ khi `--allow-rate-mismatch`" rule from the design doc:
scoring the *full* suite across a genuine cross-rate comparison is refused,
not merely discouraged.

#### The one distinction in this whole document that is easy to get backwards: `sparc`

The published page calls `sparc` (Spectral Arc Length) "scale-free" and
reports it as the strongest single separator in its own numbers (AUROC
0.879). That is true, and `RATE_FREE` does not contradict it -- but "scale
free" on the page means invariant to a movement's **amplitude and duration**,
which is exactly what lets `sparc` compare a fast wipe against a slow
insertion fairly. It says nothing about frame rate.

`sparc` (`metrics/smoothness.py`) builds its frequency axis from `fs = 1/dt`
and keeps only spectral bins below a **fixed absolute** cutoff `fc = 10 Hz`.
Changing `dt` changes which physical frequencies that fixed bin grid
represents, so it changes *which portion of the same normalised spectrum*
the arc-length sum runs over -- not a clean multiplicative rescaling of the
result, and not something a single `dt_exponent` can capture. Its own spec
declares `dt_exponent=None` for exactly this reason. Comparing `sparc` scored
at 10 fps against `sparc` scored at 16 fps is exactly the confident-but-wrong
comparison this document exists to prevent, even though comparing it between
two *same-rate* clips of very different speed or duration (what the page
actually did) is legitimate and is why it separates so well.

**`sparc` is scale-free (amplitude/duration-invariant) but not
frame-rate-invariant. `log_dimensionless_jerk` is the frame-rate-invariant
one** (`dt_exponent=0`, proved -- not merely measured -- by algebraic
cancellation in `metrics/smoothness.py:20-32`: `duration**5` contributes
`dt**-5`, the squared-jerk integral contributes `dt**+5`, and `path_len`'s
own `dt` cancels separately against the `/dt` inside the speed used to
compute it). Getting this backwards -- treating "the page calls it
scale-free" as "therefore comparable across the whole benchmark matrix" --
produces a confident, invalid comparison, which is precisely the failure
mode this whole document is trying to close.

### 4. `resample:<hz>` -- explicit, opt-in trajectory resampling

The last resort, for when the full suite is wanted across clips that
genuinely differ in native rate and an anchor re-encode (layer 2) isn't
available or doesn't apply (e.g. resampling a *generated* clip, not a real
one). `core/resample.py` interpolates the joint-angle trajectory `q(t)` (and
`q_raw`/`sigma`/any per-frame `aux` a robot's FK needs) from its native grid
onto a common target rate, using monotone PCHIP interpolation, strictly
*after* the pose reader has run and *before* forward kinematics -- see that
module's docstring for the full argument, including why plain video-frame
resampling was rejected (it silently changes which pixels are decoded
without updating any metadata -- defect D1) and why PCHIP specifically
(shape-preserving, never overshoots between samples, unlike a plain cubic
spline).

Wired into `core/scorer.py::Scorer(..., rate_policy="resample:<hz>")`:
`score_readout` resamples the readout and clip to `<hz>` before FK runs, and
the returned `ScoredClip.clip` carries the resampled `ClipSpec` --
`dt_source="resampled"`, never a native-rate value -- so nothing downstream
can mistake a resampled clip's provenance for a natively-sampled one.

Three things this layer insists on, all covered by `tests/test_rate_policy.py`:

* **Downsampling only, by default.** Interpolating onto a *finer* grid
  invents frames the reader never produced; since PCHIP's whole point is to
  avoid inventing extrema, an upsampled trajectory reads as unrealistically
  smooth -- upsampling systematically flatters apparent smoothness.
  Downsampling can only discard information, never invent it, which is the
  safe direction. Upsampling raises `UpsampleRefusedError` unless
  `allow_upsample=True` is passed explicitly.
* **Integer decimation delegates to `ClipSpec.subsample(k)`.** When the
  native/target ratio is a whole number, every kept frame is a real sample --
  no interpolation is needed or wanted, and the existing, tested
  `subsample` is what does it; this module never reimplements that
  arithmetic.
* **The noise-spectrum caveat is not hidden.** A jerk (or any
  derivative-based metric) computed on a resampled clip is **not**
  comparable to one computed on a natively sampled clip at the same rate --
  interpolation reshapes whatever error spectrum the pose reader produced.
  This is exactly the kind of caveat this repository documents rather than
  papering over: `resample_readout` emits a `UserWarning` restating it every
  time an actual (non-no-op) interpolation runs.

One structural gap, stated plainly rather than worked around silently:
`ClipSpec.dt_source` is typed as a closed `Literal` (`core/clip.py`) that
does not include a "this came from resampling" value, and `core/clip.py` is
out of scope for this change. `core/resample.py` tags resampled clips with
`dt_source="resampled"`, a string outside that `Literal`. Nothing in this
codebase validates `dt_source` against the `Literal` at runtime (it is a
plain `str` field, and this repository's CI runs `ruff`, not a type
checker), so this works correctly today and does not claim a native rate --
which is the property that was asked for -- but it is a gap a future change
to `core/clip.py` should close formally by adding `"resampled"` to
`DtSource`.

## Metrics that must never be compared across frame rates, and why

Beyond the obviously `dt`-scaling metrics (speed `dt_exponent=1`, most
accelerations/energies `2`, jerk `3`, ...), five metrics carry
`dt_exponent=None` specifically because they are **not** a clean power law in
`dt` -- comparing them across rates is not just imprecise, it can flip sign
or jump discontinuously:

| key | why `dt_exponent=None` |
|---|---|
| `accel_violation_frac` | thresholds a `dt`-dependent acceleration against a **fixed** physical constant (`accel_bound`); a measured 2x `dt` error drives it from `0.000` to `0.3875` on a representative trajectory -- a near-discontinuous jump, not a gentle nudge (`tests/test_dt_invariance.py::test_wrong_dt_is_detected`) |
| `vel_violation_frac` | same reasoning, thresholds `dq/dt` against a fixed URDF velocity limit |
| `no_teleport_frac` | same reasoning, thresholds a Cartesian speed against a fixed constant |
| `total_energy_tstd` | sums a `dt^-2`-scaling kinetic term with a `dt^0` potential term -- no single exponent describes the sum; measured 1.397x under a 2x `dt` error, neither the 4x nor the 1x a homogeneous metric would show |
| `sparc` | fixed absolute Hz cutoff in its frequency axis -- see the dedicated section above |

`docs/METRICS.md` documents each of these in full; this table exists so the
rate-policy argument doesn't require cross-referencing five separate metric
write-ups to see the shape of the problem.

## Summary: which layer for which claim

| Claim being made | Layer to use |
|---|---|
| "This generated episode drifted from its own ground truth" | 1. `paired` (default) |
| "Generated motion is jerkier than real motion, at this generator's own native rate" | 2. Anchor re-encode (published page's method; ported separately) |
| "This structural/geometric property holds regardless of frame rate" | 3. `rate_free` |
| "I need the full suite across two clips that genuinely differ in native rate and no anchor is available" | 4. `resample:<hz>`, opt-in, with the noise-spectrum caveat attached to every result |
