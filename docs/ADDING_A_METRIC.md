# Adding a metric

A metric is one implementation of the `Metric` protocol
(`src/kinescore/core/metric.py`): an object with a `spec: MetricSpec` and a
`compute(ctx: MetricContext) -> MetricValue` method. Read
`core/metric.py`'s module docstring first — it explains the two structural
problems (a varying term set, defect D3; a hidden timebase, defect D1) this
contract exists to make impossible, which is why every field of `MetricSpec`
below is mandatory rather than optional convenience.

## Subclass `BaseMetric`/`SafeMetric`, don't implement `Metric` directly

Every metric in `src/kinescore/metrics/` subclasses
`kinescore.metrics._base.SafeMetric` (which itself subclasses
`core.metric.BaseMetric`), not the bare `Metric` protocol. `BaseMetric`
handles `unavailable_reason` checking (missing inputs, `unobservable_when`
flags, `min_frames`) for you, so your subclass only implements `_compute`,
which can assume every input named in `spec.requires` is present and
`ctx.n_frames >= spec.min_frames`.

`SafeMetric` closes one more gap: `MetricValue` documents an invariant —
"`value` is `NaN` **exactly** when `reason` is set" — that `BaseMetric._ok`
does not itself enforce (it wraps whatever float you give it). Several
formulas can produce a `NaN` from *valid* inputs that make the arithmetic
degenerate (a stationary end-effector zeroing `dimensionless_jerk`'s
path-length denominator; an empty amplitude-threshold band in `sparc`).
`SafeMetric._ok` demotes a non-finite result to
`MetricValue.unavailable(key, "degenerate_input:non_finite_result")`
automatically. **Use `SafeMetric`, not `BaseMetric`, unless you have a
specific reason not to** — every metric in this package does.

## `MetricSpec`, field by field

```python
MetricSpec(
    key="my_metric_key",
    units="m/s^2",
    dt_exponent=2,
    direction="lower_better",
    requires=frozenset({"P"}),
    min_frames=3,
    unobservable_when=(),
    perframe=False,
    description="...",
)
```

- **`key`** — output column name, must be unique across the registry
  (`register()` raises `ValueError` on a collision) and alphanumeric/underscore
  (`MetricSpec.__post_init__` enforces this).
- **`units`** — physical units string (`"mm"`, `"m/s^3"`, `"rad"`, ...).
  `"fraction"` is special-cased by the reference normaliser (`*_frac` keys use
  an absolute tolerance, not a relative one — see
  `reference/normalize.py::_key_score`).
- **`dt_exponent`** — the power of `1/dt` your metric's value scales by. A
  speed is `1`, an acceleration `2`, a jerk `3`; a purely geometric quantity
  (no derivative at all) is `0`. **This is the mandatory declaration** the
  next section covers.
- **`direction`** — `"lower_better"` or `"higher_better"`.
- **`requires`** — a `frozenset` drawn from `core.metric.Requires`: `P`
  (keypoint positions), `R` (rotations, needs `Capability.ROTATIONS`), `q`
  (squashed/safe joint angles), `q_raw` (unsquashed — only present for a
  `"raw_rad"` reader; see D7 in [PROVENANCE.md](PROVENANCE.md)), `colliders` /
  `support_polygon` (robot capabilities). If your metric needs something *not*
  on this list (`robot.vel_limits`, `robot.effort_limits` are the two existing
  examples — see `metrics/joint_dynamics.py::VelViolationFrac`/`EffortProxy`),
  check it by hand inside `_compute` and return
  `MetricValue.unavailable(key, "missing_input:<name>")` explicitly — do not
  extend the `Requires` literal for a one-off optional field.
- **`min_frames`** — the smallest `T` below which your formula is degenerate
  (a `std` needs ≥2 samples; `sparc`/`log_dimensionless_jerk` need `MIN_T=8`
  for a stable spectrum/3rd derivative). Get this right the first time — a
  metric that returns a bare `NaN` at exactly `T = min_frames - 1` instead of
  an explicit `too_few_frames` reason has broken the "`NaN` iff `reason`"
  invariant, and `SafeMetric` will catch it (as `degenerate_input:...`) but
  the message is less informative than a correctly-set `min_frames` would
  have produced up front.
- **`unobservable_when`** — `("flag=value", ...)` strings checked against
  `MetricContext.flags`. The one existing case is
  `("limit_semantics=squashed",)` on `LimitViolationFrac`/`LimitExcessRad` —
  read `metrics/joint_limits.py`'s module docstring for exactly why this is
  declared **in addition to** (not instead of) `requires={"q_raw"}`: the
  reason string should read `unobservable:...` (a structural,
  head-architecture fact) rather than `missing_input:...` (which reads like a
  data-quality accident) whenever the true cause is architectural.
- **`perframe`** — set `True` if your metric also has a meaningful per-frame
  array to emit into the sidecar NPZ / `MetricSuite.quantity_keys` (for
  distributional W1/KFD comparison — see `mean_jerk_mps3` for the pattern:
  `self._ok(j.mean(), perframe=perframe_array)`).
- **`description`** — a full sentence, not a fragment; it is what
  `MetricSuite.describe()` surfaces to `kinescore describe` and what a future
  reader of `docs/METRICS.md` starts from. Every existing metric's
  `description` states the formula, the units, and *why* the declared
  `dt_exponent` is what it is (or why it's `None`) — match that.

## `dt_exponent` is numerically verified — get it right

**`tests/test_metric_registry_conformance.py::test_dt_exponent_conformance`
is auto-parametrized over every registered metric and numerically checks
your declared `dt_exponent`, not just your description of it.** It holds a
fixed synthetic trajectory (`P`, `R`, `q`, `q_raw`) constant and evaluates
your metric at `dt=0.1` and `dt=0.3` (`k=3`), then asserts

```
metric(dt=0.1*3) == metric(dt=0.1) / 3**dt_exponent
```

to `rtol=1e-6` — tight, because (per `metrics/ops.py`'s module docstring)
every derivative in this package is built from
`fd(x, dt) = (x[1:] - x[:-1]) / dt` applied some fixed number of times, and
holding the *sample array* fixed while only changing the declared `dt` value
scales the result by a rational factor with **no other source of numerical
error**. If your metric divides by `dt` a different number of times than you
declared, this test fails — on the metric you just added, in CI, not three
releases later as "why does this number look different at 10fps vs 30fps."

Two escape hatches, both requiring a documented reason, not just a value:

- **`dt_exponent=None`** — for a metric that is not a homogeneous power law in
  `dt` at all. Two distinct reasons this happens in the existing metrics, and
  your `description` must say which:
  1. **A threshold against a fixed physical constant**
     (`accel_violation_frac`, `vel_violation_frac`, `no_teleport_frac`,
     `self_collision_frac` is the counter-example — see below) — the
     underlying quantity scales with `dt`, but comparing it against an
     absolute bound is a discontinuous crossing, not a smooth rescaling.
  2. **A mix of terms with different intrinsic exponents added together**
     (`total_energy_tstd`: `dt^-2` kinetic + `dt^0` potential — see its
     module docstring for the exact algebra and the measured 1.397x shift
     under a 2x `dt` error, neither the clean 4x nor 1x a homogeneous metric
     would show).
  
  **Not every threshold gets `None`** — `self_collision_frac` thresholds
  `penetration_mm` against `pen_eps_m`, but that's a **spatial** threshold
  with no `dt` involved anywhere in the computation, so it legitimately
  declares `dt_exponent=0` and *is* checked by the conformance test. Ask "does
  `dt` appear anywhere in this formula at all" before reaching for `None`.
- **A proven-zero exponent, not just a measured one** — `log_dimensionless_jerk`
  declares `dt_exponent=0` not because it happened to measure close to zero,
  but because the algebra cancels exactly (`duration^5` contributes `-5`, the
  squared-jerk integral contributes `+5`, `path_len`'s `dt` cancels against
  the `/dt` inside its own speed computation — see `metrics/smoothness.py`'s
  module docstring for the full derivation). If you can prove your exponent
  algebraically, say so in `description` the same way; if you can only
  measure it empirically, `dt_exponent=None` is the honest declaration, not a
  measured integer.

## Adding a term to the PIS set changes `suite_id` — intentionally

`metrics/suites.py::INVARIANT_V1` is `MetricSuite(name="invariant_v1",
metrics=[...all 28...], invariant_keys=[...the 10 PIS terms...])`.
`MetricSuite.suite_id` (`core/suite.py`) is a SHA-256 hash over `name`, the
full ordered `output_keys`, and the sorted `invariant_keys` — so **adding,
removing, or renaming any metric key in the suite changes `suite_id`, whether
or not you also touched `invariant_keys`.**

This is not a bug to work around. `core/suite.py`'s module docstring states
the reasoning directly: a `suite_id` is what makes "two numbers are
comparable" checkable rather than assumed — `bench/stats.py::aggregate`
refuses to pool rows whose `run.suite_id` differs (`ValueError`, unless the
caller explicitly passes `allow_mixed_suites=True`, which stamps a
non-`None` `"warning"` on the result so it can't be missed), and
`reference/normalize.py::assert_comparable` raises `ComparabilityError`
across two `InvarianceResult`s from different suites. **If you add a metric
to the suite (even one *not* in `invariant_keys`), every existing
`RealMotionReference` built against the old `suite_id` stops being usable for
`invariance_score`** until you rebuild it (`RealMotionReference.build(new_suite,
...)`) — this is intentional: the alternative is silently comparing two
different benchmarks, which is exactly defect D3 recurring.

Practically:

1. Register your new metric normally (it becomes part of `output_keys`
   automatically once you add its key to `metrics/suites.py::_ALL_METRIC_KEYS`).
2. Decide deliberately whether it belongs in `_INVARIANT_KEYS` (the PIS
   term set) — see `metrics/suites.py`'s own docstring for the litmus test:
   task-*invariant* residuals (should be small/bounded for any real robot
   motion) go in; task-*dependent* descriptive magnitudes (raw speed, raw
   accel) do not.
3. Rebuild every `RealMotionReference` you care about against the new
   `INVARIANT_V1.suite_id` before scoring anything for comparison — see
   [REGENERATING_GOLDENS.md](REGENERATING_GOLDENS.md).
4. Add the metric's row to `docs/METRICS.md` (formula, units, `dt` exponent,
   direction, detects, does-NOT-detect — the mandatory entry every metric in
   that document has).
