# Benchmarking: config → run → CSV/traces → reading the numbers

This covers scoring already-generated video against a trained reader — see
[DATA_PREP.md](DATA_PREP.md) for getting data in place first and
[TRAINING.md](TRAINING.md) for producing the reader `--reader` points at.

## The pipeline, in the order the CLI actually chains it

```
kinescore data pull / ingest / verify        # DATA_PREP.md
kinescore anchor build                       # optional: rate-matched real footage, see below
kinescore bench run --config configs/benchmark.yaml
    → out/<run_id>/bench_manifest.parquet    # matrix expansion; stops here on purpose
  -- or, for an ad hoc directory --
kinescore manifest --root <dir> --out <manifest>

kinescore score --manifest <manifest> --robot <robot> --reader <ckpt> --out out/<run>/
    → out/<run>/results.jsonl [+ traces.npz/traces_index.jsonl if --traces]

kinescore aggregate out/<run>/               → out/<run>/stats.json  {results, separation, cache_ranking}
kinescore report out/<run>/stats.json        → report.html / report.md

kinescore export --results out/<run> --out <dir>   # parallel: per-cell CSVs, does not need aggregate
kinescore rank out/<run> --metric <key>            # parallel: sort individual clips/episodes
```

`aggregate`, `export`, and `rank` are three **independent** consumers of the
same `results.jsonl` + manifest pair — none depends on another's output.
`kinescore bench run` deliberately stops after writing the manifest; scoring
is a separate `kinescore score` call the operator chains manually (so a
manifest can be inspected, filtered, or reused across multiple scoring runs
without rebuilding it).

## Config → matrix: `kinescore bench run`

```bash
kinescore bench run --config configs/benchmark.yaml \
  --robot-map configs/robot_map.yaml \
  --out out/<run_id> \
  [--only robot=fourier_gr1] [--dry-run] [--cells-out cells.json]
```

`BenchConfig` (`bench/config.py`) declares five axes
(`axes.{robot,view,horizon,cache,generator}`) expanded into a Cartesian
product of `Cell`s by `bench/matrix.py::expand`, minus **N/A** cells
(`na_cells` — a `(robot, generator)` pair `robot_map.yaml` doesn't claim is
automatically N/A; `axes.na_cells` in the config adds cells that exist for
some *other* reason, e.g. a generator with no multiview tree at all). N/A
cells are always reported, never silently absent — a `--dry-run` run prints
the full cell table including them, so "why is this cell missing" is never a
question you have to reverse-engineer. `--only AXIS=VALUE` (repeatable)
filters after expansion, without re-deriving N/A logic.

**`axes.robot`, not `axes.embodiment`.** Every axis value is validated
against the live robot registry (`kinescore.robots.available_robots()`) at
config-parse time — a typo'd or unregistered robot name fails here, not
three cells into a run. See [DATA_PREP.md](DATA_PREP.md) for why `robot` had
to become the primary axis instead of `embodiment`.

## Scoring: `kinescore score`

```bash
kinescore score --manifest <manifest> \
  --robot <robot> --reader <ckpt.pt> --suite all_metrics \
  --out out/<run>/ [--resume | --force] [--traces]
```

Every manifest row's timebase is cross-checked against `ffprobe` before
scoring (`apply_resolved_timebase` — the probe wins over any `--fps`/`--dt`
override, never the reverse; see the rate-policy section below for why this
is a hard check, not a warning). A clip that fails to decode or score is
recorded via `bench/store.py::failed_record` (`status="failed"`, every metric
`null` with the exception as its reason) and the run continues — a corrupt
clip does not abort the whole benchmark. `--resume` keys off `(clip.path,
suite_id, reader_id)` read from what's actually on disk in
`results.jsonl`, so a clip already scored under a different suite or reader
checkpoint scores again rather than colliding. `--traces` additionally writes
`traces.npz` (per-frame arrays, e.g. `mean_jerk_mps3`'s trace) +
`traces_index.jsonl` — kept out of `results.jsonl` on purpose, since that
record must stay flat/scalar (see
[legacy_docs/SCHEMA.md](../legacy_docs/SCHEMA.md)).

`--suite` selects a `MetricSuite` (`metrics/suites.py`):

| suite | metrics | when |
|---|---|---|
| `invariant_v1` | 26, frozen (`suite_id` is what every prior published number was computed under — never extended) | reproducing a specific prior comparison |
| `all_metrics` | 28 (`invariant_v1` + `torque_frac_rated` + `rigidity_worst_bone_mm`) | **the suite to score a fresh run with** |
| `rate_free` | 9, `dt_exponent == 0` only, membership derived from the live registry | comparing clips at genuinely different, unmatched frame rates — see below |

`kinescore describe --suite <name> [--json]` prints exactly what a suite
contains (key, units, `dt_exponent`, direction, PIS membership) — the
authoritative, always-current alternative to a hand-maintained list; see
[METRICS.md](METRICS.md) for the full per-metric reference.

## Reading the numbers: `aggregate` → `report`

`kinescore aggregate` computes two logically distinct things, kept separate
rather than folded into one score:

- **Paired** (`bench/stats.py`) — per-episode `delta = phi(pred) - phi(gt)`,
  bootstrap CI, Wilcoxon signed-rank p-value. This is "did *this* generated
  episode drift from its own ground truth" — the claim that needs no rate
  assumption beyond pairing (see below).
- **Separation** (`bench/separation.py`) — unpaired AUROC + Cliff's delta
  between the full real and full generated distributions, oriented so `1.0`
  always means "generated is worse" (read off the metric's declared
  `direction`, never assumed). This is "can real and generated be told apart
  at all" — a classification question, not a pairing question. A row that
  can't be computed (fewer than `DEFAULT_MIN_EPISODES=5` episodes on either
  side) returns every numeric field `None` + a reason, never a fabricated
  `0.5`.

`kinescore report stats.json` renders both tables (plus cache ranking) as a
template-free, self-contained HTML or Markdown file.

### A measured result, to calibrate what these numbers look like

GR-1 vs DreamDojo, 842 content-deduplicated ground-truth-vs-generated pairs:
13 rulers separate real from generated at `p < 1e-8` after Holm correction,
best single separator AUROC **0.849** (`mean_angacc_radps2`).
`limit_violation_frac`, `limit_excess_rad`, and `self_collision_frac` all
measured **exactly 0 across all 2007 clips** scored (a real `raw_rad`
reading, not a squashed structural zero — see [TRAINING.md](TRAINING.md)).
`effort_proxy` is `NaN` with reason `missing_input:effort_limits` for every
GR-1 clip, because the GR-1 URDF declares no effort limits at all
(`GR1Spec.effort_limits is None` always) — this is the correct behaviour,
not a bug to chase: reporting `0.0` instead would read as "measured,
perfect," which would be a false claim about a robot whose URDF was never
asked the question.

### Bimanual/ALOHA: registered, not yet scored

`aloha_bimanual` is a fully registered, constructible `RobotSpec` (see
[ARCHITECTURE.md](ARCHITECTURE.md)) and `configs/benchmark.yaml` includes it
in `axes.robot`. It has **not** been scored end-to-end yet: no trained
`aloha_bimanual` reader checkpoint exists on disk, which `configs/
benchmark.yaml` marks honestly (`reader: null`, `reader_status: untrained`
— see `kinescore.bench.config.RobotConfig.reader_status`) rather than
naming a filename that was never a real file. As of this writing `dense/
bimanual/output/**` — the real generated ALOHA data — is **not** downloaded
locally, even though it exists on HuggingFace: roughly 150 episodes across
five branches under `dense/bimanual/output/`. Do not confuse "not
downloaded yet" with "no generated data exists" — the two are different,
and only the first is currently true. Its `dreamdojo` cell also resolves
through the underscored `single_view` subtree specifically for this robot
(`sources.dreamdojo.view_dir`'s per-robot override), not the `singleview`
spelling every other robot uses there — `singleview`'s 150 aloha_bimanual
episodes are pred-only (no ground truth anywhere), while `single_view`'s
103/85 episodes (makovian/non_makovian) are real, paired data.

## Rate policy: the operative rules

The benchmark matrix spans genuinely different native frame rates —
ctrlworld 5fps (with a measured, real anomaly: 2 episodes inside one
ctrlworld cell probe at 30fps, 6x the other 394 clips in that same cell),
dreamdojo 10fps, dreamgen 16fps, real GR-1 teleop 20fps, real ALOHA 30fps.
Most physics metrics here are derivatives (speed, accel, jerk, energy,
momentum), and a finite-difference derivative scales with `dt` raised to its
declared `dt_exponent` — comparing a 16fps clip's raw jerk against a 10fps
clip's raw jerk is not comparing two measurements of the same quantity.

**This was measured, not assumed, and the measurement is why these are hard
failures, not warnings.** Scoring real GR-1 teleop at its native 20fps
against dreamdojo output at 10fps gave real jerk ~117 m/s³ vs generated
~16–23 m/s³ — real looks 5x jerkier — and an AUROC of **0.00** (not 0.5:
every real clip ranked worse than every generated one). Re-encoding the real
anchor to dreamdojo's 10fps flipped it: real 15.7 vs generated 21.3, delta
+5.5 [4.7, 7.0], p≈2×10⁻²⁰. Same footage, opposite conclusion, purely from
respecting `dt`. Jerk's `dt_exponent=3` means the ctrlworld 30fps anomaly
above (a 6x rate ratio against its own cell) would inflate raw jerk by
**216x** if compared naively; the cross-generator worst case (ctrlworld 5fps
vs dreamgen 16fps, 3.2x ratio) is **~33x**.

Three layers, use in this order:

1. **`paired` (default).** Score every generated clip against **its own**
   ground truth — same scene, task, episode, native frame rate. `dt` cancels
   within the pair because both clips share it; this is where every headline
   paired-delta number in this doc comes from, and it's the only layer that
   supports the full `all_metrics` suite. `bench/manifest.py::verify_manifest`
   hard-fails (not warns) a gt/pred pair whose two members were probed at
   different frame rates — nothing downstream can tell "matched" from
   "silently mismatched" once numbers are computed, so it has to be caught
   before scoring.
2. **Anchor re-encode** — `kinescore anchor build` (`cmd_anchor.py`) +
   `kinescore bench noise-floor` (`bench/noise_floor.py`). Re-encodes real
   reference footage to a generator's exact fps/resolution/compression
   quality via `ffmpeg`, so a genuine cross-generator claim ("generated is
   27% jerkier than frame-rate-matched real motion") is possible. Comes with
   a built-in caveat check: re-encoding also changes compression artifacts,
   not only frame rate, so `noise-floor` measures a **null delta** — real
   footage re-encoded against itself, per-episode CRF varied deterministically
   (`base + episode_num % mod`, defaults 23/12) — so a genuine gap can be
   told apart from a re-encode artifact. `below_floor(observed_delta, floor)`
   checks `|delta|/floor < 1.0`.
3. **`rate_free` suite** — when no anchor is available, or a claim needs to
   hold for *any* frame rate rather than one matched pair, restrict to the 9
   `dt_exponent==0` metrics (`kinescore score --suite rate_free`).
   `core/scorer.py::Scorer(rate_policy="rate_free")` enforces this
   structurally at construction — it checks every metric in *whatever suite
   was actually passed* for `dt_exponent == 0` and raises naming the
   offending keys if any aren't, rather than silently swapping suites behind
   the caller's back.

**The one distinction in this whole doc that's easy to get backwards:**
`sparc` is "scale-free" (invariant to a movement's amplitude/duration, which
is why it separates real from generated well on the published page) but is
**not** frame-rate-invariant (`dt_exponent=None` — it builds its frequency
axis from `fs=1/dt` against a fixed 10Hz cutoff, so changing `dt` changes
which physical frequencies that cutoff represents). `log_dimensionless_jerk`
is the metric that's actually frame-rate-invariant (`dt_exponent=0`, proved
algebraically, not merely measured — see [METRICS.md](METRICS.md)). Treating
"the page calls sparc scale-free" as "therefore comparable across the whole
matrix" is exactly the confident-but-wrong comparison this section exists to
prevent.

A fourth, opt-in layer (`rate_policy="resample:<hz>"`, `core/resample.py`)
PCHIP-interpolates the joint trajectory `q(t)` onto a common rate after the
reader runs and before FK — downsampling only by default
(`UpsampleRefusedError` unless `allow_upsample=True`, since upsampling
invents frames a reader never produced and systematically flatters apparent
smoothness); every resample emits a `UserWarning` restating that a resampled
clip's derivative metrics are not comparable to a natively-sampled one. This
is the last resort, for the full suite across clips that genuinely differ in
rate with no anchor available — reach for `rate_free` first.

Metrics that carry `dt_exponent=None` and must never be compared across
rates for a *different* reason (thresholding a `dt`-dependent quantity
against a fixed physical constant, or summing terms with different
exponents): `accel_violation_frac`, `vel_violation_frac`, `no_teleport_frac`,
`total_energy_tstd`, `sparc`, `torque_frac_rated`. Full reasoning per metric
in [METRICS.md](METRICS.md).

## CSV export and per-clip ranking

```bash
kinescore export --results out/<run> --out <dir> [--sort-by mean_jerk_mps3]
kinescore rank out/<run> --metric mean_jerk_mps3 [--paired] [--top 20]
```

`export` writes one CSV per benchmark cell, **mirroring the input tree**
(`<cache>/<embodiment>/<view>/<generator>/<horizon>/clips.csv`) plus one
`SUMMARY.csv` at the run root — never scores anything, safe to rerun. Each
CSV's first line is a `#`-prefixed self-describing comment (sort key +
direction, plus a rigidity headline note when applicable). `rank` sorts by
one physical-unit ruler at a time (`--paired` sorts by the per-episode
`pred - gt` delta instead of the raw value) — there is deliberately no
composite score to sort by; `kinescore describe`'s per-metric `direction` is
what `rank` reads to know which end is "worst" without a hardcoded
per-metric special case.
