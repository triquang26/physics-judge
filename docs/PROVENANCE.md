# Provenance

`kinescore` is a merge of two research codebases (the "sources") into one
standalone benchmark:

- **A** = `Marionette-ciasc`, specifically `judge/{fk,physics,pixel_judge,scorer}.py`.
  Its git HEAD is `13b97e38cb40eefd7f91315c2eeba06b3ae6acb3`, but the working tree had
  **36 dirty files** at snapshot time (`git status --porcelain` was not clean),
  so **the SHA-256 in `provenance/sources.sha256` is authoritative, not the
  commit** — see `provenance/git_state.json`'s `ciasc_A` block.
- **B** = `Marionette-fkjepa`, specifically `models/evaluation/motion_reference.py`,
  `models/kinematics/gr1_fk.py`, `models/physics/smoothness.py` (plus other files
  cited below; the snapshot scope was later extended to cover all of them — see [(a3)](#a3-previously-unhashed-sources--gap-now-closed)).
  Its `.git` is a **dead worktree pointer** (`gitdir:` points at a path that no
  longer resolves — `PermissionError` on this host), so B has **no citable commit
  at all**. SHA-256 is the *only* provenance B can offer.
- **D** = `Marionette/models/evaluation/` was considered and **not used** — see
  [(c)](#c-not-ported-and-why).

This document is generated from, and should be cross-checked against:
`provenance/sources.sha256` (hash ledger), `provenance/copy_manifest.tsv`
(suggested destinations), `provenance/git_state.json` (commit/dirty state),
`tools/source_snapshot.py` (what produced the above), `tools/gen_golden.py`
(what produced `tests/golden/*.npz`), `tools/ast_diff.py` (the verbatim-body
checker), and `tests/golden/MANIFEST.json`.

## (a) Source → destination table

### (a1) Files snapshotted by `tools/source_snapshot.py`

These files are hashed in `provenance/sources.sha256`, which covers all 18 cited sources —
`tools/source_snapshot.py`'s `SOURCES_A`/`SOURCES_B` constants are the sole
source of truth for which bytes were hashed. Every row below is verifiable by
re-running that script and diffing against `provenance/sources.sha256`.

| # | Source | LOC | SHA-256 (source) | Destination(s) | Classification |
|---|---|---:|---|---|---|
| 1 | A `judge/fk.py` | 427 | `36a07ab83ec33f58295c285a84a6a818e89b230a8bde5cd9cf163fe5b7579b9c` | `src/kinescore/robots/franka/fk.py` (412 LOC) + `src/kinescore/robots/franka/constants.py` (121 LOC) | **verbatim** |
| 2 | A `judge/physics.py` | 399 | `fba2ff609601053d5e6c05731765f6dc2435d723781c6956441849320387d188` | `src/kinescore/metrics/{rigidity,temporal,angular,energy,joint_limits,joint_dynamics}.py`, `src/kinescore/metrics/ops.py` (`fd`, `vee`) | **adapted** |
| 3 | A `judge/pixel_judge.py` | 572 | `4de2fb1579245f7f7e91affcef3d1a275d10ba61dc441d22089c6c82087d4b33` | `src/kinescore/heads/attentive.py`, `heads/mlp.py`, `backbones/pooling.py` (verbatim); `heads/ranges.py::squash_to_limits` (verbatim); `backbones/dino.py` (adapted); `readers/checkpoint.py` (adapted) | **mixed** — see per-file note below |
| 4 | A `judge/scorer.py` | 274 | `e96cd676666a0a046ad76aeb8aa9b34022f057e254c5d297d0a2572cb7faf66f` | `src/kinescore/core/scorer.py` (138 LOC) | **rewritten** |
| 5 | B `models/evaluation/motion_reference.py` | 318 | `3d1fdc755ecc02bb255bdacb42f61a38fe1951282d2b866080cbf3f57819fba4` | `src/kinescore/reference/{fingerprint,normalize,distances}.py` | **mixed** — see per-file note below |
| 6 | B `models/kinematics/gr1_fk.py` | 378 | `746b8e4d3bebce2e66e7759126eb4982e077f8a3d1801026369ef00ddad4a211` | `src/kinescore/robots/gr1/fk.py` (400 LOC) | **verbatim**, except one rename (see below) |
| 7 | B `models/physics/smoothness.py` | 243 | `5bb6262cc0c85016879b319ecfd90c5567d1022541755959c0a1d8083556181f` | `src/kinescore/metrics/smoothness.py` (231 LOC) | **adapted** |

Per-file notes:

- **#1 `fk.py`.** `robots/franka/fk.py`'s own docstring: "the FK math (every
  method body below) is unchanged character-for-character; the only edits are
  the import lines and the extraction of the module-level constant tables into
  `kinescore.robots.franka.constants`." Confirmed by reading both files: every
  method (`forward`, `forward_transforms`, `ee_pose`, `_joint_tensor`,
  `_compute_rest_bones`, `_matrix_to_rotvec`, ...) is line-for-line identical
  to what the docstring claims; `tools/ast_diff.py` is the tool that checks
  this mechanically (see [(d)](#d-a-scoped-equivalence-claim)).
- **#2 `physics.py`.** Every formula (`velocity`/`acceleration`/`jerk` via
  `fd`, `angular_velocity` via `vee`, `kinetic_energy`, `momentum_continuity`,
  `limit_violation`'s `relu(q-hi)+relu(lo-q)` excess term, `joint_feasibility`)
  is a verbatim port at the arithmetic level — every metric module's docstring
  says "verbatim port of ..." for its specific formula. The **classification
  is "adapted" rather than "verbatim"** because: the code is restructured
  from two monolithic classes (`KinematicConsistency`, `PhysicsConsistency`)
  into one `BaseMetric` subclass per quantity with a declared `MetricSpec`;
  `joint_limits.py` retargets the excess computation from `q` to `q_raw`
  (defect D7 — a real behaviour change, not merely a refactor); `rigidity.py`
  adds the `bone_set="rigid"` default (defect D9); `RigidityWobble` uses
  population (`ddof=0`) rather than sample std. `bone_set="all"` and (in
  effect) the population-std change are what make the *legacy* numbers still
  reproducible for comparison — see [(b)](#b-intentional-behaviour-changes).
- **#3 `pixel_judge.py`.** `heads/attentive.py::AttentivePoseHead.forward` is
  verbatim (its own docstring: "the einsum / concat / MLP order is copied line
  for line"; `cam_emb` stays conditional on `n_cams>1` specifically so real
  checkpoints still `load_state_dict(strict=True)`) **except** the D4 token
  guard, which is new. `heads/mlp.py::DinoPoseHead` is verbatim.
  `backbones/pooling.py::pool_patch_tokens` is verbatim. `heads/ranges.py`'s
  `squash_to_limits` is a verbatim port of `predict_pose`'s sigmoid-squash
  line. `backbones/dino.py` is adapted: `encode_latent`/`encode_latent_grad`
  (the diffusers/SVD latent-decode path) are **deleted, not adapted** (see
  [(c)](#c-not-ported-and-why)), and view splitting goes through
  `ViewLayout.view_height` instead of a bespoke `H % n_cams` check.
  `readers/checkpoint.py`'s `save`/`load` are adapted from
  `PixelPhysicsJudge.save`/`.load` (defect D5).
- **#4 `scorer.py`.** Classified **rewritten**, not adapted: the source's
  `Scorer` had no `ClipSpec`, `MetricContext`, `ViewLayout`, or
  `limit_semantics` flag to compose — those types do not exist in the source
  at all. `core/scorer.py`'s cross-checks (`reader.robot_name == robot.name`,
  `_check_layout`, threading `dt` from the clip with no default) are new
  structure built to close D1/D2/D4/D7, not a line-for-line port with fixes
  layered on top.
- **#5 `motion_reference.py`.** `reference/distances.py`'s `profile_w1` and
  `kfd_approx` are adapted verbatim-math ports (`torch.quantile` →
  `np.quantile`+subsampling is the only numeric-path change; `_frechet` is
  renamed `kfd_approx` with an honest docstring, math untouched).
  `reference/fingerprint.py` (the `RealMotionReference` class itself —
  construction, serialization) is **rewritten**: schema 2 adds `dt`,
  `term_keys`, `floors` (defects D2, D3, D3b) that have no analogue in the
  source's `save()`/`load()`. `reference/normalize.py::invariance_score` is
  **rewritten** for the same reason (fixed term set, floored denominator) —
  but it also ships `legacy=True`, an intentionally-preserved verbatim replica
  of the source's `invariance_score` body, for regression/provenance
  comparison (see `tests/test_reference_golden.py`,
  `tests/test_pis_term_set.py`).
- **#6 `gr1_fk.py`.** `robots/gr1/fk.py`'s docstring: "Every method body below
  is unchanged from the source **except** `GR1FK.hand_flexion_mean`, which is
  the source's `state_to_gripper` renamed (identical implementation, zero
  numeric change)" — the rename exists because the old name/docstring implied
  a `[0,1]`-normalised gripper fraction that the (unchanged) implementation
  never actually produced; see the method's own docstring for the full
  argument. Confirmed by reading: `keypoints_fk`, `forward_transforms`,
  `ee_pose`, `_full_theta`, `_compute_rest_bones`, `chain_depths`,
  `fingers_fk`, `_matrix_to_rotvec`, `state_to_q17` are unchanged.
- **#7 `smoothness.py`.** `sparc`, `dimensionless_jerk`, `log_dimensionless_jerk`
  are verbatim ports (pure numpy, fp64, no torch) — see the module docstring's
  citations (Balasubramanian et al. 2015; Hogan & Sternad 2009). Classified
  "adapted" because they are wrapped in `Sparc`/`LogDimensionlessJerk`
  `BaseMetric` classes with a `MetricSpec`, and the aggregation over
  end-effector sites is generalised from the source's hardcoded left/right
  loop to `robot.ee_sites()`.

### (a2) Destination modules with no claimed source (new code)

Everything below has no "ported"/"verbatim"/"adapted" claim anywhere in its
docstring and is new code written for this merge — the frozen contracts, the
robot-wrapper (`*Spec`) classes, and new infrastructure:

`src/kinescore/core/{clip,metric,robot,reader,suite}.py` (the frozen
contracts), `src/kinescore/robots/{franka,gr1}/spec.py` (the `RobotSpec`
wrappers around the ported FK classes), `src/kinescore/robots/{urdf,synthetic}.py`,
`src/kinescore/readers/{squashed,heteroscedastic,ensemble,_frames}.py` (new
compositions of ported heads), `src/kinescore/heads/views.py` (the D4 fix,
`ViewEmbedding`), `src/kinescore/metrics/{suites,_base}.py`,
`src/kinescore/bench/store.py`, `src/kinescore/bench/runner.py`,
`src/kinescore/paths.py`.

### (a3) Previously-unhashed sources — gap now closed

An earlier revision of `tools/source_snapshot.py` enumerated only **seven**
source paths, while the ported package cited **fifteen**. Eight
"ported"/"verbatim" docstring claims therefore had no SHA-256 backing them: they
were only as strong as the docstring's own word. A provenance record that is
silently incomplete is worse than none, because a reader reasonably assumes it
is exhaustive.

`SOURCES_A` / `SOURCES_B` now enumerate **all 18** cited source files, and
`provenance/sources.sha256` was regenerated against the live checkouts. The
modules listed below — `robot_colliders.py`, `feasibility.py`,
`pixel_judge_gr1.py`, `readout_v2.py`, `readout_v2_scorer.py`,
`eval/bench/{manifest,scoring,stats}.py`, `corruptions.py`, and the two
`scripts/` training entry points — are now hashed like every other source, so
their claims are checkable by re-running the tool rather than taken on trust.

`tests/test_provenance_complete.py` pins this: it cross-checks the enumerated
list against every source path mentioned in `src/`, so a future port that cites
a new source file without adding it to the snapshot scope fails CI rather than
quietly reintroducing the gap.

The destination-to-source claims themselves are unchanged from when they were
first recorded:

| Destination | LOC | Docstring's claimed source | Claim |
|---|---:|---|---|
| `src/kinescore/robots/gr1/colliders.py` | 160 | B `models/physics/robot_colliders.py` | "Ported **verbatim** ... no line of the class body below is changed" |
| `src/kinescore/metrics/feasibility.py` | 335 | B, `MechanicalFeasibility` (exact file path not cited) | "Ports the per-check scorers of Marionette-fkjepa's `MechanicalFeasibility` ... as four standalone metrics" |
| `src/kinescore/heads/disentangled.py` | 90 | B `models.evaluation.pixel_judge_gr1.DisentangledPoseHead` | "Ported **verbatim** (module math unchanged)" |
| `src/kinescore/heads/heteroscedastic.py` | 200 | B `models.posendf.readout_v2` | "Ported from ... Only the inference-time surface is ported" |
| `src/kinescore/heads/ranges.py` (`clamp_for_fk` only; `squash_to_limits` is covered by (a1)#3) | — | B `models.posendf.readout_v2.clamp_for_fk` | "ported verbatim" |
| `src/kinescore/bench/manifest.py`, `src/kinescore/video/probe.py` (`ffprobe`) | 289, 187 | B `Marionette-fkjepa/eval/bench/manifest.py` | "Ported from ... its `ffprobe` is good and lives on unchanged" |
| `src/kinescore/bench/stats.py` | 406 | B `Marionette-fkjepa/eval/bench/stats.py` | "unchanged logic (numpy/pandas/scipy), carried over as-is", except `load_scores` ("rewritten") and `aggregate`+its suite-mixing guard ("new, not ported") |
| `src/kinescore/video/corruptions.py` | 129 | B `Marionette-fkjepa/models/evaluation/corruptions.py` | ported, with an explicit, documented RNG-reproducibility fix (every operator now takes an explicit `generator:` argument) |

None of these are contradicted by anything observed while writing this
document — they read as internally consistent, carefully-scoped claims (e.g.
`bench/stats.py` explicitly separates what is verbatim from what is new in
its own docstring) — but they are **not independently SHA-256-verifiable
today**. If `docs/PROVENANCE.md` is treated as a compliance artifact, the fix
is to extend `tools/source_snapshot.py`'s `SOURCES_A`/`SOURCES_B` to include
these paths and re-run it while the B checkout is still reachable (its `.git`
is already a dead pointer — see the top of this document — so this window is
closing, not open-ended).

## (b) Intentional behaviour changes

One subsection per defect. Each states **was** / **is** / **why** / **test
that pins it** / **do old artifacts still load**.

### D1 — `dt` hardcoded by omission

- **Was.** `PhysicsConsistency.__init__(dt=0.2)` and `PhysicsReferee.__init__(dt=0.2)`
  defaulted silently. Two call sites (`stress_test_judge.py:49-50`,
  `viz_stress_gif.py:43-44`) never passed one and had no CLI flag, while the
  training loader sampled a random frame skip of 1 or 2 — so roughly half the
  clips scored were truly `dt=0.4`, not `0.2`.
- **Measured impact.** Speed scales 2x, acceleration 4x, jerk 8x for a 2x `dt`
  error; `accel_violation_frac` moves `0.000 → 0.3875` on a representative
  trajectory (`AccelViolationFrac`'s own default `accel_bound=5.0`).
- **Is.** `dt` lives in `ClipSpec` (`src/kinescore/core/clip.py`) as a
  keyword-only field with **no default**, validated by `validate_dt` (rejects
  non-finite, ≤0, and anything above 10s — the last catches "passed fps where
  dt was expected"). `MetricContext.dt` is likewise required with no default.
  `ClipSpec.subsample(k)` is the *only* supported decimation path (it scales
  `dt` for you); `video/reader.py::load_rgb` only drops frames via
  `clip.stride`, and raises if the decoded frame count disagrees with what the
  spec declares. Every derivative in `src/kinescore/metrics/ops.py::fd`
  divides by `dt` exactly once per application, which is what makes a
  declared `dt_exponent` checkable to float32 precision (see
  `tests/test_metric_registry_conformance.py`).
- **Why.** A silently-wrong frame rate is invisible in the numbers alone —
  ratio metrics partly hide it, absolute ones (accel/jerk violations) don't,
  and nothing in the source recorded which `dt` a given score actually used.
- **Test that pins it.** `tests/test_dt_invariance.py::test_wrong_dt_is_detected`
  (the exact D1 scenario: same `P` array scored at the true `dt` vs `dt`
  wrong by 2x — speed/accel/jerk ratios come out exactly 2/4/8,
  `accel_violation_frac` goes `0.000 → 0.3875`, both to `rtol=1e-5`), plus
  `tests/test_dt_invariance.py::test_{mean_speed,mean_accel,mean_jerk}_dt_invariant`
  (real resampling, loose tolerance) and every parametrized case in
  `tests/test_metric_registry_conformance.py::test_dt_exponent_conformance`.
- **Do old artifacts still load?** N/A — `dt` was never serialized on its own
  in the source; this defect is about a runtime default, not a file format.
  The adjacent reference-file question is D2, below.

### D2 — reference did not serialize `dt`

- **Was.** `RealMotionReference.save()`/`.load()` persisted `inv_baseline`,
  `quantiles`, `feat_mu`, `feat_cov`, `quantity_keys`, `n_q` — never the frame
  interval the reference was built at. Every one of those fields scales as
  some power of `1/dt` (a jerk baseline as `1/dt**3`), so loading a reference
  built at one rate and scoring a run at another silently rescaled every
  metric by up to 8x, undetectable from the file or the numbers alone.
- **Is.** Schema 2 (`src/kinescore/reference/fingerprint.py::SCHEMA_VERSION = 2`)
  requires and persists `dt` (via `validate_dt`), plus `suite_id`, `term_keys`,
  `floors` (D3/D3b, below). `RealMotionReference.load` on a schema-2 file
  reads `dt` from the file. A **legacy** (schema-less, pre-D2) file has no
  `dt` field at all and `load` **raises** unless the caller passes
  `dt=` explicitly (`load(path, dt=0.2)`) — an explicit acknowledgement of
  what rate the caller believes the file was built at, not a silent guess.
  `RealMotionReference.check_rate(dt, allow_rate_mismatch=False)` raises
  `RateMismatchError` when a clip's `dt` disagrees with the reference's by
  more than `1e-6` relative, unless the caller opts in
  (`allow_rate_mismatch=True`, which downgrades to a `warnings.warn` and sets
  `InvarianceResult.rate_mismatch=True`).
- **Why.** A frame-rate mismatch between a reference and the run being scored
  against it is exactly as invisible, and exactly as damaging, as D1's
  runtime default — just one layer further from the metric arithmetic.
- **Test that pins it.** `tests/test_reference_dt_roundtrip.py` (schema-2
  round-trip preserves `dt`; a legacy file with no `dt` raises without an
  explicit `dt=`, and loads with a warning when one is given).
- **Do old artifacts still load?** **Yes, deliberately** — `_load_legacy` in
  `fingerprint.py` reads a pre-D2 file, requires `dt=` from the caller, sets
  `suite_id="legacy:v1"` (honestly flagging that the file predates
  suite-pinning too — see D3/D4b), derives `term_keys` from
  `sorted(d["inv_baseline"])`, and `warnings.warn`s that the floors fall back
  to unit defaults (no per-rollout residuals were persisted in that format to
  fit a median-absolute-deviation floor from). A legacy file loads; it does
  not load *silently*.

### D3 (a) — PIS averaged over a varying number of terms

Two unrelated defects in the sources share the label "D3" in this codebase's
own docstrings — this is a genuine naming collision in the code, not
duplication in this document. This subsection is (a); the manifest-side one
is (b), immediately below.

- **Was.** The source's `invariance_score` iterated `residuals.items()` and
  `continue`d past any key missing from `inv_baseline`:
  `pis = sum(scores.values()) / max(len(scores), 1)`.
  `PhysicsConsistency.invariant_residuals()` emitted 6 keys unconditionally,
  +1 with rotations, +3 with joint angles — so a clip scored with joint angles
  produced a mean over 9–10 terms, one scored from keypoints alone a mean over
  6, and **both were reported as the same number and directly compared**.
  `INVARIANT_KEYS` was declared in the source but never actually used to pin
  anything.
- **Is.** `MetricSuite` (`core/suite.py`) fixes `output_keys` (every metric
  the suite declares) and `invariant_keys` (an explicit, hand-picked subset —
  `metrics/suites.py::INVARIANT_V1` mirrors the source's own
  `INVARIANT_KEYS`, ten keys) at construction, and hashes both into
  `suite_id`. `RealMotionReference.term_keys` is `suite.invariant_keys` at
  reference-build time, not inferred from what a rollout happened to have.
  `invariance_score(..., policy=...)` in `reference/normalize.py` then
  evaluates **exactly** that fixed set for every clip: `policy="strict"`
  (default) makes the whole PIS `NaN` with a reason if any declared term is
  missing/NaN, rather than quietly averaging over fewer terms;
  `policy="available"` opts back into an average-over-present-terms but
  *records* `n_terms` against `n_terms_declared`; `policy="nan"` lets a
  missing term's `NaN` propagate through an ordinary mean.
  `n_terms == n_terms_declared` for every clip under `"strict"`/`"nan"`.
- **Why.** Two clips whose PIS differs only because one happened to lack
  joint-angle data would silently look like two clips of different physical
  quality.
- **Test that pins it.** `tests/test_pis_term_set.py` — in particular
  `test_fixed_term_set_gives_same_n_terms_with_and_without_a_key` (both
  report `n_terms == n_terms_declared == 3`; the incomplete one is `NaN` with
  a `"missing_terms"` reason, not silently averaged over 2) and
  `test_legacy_replica_reproduces_the_varying_term_count`, which **verifies
  the legacy 3-vs-2 asymmetry is reproduced** by `invariance_score(...,
  legacy=True)` — the exact analogue of the brief's "7-vs-10" scenario,
  confirmed against a synthetic 3-term suite rather than the real 10-term
  `INVARIANT_V1` (both demonstrate the same defect mechanism). Also
  `tests/test_reference_golden.py::test_legacy_replica_reproduces_source_pis_scores_and_n_terms`
  against the real golden fixture.
- **Do old artifacts still load?** N/A — this is scoring-time logic, not a
  file format; the adjacent file-format question is D2.

### D3 (b) — manifest builder silently overrode the probed frame rate

This is the codebase's *other* "D3" — a distinct defect from (a), sharing the
label. It lives in `bench/manifest.py` and `video/probe.py`.

- **Was.** `Marionette-fkjepa/eval/bench/manifest.py`'s row builder called
  `ffprobe` on every clip, read `w`/`h`/`n_frames` from the result — and then
  **overwrote** `fps` with whatever a hand-maintained per-family config table
  said, discarding the probed value outright. The config file itself carries
  the scar tissue from this: one entry reads `dreamgen: 16.0  # PROBED: ...
  NOT 10 fps`, i.e. someone had already been bitten by a stale table entry and
  left a comment instead of fixing the code path that let it happen.
- **Is.** `video/probe.py::resolve_timebase` always calls `ffprobe`; if a
  table/CLI value is also given, disagreement beyond `probe_tolerance`
  (default 1%) is a hard `TimebaseError` naming the file and both numbers —
  not a silent override in either direction. `bench/manifest.py::_row` routes
  every row through `resolve_timebase` instead of reimplementing the
  override. The manifest's `fps_probed` column is kept *even when a
  table/CLI value won*, so a resolved-but-suspicious row stays auditable.
- **Why.** The fix is not "trust the probe over the table" — a table can be
  right when a container's timestamps are lying — it is that both are
  **always available and cross-checked**, so a disagreement is a loud error
  naming both values instead of an invisible substitution.
- **Test that pins it.** `tests/test_video_probe.py::TestResolveTimebase`
  (`test_raises_when_table_disagrees_with_probe`,
  `test_fps_and_dt_args_are_mutually_exclusive`,
  `test_dt_arg_within_tolerance_is_accepted`), `tests/test_manifest.py`.
- **Do old artifacts still load?** N/A — this is a manifest-build-time check,
  not a persisted format.

### D3b — zero-baseline blowup

- **Was.** `denom = base*(tol-1)+1e-8`. When a key's real-data baseline is
  ~0 (e.g. `rigidity_wobble_mm` on a clean reader), the denominator collapses
  to `~1e-8`, the per-key score saturates to `1.0` on any tiny excess, and
  then dominates the unweighted PIS mean regardless of how physical the rest
  of the rollout is.
- **Is.** `denom = max(base, floors[k]) * (tol - 1.0)`. `floors[k]` is fit at
  reference-build time (`RealMotionReference.build`) as
  `max(median_absolute_deviation(real values for k), UNIT_FLOORS.get(k, 0.0))`
  — e.g. `UNIT_FLOORS["rigidity_wobble_mm"] = 0.5` (mm). `*_frac` keys
  (already in `[0,1]`) are untouched — they use the absolute `frac_tol` path,
  which the floor fix never applies to.
- **Why / no-op guarantee.** When `base > floors[k]`, `max(base, floors[k]) ==
  base`, so the fixed denominator equals the old one to within `1e-8` (the
  removed epsilon) — **verified**: `tests/test_pis_zero_baseline.py::test_well_conditioned_case_is_a_numerical_no_op`
  checks `abs(fixed_denom - legacy_denom) <= 1e-8` and that the resulting PIS
  values agree to `abs_tol=1e-6` across four different `v`. On the
  pathological case (`base=1e-9, floor=0.5, v=1e-6`),
  `test_legacy_denominator_saturates_near_zero_baseline` confirms the legacy
  path saturates to `pis == 1.0` while the fixed path gives `pis < 1e-3`.
- **Test that pins it.** `tests/test_pis_zero_baseline.py` (all four cases
  above).
- **Do old artifacts still load?** A legacy (schema-1) reference has no
  per-rollout residuals persisted to fit a MAD floor from, so its
  `floors` fall back to `UNIT_FLOORS` defaults only (`{k: UNIT_FLOORS.get(k,
  0.0) for k in term_keys}` in `_load_legacy`) — a documented, warned-about
  degradation, not a crash.

### D4 — multiview cache/score desync with no assertion

- **Was.** Cache-time views came from separate files concatenated on the
  token axis; score-time views came from splitting one frame on the height
  axis. The source's only guard, `AttentivePoseHead.forward`'s `N %
  self.n_cams != 0` check, lived **inside** the `n_cams > 1` branch — so the
  single-camera case (the overwhelmingly common one) had **no guard at all**,
  and a 1-camera head could silently consume a 147-token 3-camera feature
  grid (`147 % 1 == 0` never even entered the branch that checks anything).
- **Is.** `ViewLayout.assert_tokens` (`core/clip.py`) checks the token count
  in both directions: with `tokens_per_view` known it asserts the *exact*
  count (`n_views * tokens_per_view`), not mere divisibility.
  `AttentivePoseHead.forward` asserts unconditionally at the top, before the
  `n_cams > 1` branch, so it fires for `n_cams == 1` too — the source's
  `cam_emb` stays conditional on `n_cams > 1` (unchanged, for checkpoint
  compat: `judge_v3l`/`judge_reward` have no `cam_emb` key at all).
  `heads/views.py::ViewEmbedding` is the new, standalone version used by
  heads that never had multiview support (`ReadoutV2Head`,
  `DisentangledPoseHead`, `DinoPoseHead`) — its bias is always present but
  zero-init, so a single-view layout is bit-identical to no embedding at all.
  Caches are self-describing via `ViewLayout.key`.
- **Why.** A shape mismatch that happens to be evenly divisible is invisible
  arithmetic corruption, not a crash — the worst kind of bug.
- **Test that pins it.** `tests/test_multiview_layout.py` — both
  `ViewEmbedding` and `AttentivePoseHead` are checked in both directions
  (`test_view_embedding_1view_head_fed_147_tokens_raises`,
  `test_attentive_head_1view_fed_147_tokens_raises`, and the 3-view analogues),
  plus the zero-init no-op guarantee
  (`test_view_embedding_v1_is_bit_identical_to_no_embedding`,
  `test_attentive_head_v1_output_bit_identical_with_and_without_zero_cam_emb`).
- **Do old artifacts still load?** Yes — `AttentivePoseHead`'s `cam_emb`
  stays conditional specifically so `judge_v3l`/`judge_reward` (no `cam_emb`
  key) and `judge_v3l_mv` (has one, shape `(3, embed_dim)`) all still
  `load_state_dict(strict=True)` — see D5's checkpoint tests, which exercise
  exactly this.

### D4b — reference's quantity-key set depended on iteration order

- **Was.** `qkeys = [k for k in samples_list[0]]` — whichever motion
  quantities the *first* real rollout happened to carry became the
  reference's permanent quantity set. A later rollout missing one of those
  keys raised a bare `KeyError` from deep inside `torch.stack`.
- **Is.** `RealMotionReference.quantity_keys` is `suite.quantity_keys` — a
  property of the `MetricSuite`, fixed before any real data is seen.
  `RealMotionReference.build` validates every rollout against the full
  declared set *up front*, raising one message listing exactly which
  rollouts are missing which keys (`reference/fingerprint.py::build`,
  the `missing_qty` block), rather than an accidental subset silently
  becoming "the reference" or a `KeyError` surfacing later at score time.
- **Why.** Same shape as D3(a): a set that should be a declared property of
  the suite was instead an accident of whichever rollout happened to be
  iterated first.
- **Test that pins it.** Covered inside `tests/test_reference_golden.py`
  (`RealMotionReference.build`'s validation path) and by construction in
  `tests/test_pis_term_set.py`'s reference-building helper, which always goes
  through the suite-declared key list.
- **Do old artifacts still load?** Same legacy path as D2/D3b — a schema-1
  file's `quantity_keys` is whatever was persisted in the file (no
  suite-pinning existed then), loaded as-is with the `suite_id="legacy:v1"`
  flag and a `warnings.warn`.

### D5 — checkpoint dropped `hidden`/`dropout`

- **Was.** The source's `save()` wrote a 12-13-key cfg covering backbone/pooling
  settings but never `hidden` or `dropout`. `load()` reconstructed the head
  with `hidden` hardcoded to `AttentivePoseHead`'s constructor default (512).
  A checkpoint trained with `hidden != 512` was therefore **permanently
  unloadable** — `load_state_dict(strict=True)` fails on the `mlp.0.*` shape.
- **Is.** `readers/checkpoint.py::save` persists both fields. `load` still
  tolerates their absence in an old file: `hidden` is **inferred** from
  `state_dict["mlp.0.bias"].shape[0]`, `n_heads` from
  `state_dict["score.weight"].shape[0]`, `embed_dim` from
  `state_dict["norm.weight"].shape[0]` — cross-checked against the cfg even
  when the cfg *does* carry them (`_infer_or_check`: a hand-edited or
  corrupted cfg fails loudly naming both the declared and inferred values,
  rather than constructing a head whose declared shape doesn't match its own
  weights). `dropout` defaults to `0.1` when absent (shape-inert, cannot break
  loading). **`cfg.get("n_cams", 1)` is preserved exactly** — `judge_v3l` and
  `judge_reward` predate the multiview head and have no `n_cams` key at all.
- **Why.** A checkpoint that cannot be reloaded because of a save-side
  omission — not because the weights are bad — is the worst kind of format
  bug: it destroys work that was never actually broken.
- **Test that pins it.** `tests/test_checkpoint_legacy_cfg.py` (synthesizes
  the *exact* 12-key cfg the real `judge_v3l` file has — verified by loading
  the actual checkpoint — including the `hidden=256`-instead-of-512 case the
  source could never have reloaded at all) and
  `tests/test_checkpoint_roundtrip.py` (`test_roundtrip_cross_check_raises_on_corrupted_cfg`
  et al.). `@pytest.mark.ckpt`-gated tests
  (`test_real_production_checkpoints_load`) verify all **three** real
  checkpoints (`judge_v3l`, `judge_v3l_mv`, `judge_reward`) load with
  `strict=True` when a `KINESCORE_CIASC_ROOT` checkout is available.
- **Do old artifacts still load, bit-for-bit?** All three load with
  `strict=True` (verified above). **A forward-output match against
  `tests/golden/golden_ckpt_head.npz` is verified for two of the three**
  (`judge_v3l`, `judge_v3l_mv` — `judge_reward` has no golden fixture at all,
  see `tools/gen_golden.py::_CKPTS`), via
  `test_real_checkpoint_forward_matches_golden`, and the comparison is
  `torch.testing.assert_close(atol=1e-4, rtol=1e-4)` — **numerically close,
  not literal bit-identity.** This is a minor correction to how "bit-for-bit"
  should be read: it means "agrees with the golden recording of the source's
  own `AttentivePoseHead.forward` to 1e-4", which is the tightest claim
  possible once fp32 arithmetic and two separate process runs are involved,
  not exact binary equality.

### D7 — `limit_violation` structurally always 0

- **Was.** A sigmoid-squashed head (`q = lo + (hi-lo)*sigmoid(raw)`) cannot
  exceed `[lo, hi]` by construction, so `limit_violation(q)` read "perfect"
  (`0.0`) for **every clip ever scored with that head family** — which was
  every real checkpoint (`judge_v3l`, `judge_v3l_mv`, `judge_reward` all use
  the squash) — while measuring nothing about the video at all.
- **Is.** `limit_semantics` (`"squashed"` | `"raw_rad"`) is a declared field
  on every `PoseReader` (`core/reader.py`), propagated into
  `MetricContext.flags`. `LimitViolationFrac`/`LimitExcessRad`
  (`metrics/joint_limits.py`) require `q_raw` (present only for a `"raw_rad"`
  reader) **and** declare `unobservable_when=("limit_semantics=squashed",)`
  as defence in depth — so the unavailability reason reads
  `unobservable:limit_semantics=squashed` (a structural, head-architecture
  fact) rather than `missing_input:q_raw` (which would read like a
  data-quality accident) whenever the true cause is "this head family cannot
  violate its limits by construction". A squashed reader **never** reports
  `0.0` for these metrics — it reports `NaN` with that reason.
  `LimitHeadroomRad` is the complementary, always-observable substitute
  (distance from the always-safe `q` to the nearest limit — the closest a
  squashed head can express "was this straining against its limits").
- **Why.** `0.0` and "not measured" are not the same claim, and conflating
  them here would misreport a structural blind spot as a clean bill of
  health for the majority of real checkpoints in production.
- **Test that pins it.** `tests/test_limit_semantics.py`,
  `tests/test_reader_limit_semantics.py` (`test_heteroscedastic_reader_exposes_q_raw_and_clamp_magnitude`,
  `test_heteroscedastic_reader_clamp_is_zero_when_raw_in_range`).
- **Do old artifacts still load?** N/A (runtime metric semantics, not a file
  format) — but every real checkpoint predates the `limit_semantics` field
  entirely; `readers/checkpoint.py::_LEGACY_LIMIT_SEMANTICS = "squashed"` is
  the historically-accurate default applied when a loaded cfg has no such
  key, "because every real checkpoint at this format was trained with the
  sigmoid squash".

### D9 — gripper contaminates rigidity (found during this migration)

- **Was.** The Franka's `bone_pairs` in the source is every *consecutive*
  keypoint pair, which puts `panda_leftfinger -> panda_rightfinger` in the
  list. That bone's rest length is **exactly 0.0 m** (the fingers coincide
  when the gripper is closed, the pose `_compute_rest_bones` measures from).
  Two neighbouring bones (`panda_hand -> panda_leftfinger`, 0.0584 m;
  `panda_rightfinger -> panda_hand_tcp`, 0.045 m) carry the same defect for a
  less obvious reason: both finger links translate along the *prismatic*
  finger joints as the gripper opens, so both bones' realised lengths track
  gripper actuation too, not arm flex — despite non-zero rest length.
- **Measured (reproduced).** A perfectly rigid, motionless arm holding its
  gripper **open** for every frame scores `rigidity_residual_mm ≈ 15.37`
  (`pytest.approx(15.37, rel=0.02)`); **closed**, `0.00`; a `0→1` opening
  **ramp** over the clip, `≈7.2` (`pytest.approx(7.2, rel=0.10)` — roughly
  half the held-open figure, because the residual is a mean over frames and a
  ramp averages the contamination over its intermediate openings — asserting
  15.37 against a ramp would be comparing two different quantities, per the
  test's own docstring). With the fix (`bone_set="rigid"`), all three cases
  are `pytest.approx(0.0, abs=1e-6)`.
- **Is.** `RobotSpec` (`core/robot.py`) exposes both `bone_pairs`/`bone_lengths`
  (the full legacy set) and `rigid_bone_pairs`/`rigid_bone_lengths` (degenerate
  bones dropped) — `rigidity_residual_mm`/`rigidity_wobble_mm` metrics use
  `rigid_bone_pairs` by default (`bone_set="rigid"`); `bone_set="all"` exists
  purely to reproduce the legacy (contaminated) numbers, registered as
  `rigidity_residual_all_mm`/`rigidity_wobble_all_mm`. `FrankaSpec._rigid_bone_mask`
  applies two independent rules, both of which a bone must pass: (1)
  **structural** — neither endpoint is in `ACTUATED_LINKS` (for the Panda,
  exactly the two prismatic fingers); this is read off the kinematic chain
  and **transfers to any robot**; (2) **degenerate-length** — rest length
  exceeds `DEGENERATE_BONE_M` (1 mm), the general safety net in `core/robot.py`
  that catches a future robot's coincident-keypoint bone even if rule (1)
  doesn't apply. On the Panda, rule (1) alone drops bones 4, 5, 6; rule (2)
  alone would drop only bone 5 — the *intersection* is what makes the
  exclusion principled rather than a distance threshold tuned to one robot
  (a distance cutoff between 0.0584 m and 0.088 m only separates the two on
  this robot by coincidence). `FrankaSpec.__init__` warns, by name, for every
  bone `rigid_bone_pairs` drops.
- **Why.** Grasping clips were penalised for grasping — the exact opposite of
  what a rigidity residual is supposed to detect.
- **Test that pins it.** `tests/test_rigidity_gripper_contamination.py` (two
  tiers: a hand-built `FakeRobot` isolating the mechanism, CPU-only, always
  runs; the literal Franka geometry reproducing the measured 15.37/7.2 mm
  figures, `ckpt`/URDF-gated), `tests/test_robot_degenerate_bones.py`,
  `tests/test_rigidity_dt_free.py`, `tests/test_fk_rest_pose.py`.
- **Do old artifacts still load?** N/A — this is a metric-computation choice
  (which bones to include), not a persisted file format; `bone_set="all"`
  reproduces legacy numbers exactly for any FK output, old or new.

## (c) Not ported, and why

- **Source D** (`Marionette/models/evaluation/`) — considered and rejected:
  it is missing `__init__.py` and `consistency_metrics.py`, and **cannot even
  be imported**, let alone hashed or golden-tested against.
- **The diffusers/SVD latent path** — `PixelPhysicsJudge.encode_latent` /
  `.encode_latent_grad` in `judge/pixel_judge.py` decoded chunks of a
  duck-typed Stable-Video-Diffusion `vae` latent (`(T,4,24,40)`) to pixels
  before running DINO. Deleted, not adapted: kinescore's contract is frames
  in, joints out (`core/reader.py`), which is what makes the benchmark
  model-agnostic — "no `diffusers` dependency, not even optional" is a
  `pyproject.toml`-level guarantee (`backbones/dino.py`'s docstring makes the
  same point explicitly), and the video-file contract (`ffprobe` + a decoder)
  supersedes needing a latent path at all.
- **`GeometryReferee` / `geometry_head` / `camera_solver`** — need a depth
  stack (multi-view or monocular depth estimation) that this benchmark does
  not have; a genuinely separate axis of physical plausibility (camera/scene
  geometry, not robot kinematics) left for a future package rather than
  half-ported here.
- **~58 numbered experiment driver scripts** in source A (`scripts/*.py`,
  the dirty-file list in `provenance/git_state.json` gives a sense of the
  volume) — one-off training/analysis entry points, not benchmark logic.
- **`configs/*.yaml`** — stale, machine-specific paths (dataset roots,
  checkpoint directories on the original authors' hosts); superseded by
  `kinescore.paths`'s environment-variable resolution
  (`KINESCORE_ASSETS`/`KINESCORE_CKPT_DIR`/etc., no hardcoded fallback paths
  at all — see `docs/ADDING_A_ROBOT.md`'s asset policy and `paths.py`'s own
  docstring for the class of bug a baked-in default path causes).

## (d) A scoped equivalence claim

This document deliberately does **not** claim "byte-for-byte with the
sources" as a blanket statement — that claim would be both false (most of the
codebase is adapted or rewritten, on purpose, to fix the defects in (b)) and,
where it *is* true, unverifiable without a mechanical check. Instead:

- **AST-identical, verified by `tools/ast_diff.py`.** `tools/ast_diff.py`
  parses two files, collects every `FunctionDef`/`AsyncFunctionDef` by
  dotted qualified name, strips each function's leading docstring, and
  compares the `ast.unparse`'d remainder — i.e. it proves the *executable
  body* is unchanged modulo docstrings/comments/import lines, function by
  function, not just "the file looks similar". Its own `--self-test` mode
  checks `A judge/fk.py` against `B models/kinematics/fk.py` (a sibling file
  in B, byte-identical to A's `judge/fk.py` per `copy_manifest.tsv`'s note)
  and asserts zero differences, as a sanity check on the tool itself. This is
  the mechanical backing for the **verbatim** rows in (a1): #1 (`fk.py`), the
  arithmetic-level claims inside #2/#3/#7 (individual ported functions, not
  the containing modules, since those were restructured), and #6 (`gr1_fk.py`,
  modulo the one documented rename).
- **Numerically pinned, verified by golden fixtures.** `tools/gen_golden.py`
  freezes the *old* judge's numeric output (not its source bytes) into
  `tests/golden/*.npz`, generated by importing and executing the actual
  source repositories under a pinned seed (`torch.manual_seed(0)`,
  `torch.use_deterministic_algorithms(True)`, fp32, CPU-only — see its own
  module docstring for the full determinism contract). `tests/golden/MANIFEST.json`
  records, per fixture: which source qualnames it exercised, which source
  files' SHA-256 it was generated against, and the RNG seed(s) used. Where a
  golden fixture's inputs are re-run through `src/kinescore` (`tools/gen_golden.py
  --diff`, via `_DIFF_ADAPTERS`) and match, that is the numeric analogue of an
  AST diff — for behaviour that intentionally changed (D1–D9), the comparison
  is instead against the *fixture's own reproduced-legacy-number* assertions
  inside the relevant test file (e.g. `test_franka_reproduces_measured_15_37mm`),
  not against `--diff`'s legacy-vs-new table, which only checks
  currently-registered adapters (`_DIFF_ADAPTERS` is empty as shipped in this
  script — see [REGENERATING_GOLDENS.md](REGENERATING_GOLDENS.md)).
- **Everything else differs exactly as enumerated in (b) and (a1)'s
  per-file notes** — restructured into `MetricSpec`/`RobotSpec`/`PoseReader`
  classes, retargeted at `q_raw`, given a default `bone_set="rigid"`, given a
  `dt` with no default, etc. Where a fix changes a reported number, both the
  legacy and corrected values remain reachable (`bone_set="all"`,
  `legacy=True`, the golden fixtures) specifically so a reviewer can see which
  published numbers moved and why, rather than the fix silently overwriting
  the old behaviour with no way back to compare against it.
