# Changelog

All notable changes to `kinescore` are documented here. This project does not
yet have prior published releases — `0.1.0` is the initial standalone
extraction from two research codebases (`Marionette-ciasc`, `Marionette-fkjepa`),
merged and fixed as one benchmark. See
[legacy_docs/PROVENANCE.md](legacy_docs/PROVENANCE.md) for the full
defect-by-defect writeup (was/is/why/test/legacy-compat) behind every entry
below.

## Unreleased

### Docs and scripts reorganised

- `docs/` cut down to six current files (`README.md` at the repo root plus
  `docs/{DATA_PREP,TRAINING,BENCHMARKING,ARCHITECTURE,METRICS}.md`), written
  fresh against the post-refactor code: `core/registry.py`'s `Registry[T]`,
  `core/contracts.py`'s five extension points, the `axes.robot` (not
  `axes.embodiment`) benchmark matrix, `aloha_bimanual` as a fourth
  registered robot, and the D7-addendum removal of the squashed pose-reader
  path. Everything else (`PROVENANCE`, `DECISIONS`, `RATE_POLICY`, `SCHEMA`,
  `ADDING_A_ROBOT`, `ADDING_A_METRIC`, `ADDING_ALOHA_NOTES`, `MODIFYING`,
  `REGENERATING_GOLDENS`, plus the now-superseded `DATA_LAYOUT`/`USAGE`)
  moved verbatim to `legacy_docs/` — a decision record, not usage docs (see
  `legacy_docs/README.md`).
- `scripts/` created: the ad-hoc Airbot MMK2 cache/train runbooks that were
  living outside the repo (`kinescore_runtime/*.sh`) were copied in, renamed
  from opaque `run_cache_camhigh_A.sh`-style names to
  `<verb>_<robot>_<dataset>_<variant>.sh`, and given a What/Why/Input/Output
  header — the rename surfaced that "camhigh A/B" is not two physical
  cameras but one camera spelled two ways (`cam_head_rgb` vs `cam_high_rgb`)
  across two source-dataset naming conventions, and that the squashed-head
  training script among them calls a CLI subcommand (`kinescore train`) that
  no longer exists. `scripts/convert_lerobot_to_train.py` is new: a single
  LeRobot v1/v2 -> kinescore-training-contract converter covering all three
  robots with training data (`fourier_gr1`, `airbot_mmk2`, `aloha_bimanual`),
  verified against real on-disk data for each.
- `tools/check_repo_hygiene.py` gained a fifth check
  (`check_scripts_have_header`) enforcing the header above on every file in
  `scripts/`, with `tests/test_repo_hygiene.py` coverage for both the
  positive and negative case.

## 0.1.0

### Fixed (defects inherited from the source codebases)

- **D1 — `dt` hardcoded by omission.** The source's `PhysicsConsistency`/
  `PhysicsReferee` defaulted `dt=0.2` silently; two call sites never passed
  one, while a training loader decimated frames at a random stride of 1 or
  2 — roughly half the corpus was scored at a `dt` wrong by 2x, inflating
  speed 2x, acceleration 4x, jerk 8x, and driving `accel_violation_frac` from
  `0.000` to `0.3875` on a representative trajectory. `dt` now lives on
  `ClipSpec` as a required, keyword-only field with no default;
  `ClipSpec.subsample(k)` is the only supported decimation path.
- **D2 — reference did not serialize `dt`.** Baselines and quantiles scale
  as `1/dt^n`; a reference built at one frame rate could silently score a run
  at another. Reference schema 2 persists `dt`; loading a pre-D2 file raises
  unless `dt=` is passed explicitly; scoring against a mismatched rate raises
  `RateMismatchError` unless the caller opts in.
- **D3 — PIS averaged over a varying number of terms.** A clip scored with
  joint angles produced a Physical Invariance Score averaged over 9–10 terms;
  one scored from keypoints alone, over 6 — both reported and compared as the
  same number. The PIS term set is now fixed at reference-build time from the
  scoring suite's declared `invariant_keys`; a missing declared term makes
  the whole score `NaN` with a reason by default, never a silent shrink.
- **D3 (manifest variant) — a config table silently overrode the probed
  frame rate.** The source's manifest builder probed every clip with
  `ffprobe`, then discarded the probed `fps` in favour of a hand-maintained
  per-family lookup table. The probe and any table/CLI override are now
  always cross-checked; disagreement beyond tolerance is a hard error naming
  the file and both values.
- **D3b — zero-baseline PIS blowup.** A near-zero real-data baseline
  collapsed the PIS denominator to the source's fixed `1e-8` epsilon,
  saturating that key's score to `1.0` and dominating the aggregate
  regardless of how physical the rest of the rollout was. The denominator now
  floors at a per-key, data-fitted median-absolute-deviation (or a unit-aware
  default), with a verified numerical no-op for any well-conditioned key.
- **D4 — multiview cache/score desync with no assertion.** The only guard
  (`N % n_cams != 0`) lived inside the multi-camera branch, so the
  overwhelmingly common single-camera case had no check at all, and a 1-view
  head could silently consume a 3-view feature grid. Camera layout is now a
  first-class, self-describing `ViewLayout` asserted in both directions,
  unconditionally, before any camera-count branching.
- **D4b — reference's distributional-comparison key set depended on
  iteration order.** Whichever motion quantities the first real rollout
  happened to carry became the reference's permanent quantity set; a later
  rollout missing one raised a bare `KeyError` deep inside a stacking call.
  The quantity set is now a declared property of the scoring suite, validated
  against every rollout up front with a single message naming every offender.
- **D5 — checkpoint format dropped `hidden`/`dropout`.** A checkpoint trained
  with a non-default hidden width could never be reloaded at all. Both fields
  are now persisted; loading an old file without them infers `hidden` from
  the saved weights' own shape (cross-checked against the cfg, never silently
  trusted); `n_cams` absence in a legacy cfg still defaults to `1` exactly as
  before, preserving load compatibility for all three real production
  checkpoints.
- **D7 — `limit_violation` structurally always zero.** A sigmoid-squashed
  pose head cannot exceed its joint limits by construction, so this metric
  read "perfect" for every clip ever scored with such a head — every real
  production checkpoint — while measuring nothing about the video. Joint-limit
  metrics now require the reader's unsquashed output and are explicitly
  declared unobservable (reported as `null`, never `0.0`) whenever the reader
  architecture makes that structurally impossible; a headroom-to-limit metric
  that *is* observable under a squashed head is added alongside.
- **D9 (found during this migration) — gripper actuation contaminated
  rigidity.** The Franka's consecutive-keypoint bone chain includes a
  finger-to-finger bone with an exactly-zero rest length, plus two
  neighbouring bones that also track the prismatic finger joint. A perfectly
  rigid, motionless arm merely opening its gripper measured a rigidity
  residual of ~15.37 mm (0.00 mm closed; ~7.2 mm for a 0→1 opening ramp) —
  grasping clips were penalised for grasping. Rigidity metrics now exclude
  any bone with an actuated (non-arm-joint-driven) endpoint by default, with
  a degenerate-rest-length check as a second, independent net; the legacy,
  contaminated numbers remain reachable via an explicit `bone_set="all"` for
  provenance comparison.

### Added

- A frozen `Metric`/`MetricSpec`/`MetricContext`/`MetricValue` contract
  (`kinescore.core.metric`) under which every metric declares its units,
  `dt` exponent, required inputs, and unobservability conditions —
  numerically verified per-metric by an auto-parametrized conformance test.
- A frozen `RobotSpec` protocol (`kinescore.core.robot`) with two shipped
  implementations (Franka Panda, Fourier GR-1) plus a dependency-free
  synthetic 2-link arm for CPU-only metric testing.
- A canonical, append-only result schema (`kinescore.bench.store`) with a
  single flattener, a static per-`suite_id` key-set guarantee, and explicit
  `ok`/`failed` status recording — no more incompatible writer formats.
- `kinescore.reference` — a real-motion fingerprint (PIS, per-quantity W1,
  Kinematic Fréchet Distance) pinned to a frame rate and a suite, with an
  honestly-named `kfd_approx` (an eigendecomposition approximation) alongside
  an exact `kfd` path.
- 28 registered metrics across rigidity, temporal, angular, energy,
  joint-limit, joint-dynamics, smoothness, and mechanical-feasibility
  families, documented in full (formula, units, `dt` exponent, detects,
  does-NOT-detect) in `docs/METRICS.md`.

### Removed

- The `diffusers`/Stable-Video-Diffusion latent-decode path
  (`encode_latent`/`encode_latent_grad`) — deleted, not adapted. The input
  contract is video files; anything that can write an mp4 can be scored
  without this package knowing how it was generated, and there is no
  `diffusers` dependency anywhere, not even optional.
