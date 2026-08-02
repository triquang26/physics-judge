# Architecture

The OOP map of the package, and how to add a robot, a metric, or a benchmark
data source. This is the "how the pieces fit" doc; for the CLI-level "how do I
train/run something" doc see [TRAINING.md](TRAINING.md) and
[BENCHMARKING.md](BENCHMARKING.md).

## Layering

```
core/        the five extension-point contracts, no dependency on bench/ or cli/
robots/      RobotSpec implementations, registered lazily
readers/     PoseReader implementations (checkpoint -> Readout)
heads/       the neural nets a PoseReader wraps
metrics/     Metric implementations + suites
training/    cache-building + head-training, reads/writes checkpoints
bench/       the benchmark matrix: layout, ingest, scoring pipeline, stats
cli/         one file per subcommand, thin argparse wrappers over the above
```

`core` is lower-level than `bench` by design (`core/contracts.py`'s own
docstring states this explicitly): `bench`-side types (`Cell`, `BenchConfig`,
`SourcePlugin`) are imported into `core/contracts.py` only under
`TYPE_CHECKING`, so a real runtime import the other way would be circular.
Nothing in `core/` or `robots/` imports `torch` at *module* scope unless the
attribute genuinely needs a tensor type — `core/registry.py` and
`core/contracts.py`'s `ClipSource`/`DataLayout` ABCs are torch-free so a
CPU-only, network-free test run stays fast.

## The five contracts (`core/contracts.py`)

One module re-exports (or defines) every extension-point interface, so adding
a new robot/reader/metric/source/layout is "implement this one protocol",
never "find the right dict to append to by grep."

| contract | home module | shape |
|---|---|---|
| `RobotSpec` | `core/robot.py` | `name, n_joints, keypoint_links, bone_pairs, bone_lengths, rigid_bone_pairs/lengths, q_lo, q_hi, vel_limits, effort_limits, capabilities, urdf_sha256` + `forward_kinematics(q, aux) -> P`, `forward_transforms(q, aux) -> (P, R)`, `ee_sites()` |
| `PoseReader` | `core/reader.py` | `limit_semantics, view_layout, robot_name, reader_id` + `.read(frames) -> Readout` |
| `Metric` | `core/metric.py` | a `MetricSpec` (key/units/dt_exponent/direction/requires/...) + `_compute(ctx) -> float`, wrapped by `SafeMetric` |
| `ClipSource` | `core/contracts.py` (mirrors `bench/sources/base.py`) | `GENERATOR: ClassVar[str]` + `make_plugin(cell, data_root, config) -> SourcePlugin` |
| `DataLayout` | `core/contracts.py` (mirrors `bench/layout.py`) | `cells()`, `cell_dir(cell)`, `validate() -> list[str]` |

## `Registry[T]` (`core/registry.py`)

One generic class, reused by every axis that used to hand-roll its own
`dict[str, Callable]` + `get_x`/`available_x` pair. `Registry(kind).register(name,
factory)` stores a **zero-arg factory**, never a constructed instance and
never an eagerly-imported class — `Registry.get(name)` is what actually calls
the factory. This is what lets `robots.available_robots()` list
`"franka_panda"` without importing `pytorch_kinematics`, and what keeps a test
run that only ever asks for `"synthetic_2r"` from ever triggering a heavy,
optional-in-spirit dependency it doesn't need. `robots/__init__.py` is the
worked example every other registry in this codebase follows.

## `robots/` — 5 registered `RobotSpec`s

```python
"franka_panda"    -> robots/franka/spec.py::FrankaSpec       (lazy)
"fourier_gr1"     -> robots/gr1/spec.py::GR1Spec              (lazy)
"airbot_mmk2"     -> robots/airbot_mmk2/spec.py::AirbotMMK2Spec (lazy)
"aloha_bimanual"  -> robots/aloha/spec.py::AlohaSpec           (lazy)
"synthetic_2r"    -> robots/synthetic/spec.py::Synthetic2R     (eager -- no pytorch_kinematics dependency)
```

Every real robot (not `synthetic_2r`) follows the same shape: an FK module
(`fk.py`, differentiable forward kinematics, ported from a source research
repo where noted) + a `constants.py` (URDF joint/link names, verified against
real logged data, not assumed from a published convention) + `spec.py` (the
`RobotSpec` adapter). Bimanual robots (`GR1Spec`, `AlohaSpec`) concatenate two
per-arm chains: `keypoint_links = KEYPOINTS_LEFT + KEYPOINTS_RIGHT`,
`bone_pairs = cat([left, right + len(KEYPOINTS_LEFT)])`.

### Adding a robot

1. Verify the joint layout against **real logged data**, not the published
   convention alone — `robots/aloha/constants.py`'s module docstring is the
   template: cross-check `meta/modality.json` (if the dataset ships one)
   against a raw parquet read, and cross-check `meta/stats.json`'s per-column
   min/max against the URDF's own `<limit>` values, before trusting a slice.
2. Write `fk.py` (differentiable, `(B,T,n_joints) -> (B,T,K,3)` keypoints;
   `(B,T,n_joints) -> (P, R)` for full transforms) and `constants.py`.
3. Write `spec.py`'s class: `name` (the registry key — this is the ONE naming
   system; do not also invent a second name for the same robot elsewhere),
   `n_joints` (the count actually **predicted**, which may be less than the
   URDF's full DOF — e.g. `GR1Spec.n_joints == 17` against a 54-joint URDF,
   because the recording setup was upper-body-only teleop; see
   `robots/gr1/spec.py`'s module docstring for what that implies for every
   metric touching legs/feet), `q_lo`/`q_hi`/`vel_limits`/`effort_limits`
   (read straight off the URDF; `None` for whichever the URDF doesn't
   declare — never fabricate a limit), `capabilities` (a `frozenset[str]` of
   `Capability.{ROTATIONS,COLLIDERS,SUPPORT_POLYGON,EFFORT_LIMITS}` — declare
   **only** what you actually back with real data; a declared-but-unbacked
   capability is worse than an undeclared one, since a metric would trust it).
4. **The D9 rigid-bone rule.** A "bone" (consecutive keypoint pair) whose far
   endpoint is driven by something other than a predicted joint — a gripper
   finger, most commonly — must be excluded from `rigid_bone_pairs`, or
   `rigidity_residual_mm`/`rigidity_wobble_mm` silently measure gripper
   actuation instead of arm rigidity (a perfectly rigid, motionless Franka
   arm scored 15.37mm purely from an opening gripper, before this rule
   existed — see legacy_docs/PROVENANCE.md D9). `robots/base.py::
   structural_rigid_bone_mask` is the shared implementation; every real robot
   in this repo calls it with its own `ACTUATED_LINKS` set. A second,
   independent guard (`DEGENERATE_BONE_M = 1e-3`,
   `core/robot.py::rigid_bone_mask`) drops any bone whose rest length is
   near-zero (two links that coincide at the URDF's rest pose) regardless of
   whether it's gripper-driven.
5. Register the factory in `robots/__init__.py`, lazy-importing inside the
   factory function (never at module scope) so `available_robots()` stays
   cheap. `bench/config.py`'s `AxesConfig.robot` and `RobotConfig.spec` are
   both validated against `robots.available_robots()` at config-load time, so
   a typo'd or unregistered robot name fails at `kinescore bench run`'s
   config-parse step, not three cells into a run.
6. If the robot is meant to be **scored** (not just trained a reader for),
   add it to `configs/robot_map.yaml` (`embodiment` + `generators` — see
   [DATA_PREP.md](DATA_PREP.md)) and to a `configs/*.yaml` benchmark config's
   `axes.robot` + `robots:` reader-pin table.

Registration goes through `core/registry.py::Registry`; the per-robot
checklist items (D9 rigid-bone exclusion, capability declarations, keypoint
ordering) are the parts that actually matter and are unchanged.

## `readers/` and `heads/` — one reader family

`readers/loader.py::load_reader(path, *, robot, view_layout, device, ...)` is
the single entry point. It reads a checkpoint's `cfg` dict, and if
`checkpoint_v2.is_readout_v2_cfg(cfg)` is true, routes to
`checkpoint_v2.load_reader` (the **only** working path). Anything else raises
`NotImplementedError` naming the file — a squashed-head checkpoint
(`AttentivePoseHead`/`SquashedPoseReader`, the legacy `judge_v3l` format) is
*detected* so the error is legible, but there is no loader for it any more:
that whole path (`heads/attentive.py`, `heads/mlp.py`, `heads/disentangled.py`,
`readers/ensemble.py`, `training/trainer.py`, the `kinescore train`
subcommand) was **removed**, not merely deprecated — see
[legacy_docs/PROVENANCE.md](../legacy_docs/PROVENANCE.md)'s "D7 addendum" for
why. `core/reader.py::LimitSemantics` is now `Literal["raw_rad"]`, a
single-value type, which is the type-level expression of the same fact.

The one live head is `heads/heteroscedastic.py::ReadoutV2Head`
(`d_model`/`n_heads`/`temporal_nhead`/... — see `cmd_train_rawrad.py`'s flags)
paired with `heads/ranges.py::clamp_for_fk` (produces the safe `q` every
`RobotSpec.forward_kinematics` call gets) and, upstream of the clamp, an
unclamped `q_raw` — the field that makes `limit_violation_frac`/
`limit_excess_rad` observable at all (see [METRICS.md](METRICS.md)'s
`joint_limits` section). `Readout` (`core/reader.py`) carries both:
`q` (always safe for FK), `q_raw` (`Optional`, `None` for a reader that
doesn't expose it), `sigma` (per-joint aleatoric std, if the head is
heteroscedastic), `aux`, `extras`.

## `metrics/` — `SafeMetric` + the registry

Every metric subclasses `metrics/_base.py::SafeMetric`, which demotes any
non-finite `_compute` result to `MetricValue.unavailable(key,
"degenerate_input:non_finite_result")` instead of letting a bare `NaN` escape
uncaught. `core/metric.py::REGISTRY` is a flat `dict[str, Metric]`;
`register(metric)` raises on a duplicate key (no silent shadowing).
`MetricContext.available()` is what capability-gating reads:
`"colliders" in ctx.available()` iff `Capability.COLLIDERS in ctx.robot.
capabilities`, same shape for `"support_polygon"`. A metric declares
`requires: frozenset[str]` (from `Requires = Literal["P","R","q","q_raw",
"colliders","support_polygon"]`); `BaseMetric.compute()` checks that against
`ctx.available()` **before** calling `_compute`, so a metric body never has to
defensively check its own inputs exist.

`metrics/suites.py::MetricSuite(name, metrics, invariant_keys)` computes
`suite_id` as a `sha256` hash of the suite's name + its metrics' declared
output keys + its sorted `invariant_keys` — adding, removing, or renaming a
metric changes the hash, which is the mechanism that makes "two runs share a
`suite_id`" mean "directly comparable," enforced by
`bench/store.py::assert_uniform_schema` and `bench/stats.py::aggregate`'s
refusal to pool mismatched `suite_id`s without `--allow-mixed-suites`.

### Adding a metric

1. Pick the physical quantity and its `dt_exponent` — the power of `1/dt` the
   value scales by under a finite-difference derivative (position=0,
   velocity=1, acceleration=2, jerk=3). If the formula thresholds a
   `dt`-dependent quantity against a **fixed constant** (a violation
   fraction) or mixes terms with different `dt`-exponents (`total_energy_tstd`
   sums a `dt^-2` kinetic term with a `dt^0` potential term), the exponent is
   `None`, not a guess — see [BENCHMARKING.md](BENCHMARKING.md)'s rate-policy
   section for why this distinction is load-bearing, not decorative.
2. Subclass `SafeMetric`, declare a `MetricSpec` (key, units, `dt_exponent`,
   `direction`, `requires`, `min_frames`, `unobservable_when`). Implement
   `_compute(ctx) -> float`; let it raise or return non-finite on a genuine
   failure — `SafeMetric`/`BaseMetric.compute()` turn that into a
   `MetricValue.unavailable(...)` with a reason, you never construct that
   value by hand inside `_compute`.
3. `register(YourMetric())` at import time (see the bottom of any
   `metrics/*.py` module for the pattern — registration happens on import
   regardless of what names a module's `__all__` exports, which is why
   `rigidity.py` registers 6 variants even though `__init__.py` only names
   2).
4. `tests/test_metric_registry_conformance.py` numerically verifies every
   declared `dt_exponent` by scoring the same clip at two different `dt`s and
   checking the ratio — a wrong declaration fails CI, it does not silently
   ship.
5. Add it to a suite in `metrics/suites.py` if it should be scored by
   default. **Never add to `INVARIANT_V1`** — that suite is frozen (its
   `suite_id` is what every prior published number was computed under);
   `ALL_METRICS` is the suite new metrics join.

## `bench/` — the benchmark matrix

`Cell` (`bench/cell.py`) is keyed by **`robot`, not `embodiment`**:
`embodiment` (`humanoid`/`single_arm`/`bimanual`, the on-disk directory name)
is a derived, recorded field, because one embodiment directory can hold clips
from more than one physical robot depending on which generator wrote them —
`dense/humanoid/.../multiview/ctrlworld/` is Airbot MMK2, but
`dense/humanoid/.../dreamgen|dreamdojo/` is Fourier GR-1, a completely
different kinematic tree. `configs/robot_map.yaml` is the table that resolves
`(embodiment, generator) -> robot`; see [DATA_PREP.md](DATA_PREP.md).

| module | job |
|---|---|
| `matrix.py` | The only module that knows the five-axis grid (`robot x view x horizon x cache x generator`) exists. `expand()`/`na_cells()`/`allow_patterns()`. |
| `layout.py` | `RawHFLayout` (reads a dataset exactly as HF ships it, one of three coexisting on-disk shapes per generator — `episode_dir`/`task_episode`/`flat_or_dir`, see `configs/data_spec.yaml`) and `CanonicalLayout` (the ingested target shape, one `cell_card.json` per cell). |
| `ingest.py` / `verify.py` | `kinescore data ingest` (symlink raw -> canonical, one probe per cell) and `kinescore data verify` (check **every** clip's width/height/fps/frame-count against `configs/data_spec.yaml`, plus broken symlinks). |
| `robot_map.py` / `data_spec.py` | Load `configs/robot_map.yaml` / `configs/data_spec.yaml` — the data-driven contract `ingest`/`verify` run against, deliberately independent of any one `benchmark.yaml`. |
| `manifest.py` | `DiscoveredClip` -> probed, paired manifest rows. A `SourcePlugin` is a zero-arg callable yielding clips — adding a data source is a new plugin, not a new `if family == ...` branch. |
| `sources/*.py` | One `ClipSource` class per generator (`ctrlworld.py`, `dreamgen.py`, `dreamdojo.py`), each implementing `make_plugin(cell, data_root, config)`. |
| `runner.py` | `run(rows, scorer, results_path, ...)` — a clip that fails to decode/score is recorded via `store.failed_record` and the loop continues; a benchmark run does not die on the first corrupt file. |
| `store.py` | `ResultsStore` (append-only `results.jsonl` reader/writer), `flatten()` (the one dotted-key flattener), `assert_uniform_schema`. See [SCHEMA reference in legacy_docs/SCHEMA.md](../legacy_docs/SCHEMA.md) for the full record shape. |
| `stats.py` / `separation.py` | Paired statistics (bootstrap CI, Wilcoxon) and unpaired separation (AUROC, Cliff's delta) — two different questions ("did generated drift from its own GT" vs "can real and generated be told apart at all"), computed and reported separately, never conflated into one number. |
| `csv_export.py` / `traces.py` | `kinescore export` (one CSV per benchmark cell, mirroring the input tree) and `kinescore score --traces` (per-frame arrays in `traces.npz`, kept out of `results.jsonl` because that record must stay flat/scalar). |
| `noise_floor.py` | `kinescore bench noise-floor` — the paired re-encode null-delta, the honest ruler for "is this paired gap bigger than compression noise." |

## `training/` — cache once, train fast

`CacheBuilder` (`training/cache.py`) runs a frozen DINOv3 backbone over every
`{split}/{episode}.mp4` under a video root and writes a self-describing
`.pt` per episode (`CacheHeader`: view layout, backbone id, token shape —
the guard against a 3-view cache silently feeding a 1-view head). This is
what `kinescore cache` wraps. `training/datasets.py::load_split` then
RAM-flattens every cached episode of a split into one `(N, n_tokens, D)`
tensor — the fast path for data that fits in memory, with no preserved
episode boundary. `training/splits.py::stratified_episode_split` is the
alternative to a pre-split `train/`/`val/` layout: given one pool directory,
it groups episode ids by a scene key (default: strip a trailing episode
number) and assigns whole scenes to val, smallest-first, so a val set never
leaks the same scene the model trained on — see
[TRAINING.md](TRAINING.md#the-mm-acceptance-gate) for the measured 18.74mm
train / 162.10mm val gap an un-stratified split produced.

`RawRadTrainer`/`train_head_rawrad` (`training/trainer_rawrad.py`) is the
**only** head-training loop left (the squashed `training/trainer.py` loop was
removed with the rest of the squashed path). `kinescore train-rawrad` wraps
it. See [TRAINING.md](TRAINING.md) for the two-phase loss recipe and the
acceptance gate.

## `cli/` — one file per subcommand

Every `cli/cmd_*.py` exposes exactly `add_arguments(parser)` (import-light,
runs even under `--help`) and `run(args) -> int` (all heavy imports —
`torch`, the backbone, `pytorch_kinematics` — happen inside `run`, never at
module scope). `cli/main.py` registers each as a top-level `kinescore <name>`
subcommand; `data`, `bench`, `anchor`, `reference` additionally nest their own
action subparser (`kinescore data ingest`, `kinescore bench noise-floor`,
...). See [TRAINING.md](TRAINING.md) and [BENCHMARKING.md](BENCHMARKING.md)
for the two command chains this wiring supports end to end.
