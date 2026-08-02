# Regenerating goldens

## What a golden fixture proves

`tools/source_snapshot.py` proves which *bytes* a port started from (the
SHA-256 ledger in `provenance/sources.sha256` — see
[PROVENANCE.md](PROVENANCE.md)). `tools/gen_golden.py` proves which
*numbers* those bytes produced: "for a fixed seed, a fixed input, this exact
FK chain and this exact head, here is the output, to fp32 precision,
forever" (its own module docstring). Once `src/kinescore` exists and starts
diverging from a straight port — bug fixes, refactors, degenerate-bone
handling — there is no other way to tell "we fixed a defect" from "we
introduced one" than diffing against what the defective original *actually
did*. **You cannot fix a bug you cannot reproduce.**

## The governing rule

> Every intentional behaviour change ships **both** a `*_legacy` and a
> `*_v2` fixture (or the equivalent paired assertion), **plus** a
> [PROVENANCE.md](PROVENANCE.md) row. A golden that differs with no paired
> fixture means the port is wrong, not the golden stale.

Concretely, this repo does not literally name files `foo_legacy.npz` /
`foo_v2.npz` — the pairing takes whichever of three shapes fits the defect:

1. **One golden `.npz` (the frozen legacy numbers) + a `legacy=` /
   `bone_set="all"`-style parameter that reproduces them exactly, checked in
   a dedicated test.** This is the dominant pattern:
   - `reference/normalize.py::invariance_score(..., legacy=True)` — a
     byte-for-byte replica of the source's `invariance_score` body (defects
     D3, D3b included), used **only** for regression comparison. Pinned by
     `tests/test_pis_term_set.py::test_legacy_replica_reproduces_the_varying_term_count`
     and `tests/test_reference_golden.py::test_legacy_replica_reproduces_source_pis_scores_and_n_terms`
     (the latter against the real `tests/golden/golden_reference.npz`).
   - `metrics/rigidity.py`'s `bone_set="all"` — reproduces the legacy,
     gripper-contaminated `rigidity_residual_mm`/`rigidity_wobble_mm` exactly
     (registered under the `_all_mm`-suffixed keys, so both the fixed and the
     legacy numbers are simultaneously reachable from a live scoring run, not
     just from a frozen fixture). Pinned by
     `tests/test_rigidity_gripper_contamination.py`, which asserts the
     *literal measured figures* (15.37 mm held-open, 7.2 mm ramped, 0.00 mm
     closed) against both the fixed and legacy paths.
2. **The frozen source-numbers `.npz` itself, generated from the actual old
   codebase, compared against a synthetic reproduction of the same defect
   inside a test.** `tests/golden/golden_reference.npz` (via
   `tools/gen_golden.py::golden_reference`) freezes
   `RealMotionReference.invariance_score`/`.w1`/`.kfd` output from the *real*
   source (`Marionette-fkjepa`), including a documented asymmetry
   (`case2_missing_q`'s `kfd_eligible=False`, because `.kfd` raises
   `KeyError` on a rollout missing a declared quantity key while `.w1`
   silently skips it — see the generator's own inline comment, "this
   asymmetry is exactly the kind of behaviour a straight port must
   preserve"). There is currently no `_DIFF_ADAPTERS` entry
   (`tools/gen_golden.py`) wiring this fixture into an automated `--diff`
   comparison against `src/kinescore` — see the gap noted below.
3. **A negative-control test with no `.npz` at all**, for defects whose "old
   behaviour" is best demonstrated by construction rather than frozen output:
   `tests/test_dt_invariance.py::test_wrong_dt_is_detected` builds a
   trajectory from scratch (via `_euler_integrate_from_accel`, the exact
   discrete inverse of the acceleration finite-difference) such that scoring
   it at the true `dt` vs. a `dt` wrong by 2x reproduces the measured D1
   figures (`accel_violation_frac`: `0.000 → 0.3875`) to `rtol=1e-5`, with no
   dependency on either source checkout being reachable.

Whichever shape fits, the **PROVENANCE.md row is mandatory regardless** —
every defect subsection in [PROVENANCE.md §(b)](PROVENANCE.md#b-intentional-behaviour-changes)
names its "test that pins it" explicitly, and a new behaviour change without
a corresponding row there is exactly the "golden differs, no paired fixture"
failure mode this rule exists to prevent.

## `tools/gen_golden.py`

Generates `tests/golden/*.npz` **by importing and executing the actual
source repositories** (not `src/kinescore`) under a pinned determinism
contract — read its module docstring in full before regenerating anything:

- `torch.manual_seed(0)` before every random draw; each `golden_*` generator
  reseeds independently (`_seed_everything`), so the *order* generators run
  in cannot change any fixture's values.
- fp32 throughout, no autocast, no fp16 — even though the source's own DINO
  encode path uses fp16, because every fixture here starts from the FK/head
  layer, which the source itself always runs outside autocast.
- `torch.use_deterministic_algorithms(True)` where supported (matmul/FK on
  CPU honour it; a handful of reduction ops only warn under nondeterminism
  rather than erroring, and are left alone rather than papered over as a hard
  failure).
- CPU only. No network, no GPU, no live HF Hub.

```bash
python tools/gen_golden.py --source-a PATH_TO_MARIONETTE_CIASC \
                            --source-b PATH_TO_MARIONETTE_FKJEPA \
                            --out-dir tests/golden/
```

`--only golden_fk golden_physics ...` regenerates a subset (see
`GENERATORS` for the full list: `golden_fk`, `golden_physics`, `golden_head`,
`golden_reference`, `golden_gr1_fk`, `golden_ckpt_head`. `golden_predict_pose`
was retired along with the squashed pose-reader path it validated — see
[PROVENANCE.md](PROVENANCE.md#d7--limit_violation-structurally-always-0)'s
D7 addendum — its generator function and `.npz` output are both gone, not
merely unused). Total output is asserted under a 4 MiB budget
(`BUDGET_BYTES`) whenever a full (`--only`-less) run completes — trim a
generator before committing if you blow through it, don't raise the budget
casually.

`tests/golden/MANIFEST.json` is written automatically, and records, per
fixture: `file`, `size_bytes`, `n_arrays`, `source_qualnames` (which source
methods it exercised), `source_sha256` (which source files' hashes it was
generated against — cross-referenced against, but not a replacement for,
`provenance/sources.sha256`), and `rng_seeds`. It also records
`panda_urdf_sha256` (the exact Panda URDF the fixtures assume) and a
`versions` block (`python`, `torch`, `numpy`, `pytorch-kinematics`,
`robot-descriptions`) — **a torch/numpy upgrade can shift bit-level results
even when arithmetic is unchanged**, so `versions` is what tells a future
reader whether an observed drift is a real regression or an environment
artifact (see [MODIFYING.md](MODIFYING.md)'s "upgrade torch" row).

## `tools/gen_golden.py --diff`

```bash
python tools/gen_golden.py --source-a PATH_TO_MARIONETTE_CIASC \
                            --source-b PATH_TO_MARIONETTE_FKJEPA \
                            --out-dir tests/golden/ --diff
```

Re-runs each existing fixture's *inputs* through `src/kinescore` (not the
source repos) and prints a legacy-vs-new table (`key`, `legacy`, `new`,
`abs`, `rel` diff per array), via a registry of per-fixture adapter functions,
`_DIFF_ADAPTERS: dict[str, Callable[[dict], dict]]`.

**As shipped, `_DIFF_ADAPTERS` is empty** — every fixture reports `SKIP (no
--diff adapter registered yet)`. This is documented as the intended
degrade-gracefully behaviour in the script's own module docstring ("It is
expected to report 'unavailable' for every fixture until another agent lands
the matching `kinescore` code"), not a bug — but it means `--diff` currently
provides **no actual legacy-vs-new comparison out of the box**, despite
`src/kinescore` now existing and every metric/robot/reference module being
implemented and tested. Wiring `_DIFF_ADAPTERS` (one small function per
fixture name, mapping the golden `.npz` dict to the equivalent
`src.kinescore`-computed dict) is real, currently-unclaimed follow-up work —
not something this documentation task is scoped to implement, since doing so
would mean writing/modifying `tools/gen_golden.py`, which is outside this
task's `docs/**`-only ownership.

## Regenerating after a source change

If either source checkout (`Marionette-ciasc`/`Marionette-fkjepa`) changes in
a way that should be reflected in the fixtures — and you have both checkouts
reachable — the sequence is:

1. Re-run `tools/source_snapshot.py` first, to get a fresh
   `provenance/sources.sha256` (this is the only way to know whether the
   *bytes* actually changed, independent of `gen_golden.py`'s numeric
   output).
2. Re-run `tools/gen_golden.py` (full or `--only` the affected fixtures).
3. Diff `tests/golden/MANIFEST.json`'s `fixtures.*.source_sha256` against the
   new `provenance/sources.sha256` — they should now agree; if they don't,
   something read a different file than what was snapshotted.
4. Run the full test suite (`pytest`) — every golden-fixture-comparison test
   (`test_fk_parity_franka.py`, `test_head_golden.py`,
   `test_reference_golden.py`, the `ckpt`-marked cases in
   `test_checkpoint_roundtrip.py`) will now compare against the new numbers.
   A test failure here means either the source genuinely changed behaviour
   (expected — update the affected [PROVENANCE.md](PROVENANCE.md) row if the
   change is worth documenting) or the regeneration picked up an unintended
   difference (investigate before committing).

**Remember: B (`Marionette-fkjepa`)'s `.git` is a dead worktree pointer** —
see `provenance/git_state.json`'s `fkjepa_B` block. This does not block
*reading files* from that checkout (which is all `source_snapshot.py`/
`gen_golden.py` need), but it means B contributes **no citable commit** to
any future snapshot, only file-level SHA-256 — and if the B checkout itself
becomes unreachable (moved, deleted, host decommissioned), regenerating
anything that touches B's sources stops being possible at all. The
[PROVENANCE.md §(a3)](PROVENANCE.md#a3-claimed-but-unhashed-sources) gap
(several destination files cite a B source file that
`tools/source_snapshot.py`'s `SOURCES_B` never hashed) should be closed
*before* that window closes, not after.
