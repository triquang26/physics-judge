# Legacy docs

This directory is a **decision record, not usage documentation.** Nothing here
is guaranteed to describe the code as it exists today, and several files
explicitly document decisions that were later reversed (see PROVENANCE.md's
"D7 addendum" for the most consequential example: the squashed pose-reader
path this repo shipped with was later removed entirely).

If you are trying to *use* kinescore -- train a reader, run a benchmark, add a
robot or metric -- read the six docs in `../docs/` instead:
[`README.md`](../README.md), [`docs/DATA_PREP.md`](../docs/DATA_PREP.md),
[`docs/TRAINING.md`](../docs/TRAINING.md),
[`docs/BENCHMARKING.md`](../docs/BENCHMARKING.md),
[`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md),
[`docs/METRICS.md`](../docs/METRICS.md). Those are kept current against the
code; nothing in this directory is.

Come here when you need to know **why** something is the way it is, or what
was tried and rejected before landing on the current design.

| file | what it records |
|---|---|
| `PROVENANCE.md` | Line-by-line audit of every file ported from the source research codebases (Marionette-ciasc, Marionette-fkjepa, fkjepa) -- what was copied verbatim, what was adapted, and why. Numbered defects (D1-D9 and their addenda) are the single best source for "why does this code check X" questions. |
| `DECISIONS.md` | Design decisions made during the port, framed as "we considered A and B, chose B, because." Includes the no-squash decision (D-B), the multiview packing survey (D-G), and the Airbot MMK2 hand-exclusion evidence (D-H). |
| `RATE_POLICY.md` | The full frame-rate-mismatch investigation: which generators run at which fps, measured jerk/energy inflation from comparing across rates, and the three-layer policy (`paired` / `RATE_FREE` / explicit `resample:`) that came out of it. The **operative rules** from this file are absorbed into `docs/BENCHMARKING.md`; read this one for the measurements and reasoning behind them. |
| `SCHEMA.md` | The `results.jsonl` row schema as it stood when written. Cross-check against `kinescore.bench.store` before trusting a field name. |
| `ADDING_A_ROBOT.md` | Walkthrough of adding `GR1Spec`/`Synthetic2R` to the (now-superseded) per-axis `_FACTORIES` registration pattern. The checklist items (D9 rigid-bone exclusion, capability declarations, keypoint ordering) are still conceptually right; the registration mechanics now go through `kinescore.core.registry.Registry` -- see `docs/ARCHITECTURE.md`. |
| `ADDING_A_METRIC.md` | Walkthrough of adding a metric to the (now-superseded) suite-construction path. The `MetricSpec`/`SafeMetric`/NaN-with-reason contract it describes is still current; see `docs/ARCHITECTURE.md` for how a metric is wired into a suite today. |
| `ADDING_ALOHA_NOTES.md` | Handoff notes from the round that hypothesised the ALOHA 14/42-D joint layout from the published ACT convention alone, before it was verified against real parquet data. Superseded by the verification recorded in `src/kinescore/robots/aloha/constants.py`'s own module docstring -- read that file for the confirmed layout, this one for how the hypothesis was formed. |
| `REGENERATING_GOLDENS.md` | What a golden fixture proves and the pairing rule that governs changing one (every behavior change ships both a legacy-reproducing fixture and a PROVENANCE.md row). Still the right mental model for `tests/golden/`; file paths may have moved. |

None of these files are maintained going forward. If you find yourself editing
one to keep it accurate, that is a sign the fact belongs in `../docs/`
instead -- add it there and leave this file as the historical record it is.
