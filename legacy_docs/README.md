# Legacy docs

A **decision record, not usage documentation.** Nothing here is guaranteed to
describe the code as it exists today; several entries document decisions that
were later reversed (see PROVENANCE.md's D7 addendum: the squashed pose-reader
path this repo shipped with was removed entirely).

To *use* kinescore — train a reader, run a benchmark, add a robot or metric —
read `../docs/` instead. Come here for **why** something is the way it is, or
what was tried and rejected.

These four are kept because source and tests cite them directly:

| file | what it records |
|---|---|
| `PROVENANCE.md` | Line-by-line audit of every file ported from the source research codebases. The numbered defects (D1–D12) are the best source for "why does this code check X". Cited across `src/` and validated by `tests/test_provenance_complete.py`. |
| `DECISIONS.md` | Design decisions as "considered A and B, chose B, because" — the no-squash decision (D-B), multiview packing (D-G), Airbot hand-exclusion (D-H). |
| `SCHEMA.md` | The `results.jsonl` row schema. Cross-check against `kinescore.bench.store` before trusting a field name. |
| `ADDING_ALOHA_NOTES.md` | How the ALOHA joint layout was hypothesised before verification. The confirmed layout is in `src/kinescore/robots/aloha/constants.py`'s docstring — read that for truth, this for the reasoning. |

Not maintained going forward. If you find yourself editing one to keep it
accurate, the fact belongs in `../docs/` instead.
