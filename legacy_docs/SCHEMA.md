# Result schema

The canonical result record, defined by
`src/kinescore/core/scorer.py::ScoredClip.to_record` plus two fields
`src/kinescore/bench/store.py` adds on top (`schema_version`, `status`).
There is exactly **one** record shape and exactly **one** flattener
(`bench/store.py::flatten`) — both by design, not by accident. Read
`bench/store.py`'s module docstring first: a prior extraction of this
benchmark shipped **two incompatible `metrics.json` formats** (one writer put
metric keys at the top level, another nested them under `"metrics"`), and the
aggregator that consumed them had only ever been tested against one of the
two — rows from the other writer silently produced empty/NaN aggregates
instead of an error. Everything below is the structural fix.

## The record

```json
{
  "clip": {
    "path": "...", "fps": 10.0, "dt": 0.1, "n_frames": 64,
    "width": 224, "height": 672, "dt_source": "ffprobe",
    "view_layout": "3x49:exterior_1+exterior_2+wrist", "stride": 1,
    "codec": "h264", "sha1": null
  },
  "run": {
    "robot": "franka_panda",
    "reader_id": "readout_v2/3x49:exterior_1+exterior_2+wrist/checkpoint.pt",
    "limit_semantics": "raw_rad",
    "suite_id": "sha256:...",
    "suite_name": "invariant_v1"
  },
  "coverage": {
    "n_frames_scored": 64,
    "gate_coverage": 1.0
  },
  "metrics": {
    "rigidity_residual_mm": 0.42,
    "limit_violation_frac": 0.03,
    "...": "... every declared metric key ..."
  },
  "metrics_unavailable": {
    "...": "... e.g. \"unobservable:limit_semantics=squashed\" if the field above is ever \"squashed\" (see below); \"missing_input:...\"; \"too_few_frames:...\" ..."
  },
  "schema_version": 1,
  "status": "ok"
}
```

- **`clip`** — `ClipSpec.as_row()` (`core/clip.py`) verbatim: `path`, `fps`,
  `dt`, `n_frames`, `width`, `height`, `dt_source`
  (`"ffprobe"|"fps_arg"|"dt_arg"|"table"|"synthetic"`, provenance of the
  timebase — see D1/D3(b) in [PROVENANCE.md](PROVENANCE.md)), `view_layout`
  (`ViewLayout.key`, a stable identity string), `stride`, `codec`, `sha1`.
- **`run`** — `robot` (`RobotSpec.name`), `reader_id` (stable identity
  string, e.g. `"readout_v2/3x49:exterior_1+exterior_2+wrist/checkpoint.pt"`),
  `limit_semantics` (`"squashed"|"raw_rad"` — see D7; every reader currently
  shippable is `"raw_rad"` as of the D7 addendum below, since the squashed
  pose-reader path was removed — `"squashed"` remains a valid value of the
  field's type for a hypothetical future reader that declares it, not a
  currently-producible one), `suite_id` (`MetricSuite.suite_id`, a
  `"sha256:"`-prefixed hash of the suite's declared term set — see D3(a)),
  `suite_name`.
- **`coverage`** — `n_frames_scored` (after any confidence-gate drop),
  `gate_coverage` (fraction of frames surviving a heteroscedastic reader's
  confidence gate, `1.0` when no gate is used — see `core/scorer.py::Scorer`'s
  `gate` parameter).
- **`metrics`** — `{key: value_or_None}` for **every** key
  `MetricSuite.output_keys` declares, for **every** clip, regardless of what
  inputs that clip's reader/robot could supply
  (`SuiteResult.scalars()`'s docstring). `None`, not `NaN` — JSON has no
  `NaN` literal, so this is what makes the file portable to a non-Python
  reader; `ResultsStore.append` calls `json.dumps(..., allow_nan=False)`
  specifically so a metric that leaked a raw float `NaN` (instead of the
  required `None`+reason) crashes loudly here, at write time, rather than
  writing an invalid JSON token that only a lenient parser would silently
  accept downstream.
- **`metrics_unavailable`** — `{key: reason}` **only** for keys that are
  `None` in `metrics` (`SuiteResult.reasons()`). An available metric never
  appears here.
- **`schema_version`** — `bench/store.py::SCHEMA_VERSION` (currently `1`).
  Bumped whenever the record shape changes, so an aggregator reading an old
  `results.jsonl` fails loudly instead of misinterpreting a renamed/removed
  field as missing data. `ResultsStore.append` stamps this via
  `rec.setdefault("schema_version", SCHEMA_VERSION)` if the caller didn't set
  it.
- **`status`** — one of `bench/store.py::STATUSES = ("ok", "skipped",
  "failed")`. `"skipped"` is `--resume` bookkeeping at the
  `bench.runner`/caller level — it never actually appears as a value inside
  `results.jsonl` (a skipped clip is not written again at all); it is listed
  in `STATUSES` only so both places agree on the vocabulary.
  `ResultsStore.append` **rejects** (`ValueError`) any record whose `status`
  is not one of these three.

## Invariants

1. **Unavailable ⇒ `null`, never `0`, never omitted.** A metric that could
   not be computed for a clip is present in `metrics` with value `null` and
   present in `metrics_unavailable` with a human-readable reason string
   (`"missing_input:<name>"`, `"unobservable:<flag>=<value>"`,
   `"too_few_frames:<T><<min_frames>"`, `"degenerate_input:non_finite_result"`,
   or `"error:<ExceptionType>:<message>"`). It is **never** silently dropped
   from the `metrics` dict (that would violate the static-schema guarantee
   below) and **never** coerced to `0.0` (which would read as "measured,
   perfect" — see D7 in [PROVENANCE.md](PROVENANCE.md) for exactly why that
   distinction is load-bearing). This is enforced at the `MetricValue` level
   (`core/metric.py`: "`value` is `NaN` **exactly** when `reason` is set") and
   again at the JSON-write boundary (`allow_nan=False`, above).
2. **Every row of a given `suite_id` has the same metric key set.** Checked
   by `bench/store.py::assert_uniform_schema`, which raises naming both
   diverging key sets the moment it finds a mismatch — this is the exact
   check that would have caught the two-incompatible-formats defect the
   module docstring describes, at the point a bad row is *read*, not three
   functions downstream when an aggregate silently comes out over the wrong
   count of terms. It holds by construction for any row produced through
   `MetricSuite.evaluate` (which always emits exactly `output_keys`) and
   through `failed_record` (which fills `dict.fromkeys(scorer.suite.output_keys)`
   — the *same* key source, so a clip that failed before scoring even began
   still produces a row indistinguishable in shape from a successful one).
3. **Failures are recorded with `status="failed"`, not dropped.**
   `bench/store.py::failed_record` builds the *same* top-level shape as a
   successful record — same `clip`/`run` blocks (from whatever `ClipSpec.as_row()`
   or raw manifest row was available before the failure), `metrics` filled
   with `None` for every declared key, `metrics_unavailable` filled with the
   error string for every declared key. Its own docstring states the
   reasoning: "simply omitting the row (or writing a bare `{"path": ...,
   "error": ...}`) would violate the 'every row with the same `suite_id` has
   the same metrics key set' rule and would make a failed run look like a
   shorter, cleaner one — a silently-shorter results file is how a benchmark
   lies." `bench/runner.py::run` is the caller that actually reaches this
   path: a clip whose decode or scoring raises is caught, written via
   `failed_record`, and the loop continues — a benchmark run does not die on
   the first corrupt file.

## File format

`results.jsonl` — one JSON object per line, **not** one big JSON array and
**not** one file per clip. Appendable: a run that dies partway through leaves
a prefix of valid, individually-parseable rows instead of either an
unparseable half-written array or thousands of tiny files to glob
(`bench/store.py`'s module docstring). `ResultsStore` (`bench/store.py`) is
the append-only reader/writer:

- `ResultsStore.append(record)` — validates `status`, stamps
  `schema_version` if absent, appends one `json.dumps(..., allow_nan=False)`
  line.
- `ResultsStore.iter_records()` — yields each parsed JSON object in file
  order; a missing file yields nothing (not an error), so
  `existing_keys()` on a fresh output directory is just an empty set.
- `ResultsStore.existing_keys()` — `{record_key(rec) for rec in self}`, what
  `--resume` skips. `record_key(record) = (clip.path, run.suite_id,
  run.reader_id)` — a clip is re-scoreable by a different reader checkpoint
  or a different metric suite without those results colliding.
- `ResultsStore.truncate()` — empties the file; what `--force` uses before a
  fresh run.

## `flatten` — the one flattener

`bench/store.py::flatten(record) -> {"a.b.c": value}` recursively dots every
nested key (`clip.dt`, `run.suite_id`, `metrics.mean_jerk_mps3`,
`metrics_unavailable.mean_jerk_mps3`). It is used both by whatever writes a
flat table (`bench/stats.py::load_scores`, for a tidy `pandas.DataFrame`) and
by anything that wants to *describe* the schema (`tests/test_schema.py`), so
the two can never drift apart the way the source's two incompatible writers
did. Leaf values, including `None`, pass through unchanged — `flatten` only
restructures keys, it never coerces a value (in particular it never turns a
missing metric's `None` into `0`).
