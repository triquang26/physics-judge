"""One canonical result schema, and the only flattener for it.

A prior extraction of this benchmark shipped two incompatible ``metrics.json``
formats -- one script wrote per-clip files with the metric keys at the top
level, another wrote them nested under a ``"metrics"`` key, and the aggregator
that consumed them had only ever been tested against one of the two. Rows
written by the other writer silently produced empty/NaN aggregates instead of
an error, because the aggregator's key lookup just didn't find anything and
moved on.

The fix here is structural, not a convention to remember:

* There is exactly **one** record shape, defined by
  :meth:`~kinescore.core.scorer.ScoredClip.to_record` (frozen, in
  ``kinescore.core``) plus the two fields this module adds on top --
  ``schema_version`` and ``status``. Nothing downstream is allowed to invent a
  second shape.
* There is exactly **one** flattener, :func:`flatten`. It is used both by
  whatever writes a flat table (``bench.stats.load_scores``) and by anything
  that wants to *describe* the schema (``tests/test_schema.py``), so the two
  can never drift apart the way the two incompatible writers did.
* Every row from the same ``suite_id`` has the same metric key set, checked by
  :func:`assert_uniform_schema` -- including rows for clips that failed before
  a :class:`~kinescore.core.suite.SuiteResult` ever existed. See
  :func:`failed_record`.

``results.jsonl`` (one JSON object per line) rather than one big JSON array or
per-clip files: appendable, so a run that dies partway through leaves a
prefix of valid, individually-parseable rows instead of either an unparseable
half-written array or thousands of tiny files to glob.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from typing import Any

__all__ = [
    "SCHEMA_VERSION", "STATUSES", "flatten", "record_key", "failed_record",
    "iter_records", "assert_uniform_schema", "ResultsStore",
]

#: Bumped whenever the record shape changes. Carried on every row so an
#: aggregator reading an old results.jsonl fails loudly instead of
#: misinterpreting a renamed/removed field as missing data.
SCHEMA_VERSION = 1

#: A row's lifecycle. ``skipped`` is for ``--resume`` bookkeeping at the
#: caller level (kinescore.bench.runner) -- a skipped clip is *not* written
#: again, so it is not a status that appears inside results.jsonl itself,
#: only in the runner's summary. It is listed here because both places must
#: agree on the vocabulary.
STATUSES = ("ok", "skipped", "failed")


def flatten(record: dict) -> dict[str, Any]:
    """Recursively flatten a nested record into ``{"a.b.c": value}``.

    The one flattener, used everywhere a flat view of a record is needed
    (``bench.stats.load_scores`` for a tidy DataFrame, ``tests/test_schema.py``
    for the exact-key-set assertion). A record produced by
    :meth:`ScoredClip.to_record` plus this module's ``schema_version``/
    ``status`` flattens to keys like ``clip.dt``, ``run.suite_id``,
    ``metrics.mean_jerk_mps3``, ``metrics_unavailable.mean_jerk_mps3``.

    Leaf values (including ``None``) are passed through unchanged -- this
    function only restructures keys, it never coerces a value (in particular
    it never turns a missing/unavailable metric's ``None`` into ``0``).
    """
    out: dict[str, Any] = {}

    def _walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(f"{prefix}.{k}" if prefix else str(k), v)
        else:
            out[prefix] = obj

    _walk("", record)
    return out


def record_key(record: dict) -> tuple[str | None, str | None, str | None]:
    """Identity of one result row: ``(clip path, suite_id, reader_id)``.

    A clip is re-scoreable by a different reader checkpoint or a different
    metric suite without those results colliding; ``--resume`` uses this key
    to decide whether a manifest row has already been scored *by this run's
    reader+suite* rather than merely "this path appears somewhere in the
    file".
    """
    clip = record.get("clip") or {}
    run = record.get("run") or {}
    return (clip.get("path"), run.get("suite_id"), run.get("reader_id"))


def failed_record(clip_row: dict, scorer: Any, error: str) -> dict:
    """Canonical record for a clip that failed before a ``SuiteResult`` existed.

    The runner cannot call ``ScoredClip.to_record()`` for a clip whose
    decoding or scoring raised -- there is no ``ScoredClip``. Simply omitting
    the row (or writing a bare ``{"path": ..., "error": ...}``) would violate
    the "every row with the same suite_id has the same metrics key set" rule
    and would make a failed run look like a shorter, cleaner one -- a
    silently-shorter results file is how a benchmark lies.

    Instead this builds the same top-level shape as a successful record, using
    ``scorer.suite.output_keys`` (the suite's fixed term set, known without
    ever having scored anything) to fill ``metrics`` with ``None`` and
    ``metrics_unavailable`` with ``error`` for every key.

    Parameters
    ----------
    clip_row:
        Any dict describing the clip -- a real ``ClipSpec.as_row()`` if a spec
        was built before the failure, or the raw manifest row if not (e.g. the
        spec construction itself is what failed). Either way it becomes the
        record's ``"clip"`` block verbatim.
    scorer:
        The :class:`~kinescore.core.scorer.Scorer` that was going to be used,
        for ``robot``/``reader_id``/``limit_semantics``/``suite_id``/
        ``suite_name`` and the metric key set.
    error:
        Human-readable failure reason, e.g.
        ``"error:RuntimeError:CUDA out of memory"``. Stored verbatim as every
        metric's unavailability reason.
    """
    keys = scorer.suite.output_keys
    return {
        "clip": dict(clip_row),
        "run": {
            "robot": scorer.robot.name,
            "reader_id": scorer.reader.reader_id,
            "limit_semantics": scorer.reader.limit_semantics,
            "suite_id": scorer.suite.suite_id,
            "suite_name": scorer.suite.name,
        },
        "coverage": {"n_frames_scored": 0, "gate_coverage": 0.0},
        "metrics": dict.fromkeys(keys),
        "metrics_unavailable": dict.fromkeys(keys, error),
        "status": "failed",
    }


def iter_records(path: str) -> Iterator[dict]:
    """Yield each JSON record from a ``results.jsonl`` file, in file order.

    Blank lines are skipped (harmless if a writer's last line ended with a
    trailing newline). Missing file yields nothing rather than raising, so
    ``ResultsStore.existing_keys()`` on a fresh output directory is just an
    empty set instead of a special case the caller has to handle.
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def assert_uniform_schema(records: Iterable[dict]) -> None:
    """Raise if two records sharing a ``suite_id`` have different metric keys.

    This is the check that would have caught the two-incompatible-formats
    defect described in the module docstring before it ever reached an
    aggregator: fail at the point a bad row is read, not three functions
    downstream when an average silently comes out over the wrong count of
    terms.
    """
    seen: dict[str, frozenset] = {}
    for rec in records:
        sid = rec.get("run", {}).get("suite_id")
        keys = frozenset((rec.get("metrics") or {}).keys())
        if sid in seen and seen[sid] != keys:
            raise ValueError(
                f"suite_id {sid!r} has inconsistent metrics key sets across "
                f"rows: {sorted(seen[sid])} vs {sorted(keys)}. Two suites "
                f"with the same id must be the same benchmark.")
        seen[sid] = keys


class ResultsStore:
    """Append-only, resumable ``results.jsonl`` writer/reader.

    Parameters
    ----------
    path:
        Location of the results file. Parent directory is created on first
        :meth:`append`.
    """

    def __init__(self, path: str) -> None:
        self.path = path

    def iter_records(self) -> Iterator[dict]:
        yield from iter_records(self.path)

    def existing_keys(self) -> set:
        """``{record_key(rec) for rec in self}`` -- what ``--resume`` skips."""
        return {record_key(rec) for rec in self.iter_records()}

    def append(self, record: dict) -> None:
        """Write one record as a line, stamping ``schema_version`` if absent.

        ``allow_nan=False`` turns a metric that leaked a raw ``NaN`` (instead
        of the required ``None`` + reason) into an immediate, loud crash here
        rather than a ``NaN`` token silently written into the file -- standard
        JSON has no ``NaN`` literal, so a strict downstream parser (e.g. most
        non-Python readers) would otherwise fail far from the cause, or worse,
        a lenient one would parse it as a Python float and let it flow into an
        aggregate.
        """
        rec = dict(record)
        rec.setdefault("schema_version", SCHEMA_VERSION)
        if rec.get("status") not in STATUSES:
            raise ValueError(
                f"record status must be one of {STATUSES}, got {rec.get('status')!r}")
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, allow_nan=False))
            f.write("\n")

    def truncate(self) -> None:
        """Empty the results file -- what ``--force`` uses before a fresh run."""
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        open(self.path, "w").close()
