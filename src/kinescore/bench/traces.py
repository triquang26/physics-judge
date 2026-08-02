"""Per-frame trace sidecar: ``results.jsonl`` stays scalar-only, traces live here.

Why a sidecar, not a new column in ``results.jsonl``
--------------------------------------------------------
``legacy_docs/SCHEMA.md`` fixes ``results.jsonl`` to a static, flat, one-JSON-object-
per-line schema -- every row the same key set
(``bench/store.py::assert_uniform_schema``), every value a JSON scalar
(``ResultsStore.append``'s ``allow_nan=False``). Embedding a per-frame array
in a metric column would break the "flat" half of that contract for every
consumer that reads ``results.jsonl`` as a table, and would bloat a
2007-row, sub-megabyte file into something an aggregator has to special-case.
Nothing in this module writes to ``results.jsonl``; a run with ``--traces``
and a run without it produce byte-identical ``results.jsonl`` files.

Layout
------
One ``.npz`` per run (conventionally ``<out>/traces.npz``, next to
``results.jsonl`` in the same run directory) holds every clip's per-frame
arrays. Entries are named ``"<clip_id>/<metric_key>.npy"``, where
``clip_id`` is a 16-hex-char ``sha1`` of the clip's identity tuple
(``path``, ``suite_id``, ``reader_id`` -- the same identity
``bench/store.py::record_key`` uses for ``results.jsonl``). Hashing the
identity (rather than an incrementing counter) makes a write idempotent by
construction: the same clip, scored by the same reader+suite, always lands
on the same entry names, so writing it twice (e.g. a crash between the npz
append and the index update below) overwrites rather than duplicates.

A companion JSON-Lines index (``<out>/traces_index.jsonl`` -- one JSON object
per clip per line, the same appendable convention ``results.jsonl`` itself
uses, and for the same reason: see :class:`TraceStore`'s docstring for why a
single growing JSON object per run does not scale to thousands of clips) is
the self-describing half. Each line records ``clip_id`` (the npz entry
prefix) plus ``path``, ``dt``, ``fps``, ``n_frames`` (the *scored* frame
count -- post-confidence-gate, see
:class:`~kinescore.core.scorer.ScoredClip`), ``suite_id``, ``reader_id``,
and, per metric, ``units``, ``length``, ``first_frame_index`` and
``dt_exponent``. Both files are plain zip/JSON --
``np.load("traces.npz")`` and ``[json.loads(l) for l in
open("traces_index.jsonl")]`` work with no kinescore import, so a notebook
can plot a trace without this package installed. See :func:`load_example`
below for the worked example (also runnable standalone, no kinescore
import, as its own docstring shows).

Resumability
------------
:meth:`TraceStore.append` no-ops for a clip identity already known -- either
durably committed to the index or still pending in the current process (see
:meth:`TraceStore.checkpoint`) -- the same skip semantics ``--resume`` gets
for ``results.jsonl`` via ``ResultsStore.existing_keys()``. In practice
``bench/runner.py`` never even calls it for a resumed-and-skipped row (the
row is skipped before scoring happens at all), so this is defence in depth,
not the primary mechanism.

Alignment: a k-th derivative loses k frames
-----------------------------------------------
Every trace in this module is shorter than or equal to ``n_frames`` (the
clip's *scored* frame count) and ``first_frame_index`` states, unambiguously,
which original frame index ``trace[0]`` corresponds to:

* ``"same_length"`` -- no frames dropped (a purely per-frame geometric quantity,
  e.g. ``rigidity_worst_bone_mm``). ``length == n_frames``,
  ``first_frame_index == 0``.
* ``"front"`` -- every dropped frame comes from the start of the clip (an
  n-th order finite-difference chain, e.g. ``mean_jerk_mps3`` drops 3 via
  three applications of :func:`kinescore.metrics.ops.fd`).
  ``length == n_frames - drop``, ``first_frame_index == drop``.
* ``"interior"`` -- one frame dropped from *each* end
  (``torque_frac_rated``'s central-difference interior slice,
  ``ratio[:, 1:-1]`` -- see ``metrics/torque.py::joint_torques``'s docstring
  for why the endpoints are a static-hold approximation, not a real
  measurement). ``length == n_frames - 2``, ``first_frame_index == 1``.

This module is the single place that alignment knowledge lives: it cannot be
declared on :class:`~kinescore.core.metric.MetricSpec` without adding a field
to the frozen ``core/metric.py`` contract, which is out of this task's scope
(and would be a bigger, riskier change for a fact that is really "how this
particular metric's arithmetic is shaped", not part of the metric/suite
identity contract ``MetricSpec`` exists to fix). See :data:`ALIGNMENTS`.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kinescore.core.metric import get_metric

__all__ = [
    "ALIGNMENTS", "TRACE_INDEX_SCHEMA_VERSION", "first_frame_index",
    "ClipTraces", "clip_traces_from_scored", "TraceStore",
]

#: Bumped whenever a ``traces_index.jsonl`` line's shape changes -- the same
#: role ``bench/store.py::SCHEMA_VERSION`` plays for ``results.jsonl``,
#: stamped on every line so a downstream reader can tell old and new shapes
#: apart rather than guessing from which fields happen to be present.
TRACE_INDEX_SCHEMA_VERSION = 1

#: How each per-frame-emitting metric's dropped frames are distributed across
#: the clip -- see the module docstring's "Alignment" section. A metric key
#: absent here is a programming error in this module (every metric whose
#: ``spec.perframe`` is ``True`` must be listed) -- :func:`first_frame_index`
#: raises rather than guessing a convention for an unclassified metric.
ALIGNMENTS: dict[str, str] = {
    "rigidity_worst_bone_mm": "same_length",
    "rigidity_worst_bone_all_mm": "same_length",
    "mean_jerk_mps3": "front",
    "mean_speed_mps": "front",
    "mean_accel_mps2": "front",
    "accel_violation_frac": "front",
    "no_teleport_frac": "front",
    "torque_frac_rated": "interior",
}

#: Number of frames each ``"front"``/``"interior"`` metric drops, one finite
#: difference at a time (0 for ``"same_length"``) -- used only to sanity-check
#: :data:`ALIGNMENTS` against the actual trace length in
#: :func:`first_frame_index` (a metric whose real drop doesn't match its
#: declared alignment kind is a bug in this table, not a fact about the
#: clip, and should fail loudly rather than silently mislabel the trace).
_EXPECTED_DROP: dict[str, int] = {
    "rigidity_worst_bone_mm": 0,
    "rigidity_worst_bone_all_mm": 0,
    "mean_jerk_mps3": 3,
    "mean_speed_mps": 1,
    "mean_accel_mps2": 2,
    "accel_violation_frac": 2,
    "no_teleport_frac": 1,
    "torque_frac_rated": 2,
}


def first_frame_index(metric_key: str, n_frames: int, length: int) -> int:
    """0-based index into the clip's (scored) frame axis that ``trace[0]`` is for.

    Raises ``KeyError`` for a metric absent from :data:`ALIGNMENTS`, and
    ``ValueError`` if the observed ``n_frames - length`` doesn't match that
    metric's expected drop (:data:`_EXPECTED_DROP`) -- both are "this
    module's bookkeeping is out of sync with a metric's real arithmetic",
    which must surface as a hard failure rather than a silently wrong
    alignment written into a run's traces.
    """
    kind = ALIGNMENTS[metric_key]
    drop = n_frames - length
    expected = _EXPECTED_DROP[metric_key]
    if drop != expected:
        raise ValueError(
            f"{metric_key}: trace length {length} implies {drop} frame(s) "
            f"dropped from n_frames={n_frames}, but ALIGNMENTS/_EXPECTED_DROP "
            f"declares {expected} -- this metric's arithmetic changed "
            f"without updating bench/traces.py's alignment table.")
    if kind == "same_length":
        return 0
    if kind == "front":
        return drop
    if kind == "interior":
        if drop % 2 != 0:
            raise ValueError(
                f"{metric_key}: 'interior' alignment expects an even frame "
                f"drop (split evenly across both ends), got {drop}")
        return drop // 2
    raise ValueError(f"{metric_key}: unknown alignment kind {kind!r}")


@dataclass(frozen=True)
class ClipTraces:
    """One clip's per-frame arrays, plus the metadata needed to interpret them.

    Built by :func:`clip_traces_from_scored`; consumed by
    :meth:`TraceStore.append`. Deliberately holds plain Python/numpy values
    (no torch, no kinescore ``Metric``/``MetricSpec`` objects) -- everything
    that reaches :class:`TraceStore` is already what gets serialised.
    """

    record_key: tuple[Any, Any, Any]
    path: str
    dt: float
    fps: float
    n_frames: int
    suite_id: str
    suite_name: str
    reader_id: str
    arrays: dict[str, np.ndarray]
    meta: dict[str, dict[str, Any]] = field(default_factory=dict)


def clip_traces_from_scored(scored: Any) -> ClipTraces | None:
    """Build :class:`ClipTraces` from a :class:`~kinescore.core.scorer.ScoredClip`.

    Reads ``scored.result.perframe()`` -- the dict of already-available,
    non-``None`` per-frame arrays :meth:`~kinescore.core.suite.SuiteResult.perframe`
    collects (unavailable metrics, and metrics that don't declare
    ``perframe=True`` at all, are simply absent from it, never a fabricated
    zero array). Each array is tagged with its alignment via
    :func:`first_frame_index` and cast to ``float32`` (the arrays already
    are float32 coming out of every metric here, but this is a hard
    guarantee independent of that).

    Returns ``None`` (not an empty :class:`ClipTraces`) when the suite
    declared no ``perframe`` metrics, or none of them produced an array for
    this clip (e.g. every one was unavailable) -- so a caller can skip the
    write entirely rather than persisting an empty shell that would still
    cost a zip-entry-less clip slot in the index.
    """
    samples = scored.result.perframe()
    if not samples:
        return None

    n_frames = int(scored.n_frames_scored)
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, dict[str, Any]] = {}
    for key, arr in samples.items():
        values = np.asarray(arr, dtype=np.float32).reshape(-1)
        idx = first_frame_index(key, n_frames, int(values.shape[0]))
        # units/dt_exponent are declared on the registered Metric's spec, not
        # on MetricValue -- look the metric up by key rather than needing
        # the suite object threaded through this function.
        metric_spec = get_metric(key).spec
        arrays[key] = values
        meta[key] = {
            "units": metric_spec.units,
            "length": int(values.shape[0]),
            "first_frame_index": idx,
            "dt_exponent": metric_spec.dt_exponent,
        }

    if not arrays:
        return None

    clip = scored.clip
    return ClipTraces(
        record_key=(clip.path, scored.result.suite_id, scored.reader_id),
        path=clip.path, dt=float(clip.dt), fps=float(clip.fps),
        n_frames=n_frames, suite_id=scored.result.suite_id,
        suite_name=scored.result.suite_name, reader_id=scored.reader_id,
        arrays=arrays, meta=meta,
    )


def _clip_id(key: tuple[Any, Any, Any]) -> str:
    """Stable 16-hex-char id for a clip identity -- the npz entry prefix.

    See the module docstring's "Layout" section for why this is a hash of
    the identity rather than an incrementing counter.
    """
    raw = "␟".join("" if p is None else str(p) for p in key)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


#: How many clips' worth of appends to buffer before durably committing
#: (see :meth:`TraceStore.checkpoint`). Re-opening a zip in append mode costs
#: roughly O(existing entry count) -- committing every single append makes a
#: 2007-clip run O(n^2) in the number of clips (measured: ~27s for 1000
#: reopen-per-append writes vs ~0.4s holding the handle open) -- so this
#: batches the durability cost instead of paying it on every call.
_ZIP_CHECKPOINT_EVERY = 100


class TraceStore:
    """Append-only, resumable per-frame-trace sidecar, mirroring ``ResultsStore``.

    Parameters
    ----------
    path:
        ``.npz`` file location (conventionally ``<out>/traces.npz``, next to
        ``results.jsonl``). The JSON index lives alongside it, at the same
        path with the extension replaced by ``_index.jsonl`` (one JSON
        object per line, per clip -- ``traces.npz`` -> ``traces_index.jsonl``
        -- the same appendable convention ``results.jsonl`` uses, and for the
        same reason: an ``O(1)``-per-call write instead of rewriting a
        growing dict from scratch on every clip).

    Durability vs throughput
    -------------------------
    A clip's zip entries (``zf.writestr``) land in the underlying file
    immediately, but a zip's *central directory* -- the index a reader
    (including ``np.load``) actually consults to know what entries exist --
    is only written when the archive is closed. Closing and reopening after
    *every* clip is what made the naive version of this class O(n^2) (see
    :data:`_ZIP_CHECKPOINT_EVERY`); holding the handle open across many
    appends and committing every :data:`_ZIP_CHECKPOINT_EVERY` clips (via
    :meth:`checkpoint`) is what makes this fast. The trade-off is durability
    granularity: a crash between two checkpoints loses at most
    :data:`_ZIP_CHECKPOINT_EVERY` clips' traces (never a corrupt file, never
    a wrong-but-present entry -- just absent, exactly like an unscored clip).
    A caller that wants a hard guarantee everything appended so far is on
    disk calls :meth:`checkpoint` (or :meth:`close`, or uses this as a
    context manager) explicitly; ``bench.runner.run`` always closes the
    store in a ``finally`` block so a normal (or exceptional) end of a
    scoring run never leaves a checkpoint-interval's worth of traces
    unflushed for no reason.

    The index is written to lag the zip, never lead it: an append's index
    record is only flushed to ``_index.jsonl`` as part of the *same*
    :meth:`checkpoint` call that commits its zip entries, so a reader of the
    index (this class's own :meth:`has`/``--resume`` aside, which also
    consults in-memory pending state) never sees a clip claimed as
    available whose arrays did not actually land.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        base, _ext = os.path.splitext(path)
        self.index_path = base + "_index.jsonl"
        self._known: dict[str, dict] | None = None
        self._zf: zipfile.ZipFile | None = None
        #: Index records for clips already written into the currently-open
        #: zip handle but not yet durably committed (see class docstring).
        self._pending: list[dict] = []

    def _ensure_known(self) -> dict[str, dict]:
        if self._known is None:
            known: dict[str, dict] = {}
            if os.path.exists(self.index_path):
                with open(self.index_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rec = json.loads(line)
                            known[rec["key"]] = rec
            self._known = known
        return self._known

    def has(self, record_key: tuple[Any, Any, Any]) -> bool:
        """Whether ``record_key`` already has traces written (for ``--resume``).

        Checks both the durably-committed index and this instance's own
        not-yet-checkpointed pending appends, so calling ``append`` twice in
        a row for the same clip within one run is always a no-op, even
        before the next checkpoint.
        """
        key_str = self._key_str(record_key)
        if key_str in self._ensure_known():
            return True
        return any(p["key"] == key_str for p in self._pending)

    @staticmethod
    def _key_str(record_key: tuple[Any, Any, Any]) -> str:
        return "␟".join("" if p is None else str(p) for p in record_key)

    def _ensure_zip_open(self) -> zipfile.ZipFile:
        if self._zf is None:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._zf = zipfile.ZipFile(self.path, "a",
                                       compression=zipfile.ZIP_DEFLATED)
        return self._zf

    def checkpoint(self) -> None:
        """Durably commit every append since the last checkpoint.

        Closes (and, on the next :meth:`append`, transparently reopens) the
        npz zip -- writing its central directory, the part a zip reader
        needs to see any entry at all -- and only *then* flushes the
        buffered index records for those now-durable entries. Safe to call
        with nothing pending (a no-op).
        """
        if self._zf is not None:
            self._zf.close()
            self._zf = None
        if not self._pending:
            return
        parent = os.path.dirname(self.index_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.index_path, "a") as f:
            for record in self._pending:
                f.write(json.dumps(record))
                f.write("\n")
        known = self._ensure_known()
        for record in self._pending:
            known[record["key"]] = record
        self._pending = []

    def close(self) -> None:
        """Finalize the store: commit every pending append. Idempotent."""
        self.checkpoint()

    def __enter__(self) -> TraceStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def append(self, traces: ClipTraces) -> None:
        """Write one clip's traces. No-op if this clip is already known.

        Buffers into the open zip handle and the in-memory pending list;
        durably lands on disk at the next :meth:`checkpoint` (automatic
        every :data:`_ZIP_CHECKPOINT_EVERY` calls, or explicit via
        :meth:`checkpoint`/:meth:`close`).
        """
        if self.has(traces.record_key):
            return

        clip_id = _clip_id(traces.record_key)
        zf = self._ensure_zip_open()
        for metric_key, arr in traces.arrays.items():
            entry = f"{clip_id}/{metric_key}.npy"
            buf = io.BytesIO()
            np.save(buf, arr)
            zf.writestr(entry, buf.getvalue())

        self._pending.append({
            "schema_version": TRACE_INDEX_SCHEMA_VERSION,
            "key": self._key_str(traces.record_key),
            "clip_id": clip_id,
            "path": traces.path,
            "dt": traces.dt,
            "fps": traces.fps,
            "n_frames": traces.n_frames,
            "suite_id": traces.suite_id,
            "suite_name": traces.suite_name,
            "reader_id": traces.reader_id,
            "traces": traces.meta,
        })
        if len(self._pending) >= _ZIP_CHECKPOINT_EVERY:
            self.checkpoint()


def load_example(out_dir: str, metric_key: str = "mean_jerk_mps3"):
    """Worked example: load one clip's trace and return ``(t, values)`` to plot.

    Plain ``numpy``/``json`` only -- no kinescore import needed to reproduce
    this by hand in a notebook:

    .. code-block:: python

        import json
        import numpy as np

        # one JSON object per line, per clip -- same convention as
        # results.jsonl
        entries = [json.loads(line)
                  for line in open("<out>/traces_index.jsonl")]
        npz = np.load("<out>/traces.npz")

        # pick any clip that has this metric
        entry = next(e for e in entries if "mean_jerk_mps3" in e["traces"])
        trace_meta = entry["traces"]["mean_jerk_mps3"]
        values = npz[f"{entry['clip_id']}/mean_jerk_mps3"]

        dt = entry["dt"]
        t0 = trace_meta["first_frame_index"] * dt
        t = t0 + dt * np.arange(len(values))

        import matplotlib.pyplot as plt
        plt.plot(t, values)
        plt.xlabel("time (s)")
        plt.ylabel(f"mean_jerk_mps3 ({trace_meta['units']})")
        plt.show()

    This function does the same thing, for the first clip found that has
    ``metric_key``, and returns ``(t, values)`` instead of plotting -- useful
    from a test or a REPL. Raises ``FileNotFoundError`` if ``out_dir`` has no
    ``traces.npz``/``traces_index.jsonl`` (i.e. the run was scored without
    ``--traces``), and ``KeyError`` if no clip in the run has ``metric_key``.
    """
    index_path = os.path.join(out_dir, "traces_index.jsonl")
    npz_path = os.path.join(out_dir, "traces.npz")
    npz = np.load(npz_path)

    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if metric_key not in entry["traces"]:
                continue
            trace_meta = entry["traces"][metric_key]
            values = npz[f"{entry['clip_id']}/{metric_key}"]
            dt = entry["dt"]
            t0 = trace_meta["first_frame_index"] * dt
            t = t0 + dt * np.arange(len(values))
            return t, values

    raise KeyError(f"no clip in {out_dir!r} has a trace for {metric_key!r}")
