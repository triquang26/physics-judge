"""Score every manifest row, streaming results to disk as it goes.

Iterates rows from :mod:`kinescore.bench.manifest`, decodes frames via
:func:`kinescore.video.reader.load_rgb`, scores them with a
:class:`~kinescore.core.scorer.Scorer`, and appends one
:class:`~kinescore.bench.store.ResultsStore` record per clip.

The load-bearing property is that **a clip that fails does not crash the
run** -- it is recorded with ``status="failed"`` (see
:func:`~kinescore.bench.store.failed_record`) and the loop continues. A
benchmark runner that dies on the first corrupt file and has to be restarted
from scratch is worse than one that finishes and tells you honestly that 3 of
1000 clips failed to decode; a runner that dies and *silently* drops the
clips it already scored on restart (no ``--resume``) is worse still, which is
why resuming is keyed off what's actually on disk rather than an in-memory
count.
"""
from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from kinescore.bench.store import ResultsStore, failed_record
from kinescore.core.clip import ClipSpec, ViewLayout

__all__ = ["run"]

#: Called after each row as ``progress(row, status)`` with
#: ``status in {"ok", "failed", "skipped"}``, if given to :func:`run`.
ProgressCallback = Callable[[dict, str], None]

#: ViewLayout is a frozen dataclass, so one shared instance is a safe default.
_SINGLE_VIEW = ViewLayout()


def run(rows: list[dict], scorer: Any, results_path: str, *,
        resume: bool = False, force: bool = False, max_frames: int = 0,
        view_layout: ViewLayout = _SINGLE_VIEW,
        progress: ProgressCallback | None = None,
        trace_store: Any = None) -> dict[str, int]:
    """Score every row in ``rows`` with ``scorer``, appending to ``results_path``.

    Parameters
    ----------
    rows:
        Manifest rows, e.g. from :func:`kinescore.bench.manifest.build_manifest`
        or :func:`~kinescore.bench.manifest.load_manifest`. Each row must have
        at least the :data:`~kinescore.bench.manifest.ROW_KEYS` fields needed
        to reconstruct a :class:`~kinescore.core.clip.ClipSpec`
        (``path``, ``fps``, ``n_frames``, ``w``, ``h``).
    scorer:
        A composed :class:`~kinescore.core.scorer.Scorer`.
    results_path:
        ``results.jsonl`` location (see
        :class:`~kinescore.bench.store.ResultsStore`).
    resume:
        Skip rows whose ``(path, suite_id, reader_id)`` already appears in
        ``results_path`` -- see
        :func:`~kinescore.bench.store.record_key`. Mutually exclusive with
        ``force``.
    force:
        Truncate ``results_path`` before scoring, discarding any prior run.
        Mutually exclusive with ``resume``.
    max_frames:
        Forwarded to :func:`~kinescore.video.reader.load_rgb`.
    view_layout:
        Camera packing to attach to every reconstructed
        :class:`~kinescore.core.clip.ClipSpec`. Must match what
        ``scorer.reader`` expects (checked by
        :meth:`~kinescore.core.scorer.Scorer.score`); pass the same
        ``ViewLayout`` used to build the manifest.
    trace_store:
        Optional :class:`~kinescore.bench.traces.TraceStore`. ``None`` (the
        default) is a complete no-op -- ``results.jsonl`` and everything
        else this function does is byte-identical whether or not a caller
        ever passes this. When given, every successfully-scored ("ok") row
        additionally has :func:`~kinescore.bench.traces.clip_traces_from_scored`
        run against it and, if that produced any traces, appended to the
        store. A clip that fails to decode/score never reaches this (there
        is no :class:`~kinescore.core.scorer.ScoredClip` to build traces
        from), matching ``results.jsonl``'s own "no ``ScoredClip``, no
        traces" story for a failed row.

    Returns
    -------
    dict
        ``{"n_total", "n_scored", "n_skipped", "n_failed"}``.

    Raises
    ------
    ValueError
        If both ``resume`` and ``force`` are set.
    """
    if resume and force:
        raise ValueError("resume and force are mutually exclusive")

    from kinescore.video.reader import load_rgb

    store = ResultsStore(results_path)
    if force:
        store.truncate()
    done = store.existing_keys() if resume else set()

    n_scored = n_skipped = n_failed = 0
    try:
        for row in rows:
            key = (row.get("path"), scorer.suite.suite_id, scorer.reader.reader_id)
            if resume and key in done:
                n_skipped += 1
                if progress:
                    progress(row, "skipped")
                continue

            try:
                clip = ClipSpec.from_fps(
                    path=row["path"], fps=float(row["fps"]),
                    n_frames=int(row["n_frames"]), width=int(row["w"]),
                    height=int(row["h"]),
                    dt_source=row.get("dt_source", "table"),
                    view_layout=view_layout, codec=row.get("codec"),
                    sha1=row.get("sha1"))
                frames = load_rgb(clip, max_frames=max_frames)
                scored = scorer.score(frames, clip)
            except Exception as exc:  # noqa: BLE001
                n_failed += 1
                reason = "".join(
                    traceback.format_exception_only(type(exc), exc)).strip()
                store.append(failed_record(row, scorer, f"error:{type(exc).__name__}:{reason}"))
                if progress:
                    progress(row, "failed")
                continue

            record = dict(scored.to_record())
            record["status"] = "ok"
            store.append(record)
            if trace_store is not None:
                from kinescore.bench.traces import clip_traces_from_scored
                traces = clip_traces_from_scored(scored)
                if traces is not None:
                    trace_store.append(traces)
            n_scored += 1
            if progress:
                progress(row, "ok")
    finally:
        # A trace_store batches its durability checkpoints (see
        # TraceStore.checkpoint's docstring) for throughput -- without this,
        # up to _ZIP_CHECKPOINT_EVERY clips' worth of traces from the tail
        # of a run would sit unflushed forever. Runs whether the loop above
        # finished normally or raised, mirroring `store` (ResultsStore),
        # which is already durable per-row and needs no equivalent flush.
        if trace_store is not None:
            trace_store.close()

    return {"n_total": len(rows), "n_scored": n_scored,
            "n_skipped": n_skipped, "n_failed": n_failed}
