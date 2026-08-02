"""Turn a scored ``results.jsonl`` into a mirrored tree of per-clip CSVs.

Joins a run's ``results.jsonl`` against the manifest that produced it, and
emits **one CSV per leaf directory of the input tree, mirroring it exactly**
(``dense/humanoid/singleview/dreamgen/makovian/clips.csv``), plus one
``SUMMARY.csv`` at the run root. Never scores anything itself; re-running
:func:`export_csvs` against the same inputs is idempotent. See
``legacy_docs/DECISIONS.md`` D-I for why the join is needed, why grouping is
derived from each clip's path rather than its `family`, and the
suite-name-vs-suite-id / fps-comparability notes stamped into every file.
"""
from __future__ import annotations

import csv
import math
import os
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

from kinescore.bench.manifest import load_manifest
from kinescore.bench.store import assert_uniform_schema, iter_records

__all__ = [
    "GROUP_KEYS", "UNMATCHED_GROUP", "group_key_for_path",
    "relative_to_data_root", "ClipRow", "GroupSummary", "ExportResult",
    "build_clip_rows", "group_rows", "sort_group", "summarize_group",
    "write_clip_csv", "write_summary_csv", "export_csvs",
]

# ── grouping ─────────────────────────────────────────────────────────────

#: The five path segments that identify a benchmark cell, in output-tree order.
GROUP_KEYS = ("cache", "embodiment", "view", "generator", "horizon")

#: A path that doesn't match the known ``video_gen_physics/...`` layout is
#: grouped here instead of being dropped -- see :func:`group_key_for_path`.
UNMATCHED_GROUP = "_unmatched"

_GROUP_RE = re.compile(
    r"video_gen_physics/(?P<cache>[^/]+)/(?P<embodiment>[^/]+)/"
    r"(?:input|output)/(?P<view>[^/]+)/(?P<generator>[^/]+)/"
    r"(?P<horizon>[^/]+)/")


def group_key_for_path(rel_path: str) -> tuple[str, ...]:
    """The output-tree path segments this clip's CSV belongs under.

    Mirrors ``video_gen_physics/<cache>/<embodiment>/{input,output}/<view>/
    <generator>/<horizon>/...``, dropping the ``video_gen_physics`` prefix
    and the ``input``/``output`` segment. A path that doesn't match (an
    untaught family) is grouped under :data:`UNMATCHED_GROUP` instead of
    dropped -- see ``legacy_docs/DECISIONS.md`` D-I.
    """
    norm = rel_path.replace(os.sep, "/")
    m = _GROUP_RE.search(norm)
    if m:
        return tuple(m.group(k) for k in GROUP_KEYS)
    parts = [p for p in norm.split("/") if p][:2]
    return (UNMATCHED_GROUP, *parts)


def relative_to_data_root(path: str, data_root: str) -> str:
    """POSIX-style ``path`` relative to ``data_root``.

    Falls back to the absolute (slash-normalised) path when ``path`` isn't
    actually under ``data_root``, rather than raising -- a stray file outside
    the declared data root should still produce a row, not crash the export.
    """
    root = os.path.abspath(data_root)
    abs_path = os.path.abspath(path)
    rel = os.path.relpath(abs_path, root)
    if rel.startswith(".."):
        return abs_path.replace(os.sep, "/")
    return rel.replace(os.sep, "/")


# ── per-clip row ─────────────────────────────────────────────────────────

#: Column order for the identity/timebase/provenance block of a per-clip CSV
#: (metric columns and ``status``/``failure_reason`` are appended by
#: :func:`write_clip_csv`). Matches the CSV contract in the run brief.
CLIP_BASE_COLUMNS: tuple[str, ...] = (
    "episode", "path", "role",
    "fps_probed", "dt", "n_frames", "width", "height", "codec",
    "robot", "reader_id", "suite_id", "limit_semantics",
)
CLIP_TAIL_COLUMNS: tuple[str, ...] = ("status", "failure_reason")

#: The rigidity column a report should quote, not ``rigidity_residual_mm``/
#: ``rigidity_wobble_mm`` -- see ``legacy_docs/DECISIONS.md`` D-A.
HEADLINE_RIGIDITY_METRIC = "rigidity_worst_bone_mm"

#: Printed in every leading ``#`` comment -- a suite is "which rulers were
#: computed", never "which clips were scored" (that's the config axes,
#: encoded in this file's own path). See ``legacy_docs/DECISIONS.md`` D-I.
SUITE_MEANING_NOTE = (
    "suite=which rulers were computed (see suite_id/suite_name); "
    "which clips were scored is decided by the config axes "
    "(embodiment/view/horizon/cache/generator) encoded in this file's own "
    "path, not by the suite")


@dataclass(frozen=True)
class ClipRow:
    """One scored (or never-scored) clip, ready to render as a CSV row."""

    group: tuple[str, ...]
    episode: str
    path: str
    role: str
    fps_probed: float | None
    dt: float | None
    n_frames: int | None
    width: int | None
    height: int | None
    codec: str | None
    robot: str
    reader_id: str
    suite_id: str
    suite_name: str
    limit_semantics: str
    metrics: dict[str, float | None] = field(default_factory=dict)
    metrics_unavailable: dict[str, str] = field(default_factory=dict)
    status: str = "unknown"
    failure_reason: str = ""


def _manifest_index(manifest_rows: Sequence[dict]) -> dict[str, dict]:
    return {r["path"]: r for r in manifest_rows if r.get("path")}


def _merged_clip_block(manifest_row: dict, record_clip: dict) -> dict:
    """Manifest row overlaid with the record's own ``clip`` block (wins on overlap).

    See ``legacy_docs/DECISIONS.md`` D-I for why the join exists.
    """
    return {**manifest_row, **record_clip}


def _clip_row_from(*, group: tuple[str, ...], merged: dict, run: dict,
                   metrics: dict, metrics_unavailable: dict, status: str,
                   failure_reason: str, path_rel: str) -> ClipRow:
    return ClipRow(
        group=group,
        # episode/role come from `merged`, not the bare manifest, since a
        # failed record's own `clip` block IS a manifest row (D-I) and
        # carries both even with no external manifest supplied.
        episode=str(merged.get("episode", "")),
        path=path_rel,
        role=str(merged.get("role", "")),
        fps_probed=merged.get("fps_probed"),
        dt=merged.get("dt"),
        n_frames=merged.get("n_frames"),
        width=merged.get("width", merged.get("w")),
        height=merged.get("height", merged.get("h")),
        codec=merged.get("codec"),
        robot=run.get("robot", ""),
        reader_id=run.get("reader_id", ""),
        suite_id=run.get("suite_id", ""),
        suite_name=run.get("suite_name", ""),
        limit_semantics=run.get("limit_semantics", ""),
        metrics=dict(metrics), metrics_unavailable=dict(metrics_unavailable),
        status=status, failure_reason=failure_reason,
    )


def build_clip_rows(records: Sequence[dict], manifest_rows: Sequence[dict],
                    data_root: str, *,
                    unscored_reason: str | None = None) -> list[ClipRow]:
    """Join scored ``records`` with ``manifest_rows`` into :class:`ClipRow` s.

    Parameters
    ----------
    records:
        Parsed ``results.jsonl`` rows (``kinescore.bench.store.iter_records``).
    manifest_rows:
        Rows from the manifest that produced ``records`` -- identity fields a
        successful record's own ``clip`` block never carries (D-I). May be empty.
    data_root:
        ``$KINESCORE_DATA_ROOT`` -- every clip's path is written relative to
        this (see :func:`relative_to_data_root`).
    unscored_reason:
        If given, every manifest row whose path never appears in ``records``
        (discovered but never scored) is emitted as its own ``status="skipped"``
        row with this as ``failure_reason`` and every metric column empty --
        lets a cell with *no reader at all* still produce a CSV. ``None``
        (default) omits never-scored rows.

    Returns
    -------
    list of :class:`ClipRow`, unsorted (grouping/sorting is
    :func:`group_rows`/:func:`sort_group`'s job).
    """
    man_idx = _manifest_index(manifest_rows)
    seen_paths: set[str] = set()
    rows: list[ClipRow] = []

    for rec in records:
        clip = dict(rec.get("clip") or {})
        path = clip.get("path")
        if path is None:
            continue
        seen_paths.add(path)
        manifest_row = man_idx.get(path, {})
        merged = _merged_clip_block(manifest_row, clip)
        run = rec.get("run") or {}
        metrics = rec.get("metrics") or {}
        unavailable = rec.get("metrics_unavailable") or {}
        status = rec.get("status", "unknown")
        failure_reason = ""
        if status == "failed":
            failure_reason = next(iter(unavailable.values()), "") or ""

        path_rel = relative_to_data_root(path, data_root)
        rows.append(_clip_row_from(
            group=group_key_for_path(path_rel),
            merged=merged, run=run, metrics=metrics,
            metrics_unavailable=unavailable, status=status,
            failure_reason=failure_reason, path_rel=path_rel))

    if unscored_reason is not None:
        for path, manifest_row in man_idx.items():
            if path in seen_paths:
                continue
            path_rel = relative_to_data_root(path, data_root)
            rows.append(_clip_row_from(
                group=group_key_for_path(path_rel),
                merged=manifest_row, run={}, metrics={}, metrics_unavailable={},
                status="skipped", failure_reason=unscored_reason,
                path_rel=path_rel))

    return rows


def group_rows(rows: Sequence[ClipRow]) -> dict[tuple[str, ...], list[ClipRow]]:
    """Bucket ``rows`` by :attr:`ClipRow.group`, preserving encounter order."""
    groups: dict[tuple[str, ...], list[ClipRow]] = {}
    for row in rows:
        groups.setdefault(row.group, []).append(row)
    return groups


def _badness(value: float | None, direction: str) -> float | None:
    """Orient a raw metric value so *higher = worse*; ``None`` stays ``None``."""
    if value is None:
        return None
    return value if direction == "lower_better" else -value


def sort_group(rows: Sequence[ClipRow], metric_key: str,
               direction: str) -> list[ClipRow]:
    """Sort ``rows`` worst-first by ``metric_key``, oriented by ``direction``.

    Rows whose ``metric_key`` is unavailable (``None`` -- unscored, failed,
    or the metric was structurally unobservable for that clip) sort last,
    never coerced to a value that would place them arbitrarily among the
    scored rows.
    """
    def key(row: ClipRow) -> tuple[bool, float]:
        b = _badness(row.metrics.get(metric_key), direction)
        return (b is not None, b if b is not None else 0.0)

    return sorted(rows, key=key, reverse=True)


# ── rendering ────────────────────────────────────────────────────────────

def _fmt(x: object) -> str:
    """Render one cell: ``None``/``NaN`` -> empty string, else a compact repr.

    Never renders a missing value as ``0`` -- see the module docstring.
    """
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x):
            return ""
        return f"{x:.6g}"
    return str(x)


def _headline_rigidity_note(metric_keys: Sequence[str]) -> str:
    """``" | headline_rigidity=..."`` suffix, empty when this suite lacks the column."""
    if HEADLINE_RIGIDITY_METRIC not in metric_keys:
        return ""
    return (f" | headline_rigidity={HEADLINE_RIGIDITY_METRIC} "
           f"(quote this, not rigidity_residual_mm/rigidity_wobble_mm -- "
           f"see legacy_docs/DECISIONS.md D-A)")


def write_clip_csv(path: str, rows: Sequence[ClipRow],
                   metric_keys: Sequence[str], *, sort_by: str,
                   direction: str) -> None:
    """Write one leaf CSV: identity/timebase/provenance + every metric.

    The first line is a ``#``-prefixed comment (self-describing: sort key +
    direction, the headline rigidity column when present) that standard CSV
    readers defaulting to ``comment='#'`` skip transparently. A single
    ``clips.csv`` is always one benchmark cell, so every row shares one
    nominal fps and within-file rate-dependent comparisons are safe -- the
    cross-fps caveat belongs on ``SUMMARY.csv`` (:func:`write_summary_csv`),
    not here.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    columns = list(CLIP_BASE_COLUMNS)
    for key in metric_keys:
        columns.append(key)
        columns.append(f"{key}_reason")
    columns.extend(CLIP_TAIL_COLUMNS)

    with open(path, "w", newline="") as f:
        f.write(f"# sort_by={sort_by} direction={direction} "
               f"order=worst_first n_rows={len(rows)}"
               f"{_headline_rigidity_note(metric_keys)} | {SUITE_MEANING_NOTE}\n")
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {
                "episode": row.episode, "path": row.path, "role": row.role,
                "fps_probed": _fmt(row.fps_probed), "dt": _fmt(row.dt),
                "n_frames": _fmt(row.n_frames), "width": _fmt(row.width),
                "height": _fmt(row.height), "codec": row.codec or "",
                "robot": row.robot, "reader_id": row.reader_id,
                "suite_id": row.suite_id,
                "limit_semantics": row.limit_semantics,
                "status": row.status, "failure_reason": row.failure_reason,
            }
            for key in metric_keys:
                out[key] = _fmt(row.metrics.get(key))
                out[f"{key}_reason"] = row.metrics_unavailable.get(key, "") or ""
            writer.writerow(out)


@dataclass(frozen=True)
class GroupSummary:
    """One :data:`SUMMARY.csv <write_summary_csv>` row: one cell's rollup."""

    group: tuple[str, ...]
    n_clips: int
    n_ok: int
    n_failed: int
    n_skipped: int
    robot: str
    reader_id: str
    suite_id: str
    suite_name: str
    fps: float | None
    medians: dict[str, float | None]


def summarize_group(rows: Sequence[ClipRow],
                    metric_keys: Sequence[str]) -> GroupSummary:
    """Roll ``rows`` (one benchmark cell) up into a :class:`GroupSummary`.

    Medians are taken over available (non-``None``) values only. ``robot``/
    ``reader_id``/``suite_id``/``suite_name`` are joined with ``"|"`` if a
    group somehow mixes more than one (it never should within one scored
    run) -- recording that plainly lets a reader notice the anomaly instead
    of it being hidden. See ``legacy_docs/DECISIONS.md`` D-I for why both
    ``suite_id`` and ``suite_name`` are needed to tell two folders apart.
    """
    n_ok = sum(1 for r in rows if r.status == "ok")
    n_failed = sum(1 for r in rows if r.status == "failed")
    n_skipped = sum(1 for r in rows if r.status == "skipped")
    robots = sorted({r.robot for r in rows if r.robot})
    readers = sorted({r.reader_id for r in rows if r.reader_id})
    suite_ids = sorted({r.suite_id for r in rows if r.suite_id})
    suite_names = sorted({r.suite_name for r in rows if r.suite_name})
    fpses = [r.fps_probed for r in rows if r.fps_probed is not None]
    medians = {
        key: (statistics.median(vals) if (vals := [
            r.metrics.get(key) for r in rows if r.metrics.get(key) is not None
        ]) else None)
        for key in metric_keys
    }
    return GroupSummary(
        group=rows[0].group if rows else (),
        n_clips=len(rows), n_ok=n_ok, n_failed=n_failed, n_skipped=n_skipped,
        robot="|".join(robots), reader_id="|".join(readers),
        suite_id="|".join(suite_ids), suite_name="|".join(suite_names),
        fps=statistics.median(fpses) if fpses else None, medians=medians)


def _fps_caveat(summaries: Sequence[GroupSummary]) -> str:
    """Warn when this file mixes rows recorded at different frame rates.

    See ``legacy_docs/DECISIONS.md`` D-I for why this matters (a real ~4x jerk gap
    that was entirely ``dt_exponent``, not physics). Silent when every group
    shares one fps or the file is empty.
    """
    fpses = sorted({round(s.fps, 6) for s in summaries if s.fps is not None})
    if len(fpses) <= 1:
        return ""
    listed = ", ".join(str(f) for f in fpses)
    return (f" | WARNING fps varies across rows in this file ({listed}) -- "
           f"median_* columns for any rate-dependent metric (dt_exponent != "
           f"0, see docs/METRICS.md) are NOT comparable across rows without "
           f"correcting by (fps_a/fps_b)**dt_exponent (docs/BENCHMARKING.md)")


def write_summary_csv(path: str, summaries: Sequence[GroupSummary],
                      metric_keys: Sequence[str]) -> None:
    """Write ``SUMMARY.csv``: one row per benchmark cell.

    ``suite_id``/``suite_name`` are their own columns -- see
    :func:`summarize_group`. The leading ``#`` comment adds the headline
    rigidity note and (unlike a single ``clips.csv``, which is always one
    fps) the cross-fps comparability warning, since this file's rows
    routinely span several generators at different native rates.
    """
    columns = ["group", "n_clips", "n_ok", "n_failed", "n_skipped",
              "robot", "reader", "suite_id", "suite_name", "fps"] + [
        f"median_{k}" for k in metric_keys]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write(f"# one row per benchmark cell, n_groups={len(summaries)}"
               f"{_headline_rigidity_note(metric_keys)}"
               f"{_fps_caveat(summaries)} | {SUITE_MEANING_NOTE}\n")
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for s in summaries:
            out = {
                "group": "/".join(s.group), "n_clips": s.n_clips,
                "n_ok": s.n_ok, "n_failed": s.n_failed,
                "n_skipped": s.n_skipped, "robot": s.robot,
                "reader": s.reader_id, "suite_id": s.suite_id,
                "suite_name": s.suite_name, "fps": _fmt(s.fps),
            }
            for key in metric_keys:
                out[f"median_{key}"] = _fmt(s.medians.get(key))
            writer.writerow(out)


@dataclass(frozen=True)
class ExportResult:
    """What :func:`export_csvs` wrote, for a CLI to report or a test to assert on."""

    out_dir: str
    n_rows: int
    n_groups: int
    metric_keys: tuple[str, ...]
    csv_paths: tuple[str, ...]
    summary_path: str


def export_csvs(results_path: str, out_dir: str, *, data_root: str,
                manifest_path: str | None = None,
                sort_by: str = "mean_jerk_mps3",
                suite_name: str = "invariant_v1",
                unscored_reason: str | None = None) -> ExportResult:
    """Read ``results_path``, write the mirrored CSV tree under ``out_dir``.

    Never re-scores anything; idempotent (every file fully rewritten).

    Parameters
    ----------
    results_path:
        A ``results.jsonl`` written by ``kinescore score``.
    out_dir:
        Root of the output tree; ``<out_dir>/<cache>/<embodiment>/<view>/
        <generator>/<horizon>/clips.csv`` per cell plus
        ``<out_dir>/SUMMARY.csv``.
    data_root:
        ``$KINESCORE_DATA_ROOT`` -- every clip's ``path`` column is relative to this.
    manifest_path:
        The manifest that produced ``results_path`` (D-I's join). ``None``
        skips it (identity columns blank for ``"ok"`` rows).
    sort_by:
        Metric key every leaf CSV is sorted by, worst-first. Must be a
        metric this run actually declares.
    suite_name:
        Fallback metric-column source only when ``results_path`` has zero
        records; otherwise the columns come from the scored data itself.
    unscored_reason:
        Forwarded to :func:`build_clip_rows`.

    Raises
    ------
    ValueError
        If ``sort_by`` is not a metric key this run declares.
    """
    import kinescore.metrics  # noqa: F401 -- populates the metric registry
    from kinescore.bench.suites import get_suite
    from kinescore.core.metric import get_metric

    records = list(iter_records(results_path))
    assert_uniform_schema(records)

    if records:
        metric_keys = tuple((records[0].get("metrics") or {}).keys())
    else:
        metric_keys = tuple(get_suite(suite_name).output_keys)

    if sort_by not in metric_keys:
        raise ValueError(
            f"--sort-by {sort_by!r} is not among this run's metrics: "
            f"{list(metric_keys)}")
    direction = get_metric(sort_by).spec.direction

    manifest_rows = load_manifest(manifest_path) if manifest_path else []
    clip_rows = build_clip_rows(records, manifest_rows, data_root,
                                unscored_reason=unscored_reason)

    csv_paths: list[str] = []
    summaries: list[GroupSummary] = []
    for group, rows in sorted(group_rows(clip_rows).items()):
        rows_sorted = sort_group(rows, sort_by, direction)
        path = os.path.join(out_dir, *group, "clips.csv")
        write_clip_csv(path, rows_sorted, metric_keys, sort_by=sort_by,
                       direction=direction)
        csv_paths.append(path)
        summaries.append(summarize_group(rows_sorted, metric_keys))

    summary_path = os.path.join(out_dir, "SUMMARY.csv")
    write_summary_csv(summary_path, summaries, metric_keys)

    return ExportResult(
        out_dir=out_dir, n_rows=len(clip_rows), n_groups=len(csv_paths),
        metric_keys=metric_keys, csv_paths=tuple(csv_paths),
        summary_path=summary_path)
