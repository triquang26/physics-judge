"""``kinescore rank``: sort clips by ONE physical-unit ruler -- never a score.

The sorting/rendering logic (:func:`~kinescore.bench.rank.badness`,
:func:`~kinescore.bench.rank.unpaired_rows`,
:func:`~kinescore.bench.rank.paired_rows`,
:func:`~kinescore.bench.rank.render_rows`) lives in :mod:`kinescore.bench.rank`
-- see that module's docstring for why there is no composite score and what
``--paired`` answers that the default mode does not. This module is the
argparse shell: parse ``--metric``/``--method``/``--paired``/``--top``/
``--format``, load and join the scores, sort, and print.
"""
from __future__ import annotations

import argparse
import sys

NAME = "rank"
HELP = "sort clips by one physical-unit ruler (no composite score exists)"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dir", help="directory written by `kinescore score` "
                                    "(results.jsonl) and `kinescore manifest` "
                                    "(bench_manifest.*)")
    parser.add_argument("--metric", required=True,
                        help="registered metric key to sort by, e.g. "
                             "mean_jerk_mps3 (bare key, not the "
                             "'metrics.' prefixed column name; see "
                             "`kinescore describe` for the full list)")
    parser.add_argument("--method", action="append", default=None,
                        help="restrict to method(s) (repeatable; default: "
                             "every method present in the results)")
    parser.add_argument("--paired", action="store_true",
                        help="sort episodes by their own paired delta "
                             "(pred - gt) instead of sorting individual "
                             "clips by raw value")
    parser.add_argument("--top", type=int, default=None,
                        help="keep only the N worst-ranked rows (default: all)")
    parser.add_argument("--format", choices=("table", "csv", "json"), default="table")
    parser.add_argument("--manifest", default=None,
                        help="override manifest path (default: auto-detect "
                             "<dir>/bench_manifest.parquet or .json)")
    parser.add_argument("--results", default=None,
                        help="override results.jsonl path (default: <dir>/results.jsonl)")


def run(args: argparse.Namespace) -> int:
    import os

    import kinescore.metrics  # noqa: F401  (side effect: populates the metric registry)
    from kinescore.bench.manifest import autodetect_manifest
    from kinescore.bench.rank import badness, paired_rows, render_rows, unpaired_rows
    from kinescore.bench.stats import load_scores
    from kinescore.core.metric import get_metric

    try:
        spec = get_metric(args.metric).spec
    except KeyError as exc:
        print(f"[rank] {exc}", file=sys.stderr)
        return 2

    if spec.dt_exponent is None:
        print(f"[rank] note: {args.metric!r} has dt_exponent=None -- do not "
             f"compare these values across clips scored at different frame "
             f"rates (see docs/METRICS.md)", file=sys.stderr)

    results_path = args.results or os.path.join(args.dir, "results.jsonl")
    if not os.path.exists(results_path):
        print(f"[rank] no results.jsonl at {results_path!r}", file=sys.stderr)
        return 1
    # required=True: rank immediately joins the manifest against
    # results.jsonl -- there is nothing useful to do without one.
    manifest_path = args.manifest or autodetect_manifest(args.dir)

    df = load_scores(results_path, manifest_path)
    if df.empty:
        print(f"[rank] no scored rows joined between {results_path!r} "
             f"and {manifest_path!r}", file=sys.stderr)
        return 1

    col = f"metrics.{args.metric}"
    if col not in df.columns:
        print(f"[rank] {args.metric!r} is not present in the joined results "
             f"(no clip declared this metric); columns available: "
             f"{sorted(c for c in df.columns if c.startswith('metrics.'))}",
             file=sys.stderr)
        return 1

    if args.paired:
        rows = paired_rows(df, col, args.method)
        rows.sort(key=lambda r: badness(r["delta"], spec.direction), reverse=True)
        columns = ["method", "episode", "delta"]
    else:
        rows = unpaired_rows(df, col, args.method)
        rows.sort(key=lambda r: badness(r["value"], spec.direction), reverse=True)
        columns = ["method", "episode", "role", "value"]

    if args.top is not None:
        rows = rows[: args.top]
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    columns = ["rank", *columns]

    print(f"[rank] {args.metric} ({spec.units}, {spec.direction}) -- "
         f"{len(rows)} row(s), worst first", file=sys.stderr)
    print(render_rows(rows, args.format, columns))
    return 0
