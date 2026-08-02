"""``kinescore export``: turn a scored run into a mirrored tree of CSVs.

Thin CLI wrapper around :func:`kinescore.bench.csv_export.export_csvs` -- see
that module's docstring for the join/grouping/no-composite-score design.
This command touches no GPU and no video file: it only reads
``results.jsonl`` (already written by ``kinescore score``) and the manifest
that produced it, so it is safe and fast to rerun after a partial or resumed
scoring run, or after changing ``--sort-by``.
"""
from __future__ import annotations

import argparse
import os
import sys

NAME = "export"
HELP = "export a scored run as one CSV per benchmark cell, mirroring the input tree"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results", required=True,
                        help="directory written by `kinescore score` "
                             "(must contain results.jsonl)")
    parser.add_argument("--out", required=True,
                        help="output directory; <out>/<cache>/<embodiment>/"
                             "<view>/<generator>/<horizon>/clips.csv per cell "
                             "plus <out>/SUMMARY.csv")
    parser.add_argument("--data-root", default=None,
                        help="$KINESCORE_DATA_ROOT override; every clip's "
                             "path column is written relative to this "
                             "(default: $KINESCORE_DATA_ROOT)")
    parser.add_argument("--manifest", default=None,
                        help="manifest (.parquet/.json) that produced "
                             "--results (default: auto-detect "
                             "<results>/bench_manifest.parquet or .json)")
    parser.add_argument("--sort-by", default="mean_jerk_mps3",
                        help="metric every leaf CSV is sorted by, worst-"
                             "first (default: mean_jerk_mps3)")
    parser.add_argument("--suite", default="invariant_v1",
                        help="fallback metric-column source when --results "
                             "has zero scored rows (default: invariant_v1)")
    parser.add_argument("--unscored-reason", default=None,
                        help="if given, every manifest row never scored "
                             "(present in --manifest but absent from "
                             "results.jsonl) is written as its own row with "
                             "status=skipped and this text as "
                             "failure_reason, instead of being omitted -- "
                             "use this for a cell with no reader at all "
                             "(e.g. --unscored-reason 'not scored: robot is "
                             "Airbot MMK2, no reader')")


def run(args: argparse.Namespace) -> int:
    from kinescore import paths
    from kinescore.bench.csv_export import export_csvs
    from kinescore.bench.manifest import autodetect_manifest
    from kinescore.cli._provenance import provenance_block, write_json

    results_path = os.path.join(args.results, "results.jsonl")
    if not os.path.exists(results_path):
        print(f"[export] no results.jsonl at {results_path!r}", file=sys.stderr)
        return 1

    # required=False: export has a documented degraded path (blank
    # episode/role identity + a printed warning) when no manifest is found,
    # unlike aggregate/rank which cannot proceed without one.
    manifest_path = args.manifest or autodetect_manifest(args.results, required=False)
    if manifest_path is None:
        print(f"[export] no manifest found under {args.results!r} and "
             f"--manifest not given; episode/role identity will be blank "
             f"for successfully-scored clips (see legacy_docs/SCHEMA.md's clip "
             f"block shape) -- pass --manifest explicitly to fix this",
             file=sys.stderr)

    data_root = args.data_root or str(paths.env_path("KINESCORE_DATA_ROOT"))

    try:
        result = export_csvs(
            results_path, args.out, data_root=data_root,
            manifest_path=manifest_path, sort_by=args.sort_by,
            suite_name=args.suite, unscored_reason=args.unscored_reason)
    except ValueError as exc:
        print(f"[export] {exc}", file=sys.stderr)
        return 1

    print(f"[export] {result.n_rows} row(s) across {result.n_groups} cell(s) "
         f"-> {args.out}")
    for path in result.csv_paths:
        print(f"[export]   {path}")
    print(f"[export] summary -> {result.summary_path}")

    prov = provenance_block(
        results=args.results, manifest=manifest_path, out=args.out,
        data_root=data_root, sort_by=args.sort_by, suite=args.suite,
        unscored_reason=args.unscored_reason, n_rows=result.n_rows,
        n_groups=result.n_groups, metric_keys=list(result.metric_keys))
    write_json(os.path.join(args.out, "export_provenance.json"), prov)

    return 0
