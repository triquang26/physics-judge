"""``kinescore aggregate``: results.jsonl + manifest -> per-method/metric stats.

Thin wrapper around :func:`kinescore.bench.stats.load_scores` +
:func:`kinescore.bench.stats.aggregate`. The one thing worth documenting here
is what happens when :func:`aggregate` raises: it does so exactly when a
method's rows span more than one ``suite_id`` (defect D3, one layer up -- see
that function's docstring), and this command does **not** catch that error
into a skip. ``--allow-mixed-suites`` is the explicit, opt-in escape hatch;
silently dropping a method/metric pair whose suite_ids disagree would hide
the exact problem the guard exists to surface.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

HELP = "compute paired physics-tax statistics from a scored run"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dir", help="directory written by `kinescore score` "
                                    "(results.jsonl) and `kinescore manifest` "
                                    "(bench_manifest.*)")
    parser.add_argument("--manifest", default=None,
                        help="override manifest path (default: auto-detect "
                             "<dir>/bench_manifest.parquet or .json)")
    parser.add_argument("--results", default=None,
                        help="override results.jsonl path (default: <dir>/results.jsonl)")
    parser.add_argument("--method", action="append", default=None,
                        help="method(s) to aggregate (repeatable; default: "
                             "every method present in the results)")
    parser.add_argument("--metric", action="append", default=None,
                        help="metrics.<key> column(s) to aggregate (repeatable; "
                             "default: every metrics.* column present)")
    parser.add_argument("--baseline", default=None,
                        help="baseline method for the second-difference comparison")
    parser.add_argument("--allow-mixed-suites", action="store_true",
                        help="pool rows whose suite_id differs instead of "
                             "raising (see kinescore.bench.stats.aggregate)")
    parser.add_argument("--bootstrap", type=int, default=10000, dest="B",
                        help="bootstrap resamples for the median CI (default: 10000)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None,
                        help="stats.json path (default: <dir>/stats.json)")


def _autodetect_manifest(dir_: str) -> str:
    for name in ("bench_manifest.parquet", "bench_manifest.json"):
        candidate = os.path.join(dir_, name)
        if os.path.exists(candidate):
            return candidate
    matches = sorted(glob.glob(os.path.join(dir_, "*manifest*")))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"no bench_manifest.parquet/.json found under {dir_!r}; pass "
        f"--manifest explicitly")


def run(args: argparse.Namespace) -> int:
    from kinescore.bench.stats import aggregate, load_scores
    from kinescore.cli._provenance import provenance_block, write_json

    results_path = args.results or os.path.join(args.dir, "results.jsonl")
    if not os.path.exists(results_path):
        print(f"[aggregate] no results.jsonl at {results_path!r}", file=sys.stderr)
        return 1
    manifest_path = args.manifest or _autodetect_manifest(args.dir)

    df = load_scores(results_path, manifest_path)
    if df.empty:
        print(f"[aggregate] no scored rows joined between {results_path!r} "
             f"and {manifest_path!r}", file=sys.stderr)
        return 1

    methods = args.method or sorted(
        m for m in df["method"].dropna().astype(str).unique().tolist())
    metrics = args.metric or sorted(
        c for c in df.columns if c.startswith("metrics."))
    if not metrics:
        print("[aggregate] no metrics.* columns in the joined results",
             file=sys.stderr)
        return 1

    results = []
    for method in methods:
        for metric in metrics:
            try:
                out = aggregate(df, method, metric, baseline=args.baseline,
                                allow_mixed_suites=args.allow_mixed_suites,
                                B=args.B, seed=args.seed)
            except ValueError as exc:
                print(f"[aggregate] {method}/{metric}: {exc}", file=sys.stderr)
                raise
            if out["n"] > 0:
                results.append(out)

    out_path = args.out or os.path.join(args.dir, "stats.json")
    prov = provenance_block(n_summaries=len(results), methods=methods,
                            metrics=metrics, baseline=args.baseline,
                            allow_mixed_suites=args.allow_mixed_suites)
    write_json(out_path, {"provenance": prov, "results": results})
    print(f"[aggregate] wrote {len(results)} method/metric summary(ies) -> {out_path}")
    return 0
