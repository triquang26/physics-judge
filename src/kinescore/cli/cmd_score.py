"""``kinescore score``: run the benchmark over a manifest, streaming to disk.

Thin CLI wrapper around :func:`kinescore.bench.runner.run`. The one piece of
logic that lives here rather than there is the timebase cross-check: every
row's ``fps``/``dt`` is re-resolved through
:func:`kinescore.cli._scoring.apply_resolved_timebase` (which calls
:func:`kinescore.video.probe.resolve_timebase`) before scoring, so
``--fps``/``--dt`` can never silently win over what the file on disk actually
is -- see that function's docstring and ``kinescore.video.probe``'s module
docstring for the defect (D3) this replaces.
"""
from __future__ import annotations

import argparse
import os
import sys

HELP = "score clips against a manifest with a trained reader"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from kinescore.cli._scoring import add_common_arguments

    parser.add_argument("--manifest", required=True,
                        help="manifest file (.parquet or .json) from "
                             "`kinescore manifest`")
    add_common_arguments(parser, require_reader=True)
    parser.add_argument("--out", required=True,
                        help="output directory; results.jsonl is written here")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true",
                        help="skip clips already scored by this reader+suite")
    resume.add_argument("--force", action="store_true",
                        help="truncate results.jsonl and rescore everything")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="cap frames per clip, applied after any stride "
                             "(0 = every probed frame)")
    parser.add_argument("--quiet", action="store_true",
                        help="don't print one line per scored clip")


def run(args: argparse.Namespace) -> int:
    from kinescore.bench.manifest import load_manifest
    from kinescore.bench.runner import run as run_bench
    from kinescore.cli._provenance import provenance_block, write_json
    from kinescore.cli._scoring import (
        apply_resolved_timebase,
        build_scorer,
        view_layout_from_args,
    )
    from kinescore.core.clip import TimebaseError

    view_layout = view_layout_from_args(args)
    rows = load_manifest(args.manifest)
    if not rows:
        print(f"[score] manifest {args.manifest!r} has no rows", file=sys.stderr)
        return 1

    print(f"[score] cross-checking timebase for {len(rows)} clip(s) "
          f"against ffprobe...")
    try:
        rows = apply_resolved_timebase(rows, fps=args.fps, dt=args.dt,
                                       view_layout=view_layout)
    except (ValueError, TimebaseError) as exc:
        print(f"[score] timebase error: {exc}", file=sys.stderr)
        raise

    scorer = build_scorer(args, view_layout)
    if scorer.reader.limit_semantics == "raw_rad":
        print(f"[score] reader {scorer.reader.reader_id!r} is raw_rad: "
              f"limit_violation_frac/limit_excess_rad are OBSERVABLE for this "
              f"run (see docs/PROVENANCE.md D7). No sigma-gate is applied by "
              f"this command -- every frame is scored; pass a `gate=` to "
              f"kinescore.core.scorer.Scorer yourself for a gated run.")

    results_path = os.path.join(args.out, "results.jsonl")

    def _progress(row: dict, status: str) -> None:
        if not args.quiet:
            print(f"[score] {status:8s} {row.get('path')}")

    summary = run_bench(rows, scorer, results_path, resume=args.resume,
                        force=args.force, max_frames=args.max_frames,
                        view_layout=view_layout, progress=_progress)
    print(f"[score] done: {summary}")

    dt_sources = sorted({r.get("dt_source") for r in rows})
    dts = sorted({r.get("dt") for r in rows})
    prov = provenance_block(
        suite_id=scorer.suite.suite_id, suite_name=scorer.suite.name,
        robot=args.robot, reader_id=scorer.reader.reader_id,
        resolved_dt=dts if len(dts) > 1 else (dts[0] if dts else None),
        dt_sources=dt_sources, summary=summary)
    write_json(os.path.join(args.out, "provenance.json"), prov)

    return 0 if summary["n_failed"] == 0 else 1
