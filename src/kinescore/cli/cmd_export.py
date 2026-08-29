"""``kinescore export``: package a scored cell for a rating UI."""
from __future__ import annotations

import argparse

NAME = "export"
HELP = "package a scored cell as numbered clips + one segments.json"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from kinescore.cli._shared import add_config_arguments, add_detector_argument

    parser.add_argument("--cell", help="cell id whose own output dir is read")
    parser.add_argument("--results", default=None,
                        help="scored output dir holding results.jsonl "
                             "(overrides the cell's own)")
    parser.add_argument("--name", required=True,
                        help="bundle directory name, e.g. fastercache_humanoid_sv")
    parser.add_argument("--dest", default=None,
                        help="where the bundle directory is written "
                             "(default: $KINESCORE_OUTPUT_DIR/web)")
    add_detector_argument(parser)
    add_config_arguments(parser)


def run(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from kinescore.cli._shared import load, resolve_cell, resolve_detectors
    from kinescore.paths import env_path
    from kinescore.video.bundle import write_bundle
    from kinescore.video.render import read_results

    if args.results:
        results_dir = Path(args.results)
    elif args.cell:
        results_dir = Path(resolve_cell(load(args), args.cell).output_dir)
    else:
        raise SystemExit("pass --cell or --results")

    rows = read_results(results_dir / "results.jsonl")
    scored = [r for r in rows if r.get("segments")]
    if not scored:
        raise SystemExit(f"no scored rows in {results_dir}")
    names = resolve_detectors(args.detectors, set(scored[0]["violations"]))

    summary_path = results_dir / "summary.json"
    summary = (json.loads(summary_path.read_text())
               if summary_path.exists() else None)
    dest = Path(args.dest) if args.dest else env_path("KINESCORE_OUTPUT_DIR") / "web"
    write_bundle(rows, dest / args.name, names, summary=summary,
                 log=lambda m: print(m))
    return 0
