"""``kinescore render``: redraw a scored cell, segment by segment."""
from __future__ import annotations

import argparse

NAME = "render"
HELP = "draw the violation timeline onto a scored cell's clips"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from kinescore.cli._shared import add_config_arguments, add_detector_argument

    parser.add_argument("--cell", help="cell id whose own output dir is read")
    parser.add_argument("--results", default=None,
                        help="scored output dir holding results.jsonl "
                             "(overrides the cell's own)")
    parser.add_argument("--out", default=None,
                        help="directory for the rendered clips "
                             "(default: <results dir>/render)")
    parser.add_argument("--flagged-only", action="store_true",
                        help="skip clips no reported detector flagged")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="playback rate of the rendered files")
    parser.add_argument("--no-reel", action="store_true",
                        help="write per-clip files only, no reels")
    add_detector_argument(parser)
    add_config_arguments(parser)


def run(args: argparse.Namespace) -> int:
    from pathlib import Path

    from kinescore.cli._shared import load, resolve_cell, resolve_detectors
    from kinescore.video.render import is_flagged, read_results, render_results

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
    if args.flagged_only:
        scored = [r for r in scored if is_flagged(r, names)]
    if not scored:
        raise SystemExit("nothing to render")

    out = Path(args.out) if args.out else results_dir / "render"
    render_results(scored, out, names, fps=args.fps, reel=not args.no_reel)
    print(f"[render] {len(scored)} clip(s) -> {out}")
    return 0
