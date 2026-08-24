"""``kinescore render``: redraw a scored cell, segment by segment."""
from __future__ import annotations

import argparse

NAME = "render"
HELP = "draw the violation timeline onto a scored cell's clips"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from kinescore.cli._shared import add_config_arguments, add_detector_argument

    parser.add_argument("--cell", required=True, help="scored cell to redraw")
    parser.add_argument("--out", default=None,
                        help="directory for the rendered clips "
                             "(default: <cell output>/render)")
    parser.add_argument("--flagged-only", action="store_true",
                        help="skip clips no reported detector flagged")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="playback rate of the rendered files")
    parser.add_argument("--no-reel", action="store_true",
                        help="write per-clip files only, not the joined reel")
    add_detector_argument(parser)
    add_config_arguments(parser)


def run(args: argparse.Namespace) -> int:
    from pathlib import Path

    from kinescore.cli._shared import load, resolve_cell, resolve_detectors
    from kinescore.video.render import is_flagged, read_results, render_results

    cell = resolve_cell(load(args), args.cell)
    rows = read_results(Path(cell.output_dir) / "results.jsonl")
    names = resolve_detectors(args.detectors, set(rows[0]["violations"]))
    if args.flagged_only:
        rows = [r for r in rows if is_flagged(r, names)]
    if not rows:
        raise SystemExit(f"cell {cell.cell_id!r}: nothing to render")

    out = Path(args.out) if args.out else Path(cell.output_dir) / "render"
    render_results(rows, out, names, fps=args.fps, reel=not args.no_reel)
    print(f"[render] {len(rows)} clip(s) -> {out}")
    return 0
