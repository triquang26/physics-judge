"""``kinescore render``: watch a scored cell, segment by segment."""
from __future__ import annotations

import argparse

NAME = "render"
HELP = "draw the violation timeline onto a scored cell's clips"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from kinescore.cli._shared import add_config_arguments

    parser.add_argument("--cell", required=True, help="scored cell to render")
    parser.add_argument("--out", default=None,
                        help="directory for the rendered clips "
                             "(default: <cell output>/render)")
    parser.add_argument("--flagged-only", action="store_true",
                        help="skip clips no detector flagged")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="playback rate of the rendered files")
    parser.add_argument("--no-reel", action="store_true",
                        help="write per-clip files only, not the joined reel")
    add_config_arguments(parser)


def _rows(results):
    import json

    if not results.exists():
        raise SystemExit(
            f"no results at {results} -- run `kinescore score` for this cell "
            f"first")
    return [json.loads(line) for line in results.read_text().splitlines()
            if line.strip()]


def _is_flagged(row) -> bool:
    return any(d.get("intervals")
               for d in (row.get("violations") or {}).values())


def run(args: argparse.Namespace) -> int:
    from pathlib import Path

    import imageio.v3 as iio
    import numpy as np

    from kinescore.cli._shared import load, resolve_cell
    from kinescore.video.overlay import render_clip

    cell = resolve_cell(load(args), args.cell)
    rows = _rows(Path(cell.output_dir) / "results.jsonl")
    if args.flagged_only:
        rows = [r for r in rows if _is_flagged(r)]
    if not rows:
        raise SystemExit(f"cell {cell.cell_id!r}: nothing to render")

    out = Path(args.out) if args.out else Path(cell.output_dir) / "render"
    out.mkdir(parents=True, exist_ok=True)
    reel = []

    for n, row in enumerate(rows, 1):
        frames = np.asarray(iio.imread(row["path"]))
        drawn = render_clip(frames, row)
        path = out / f"{row['id']}_{row['role']}.mp4"
        iio.imwrite(path, drawn, fps=args.fps, codec="libx264",
                    macro_block_size=1)
        if not args.no_reel:
            reel.append(drawn)
        print(f"[render] {n}/{len(rows)} {path.name} "
              f"{'flagged' if _is_flagged(row) else 'clean'}")

    if reel:
        joined = out / "reel.mp4"
        iio.imwrite(joined, np.concatenate(reel), fps=args.fps,
                    codec="libx264", macro_block_size=1)
        print(f"[render] reel -> {joined}")
    print(f"[render] {len(rows)} clip(s) -> {out}")
    return 0
