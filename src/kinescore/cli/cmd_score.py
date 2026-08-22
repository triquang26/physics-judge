"""``kinescore score``: judge a directory of generated clips for one cell."""
from __future__ import annotations

import argparse

NAME = "score"
HELP = "score generated clips against physics thresholds calibrated on real motion"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from kinescore.cli._shared import add_config_arguments

    parser.add_argument("--cell", help="cell id, e.g. single_arm.mv3_row.ctrlworld")
    parser.add_argument("--list", action="store_true",
                        help="print every declared cell and its status")
    parser.add_argument("--videos", default=None,
                        help="directory of clips to score (default: the "
                             "cell's canonical score tree). Searched "
                             "recursively for *.mp4")
    parser.add_argument("--checkpoint", default=None,
                        help="reader checkpoint (default: the reader's own)")
    parser.add_argument("--out", default=None,
                        help="output directory (default: the cell's own)")
    parser.add_argument("--calibration-clips", type=int, default=24,
                        help="real clips the thresholds are fitted on")
    parser.add_argument("--percentile", type=float, default=95.0,
                        help="threshold percentile over real motion")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="cap frames per clip (0 = all)")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap clips scored (0 = all)")
    parser.add_argument("--device", default="cuda")
    add_config_arguments(parser)


def _clips(root: str, limit: int) -> list[str]:
    import glob
    import os

    found = sorted(glob.glob(os.path.join(root, "**", "*.mp4"), recursive=True))
    return found[:limit] if limit else found


def run(args: argparse.Namespace) -> int:
    import json

    import torch

    from kinescore.cli._shared import load, now, resolve_cell
    from kinescore.core.context import ClipContext
    from kinescore.readers.checkpoint import ReaderExpectation, load_reader
    from kinescore.registry.provenance import (
        run_manifest,
        sha256_file,
        write_run_manifest,
    )
    from kinescore.robots import get_robot
    from kinescore.video.probe import resolve_timebase
    from kinescore.video.reader import load_rgb
    from kinescore.violations import ViolationScorer

    registry = load(args)
    if args.list or not args.cell:
        for cell_id, cell in sorted(registry.cells.items()):
            print(f"{cell_id:42s} {cell.robot:16s} {cell.reader.reader_id:34s} "
                  f"{cell.status or 'ready'}")
        return 0 if args.list else 1

    cell = resolve_cell(registry, args.cell)
    if cell.status:
        raise SystemExit(f"cell {cell.cell_id!r}: {cell.status}")

    started = now()
    layout = cell.view.layout()
    robot = get_robot(cell.robot)
    checkpoint = args.checkpoint or str(cell.reader.checkpoint_path)
    reader = load_reader(
        checkpoint, robot=robot, view_layout=layout, device=args.device,
        reader_id=cell.reader.reader_id,
        expect=ReaderExpectation(
            cell_id=cell.cell_id, robot=cell.robot,
            view_id=cell.view.view_id, n_views=layout.n_views,
            packing=layout.packing))

    def read(path: str) -> ClipContext:
        clip = resolve_timebase(path, view_layout=layout)
        cell.view.check_frame_size(clip.width, clip.height)
        frames = load_rgb(clip, max_frames=args.max_frames)
        with torch.no_grad():
            out = reader.read(frames.to(args.device))
        return ClipContext(dt=clip.dt, P=out.P.float().cpu(), robot=robot,
                           flags={"path": path})

    real_root = cell.reader.train_tree / "videos" / "val"
    real = _clips(str(real_root), args.calibration_clips)
    if not real:
        raise SystemExit(
            f"no real clips under {real_root} to calibrate thresholds on -- "
            f"run `kinescore data --reader {cell.reader.reader_id}` first")
    print(f"[score] calibrating on {len(real)} real clip(s) from {real_root}")
    scorer = ViolationScorer()
    scorer.calibrate([read(p) for p in real], pct=args.percentile)

    videos = args.videos or str(cell.score_tree)
    clips = _clips(videos, args.limit)
    if not clips:
        raise SystemExit(f"no *.mp4 under {videos}")

    out_dir = args.out or str(cell.output_dir)
    import os

    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.jsonl")
    n_failed = 0
    with open(results_path, "w") as f:
        for i, path in enumerate(clips, 1):
            try:
                report = scorer.score(read(path))
            except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
                n_failed += 1
                f.write(json.dumps({"path": path, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                print(f"[score] failed  {path}: {exc}")
                continue
            f.write(json.dumps({"path": path, "cell_id": cell.cell_id,
                                "violations": report}) + "\n")
            print(f"[score] {i}/{len(clips)} {os.path.basename(path)}")

    summary = {
        "cell_id": cell.cell_id,
        "reader_id": cell.reader.reader_id,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint),
        "videos": videos,
        "n_clips": len(clips),
        "n_failed": n_failed,
        "n_calibration_clips": len(real),
        "percentile": args.percentile,
        "thresholds": scorer.thresholds(),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    write_run_manifest(out_dir, run_manifest(
        NAME, started_at=started, sources=registry.sources, extra=summary))
    print(f"[score] {len(clips) - n_failed} scored, {n_failed} failed -> {out_dir}")
    return 0 if n_failed == 0 else 1
