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
                        help="directory of clips to score, searched "
                             "recursively for *.mp4 (default: the clips "
                             "bench/manifest.json assigns to this cell)")
    parser.add_argument("--pattern", default="*.mp4",
                        help="basename glob a --videos clip must match; a "
                             "tree that stores prediction and ground truth "
                             "side by side is scored on one of them, e.g. "
                             "full_pred.mp4")
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
    parser.add_argument("--frame-chunk", type=int, default=32,
                        help="frames encoded per backbone call (0 = whole "
                             "clip at once)")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap clips scored (0 = all)")
    parser.add_argument("--device", default="cuda")
    add_config_arguments(parser)


def _clips(root: str, limit: int, pattern: str = "*.mp4") -> list[str]:
    import glob
    import os

    found = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    return found[:limit] if limit else found


def _videos_id(path: str, root: str) -> str:
    """Unique id for a --videos clip: its path below the root, no suffix."""
    import os

    rel = os.path.relpath(path, root)
    return os.path.splitext(rel)[0].replace(os.sep, "__")


def run(args: argparse.Namespace) -> int:
    import json
    import os
    import pathlib

    import torch

    from kinescore.cli._shared import (
        load,
        now,
        resolve_cell,
        resolve_detectors,
    )
    from kinescore.core.context import ClipContext
    from kinescore.readers.checkpoint import ReaderExpectation, load_reader
    from kinescore.registry.bench import bench_root, load_bench, select
    from kinescore.registry.provenance import (
        git_state,
        run_manifest,
        sha256_file,
        write_run_manifest,
    )
    from kinescore.robots import get_robot
    from kinescore.video.probe import resolve_timebase
    from kinescore.video.reader import load_rgb
    from kinescore.video.render import read_results, render_results
    from kinescore.violations import ViolationScorer, segments
    from kinescore.violations.table import write_segment_table, write_table

    registry = load(args)
    if args.list or not args.cell:
        for cell_id, cell in sorted(registry.cells.items()):
            print(f"{cell_id:42s} {cell.robot:16s} {cell.reader.reader_id:34s} "
                  f"{cell.status or 'ready'}")
        return 0 if args.list else 1

    cell = resolve_cell(registry, args.cell)
    if cell.status:
        raise SystemExit(f"cell {cell.cell_id!r}: {cell.status}")

    checkpoint = args.checkpoint or str(cell.reader.checkpoint_path)
    if not os.path.exists(checkpoint):
        raise SystemExit(
            f"no reader checkpoint at {checkpoint} -- run "
            f"`kinescore train --reader {cell.reader.reader_id}` first, or "
            f"pass --checkpoint")

    started, git = now(), git_state()
    layout = cell.view.layout()
    robot = get_robot(cell.robot)
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
            out = reader.read(frames.to(args.device),
                              frame_chunk=args.frame_chunk)
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

    if args.videos:
        source = args.videos
        clips = [(p, {"id": _videos_id(p, args.videos)})
                 for p in _clips(args.videos, args.limit, args.pattern)]
    else:
        source = str(bench_root())
        picked = select(load_bench(), cell.select)
        if args.limit:
            picked = picked[:args.limit]
        clips = [(i.path, i.coords()) for i in picked]
    if not clips:
        raise SystemExit(
            f"no clips for {cell.cell_id!r} under {source} (select "
            f"{cell.select}) -- run `kinescore pull --what bench`, or point "
            f"--videos at a directory of *.mp4")

    all_detectors = [d.name for d in scorer.detectors]
    out_dir = args.out or str(cell.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.jsonl")
    n_failed = 0
    by_role: dict[str, int] = {}
    with open(results_path, "w") as f:
        for i, (path, coords) in enumerate(clips, 1):
            record = {"path": path, "cell_id": cell.cell_id, **coords}
            try:
                record["violations"] = scorer.score(read(path))
            except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
                n_failed += 1
                record["error"] = f"{type(exc).__name__}: {exc}"
                f.write(json.dumps(record) + "\n")
                print(f"[score] failed  {path}: {exc}")
                continue
            record["segments"] = segments.report(
                record["violations"], all_detectors)
            role = coords.get("role", "")
            by_role[role] = by_role.get(role, 0) + 1
            f.write(json.dumps(record) + "\n")
            print(f"[score] {i}/{len(clips)} {os.path.basename(path)}")

    summary = {
        "cell_id": cell.cell_id,
        "reader_id": cell.reader.reader_id,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint),
        "videos": source,
        "select": dict(cell.select),
        "n_clips": len(clips),
        "n_failed": n_failed,
        "n_scored_by_role": by_role,
        "n_calibration_clips": len(real),
        "percentile": args.percentile,
        "thresholds": scorer.thresholds(),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    write_run_manifest(out_dir, run_manifest(
        NAME, started_at=started, git=git, sources=registry.sources, extra=summary))
    print(f"[score] {len(clips) - n_failed} scored, {n_failed} failed -> {out_dir}")

    rows = read_results(pathlib.Path(out_dir) / "results.jsonl")
    scored_rows = [r for r in rows if r.get("segments")]
    if scored_rows:
        names = resolve_detectors(None, set(scored_rows[0]["violations"]))
        out = pathlib.Path(out_dir)
        print(f"[score] {write_table(scored_rows, out / 'metrics.csv', names)}")
        print(f"[score] "
              f"{write_segment_table(scored_rows, out / 'segments.csv', names)}")
        render_results(scored_rows, pathlib.Path(out_dir) / "render", names)
    return 0 if n_failed == 0 else 1
