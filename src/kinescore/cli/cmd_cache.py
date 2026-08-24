"""``kinescore cache``: encode a reader's train tree with the frozen backbone."""
from __future__ import annotations

import argparse

NAME = "cache"
HELP = "precompute frozen-backbone tokens for a reader's train tree"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from kinescore.cli._shared import add_config_arguments

    parser.add_argument("--reader", required=True, help="reader id")
    parser.add_argument("--splits", default="train,val",
                        help="comma-separated splits to encode")
    parser.add_argument("--device", default="cuda",
                        help="where the backbone runs")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap episodes per split (0 = all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-encode episodes that are already cached")
    parser.add_argument("--frame-chunk", type=int, default=32,
                        help="frames per backbone call; lower this if a long "
                             "episode exhausts GPU memory")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="cap decoded frames per episode (0 = all)")
    add_config_arguments(parser)


def run(args: argparse.Namespace) -> int:
    from kinescore.backbones.default import build_backbone
    from kinescore.cli._shared import load, now, resolve_reader
    from kinescore.registry.provenance import (
        git_state,
        run_manifest,
        write_run_manifest,
    )
    from kinescore.training.cache import CacheBuilder

    registry = load(args)
    reader = resolve_reader(registry, args.reader)
    tree = reader.train_tree
    if not tree.is_dir():
        raise SystemExit(
            f"no train tree at {tree} -- run `kinescore data --reader "
            f"{reader.reader_id}` first")

    started, git = now(), git_state()
    layout = reader.view.layout()
    backbone = build_backbone(layout, device=args.device)
    builder = CacheBuilder(backbone, layout, reader.reader_id)

    totals: dict[str, dict[str, int]] = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        totals[split] = builder.build_split(
            video_root=str(tree / "videos"),
            annotation_root=str(tree / "annotation"),
            out_root=str(reader.cache_dir), split=split, limit=args.limit,
            device=args.device, overwrite=args.overwrite,
            max_frames=args.max_frames, frame_chunk=args.frame_chunk,
            progress=lambda m: print(f"[cache] {m}"))

    write_run_manifest(reader.cache_dir, run_manifest(
        NAME, started_at=started, git=git, sources=registry.sources,
        extra={"reader_id": reader.reader_id, "device": args.device,
               "splits": totals}))
    return 0
