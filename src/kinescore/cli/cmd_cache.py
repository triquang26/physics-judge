"""``kinescore cache``: precompute frozen-backbone tokens for training.

Thin wrapper around :func:`kinescore.training.cache.precompute_cache` -- see
that module's docstring for the on-disk layout (``video_root/split/{ep}.mp4``
+ ``annotation_root/split/{ep}.json``) and why the cache files it writes are
self-describing.
"""
from __future__ import annotations

import argparse

NAME = "cache"
HELP = "precompute frozen-backbone patch tokens for training"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video-root", required=True,
                        help="root containing {split}/{episode}.mp4 clips")
    parser.add_argument("--annotation-root", required=True,
                        help="root containing {split}/{episode}.json "
                             "annotations (joint_source must be 'real')")
    parser.add_argument("--out", required=True,
                        help="output cache root; writes {out}/{split}/{episode}.pt")
    parser.add_argument("--split", action="append", default=None,
                        help="split(s) to process (repeatable; default: val, train)")
    parser.add_argument("--pattern", default="*.mp4")
    parser.add_argument("--dino-model", default="dinov3_vitl16")
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--dino-input", type=int, default=224)
    parser.add_argument("--patch-pool", type=int, default=2)
    parser.add_argument("--dino-repo-dir", default="",
                        help="local facebookresearch/dinov2 clone (DINOv2 "
                             "models only; DINOv3 loads from the HF cache)")
    parser.add_argument("--n-views", type=int, default=1)
    parser.add_argument("--view-order", default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-chunk", type=int, default=0,
                        help="encode at most this many frames of an episode "
                             "per backbone call (0 = whole episode at once, "
                             "the historical behaviour); set this on a long "
                             "episode + high --dino-input to avoid OOMing a "
                             "shared GPU -- see training/cache.py::encode_clip")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap episodes per split (0 = all)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")


def run(args: argparse.Namespace) -> int:
    from kinescore.cli._provenance import provenance_block, write_json
    from kinescore.cli._scoring import view_layout_from_args
    from kinescore.training.cache import backbone_id, build_backbone, precompute_cache

    view_layout = view_layout_from_args(args)
    backbone = build_backbone(
        dino_model=args.dino_model, embed_dim=args.embed_dim,
        dino_input=args.dino_input, patch_pool=args.patch_pool,
        dino_repo_dir=args.dino_repo_dir, view_layout=view_layout,
        device=args.device)

    splits = args.split or ["val", "train"]
    totals = {"n_done": 0, "n_skipped_existing": 0, "n_skipped_no_annotation": 0}
    for split in splits:
        summary = precompute_cache(
            video_root=args.video_root, annotation_root=args.annotation_root,
            out_root=args.out, split=split, backbone=backbone,
            view_layout=view_layout, pattern=args.pattern, limit=args.limit,
            device=args.device, overwrite=args.overwrite,
            max_frames=args.max_frames, frame_chunk=args.frame_chunk,
            progress=print)
        for k in totals:
            totals[k] += summary[k]

    print(f"[cache] total: {totals}")
    prov = provenance_block(
        backbone_id=backbone_id(backbone), view_layout=view_layout.key,
        splits=splits, **totals)
    write_json_path = f"{args.out}/cache_provenance.json"
    write_json(write_json_path, prov)
    print(f"[cache] provenance -> {write_json_path}")
    return 0
