"""``kinescore train``: fit a reader's head on its cached tokens."""
from __future__ import annotations

import argparse

NAME = "train"
HELP = "train a reader's keypoint head against forward-kinematics targets"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from kinescore.cli._shared import add_config_arguments

    parser.add_argument("--reader", required=True, help="reader id")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="windows per step")
    parser.add_argument("--window-size", type=int, default=16,
                        help="frames per window")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-late", type=float, default=5e-4)
    parser.add_argument("--lr-step-at", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--read-workers", type=int, default=4,
                        help="threads reading token windows for one batch")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap episodes loaded per split (0 = all)")
    parser.add_argument("--out", default=None,
                        help="checkpoint path (default: the reader's own)")
    add_config_arguments(parser)


def run(args: argparse.Namespace) -> int:
    import dataclasses
    import json

    from kinescore.cli._shared import load, now, resolve_reader
    from kinescore.heads.keypoint import KeypointHead
    from kinescore.readers.checkpoint import save_reader
    from kinescore.registry.provenance import (
        run_manifest,
        sha256_file,
        write_run_manifest,
    )
    from kinescore.robots import get_robot
    from kinescore.training.trainer import KeypointTrainer, TrainConfig

    registry = load(args)
    reader = resolve_reader(registry, args.reader)
    if reader.status:
        raise SystemExit(f"reader {reader.reader_id!r}: {reader.status}")
    cache_dir = reader.cache_dir
    if not cache_dir.is_dir():
        raise SystemExit(
            f"no cache at {cache_dir} -- run `kinescore cache --reader "
            f"{reader.reader_id}` first")

    started = now()
    robot = get_robot(reader.robot)
    declared = registry.robots[reader.robot].get("keypoints")
    n_keypoints = KeypointTrainer.n_keypoints(robot)
    if declared is not None and int(declared) != n_keypoints:
        raise SystemExit(
            f"robots.yaml declares {declared} keypoints for {reader.robot!r} "
            f"but its forward kinematics produces {n_keypoints}")

    layout = reader.view.layout()
    head = KeypointHead(in_dim=1024, n_keypoints=n_keypoints)
    cfg = TrainConfig(
        steps=args.steps, batch_size=args.batch_size,
        window_size=args.window_size, lr=args.lr, lr_late=args.lr_late,
        lr_step_at=args.lr_step_at, seed=args.seed,
        read_workers=args.read_workers,
        eval_every=args.eval_every, log_every=args.log_every,
        device=args.device)
    trainer = KeypointTrainer(head, robot, reader_id=reader.reader_id,
                              view_layout=layout, cfg=cfg)

    tree = reader.train_tree
    load_kwargs = {"cache_root": str(cache_dir),
                   "annotation_root": str(tree / "annotation"),
                   "limit": args.limit,
                   "progress": lambda m: print(f"[train] {m}")}
    train_episodes = trainer.load_episodes(split="train", **load_kwargs)
    try:
        val_episodes = trainer.load_episodes(split="val", **load_kwargs)
    except RuntimeError:
        val_episodes = None
        print("[train] no val split cached; training without validation")

    result = trainer.fit(
        train_episodes=train_episodes, val_episodes=val_episodes,
        progress=lambda step, loss: print(f"[train] step {step} loss {loss:.5f}"),
        on_eval=lambda step, mm: print(f"[train] step {step} val {mm:.2f} mm"))

    head.load_state_dict(result.best_state_dict)
    out = args.out or str(reader.checkpoint_path)
    cells = [c.cell_id for c in registry.cells_for_reader(reader.reader_id)]
    save_reader(out, head, cell_id=",".join(sorted(cells)), robot=reader.robot,
                view_id=reader.view.view_id, view_layout=layout,
                meta={"train_mm": result.train_mm, "val_mm": result.val_mm,
                      "best_step": result.best_step, "steps": result.steps,
                      "n_train_episodes": len(train_episodes),
                      "n_val_episodes": 0 if val_episodes is None
                      else len(val_episodes)})

    summary = {
        "reader_id": reader.reader_id,
        "checkpoint": out,
        "checkpoint_sha256": sha256_file(out),
        "train_mm": result.train_mm,
        "val_mm": result.val_mm,
        "best_val_mm": result.best_val_mm,
        "best_step": result.best_step,
        "steps": result.steps,
        "config": dataclasses.asdict(cfg),
    }
    print(f"[train] {json.dumps({k: v for k, v in summary.items() if k != 'config'})}")
    write_run_manifest(cache_dir, run_manifest(
        NAME, started_at=started, sources=registry.sources, extra=summary))
    return 0
