"""``kinescore train``: train an :class:`AttentivePoseHead` on cached tokens.

Wires together :mod:`kinescore.training.datasets` (load the cache),
:mod:`kinescore.training.trainer` (AdamW/cosine head-only loop) and
:mod:`kinescore.readers.checkpoint` (persist a checkpoint ``kinescore
score``/:func:`kinescore.cli._scoring.build_scorer` can load back).
"""
from __future__ import annotations

import argparse

HELP = "train a pose-reader head on precomputed tokens"


def add_arguments(parser: argparse.ArgumentParser) -> None:

    parser.add_argument("--cache-root", required=True,
                        help="root written by `kinescore cache` "
                             "({split}/{episode}.pt)")
    parser.add_argument("--annotation-root", required=True,
                        help="root containing {split}/{episode}.json annotations")
    parser.add_argument("--robot", required=True,
                        help="registered robot name (supplies n_joints/q_lo/q_hi/FK)")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--down-sample", type=int, required=True,
                        help="joint logs run at this multiple of the video's "
                             "frame rate (e.g. 3 for 15 Hz logs / 5 Hz video)")
    parser.add_argument("--n-views", type=int, default=1)
    parser.add_argument("--view-order", default=None)
    parser.add_argument("--dino-model", default="dinov3_vitl16",
                        help="recorded in the checkpoint's backbone cfg for "
                             "`kinescore score` to reconstruct the backbone with")
    parser.add_argument("--dino-input", type=int, default=224)
    parser.add_argument("--patch-pool", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--bs", type=int, default=2048, dest="batch_size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--limit", type=int, default=0,
                        help="cap episodes per split loaded into RAM (0 = all)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--out", required=True, help="checkpoint output .pt path")


def run(args: argparse.Namespace) -> int:
    from kinescore.cli._provenance import provenance_block, write_json
    from kinescore.cli._scoring import view_layout_from_args
    from kinescore.heads.attentive import AttentivePoseHead
    from kinescore.readers import checkpoint as ckpt_mod
    from kinescore.robots import get_robot
    from kinescore.training.datasets import load_split
    from kinescore.training.trainer import TrainConfig, train_head

    view_layout = view_layout_from_args(args)
    robot = get_robot(args.robot, device=args.device)

    print(f"[train] loading {args.train_split!r} split...")
    train_split = load_split(args.cache_root, args.annotation_root,
                             args.train_split, args.down_sample,
                             limit=args.limit, expected_view_layout=view_layout,
                             progress=print)
    val_split = None
    try:
        print(f"[train] loading {args.val_split!r} split...")
        val_split = load_split(args.cache_root, args.annotation_root,
                               args.val_split, args.down_sample,
                               limit=args.limit, expected_view_layout=view_layout,
                               progress=print)
    except RuntimeError as exc:
        print(f"[train] no validation split ({exc}); training without one")

    if train_split.n_joints != robot.n_joints:
        raise ValueError(
            f"cached joint labels have width {train_split.n_joints} but "
            f"robot {args.robot!r} has n_joints={robot.n_joints}")

    tokens_per_view = None
    if train_split.n_tokens % view_layout.n_views == 0:
        tokens_per_view = train_split.n_tokens // view_layout.n_views

    head = AttentivePoseHead(
        in_dim=train_split.embed_dim, hidden=args.hidden, n_joints=robot.n_joints,
        dropout=args.dropout, n_heads=args.n_heads, n_cams=view_layout.n_views,
        tokens_per_view=tokens_per_view)

    cfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                      weight_decay=args.weight_decay, seed=args.seed,
                      log_every=args.log_every, device=args.device)

    def _progress(step, loss):
        print(f"[train] step={step} loss={loss:.5f}")

    result = train_head(
        head, robot, train_feat=train_split.feats, train_q=train_split.q,
        train_gripper=train_split.gripper,
        val_feat=val_split.feats if val_split else None,
        val_q=val_split.q if val_split else None,
        val_gripper=val_split.gripper if val_split else None,
        cfg=cfg, progress=_progress)

    print(f"[train] final loss={result.loss_history[-1]:.5f}  "
         f"train_keypoint_mm={result.train_keypoint_mm:.2f}  "
         f"val_keypoint_mm={result.val_keypoint_mm}")

    backbone_cfg = {"dino_model": args.dino_model, "embed_dim": train_split.embed_dim,
                    "dino_input": args.dino_input, "patch_pool": args.patch_pool}
    meta = {
        "steps": result.steps, "train_keypoint_mm": result.train_keypoint_mm,
        "val_keypoint_mm": result.val_keypoint_mm,
        "train_grip_mae": result.train_grip_mae, "val_grip_mae": result.val_grip_mae,
        "n_train_episodes": train_split.n_episodes,
        "n_val_episodes": val_split.n_episodes if val_split else 0,
        "down_sample": args.down_sample,
    }
    ckpt_mod.save(args.out, head, view_layout=view_layout, robot_name=robot.name,
                 limit_semantics="squashed", backbone_cfg=backbone_cfg, meta=meta)
    print(f"[train] saved -> {args.out}")

    prov = provenance_block(robot=args.robot, view_layout=view_layout.key, **meta)
    write_json(args.out + ".provenance.json", prov)
    return 0
