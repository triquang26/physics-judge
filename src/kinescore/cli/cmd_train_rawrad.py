"""``kinescore train-rawrad``: train a :class:`ReadoutV2Head` (``raw_rad``) on
precomputed tokens.

``kinescore train`` (:mod:`kinescore.cli.cmd_train`, which trained the
*squashed* :class:`~kinescore.heads.attentive.AttentivePoseHead`) has been
removed -- see ``legacy_docs/PROVENANCE.md``'s D7 addendum -- so this is now the
package's only training subcommand, not a "raw_rad sibling" of a squashed
one. Wires together :mod:`kinescore.training.datasets`
(load the cache -- shared, unmodified), :mod:`kinescore.training.trainer_rawrad`
(the two-phase AdamW/cosine loop) and :mod:`kinescore.readers.checkpoint_v2`
(persist a checkpoint ``kinescore score``/:func:`kinescore.cli._scoring.build_scorer`
auto-routes to :class:`~kinescore.readers.checkpoint_v2.ReadoutV2PoseReader`).

``head.n_out`` is always ``robot.n_joints`` -- this command has no flag for a
wider head with auxiliary (non-FK) outputs (e.g. a gripper channel); see
``trainer_rawrad``'s module docstring for why Franka's gripper is left out of
this head entirely rather than folded into the radian limit penalty.
"""
from __future__ import annotations

import argparse

NAME = "train-rawrad"
HELP = "train a raw_rad (ReadoutV2Head) pose-reader head on precomputed tokens"


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
                        help="recorded in the checkpoint meta for provenance "
                             "only -- the ReadoutV2 reader's default backbone "
                             "cfg (checkpoint_v2.build_default_backbone) is "
                             "what `kinescore score` actually reconstructs")
    parser.add_argument("--dino-input", type=int, default=768)
    parser.add_argument("--patch-pool", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--temporal-nhead", type=int, default=8)
    parser.add_argument("--ff", type=int, default=1024)
    parser.add_argument("--n-temporal-layers", type=int, default=2)
    parser.add_argument("--t-max", type=int, default=8,
                        help="TemporalEncoder positional table size; this "
                             "command always trains with use_context=False "
                             "(T=1, see trainer_rawrad's module docstring) so "
                             "this only bounds a later context-on inference "
                             "call, not anything trained here")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--phase-a", type=int, default=1500)
    parser.add_argument("--bs", type=int, default=2048, dest="batch_size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-phase-b", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--limit-weight", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0,
                        help="cap episodes per split loaded into RAM (0 = all)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--out", required=True, help="checkpoint output .pt path")


def run(args: argparse.Namespace) -> int:
    from kinescore.cli._provenance import provenance_block, write_json
    from kinescore.cli._scoring import view_layout_from_args
    from kinescore.heads.heteroscedastic import ReadoutV2Head
    from kinescore.readers import checkpoint_v2
    from kinescore.robots import get_robot
    from kinescore.training.datasets import load_split
    from kinescore.training.trainer_rawrad import TrainConfigRawRad, train_head_rawrad

    view_layout = view_layout_from_args(args)
    robot = get_robot(args.robot, device=args.device)

    print(f"[train-rawrad] loading {args.train_split!r} split...")
    train_split = load_split(args.cache_root, args.annotation_root,
                             args.train_split, args.down_sample,
                             limit=args.limit, expected_view_layout=view_layout,
                             progress=print)
    val_split = None
    try:
        print(f"[train-rawrad] loading {args.val_split!r} split...")
        val_split = load_split(args.cache_root, args.annotation_root,
                               args.val_split, args.down_sample,
                               limit=args.limit, expected_view_layout=view_layout,
                               progress=print)
    except RuntimeError as exc:
        print(f"[train-rawrad] no validation split ({exc}); training without one")

    if train_split.n_joints != robot.n_joints:
        raise ValueError(
            f"cached joint labels have width {train_split.n_joints} but "
            f"robot {args.robot!r} has n_joints={robot.n_joints}")

    head = ReadoutV2Head(
        in_dim=train_split.embed_dim, d_model=args.d_model, n_heads=args.n_heads,
        temporal_nhead=args.temporal_nhead, ff=args.ff,
        n_temporal_layers=args.n_temporal_layers, t_max=args.t_max,
        dropout=args.dropout, n_out=robot.n_joints)

    cfg = TrainConfigRawRad(
        steps=args.steps, phase_a=args.phase_a, batch_size=args.batch_size,
        lr=args.lr, lr_phase_b=args.lr_phase_b, weight_decay=args.weight_decay,
        beta=args.beta, limit_weight=args.limit_weight, seed=args.seed,
        eval_every=args.eval_every, log_every=args.log_every, device=args.device)

    def _progress(step, loss):
        print(f"[train-rawrad] step={step} loss={loss:.5f}")

    result = train_head_rawrad(
        head, robot, train_feat=train_split.feats, train_q=train_split.q,
        val_feat=val_split.feats if val_split else None,
        val_q=val_split.q if val_split else None,
        cfg=cfg, progress=_progress)

    print(f"[train-rawrad] final loss={result.loss_history[-1]:.5f}  "
         f"train_keypoint_mm={result.train_keypoint_mm:.2f}  "
         f"val_keypoint_mm={result.val_keypoint_mm}  "
         f"best_step={result.best_step}  best_val_keypoint_mm={result.best_val_keypoint_mm}")

    head.load_state_dict(result.best_state_dict)
    head.eval()

    meta = {
        # This trainer calls the head with T=1/use_context=False for every
        # step (see trainer_rawrad's module docstring) -- its
        # TemporalEncoder never trains, so it must NOT be exercised at
        # inference either. Persisted so checkpoint_v2.resolve_use_context
        # picks it up automatically -- see that function's docstring for
        # the crash/silent-corruption this prevents.
        "use_context": False,
        "steps": result.steps, "phase_a": min(args.phase_a, max(1, args.steps // 2)),
        "best_step": result.best_step,
        "train_keypoint_mm": result.train_keypoint_mm,
        "val_keypoint_mm": result.val_keypoint_mm,
        "best_val_keypoint_mm": result.best_val_keypoint_mm,
        "n_train_episodes": train_split.n_episodes,
        "n_val_episodes": val_split.n_episodes if val_split else 0,
        "down_sample": args.down_sample,
        "beta": args.beta, "limit_weight": args.limit_weight,
        "dino_model": args.dino_model, "dino_input": args.dino_input,
        "patch_pool": args.patch_pool,
        "urdf_sha256": getattr(robot, "urdf_sha256", None),
    }
    checkpoint_v2.save(args.out, head, view_layout=view_layout,
                       robot_name=robot.name, meta=meta)
    print(f"[train-rawrad] saved -> {args.out}")

    prov = provenance_block(robot=args.robot, view_layout=view_layout.key, **meta)
    write_json(args.out + ".provenance.json", prov)
    return 0
