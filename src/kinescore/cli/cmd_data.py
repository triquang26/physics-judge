"""``kinescore data``: build a reader's canonical train tree from its source."""
from __future__ import annotations

import argparse

NAME = "data"
HELP = "materialise a reader's train tree from the corpus it declares"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from kinescore.cli._shared import add_config_arguments

    parser.add_argument("--reader", help="reader id, e.g. aloha_bimanual.mv3_row")
    parser.add_argument("--list", action="store_true",
                        help="print every declared reader and its status")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="target val fraction, split by scene")
    parser.add_argument("--seed", type=int, default=0, help="split seed")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap episodes read (0 = all); the tree is "
                             "rewritten whole, so a capped run leaves only "
                             "those episodes in it")
    parser.add_argument("--copy", action="store_true",
                        help="copy videos instead of symlinking them")
    add_config_arguments(parser)


def run(args: argparse.Namespace) -> int:
    from kinescore.cli._shared import load, now, resolve_reader
    from kinescore.registry.materialize import materialize_train_tree
    from kinescore.registry.provenance import run_manifest, write_run_manifest

    registry = load(args)
    if args.list or not args.reader:
        for reader_id, reader in sorted(registry.readers.items()):
            state = reader.status or ("ready" if reader.trainable else "no source")
            print(f"{reader_id:34s} {reader.robot:16s} {reader.view.view_id:18s} {state}")
        return 0 if args.list else 1

    reader = resolve_reader(registry, args.reader)
    started = now()
    report = materialize_train_tree(
        reader, val_ratio=args.val_ratio, seed=args.seed, limit=args.limit,
        copy=args.copy, progress=lambda m: print(f"[data] {m}"))
    write_run_manifest(report.tree, run_manifest(
        NAME, started_at=started, sources=registry.sources,
        extra={"reader_id": reader.reader_id, "robot": reader.robot,
               "view_id": reader.view.view_id, "n_train": report.n_train,
               "n_val": report.n_val, "n_skipped": len(report.skipped)}))
    return 0
