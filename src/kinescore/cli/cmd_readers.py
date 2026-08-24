"""``kinescore readers``: every trainable head and where its supervision comes from."""
from __future__ import annotations

import argparse

NAME = "readers"
HELP = "list declared readers, their corpus, and what each one scores"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from kinescore.cli._shared import add_config_arguments

    parser.add_argument("--ids", action="store_true",
                        help="print reader ids only, one per line, for loops")
    add_config_arguments(parser)


def run(args: argparse.Namespace) -> int:
    from kinescore.cli._shared import load

    registry = load(args)
    if args.ids:
        print("\n".join(sorted(registry.readers)))
        return 0
    for reader_id, reader in sorted(registry.readers.items()):
        cells = [c.cell_id for c in registry.cells_for_reader(reader_id)]
        state = reader.status or ("ready" if reader.trainable else "no source")
        ckpt = reader.checkpoint_path
        print(f"{reader_id}")
        print(f"    robot   {reader.robot}")
        print(f"    corpus  {reader.corpus}"
              + (f"  <- {reader.train.root}" if reader.train else ""))
        print(f"    view    {reader.view.view_id}  "
              f"{reader.view.n_views} view(s), packing {reader.view.packing}, "
              f"panel {reader.view.panel}")
        print(f"    state   {state}, checkpoint "
              f"{'present' if ckpt.exists() else 'missing'}")
        print(f"    scores  {', '.join(cells) or '-- nothing --'}")
    print(f"\n{len(registry.readers)} readers, {len(registry.cells)} cells")
    return 0
