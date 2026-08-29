"""``kinescore models``: what the bench holds, derived from its manifest."""
from __future__ import annotations

import argparse

NAME = "models"
HELP = "list the generators the bench scores, and their clip counts"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from kinescore.cli._shared import add_config_arguments

    parser.add_argument("--by", default="model",
                        choices=["model", "role", "method", "split", "task"],
                        help="second grouping axis")
    add_config_arguments(parser)


def run(args: argparse.Namespace) -> int:
    from collections import Counter

    from kinescore.cli._shared import load
    from kinescore.registry.bench import load_bench, select

    items = load_bench()
    registry = load(args)
    covered = {}
    for cell in registry.cells.values():
        for item in select(items, cell.select):
            covered[item.id] = cell

    axis = args.by
    rows = Counter((i.embodiment, i.view, i.model, getattr(i, axis))
                   for i in items)
    width = max(len(f"{e}/{v}/{m}") for e, v, m, _ in rows) if rows else 20
    print(f"{'embodiment/view/model':{width}s} {axis:16s} {'clips':>6s}  cell")
    for (emb, view, model, value), n in sorted(rows.items()):
        sample = next(i for i in items
                      if (i.embodiment, i.view, i.model) == (emb, view, model))
        cell = covered.get(sample.id)
        print(f"{f'{emb}/{view}/{model}':{width}s} {str(value):16s} {n:6d}  "
              f"{cell.cell_id if cell else '-- no cell --'}")

    print(f"\n{len(items)} clips, {len(covered)} covered by a declared cell")
    orphan = [i for i in items if i.id not in covered]
    if orphan:
        kinds = Counter((i.embodiment, i.view, i.model) for i in orphan)
        for k, n in sorted(kinds.items()):
            print(f"  uncovered: {'/'.join(k)}  {n} clips")
    return 0
