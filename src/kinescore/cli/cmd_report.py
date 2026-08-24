"""``kinescore report``: one table over every scored cell."""
from __future__ import annotations

import argparse

NAME = "report"
HELP = "aggregate scored cells into the benchmark table"

_AXES = ("role", "method", "split")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from kinescore.cli._shared import (
        add_config_arguments,
        add_detector_argument,
    )

    parser.add_argument("--by", default="role", choices=_AXES,
                        help="sub-partition reported per cell")
    parser.add_argument("--out", default=None,
                        help="write the table as JSON here")
    add_detector_argument(parser)
    add_config_arguments(parser)


def _rate(records: list[dict], detector: str) -> float:
    """Fraction of clips with at least one flagged interval for ``detector``."""
    if not records:
        return float("nan")
    hit = sum(1 for r in records
              if (r.get("violations") or {}).get(detector, {}).get("intervals"))
    return hit / len(records)


def run(args: argparse.Namespace) -> int:
    import json

    from kinescore.cli._shared import load, resolve_detectors

    registry = load(args)
    present: list[str] = []
    table = []

    for cell_id, cell in sorted(registry.cells.items()):
        path = cell.output_dir / "results.jsonl"
        if not path.exists():
            table.append({"cell_id": cell_id, "state": "not scored"})
            continue
        records = [json.loads(line) for line in path.read_text().splitlines()
                   if line.strip()]
        scored = [r for r in records if "violations" in r]
        for r in scored:
            for name in r["violations"]:
                if name not in present:
                    present.append(name)
        groups: dict[str, list[dict]] = {"all": scored}
        for r in scored:
            groups.setdefault(str(r.get(args.by, "")), []).append(r)
        table.append({
            "cell_id": cell_id,
            "state": "scored",
            "n_clips": len(records),
            "n_failed": len(records) - len(scored),
            "groups": {g: {"n": len(rs),
                           **{d: round(_rate(rs, d), 3) for d in present}}
                       for g, rs in groups.items()},
        })

    detectors = resolve_detectors(args.detectors, present) if present else []
    head = f"{'cell':40s} {args.by:16s} {'n':>4s}  " + "  ".join(
        f"{d:>12s}" for d in detectors)
    print(head)
    print("-" * len(head))
    for row in table:
        if row["state"] != "scored":
            print(f"{row['cell_id']:40s} {row['state']}")
            continue
        for group, stats in sorted(row["groups"].items()):
            cells = "  ".join(f"{stats.get(d, float('nan')):12.3f}"
                              for d in detectors)
            label = row["cell_id"] if group == "all" else ""
            print(f"{label:40s} {group:16s} {stats['n']:4d}  {cells}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"by": args.by, "detectors": present,
                       "reported": detectors, "cells": table}, f, indent=2)
        print(f"\n-> {args.out}")
    return 0
