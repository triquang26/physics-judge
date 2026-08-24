"""``kinescore ledger``: what is built, what is trained, what is scored."""
from __future__ import annotations

import argparse

NAME = "ledger"
HELP = "tick off every reader and cell against the artifacts on disk"

_MISSING = "-"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from kinescore.cli._shared import add_config_arguments

    parser.add_argument("--out", default=None, help="write the ledger as JSON here")
    add_config_arguments(parser)


def _tick(ok: bool) -> str:
    return "yes" if ok else _MISSING


def _n_cached(reader) -> int:
    return sum(len(list((reader.cache_dir / s).glob("*.pt")))
               for s in ("train", "val"))


def _checkpoint(reader) -> dict:
    """The reader's own checkpoint, read for its head kind and scores."""
    import torch

    from kinescore.registry.provenance import sha256_file

    path = reader.checkpoint_path
    if not path.is_file():
        return {}
    ck = torch.load(path, map_location="cpu")
    meta, cfg = ck.get("meta") or {}, ck.get("cfg") or {}
    return {"head": cfg.get("head", "keypoint"),
            "val_mm": meta.get("val_mm"), "train_mm": meta.get("train_mm"),
            "steps": meta.get("steps"), "sha256": sha256_file(path)}


def _score(cell) -> dict:
    """The cell's scored output, named by the checkpoint that produced it."""
    import json
    import os

    path = cell.output_dir / "summary.json"
    if not path.is_file():
        return {}
    s = json.loads(path.read_text())
    return {"n_clips": s.get("n_clips"), "n_failed": s.get("n_failed"),
            "by_role": s.get("n_scored_by_role") or {},
            "checkpoint": os.path.basename(s.get("checkpoint") or ""),
            "scored_sha256": s.get("checkpoint_sha256")}


def _reader_rows(registry) -> list[dict]:
    rows = []
    for reader_id, reader in sorted(registry.readers.items()):
        ck = _checkpoint(reader)
        rows.append({"reader": reader_id, "status": reader.status or "",
                     "tree": (reader.train_tree / "videos").is_dir(),
                     "cached": _n_cached(reader), **ck})
    return rows


def _cell_rows(registry, readers: dict[str, dict]) -> list[dict]:
    rows = []
    for cell_id, cell in sorted(registry.cells.items()):
        sc = _score(cell)
        trained = readers.get(cell.reader.reader_id, {}).get("sha256")
        rows.append({"cell": cell_id, "reader": cell.reader.reader_id,
                     "off_reader": bool(sc) and sc.get("scored_sha256") != trained,
                     **sc})
    return rows


def _fmt(value) -> str:
    if value is None or value == "":
        return _MISSING
    if isinstance(value, bool):
        return _tick(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _table(rows: list[dict], columns: tuple[str, ...]) -> str:
    head = [c.replace("_", " ") for c in columns]
    body = [[_fmt(r.get(c)) for c in columns] for r in rows]
    width = [max(len(h), *(len(b[i]) for b in body)) if body else len(h)
             for i, h in enumerate(head)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(head, width, strict=True)),
             "  ".join("-" * w for w in width)]
    lines += ["  ".join(c.ljust(w) for c, w in zip(b, width, strict=True)) for b in body]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    import json

    from kinescore.cli._shared import load

    registry = load(args)
    readers = _reader_rows(registry)
    cells = _cell_rows(registry, {r["reader"]: r for r in readers})

    print("readers")
    print(_table(readers, ("reader", "tree", "cached", "head", "train_mm",
                           "val_mm", "steps", "status")))
    print("\ncells")
    print(_table(cells, ("cell", "checkpoint", "n_clips", "n_failed",
                         "off_reader")))

    off = [c["cell"] for c in cells if c["off_reader"]]
    if off:
        print(f"\nscored with a checkpoint that is not the reader's own: "
              f"{', '.join(off)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"readers": readers, "cells": cells}, f, indent=2)
        print(f"\nwrote {args.out}")
    return 0
