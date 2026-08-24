"""``kinescore ledger``: one row per reader -- built, trained, scored."""
from __future__ import annotations

import argparse

NAME = "ledger"
HELP = "tick off every reader against the artifacts on disk"

_MISSING = "-"

_COLUMNS: tuple[str, ...] = (
    "reader", "robot", "corpus", "cached", "train_ep", "val_ep", "head",
    "train_mm", "val_mm", "steps", "scores", "clips", "status")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from kinescore.cli._shared import add_config_arguments

    parser.add_argument("--out", default=None,
                        help="write the ledger as JSON here")
    add_config_arguments(parser)


def _cached(reader) -> dict:
    """Feature-cache episodes per split, as `cache` left them."""
    return {s: len(list((reader.cache_dir / s).glob("*.pt")))
            for s in ("train", "val")}


def _checkpoint(reader) -> dict:
    """The reader's trained head: which kind it is, and how well it did."""
    import torch

    from kinescore.registry.provenance import sha256_file

    path = reader.checkpoint_path
    if not path.is_file():
        return {}
    ck = torch.load(path, map_location="cpu")
    meta, cfg = ck.get("meta") or {}, ck.get("cfg") or {}
    return {"head": cfg.get("head", "keypoint"),
            "train_mm": meta.get("train_mm"), "val_mm": meta.get("val_mm"),
            "steps": meta.get("steps"),
            "train_ep": meta.get("n_train_episodes"),
            "val_ep": meta.get("n_val_episodes"),
            "sha256": sha256_file(path)}


def _score(cell, trained: str | None) -> dict:
    """What one cell's scored output holds, and which checkpoint made it."""
    import json
    import os

    path = cell.output_dir / "summary.json"
    if not path.is_file():
        return {"cell": cell.cell_id, "scored": False}
    s = json.loads(path.read_text())
    return {"cell": cell.cell_id, "scored": True,
            "n_clips": s.get("n_clips"), "n_failed": s.get("n_failed"),
            "by_role": s.get("n_scored_by_role") or {},
            "checkpoint": os.path.basename(s.get("checkpoint") or ""),
            "off_reader": s.get("checkpoint_sha256") != trained}


def _rows(registry) -> list[dict]:
    """One row per declared reader, in registry order."""
    rows = []
    for reader_id, reader in sorted(registry.readers.items()):
        ck = _checkpoint(reader)
        cached = _cached(reader)
        scores = [_score(c, ck.get("sha256"))
                  for c in registry.cells_for_reader(reader_id)]
        done = [s for s in scores if s["scored"]]
        rows.append({
            "reader": reader_id, "robot": reader.robot,
            "corpus": reader.corpus, "status": reader.status or "",
            "tree": (reader.train_tree / "videos").is_dir(),
            "cached": sum(cached.values()), "cached_by_split": cached,
            **{k: v for k, v in ck.items() if k != "sha256"},
            "scores": ", ".join(s["cell"] for s in done) or None,
            "clips": sum(s["n_clips"] or 0 for s in done) or None,
            "cells": scores,
        })
    return rows


def _fmt(value) -> str:
    if value is None or value == "":
        return _MISSING
    if isinstance(value, bool):
        return "yes" if value else _MISSING
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
    lines += ["  ".join(c.ljust(w) for c, w in zip(b, width, strict=True))
              for b in body]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    import json

    from kinescore.cli._shared import load

    registry = load(args)
    rows = _rows(registry)
    print(_table(rows, _COLUMNS))

    unscored = [c["cell"] for r in rows for c in r["cells"] if not c["scored"]]
    if unscored:
        print(f"\nnot scored: {', '.join(sorted(unscored))}")
    off = [c["cell"] for r in rows for c in r["cells"]
           if c["scored"] and c["off_reader"]]
    if off:
        print(f"\nscored with a checkpoint that is not the reader's own: "
              f"{', '.join(off)}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"readers": rows}, f, indent=2)
        print(f"\nwrote {args.out}")
    return 0
