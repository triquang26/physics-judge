"""Write a scored cell out as watchable clips.

One file per clip plus a joined reel, each drawn by
:func:`~kinescore.video.overlay.render_clip`. ``score`` calls this as its last
step, so every scored cell has its segments on screen without a second
command; ``render`` calls it again to redraw an existing cell.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["is_flagged", "read_results", "render_path", "render_results"]


def read_results(results: Path) -> list[dict]:
    """The scored rows of one cell, in the order they were scored."""
    import json

    if not results.is_file():
        raise SystemExit(
            f"no results at {results} -- run `kinescore score` for this cell "
            f"first")
    return [json.loads(line) for line in results.read_text().splitlines()
            if line.strip()]


def is_flagged(row: dict, names) -> bool:
    """Whether any of ``names`` judged a segment of ``row`` a violation."""
    return any(v["violated"]
               for seg in row.get("segments") or []
               for n, v in (seg.get("detectors") or {}).items() if n in names)


def render_path(out_dir: Path, row: dict) -> Path:
    """Where ``row``'s rendered clip goes, under ``out_dir``.

    The bench flattens every clip to ``clips/<id>.mp4``, which loses which tree
    it came from. Rendering mirrors ``source_path`` instead, so a drawn clip
    sits at the same place under ``out_dir`` as its source does in the source
    repo and needs no lookup to trace. Clips scored through ``--videos`` carry
    no ``source_path``; those fall back to ``<id>_<role>.mp4``.
    """
    source = (row.get("source_path") or "").strip("/")
    if not source:
        return out_dir / f"{row['id']}_{row.get('role') or 'clip'}.mp4"
    return out_dir / Path(source).with_suffix(".mp4")


def render_results(rows: list[dict], out_dir: Path, names, *, fps: float = 5.0,
                   reel: bool = True, log=print) -> Path:
    """Draw ``rows`` into ``out_dir`` and return it.

    ``names`` are the detectors that get a row and decide the outline. With
    ``reel``, the drawn clips are also concatenated into ``reel.mp4`` in the
    order given.
    """
    import imageio.v3 as iio
    import numpy as np

    from kinescore.video.overlay import render_clip

    out_dir.mkdir(parents=True, exist_ok=True)
    joined = []

    for n, row in enumerate(rows, 1):
        drawn = render_clip(np.asarray(iio.imread(row["path"])), row, names)
        path = render_path(out_dir, row)
        path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(path, drawn, fps=fps, codec="libx264", macro_block_size=1)
        if reel:
            joined.append(drawn)
        log(f"[render] {n}/{len(rows)} {path.relative_to(out_dir)} "
            f"{'flagged' if is_flagged(row, names) else 'clean'}")

    if joined:
        iio.imwrite(out_dir / "reel.mp4", np.concatenate(joined), fps=fps,
                    codec="libx264", macro_block_size=1)
        log(f"[render] reel -> {out_dir / 'reel.mp4'}")
    return out_dir
