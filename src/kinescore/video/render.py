"""Write a scored cell out as watchable clips."""
from __future__ import annotations

from pathlib import Path

__all__ = ["is_flagged", "read_results", "reel_group", "render_path",
           "render_results", "render_root"]


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


def render_root(rows: list[dict]) -> Path | None:
    """The directory ``--videos`` was pointed at, recovered from the rows."""
    import os

    parents = [str(Path(row["path"]).parent) for row in rows
               if not (row.get("source_path") or "").strip("/")]
    return Path(os.path.commonpath(parents)) if parents else None


def render_path(out_dir: Path, row: dict, root: Path | None = None) -> Path:
    """Where ``row``'s rendered clip goes, under ``out_dir``."""
    source = (row.get("source_path") or "").strip("/")
    if source:
        return out_dir / Path(source).with_suffix(".mp4")
    below = _below_root(row, root)
    if below and below.parent != Path("."):
        return out_dir / below.with_suffix(".mp4")
    return out_dir / f"{row['id']}_{row.get('role') or 'clip'}.mp4"


def _below_root(row: dict, root: Path | None) -> Path | None:
    """``row``'s clip path relative to ``root``, or ``None`` if unrelated."""
    if root is None:
        return None
    try:
        return Path(row["path"]).relative_to(root)
    except ValueError:
        return None


def reel_group(row: dict, root: Path | None = None) -> str:
    """Which reel ``row`` belongs in: the tree it was drawn from."""
    source = (row.get("source_path") or "").strip("/")
    if source:
        return source.split("/")[0]
    below = _below_root(row, root)
    if below and below.parent != Path("."):
        return below.parts[0]
    return row.get("role") or "clip"


def render_results(rows: list[dict], out_dir: Path, names, *, fps: float = 5.0,
                   reel: bool = True, log=print) -> Path:
    """Draw ``rows`` into ``out_dir`` and return it."""
    import imageio.v2 as iio2
    import imageio.v3 as iio
    import numpy as np

    from kinescore.video.overlay import render_clip

    out_dir.mkdir(parents=True, exist_ok=True)
    root = render_root(rows)
    reels: dict[str, tuple] = {}

    def _reel(group: str):
        """The open writer for ``group``, appended to clip by clip."""
        if group not in reels:
            path = out_dir / "reel" / f"{group}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            reels[group] = (path, iio2.get_writer(
                path, fps=fps, codec="libx264", macro_block_size=1), [0])
        return reels[group]

    try:
        for n, row in enumerate(rows, 1):
            if not row.get("segments"):
                log(f"[render] skip (not scored): {row['path']}")
                continue
            drawn = render_clip(np.asarray(iio.imread(row["path"])), row, names)
            path = render_path(out_dir, row, root)
            path.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(path, drawn, fps=fps, codec="libx264",
                        macro_block_size=1)
            if reel:
                _, writer, count = _reel(reel_group(row, root))
                for frame in drawn:
                    writer.append_data(np.asarray(frame))
                count[0] += 1
            del drawn
            log(f"[render] {n}/{len(rows)} {path.relative_to(out_dir)} "
                f"{'flagged' if is_flagged(row, names) else 'clean'}")
    finally:
        for group, (path, writer, count) in sorted(reels.items()):
            writer.close()
            log(f"[render] reel {group} ({count[0]} clips) -> {path}")
    return out_dir
