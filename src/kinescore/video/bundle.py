"""Package a scored cell as numbered clips + one ``segments.json`` for the web."""
from __future__ import annotations

from pathlib import Path

from kinescore.video.render import render_root

__all__ = ["clip_entry", "write_bundle"]

#: Bucket spelling of each reduce mode.
_REDUCE = {"worst": "max"}


def clip_entry(row: dict, names: list[str], root: Path | None = None
               ) -> tuple[dict, dict]:
    """One clip's ``videos`` entry plus the detector calibration it used."""
    detectors = {
        n: {"threshold": None,
            "units": row["violations"][n].get("units", ""),
            "reduce": None}
        for n in names}
    segments = []
    for seg in row.get("segments") or []:
        entry: dict = {"start_frame": seg["start"], "end_frame": seg["end"],
                       "n_frames": seg["end"] - seg["start"] + 1}
        for n in names:
            v = seg["detectors"][n]
            detectors[n]["threshold"] = v["threshold"]
            detectors[n]["reduce"] = _REDUCE.get(v["reduce"], v["reduce"])
            entry[n] = {
                "value": v["value"],
                "ratio": (round(v["value"] / v["threshold"], 4)
                          if v["threshold"] else None),
                "violated": v["violated"],
            }
        segments.append(entry)
    video = {"source": _source(row, root),
             "n_violated": sum(1 for s in segments
                               if any(s[n]["violated"] for n in names)),
             "segments": segments}
    return video, detectors


def _source(row: dict, root: Path | None) -> str:
    """Bench ``source_path``, or the clip's path under the ``--videos`` root."""
    source = (row.get("source_path") or "").strip("/")
    if source:
        return source
    if root is not None:
        try:
            return str(Path(row["path"]).relative_to(root))
        except ValueError:
            pass
    return Path(row["path"]).name


def write_bundle(rows: list[dict], out_dir: Path, names: list[str], *,
                 summary: dict | None = None, log=print) -> Path:
    """Write ``rows`` as ``<n>.mp4`` clips plus one ``segments.json``.

    ``rows`` come from ``results.jsonl`` and keep its order, so the numbering
    is reproducible from the scored cell alone. Rows that failed to score
    carry no segments and are skipped, not renumbered around silently -- the
    skip is logged and counted in ``segments.json``.
    """
    import json
    import shutil

    from kinescore.video.probe import ffprobe

    out_dir.mkdir(parents=True, exist_ok=True)
    root = render_root(rows)
    videos, detectors, skipped = {}, {}, 0
    for row in rows:
        if not row.get("segments"):
            skipped += 1
            log(f"[export] skip (no segments): {row['path']}")
            continue
        n = len(videos) + 1
        video, detectors = clip_entry(row, names, root)
        path = out_dir / f"{n}.mp4"
        shutil.copyfile(row["path"], path)
        probe = ffprobe(str(path))
        videos[str(n)] = {"fps": probe["fps"], "n_frames": probe["n_frames"],
                          **video}
        log(f"[export] {n}/{len(rows)} {video['source']}")

    summary = summary or {}
    doc = {
        "detectors": detectors,
        "provenance": {"cell": summary.get("cell_id"),
                       "reader": summary.get("reader_id"),
                       "checkpoint": summary.get("checkpoint_sha256")},
        "n_skipped": skipped,
        "n_clips": len(videos),
        "videos": videos,
    }
    (out_dir / "segments.json").write_text(json.dumps(doc, indent=1))
    log(f"[export] {len(videos)} clips -> {out_dir}")
    return out_dir
