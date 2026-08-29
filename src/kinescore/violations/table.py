"""One flat row per scored clip, for reading violation rates out in bulk."""
from __future__ import annotations

from pathlib import Path

__all__ = ["COORDS", "flatten", "flatten_segments", "write_segment_table",
           "write_table"]

#: Clip coordinates carried through to the table, in column order.
COORDS: tuple[str, ...] = (
    "cell_id", "id", "source_path", "role", "aug_tag", "method", "embodiment",
    "view", "model", "split", "task")

def _measures(name: str, detector: dict, verdicts: list[dict]) -> dict:
    """One detector's segment verdicts and per-frame report, as scalars."""
    from kinescore.violations import DETECTORS

    spec = next((d for d in DETECTORS if d.name == name), None)
    higher_is_worse = spec.higher_is_worse if spec else True
    series = [float(v) for v in detector.get("per_frame") or []]
    values = [v["value"] for v in verdicts]
    worst = (max(values) if higher_is_worse else min(values)) if values else None
    return {"threshold": detector.get("threshold"),
            "reduce": spec.segment_reduce if spec else None,
            "n_frames": len(series),
            "n_segments": len(verdicts),
            "n_violated": sum(1 for v in verdicts if v["violated"]),
            "violated": any(v["violated"] for v in verdicts) if verdicts else None,
            "worst_segment": worst,
            "frame_fraction": detector.get("fraction"),
            "severity_median": detector.get("severity_ratio_median")}


def _verdicts(row: dict, name: str) -> list[dict]:
    return [d[name] for seg in row.get("segments") or []
            for d in [seg.get("detectors") or {}] if name in d]


def flatten(row: dict, names) -> dict:
    """One scored row as scalars, with ``names`` first among the detectors."""
    violations = row.get("violations") or {}
    ordered = list(names) + [n for n in violations if n not in names]
    flat = {c: row.get(c) for c in COORDS}
    flat["error"] = row.get("error")
    flat["n_segments"] = len(row.get("segments") or [])
    for name in ordered:
        measures = _measures(name, violations.get(name) or {},
                             _verdicts(row, name))
        for key, value in measures.items():
            flat[f"{name}_{key}"] = value
    return flat


def flatten_segments(row: dict, names) -> list[dict]:
    """One entry per segment of ``row``: the value judged, and the verdict."""
    out = []
    for index, seg in enumerate(row.get("segments") or []):
        flat = {c: row.get(c) for c in COORDS}
        flat.update({"segment": index, "start_frame": seg["start"],
                     "end_frame": seg["end"],
                     "n_frames": seg["end"] - seg["start"] + 1})
        detectors = seg.get("detectors") or {}
        ordered = list(names) + [n for n in detectors if n not in names]
        for name in ordered:
            found = detectors.get(name) or {}
            flat[f"{name}_reduce"] = found.get("reduce")
            flat[f"{name}_value"] = found.get("value")
            flat[f"{name}_threshold"] = found.get("threshold")
            flat[f"{name}_violated"] = found.get("violated")
        out.append(flat)
    return out


def _write(flat: list[dict], path: Path, fallback: list[str]) -> Path:
    import csv

    columns = list(flat[0]) if flat else fallback
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)
    return path


def write_table(rows: list[dict], path: Path, names) -> Path:
    """One row per clip, at ``path``."""
    return _write([flatten(r, names) for r in rows], path,
                  [*COORDS, "error", "n_segments"])


def write_segment_table(rows: list[dict], path: Path, names) -> Path:
    """One row per segment of every clip, at ``path``."""
    flat = [f for r in rows for f in flatten_segments(r, names)]
    return _write(flat, path, [*COORDS, "segment", "start_frame", "end_frame"])
