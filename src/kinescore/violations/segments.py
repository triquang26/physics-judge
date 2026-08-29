"""Segment-level verdicts over a scored clip."""
from __future__ import annotations

__all__ = ["SEGMENT_LEN", "bounds", "report"]

#: Frames per segment. One window of the reader's trained prediction horizon,
#: so a segment is exactly one complete unit of what the reader can see.
SEGMENT_LEN = 16


def bounds(n_frames: int, length: int = SEGMENT_LEN) -> list[tuple[int, int]]:
    """Consecutive ``[start, end]`` windows covering ``n_frames``, end inclusive.
    """
    if n_frames <= 0:
        return []
    return [(i, min(i + length, n_frames) - 1)
            for i in range(0, n_frames, max(1, length))]


def _reduce(values: list[float], how: str, higher_is_worse: bool) -> float:
    if how == "median":
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])
    return max(values) if higher_is_worse else min(values)


def report(violations: dict, names, length: int = SEGMENT_LEN) -> list[dict]:
    """One entry per segment, carrying each named detector's verdict."""
    from kinescore.violations import DETECTORS

    spec = {d.name: d for d in DETECTORS}
    series = {n: [float(v) for v in (violations.get(n) or {}).get("per_frame") or []]
              for n in names}
    n_frames = max((len(v) for v in series.values()), default=0)

    out = []
    for start, end in bounds(n_frames, length):
        entry = {"start": start, "end": end, "detectors": {}}
        for name in names:
            window = series[name][start:end + 1]
            detector = spec.get(name)
            threshold = (violations.get(name) or {}).get("threshold")
            if not window or detector is None or threshold is None:
                continue
            value = _reduce(window, detector.segment_reduce,
                            detector.higher_is_worse)
            violated = (value > threshold if detector.higher_is_worse
                        else value < threshold)
            entry["detectors"][name] = {
                "value": value, "reduce": detector.segment_reduce,
                "threshold": float(threshold), "violated": bool(violated)}
        out.append(entry)
    return out
