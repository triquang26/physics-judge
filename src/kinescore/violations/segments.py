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
            report_d = violations.get(name) or {}
            threshold = report_d.get("segment_threshold",
                                     report_d.get("threshold"))
            if not window or detector is None or threshold is None:
                continue
            value = detector.reduce_window(window)
            violated = (value > threshold if detector.higher_is_worse
                        else value < threshold)
            entry["detectors"][name] = {
                "value": value, "reduce": detector.segment_reduce,
                "threshold": float(threshold), "violated": bool(violated)}
        out.append(entry)
    return out
