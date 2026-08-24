"""Draw a scored clip's violation timeline onto its own frames.

A row of ``results.jsonl`` carries, per detector, a ``per_frame`` series and
the flagged ``intervals`` over it. Both are indexed by scored frame, one per
decoded frame, so the series can be drawn under the picture it was measured
from and read segment by segment.
"""
from __future__ import annotations

import numpy as np

__all__ = ["detector_order", "render_clip"]


def detector_order() -> tuple[tuple[str, bool], ...]:
    """Detector rows top to bottom, each with its ``higher_is_worse`` sense.

    Taken from the scorer's own list, so a detector added there appears here
    with the direction it is actually thresholded in.
    """
    from kinescore.violations import DETECTORS

    return tuple((d.name, d.higher_is_worse) for d in DETECTORS)

_HEADER_H = 26
_ROW_H = 26
_LABEL_W = 150
_VALUE_W = 128
_BG = (18, 18, 18)
_FG = (232, 232, 232)
_DIM = (120, 120, 120)
_FLAG = (54, 54, 226)
_OK = (96, 148, 72)


def _text(canvas, s, xy, colour, scale=0.42):
    import cv2

    cv2.putText(canvas, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1,
                cv2.LINE_AA)


def _severity(value: float, threshold: float, higher_is_worse: bool) -> float:
    """How close one frame sits to its threshold: 0 clean, 1 at the bound.

    ``self_collision`` is thresholded from below, so its severity is the
    reciprocal ratio -- the same convention ``Detector.report`` records.
    """
    if not threshold:
        return 0.0
    return value / threshold if higher_is_worse else threshold / max(value, 1e-9)


def _ramp(ratio: float) -> tuple[int, int, int]:
    """Severity -> BGR, green at 0 through amber at 1."""
    a = float(np.clip(ratio, 0.0, 1.0))
    return (int(_OK[0] + (40 - _OK[0]) * a),
            int(_OK[1] + (170 - _OK[1]) * a),
            int(_OK[2] + (232 - _OK[2]) * a))


def _flagged_mask(series: np.ndarray, intervals: list) -> np.ndarray:
    """Frames inside a flagged interval. Endpoints are inclusive."""
    mask = np.zeros(len(series), dtype=bool)
    for start, end in intervals:
        mask[int(start):int(end) + 1] = True
    return mask


def _measure(value: float) -> str:
    """A detector reading short enough to fit its column at any magnitude."""
    return f"{value:.0f}" if abs(value) < 1e5 else f"{value:.1e}"


def _row(canvas, y, name, higher_is_worse, detector, frame, width):
    """Draw one detector's full timeline, with the playhead at ``frame``."""
    import cv2

    series = np.asarray(detector["per_frame"], dtype=float)
    thr = float(detector["threshold"])
    flags = _flagged_mask(series, detector.get("intervals") or [])
    bar_x, bar_w = _LABEL_W, width - _LABEL_W - _VALUE_W
    top, bot = y + 5, y + _ROW_H - 7

    _text(canvas, name, (8, y + _ROW_H - 9),
          _FLAG if flags.any() else _DIM)
    for i, value in enumerate(series):
        x0 = bar_x + round(i * bar_w / len(series))
        x1 = bar_x + round((i + 1) * bar_w / len(series))
        colour = (_FLAG if flags[i]
                  else _ramp(_severity(value, thr, higher_is_worse)))
        cv2.rectangle(canvas, (x0, top), (x1 - 1, bot), colour, -1)

    for start, end in detector.get("intervals") or []:
        x0 = bar_x + round(int(start) * bar_w / len(series))
        x1 = bar_x + round((int(end) + 1) * bar_w / len(series))
        cv2.line(canvas, (x0, bot + 2), (x1 - 1, bot + 2), _FG, 1)

    px = bar_x + round((frame + 0.5) * bar_w / len(series))
    cv2.line(canvas, (px, top - 3), (px, bot + 3), _FG, 1)

    i = min(frame, len(series) - 1)
    _text(canvas, f"{_measure(series[i])}/{_measure(thr)}",
          (bar_x + bar_w + 8, y + _ROW_H - 9),
          _FLAG if flags[i] else _DIM, 0.38)


def render_clip(frames: np.ndarray, row: dict) -> np.ndarray:
    """Compose ``(N,H,W,3)`` RGB frames with ``row``'s timeline underneath.

    Returns ``(N, H + header + one row per detector, W, 3)`` RGB. Frames whose
    index falls in any flagged interval are outlined, so a segment is visible
    in the picture as well as on the bars.
    """
    import cv2

    violations = row["violations"]
    n, h, w, _ = frames.shape
    order = detector_order()
    height = _HEADER_H + h + _ROW_H * len(order)
    out = np.empty((n, height, w, 3), dtype=np.uint8)

    flagged = {name: _flagged_mask(
        np.asarray(violations[name]["per_frame"], dtype=float),
        violations[name].get("intervals") or []) for name, _ in order}
    title = (f"{row['id']}  {row['role']}  {row.get('aug_tag') or '-'}  "
             f"{row['task']}")

    for i in range(n):
        canvas = np.full((height, w, 3), _BG, dtype=np.uint8)
        canvas[_HEADER_H:_HEADER_H + h] = frames[i][:, :, ::-1]

        hits = [name for name, _ in order if flagged[name][i]]
        if hits:
            cv2.rectangle(canvas, (0, _HEADER_H), (w - 1, _HEADER_H + h - 1),
                          _FLAG, 2)
        _text(canvas, title, (8, 18), _FG)
        _text(canvas, f"f {i + 1:>3}/{n}   {' '.join(hits) or 'clean'}",
              (w - 470, 18), _FLAG if hits else _OK)

        for k, (name, higher_is_worse) in enumerate(order):
            _row(canvas, _HEADER_H + h + k * _ROW_H, name, higher_is_worse,
                 violations[name], i, w)
        out[i] = canvas[:, :, ::-1]
    return out
