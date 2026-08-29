"""Draw a scored clip's violation timeline onto its own frames."""
from __future__ import annotations

import numpy as np

__all__ = ["detector_order", "render_clip"]


def detector_order(names=None) -> tuple[tuple[str, bool], ...]:
    """Detector rows top to bottom, each with its ``higher_is_worse`` sense."""
    from kinescore.violations import DETECTORS

    sense = {d.name: d.higher_is_worse for d in DETECTORS}
    if names is None:
        return tuple(sense.items())
    return tuple((n, sense[n]) for n in names)

_HEADER_H = 58
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
    """How close one frame sits to its threshold: 0 clean, 1 at the bound."""
    if not threshold:
        return 0.0
    return value / threshold if higher_is_worse else threshold / max(value, 1e-9)


def _ramp(ratio: float) -> tuple[int, int, int]:
    """Severity -> BGR, green at 0 through amber at 1."""
    a = float(np.clip(ratio, 0.0, 1.0))
    return (int(_OK[0] + (40 - _OK[0]) * a),
            int(_OK[1] + (170 - _OK[1]) * a),
            int(_OK[2] + (232 - _OK[2]) * a))


def _verdicts(row: dict, name: str) -> list[dict]:
    """``name``'s verdict in each segment of ``row``, in time order."""
    out = []
    for seg in row.get("segments") or []:
        found = (seg.get("detectors") or {}).get(name)
        if found is not None:
            out.append({"start": seg["start"], "end": seg["end"], **found})
    return out


def _violated_mask(verdicts: list[dict], n_frames: int) -> np.ndarray:
    """Frames sitting inside a segment this detector judged a violation."""
    mask = np.zeros(n_frames, dtype=bool)
    for v in verdicts:
        if v["violated"]:
            mask[v["start"]:v["end"] + 1] = True
    return mask


def _trace(row: dict, width: int, scale: float = 0.40) -> str:
    """Where this clip came from, kept to what one header line can hold."""
    source = row.get("source_path") or ""
    if not source:
        return f"clips/{row['id']}.mp4"
    fits = max(16, int(width / (scale * 22.0)))
    return source if len(source) <= fits else "..." + source[-(fits - 3):]


def _clip_summary(by_detector: dict, order) -> str:
    """How many segments each detector judged a violation, and its worst one.
    """
    parts = []
    for name, higher_is_worse in order:
        verdicts = by_detector.get(name) or []
        if not verdicts:
            parts.append(f"{name} -")
            continue
        worst = max(_severity(v["value"], v["threshold"], higher_is_worse)
                    for v in verdicts)
        hit = sum(1 for v in verdicts if v["violated"])
        parts.append(f"{name} {verdicts[0]['reduce']} {hit}/{len(verdicts)} seg"
                     f"  worst {worst:.2f}x")
    return "   ".join(parts)


def _measure(value: float) -> str:
    """A detector reading short enough to fit its column at any magnitude."""
    return f"{value:.0f}" if abs(value) < 1e5 else f"{value:.1e}"


def _row(canvas, y, name, higher_is_worse, verdicts, n_frames, frame, width):
    """Draw one detector's segments, with the playhead at ``frame``."""
    import cv2

    bar_x, bar_w = _LABEL_W, width - _LABEL_W - _VALUE_W
    top, bot = y + 5, y + _ROW_H - 7
    any_violated = any(v["violated"] for v in verdicts)
    _text(canvas, name, (8, y + _ROW_H - 9), _FLAG if any_violated else _DIM)

    here = verdicts[0] if verdicts else None
    for v in verdicts:
        x0 = bar_x + round(v["start"] * bar_w / n_frames)
        x1 = bar_x + round((v["end"] + 1) * bar_w / n_frames)
        colour = (_FLAG if v["violated"]
                  else _ramp(_severity(v["value"], v["threshold"],
                                       higher_is_worse)))
        cv2.rectangle(canvas, (x0 + 1, top), (x1 - 2, bot), colour, -1)
        if v["start"] <= frame <= v["end"]:
            here = v
            cv2.rectangle(canvas, (x0 + 1, top), (x1 - 2, bot), _FG, 1)

    px = bar_x + round((frame + 0.5) * bar_w / n_frames)
    cv2.line(canvas, (px, top - 3), (px, bot + 3), _FG, 1)

    if here is not None:
        _text(canvas, f"{here['reduce'][:3]} {_measure(here['value'])}"
                      f"/{_measure(here['threshold'])}",
              (bar_x + bar_w + 8, y + _ROW_H - 9),
              _FLAG if here["violated"] else _DIM, 0.38)


def render_clip(frames: np.ndarray, row: dict, names=None) -> np.ndarray:
    """Compose ``(N,H,W,3)`` RGB frames with ``row``'s timeline underneath."""
    import cv2

    n, h, w, _ = frames.shape
    order = detector_order(names)
    height = _HEADER_H + h + _ROW_H * len(order)
    out = np.empty((n, height, w, 3), dtype=np.uint8)

    verdicts = {name: _verdicts(row, name) for name, _ in order}
    flagged = {name: _violated_mask(v, n) for name, v in verdicts.items()}
    title = (f"{row['id']}  {row.get('role') or '-'}  "
             f"{row.get('aug_tag') or '-'}  {row.get('task') or '-'}")
    summary = _clip_summary(verdicts, order)
    trace = _trace(row, w)

    for i in range(n):
        canvas = np.full((height, w, 3), _BG, dtype=np.uint8)
        canvas[_HEADER_H:_HEADER_H + h] = frames[i][:, :, ::-1]

        hits = [name for name, _ in order if flagged[name][i]]
        if hits:
            cv2.rectangle(canvas, (0, _HEADER_H), (w - 1, _HEADER_H + h - 1),
                          _FLAG, 2)
        _text(canvas, title, (8, 17), _FG)
        _text(canvas, f"f {i + 1:>3}/{n}   {' '.join(hits) or 'clean'}",
              (w - 470, 17), _FLAG if hits else _OK)
        _text(canvas, trace, (8, 34), _FG, 0.40)
        _text(canvas, summary, (8, 50), _DIM, 0.40)

        for k, (name, higher_is_worse) in enumerate(order):
            _row(canvas, _HEADER_H + h + k * _ROW_H, name, higher_is_worse,
                 verdicts[name], n, i, w)
        out[i] = canvas[:, :, ::-1]
    return out
