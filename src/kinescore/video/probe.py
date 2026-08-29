"""ffprobe the ground truth, and never let a config value overrule it silently.
"""
from __future__ import annotations

import subprocess

from kinescore.core.clip import ClipSpec, DtSource, TimebaseError, ViewLayout

__all__ = ["ffprobe", "resolve_timebase"]

#: ``ViewLayout`` is frozen, so one shared instance is a safe default.
_SINGLE_VIEW = ViewLayout()


def ffprobe(path: str) -> dict[str, object]:
    """Probe one video file -> ``{"w","h","codec","n_frames","fps"}``.

    Raises
    ------
    subprocess.CalledProcessError
        If ``ffprobe`` itself fails (missing binary, unreadable/corrupt file).
    """
    def _run(args: list[str]) -> str:
        return subprocess.run(
            ["ffprobe", "-v", "error", *args, path],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    out = _run(["-select_streams", "v:0", "-show_entries",
                "stream=width,height,codec_name,nb_frames,r_frame_rate",
                "-of", "default=nw=1:nk=0"])
    d: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k] = v
    w, h = int(d.get("width", 0) or 0), int(d.get("height", 0) or 0)
    codec = d.get("codec_name") or None
    rfr = d.get("r_frame_rate", "0/1")
    try:
        num, den = rfr.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except Exception:
        fps = 0.0
    nb = d.get("nb_frames", "N/A")
    if nb in ("N/A", "", "0"):
        cnt = _run(["-select_streams", "v:0", "-count_frames",
                    "-show_entries", "stream=nb_read_frames",
                    "-of", "default=nw=1:nk=1"])
        n_frames = int(cnt) if cnt.isdigit() else 0
    else:
        n_frames = int(nb)
    return {"w": w, "h": h, "codec": codec, "n_frames": n_frames, "fps": fps}


def resolve_timebase(path: str, fps_arg: float | None = None,
                      dt_arg: float | None = None,
                      fps_table: float | None = None,
                      probe_tolerance: float = 0.01, *,
                      view_layout: ViewLayout = _SINGLE_VIEW) -> ClipSpec:
    """Probe ``path`` and build its :class:`ClipSpec`, cross-checking any override.

    Parameters
    ----------
    path:
        Clip to probe.
    fps_arg, dt_arg:
        Mutually exclusive explicit overrides. Passing both is a ``ValueError``
        raised before anything is probed.
    fps_table:
        A candidate fps from a config table, if the caller has one. Treated
        exactly like ``fps_arg`` for the cross-check, with ``dt_source="table"``
        on success.
    probe_tolerance:
        Relative tolerance (fraction of the probed fps) within which an
        override is allowed to disagree with the probe -- accounts for
        container rounding (e.g. 29.97 vs 30), not a free pass for a wrong
        table entry.
    view_layout:
        Forwarded to the resulting :class:`ClipSpec` unchanged.

    Returns
    -------
    ClipSpec
        With ``dt_source`` set to whichever input actually determined ``fps``:
        ``"fps_arg"``, ``"dt_arg"``, ``"table"``, or ``"ffprobe"`` when none of
        the above were given and the probed rate was used directly.

    Raises
    ------
    TimebaseError
        If ``fps_arg``/``dt_arg``/``fps_table`` disagree with the probed fps
        beyond ``probe_tolerance``, or if neither an override nor a usable
        probed fps is available.
    """
    if fps_arg is not None and dt_arg is not None:
        raise ValueError(
            "--fps and --dt are mutually exclusive; pass at most one "
            f"(got fps_arg={fps_arg}, dt_arg={dt_arg})")

    probed = ffprobe(path)
    if probed["n_frames"] <= 0 or probed["w"] <= 0 or probed["h"] <= 0:
        raise TimebaseError(
            f"unusable stream (n_frames={probed['n_frames']}, "
            f"{probed['w']}x{probed['h']}) for {path}")
    probed_fps = float(probed["fps"])

    candidate: float | None = None
    source: DtSource = "ffprobe"
    if fps_arg is not None:
        candidate, source = float(fps_arg), "fps_arg"
    elif dt_arg is not None:
        candidate, source = 1.0 / float(dt_arg), "dt_arg"
    elif fps_table is not None:
        candidate, source = float(fps_table), "table"

    if candidate is None:
        if probed_fps <= 0:
            raise TimebaseError(
                f"{path}: ffprobe could not determine a usable frame rate "
                f"(r_frame_rate degenerate) and no fps_arg/dt_arg/fps_table "
                f"override was given")
        fps = probed_fps
    else:
        if probed_fps <= 0:
            raise TimebaseError(
                f"{path}: {source} claims fps={candidate} but ffprobe could "
                f"not measure a frame rate to cross-check it against; refusing "
                f"to trust an unverifiable override")
        rel_err = abs(candidate - probed_fps) / probed_fps
        if rel_err > probe_tolerance:
            raise TimebaseError(
                f"{path}: {source} claims fps={candidate} but ffprobe measured "
                f"fps={probed_fps} (relative error {rel_err:.1%} > tolerance "
                f"{probe_tolerance:.1%}). Fix the declared rate, or pass the "
                f"correct --fps/--dt explicitly.")
        fps = candidate

    return ClipSpec.from_fps(
        path=path, fps=fps, n_frames=int(probed["n_frames"]),
        width=int(probed["w"]), height=int(probed["h"]),
        dt_source=source, view_layout=view_layout,
        codec=probed["codec"])
