"""Re-encode REAL footage to a generator's fps/resolution -- ffmpeg work, not CLI work.

Ported from ``Marionette-fkjepa/scripts/bench/15_prepare_real_anchor.py``
(the ``real_dm`` anchor: 640x480@10fps) and
``Marionette-fkjepa/scripts/bench/34_prepare_real_dm16.py`` (the
``real_dm16`` anchor: 768x432@16fps) -- see ``legacy_docs/PROVENANCE.md`` for the
full source->destination record. Used by ``kinescore anchor build``
(:mod:`kinescore.cli.cmd_anchor`), which is the thin argparse shell around
:func:`build_anchor` below.

What "frame-rate-matched footage" actually means -- and why it is not what a
reader would assume
-------------------------------------------------------------------------------
The project page's fairest comparisons rest on "frame-rate-matched footage",
which reads, on first pass, like it should mean resampling every clip to a
common rate. It does not. Operationally, it means the opposite direction:
**re-encode the REAL anchor footage to the GENERATOR's exact fps and
resolution**, then finite-difference both the real anchor and the generated
clips at the same ``dt``. The generated clips are never touched -- a
generator's own native rate/resolution is a property of the model being
evaluated, not something to normalise away.

The reason this matters is not cosmetic. ``Marionette-fkjepa/scripts/bench/
32_non_inversion.py`` (module docstring, source lines 1-16; hashed for this
citation's provenance -- see ``legacy_docs/PROVENANCE.md`` -- but not itself
ported, since this package's ``bench/stats.py::auroc`` already provides the
tie-safe separation statistic that script computes) records the concrete
failure this fixes: pooling raw 20fps real-teleop jerk (~117 m/s^3) against
10fps generated-clip jerk (~16-23 m/s^3) collapsed the separation AUROC to
**0.00** -- "a pure fps artefact", not evidence the generator was somehow
smoother than reality. Re-encoding the real anchor to the generator's own
10fps (``real_dm``) restored an honest AUROC. The same logic produces
``real_dm16`` (768x432@16fps) for a 16fps generator family -- a *different*
rate/resolution pair, because different generators are not all the same
rate, which is exactly why this has to be a re-encode step per target
context rather than one global "the" frame rate.

The open item this closes
------------------------------
The source's own ``real_dm16`` anchor was built but **never scored** against
anything -- its ``outputs_bench/scores/`` only ever contained
``dense``/``dicache``/``fastercache``/``worldcache``/``real``/``real_dm``,
never a ``real_dm16`` row. No cross-generator, rate-matched comparison was
ever actually produced from it. ``kinescore anchor build`` is the piece that
was missing: a reusable, generic ("any target fps/resolution/CRF", not "the
two contexts one experiment happened to need") anchor builder, so a future
run against *any* generator's native rate is one command instead of a new
one-off script. Scoring the result (composing it with a trained reader/
``Scorer``) is deliberately out of this module's scope -- see ``kinescore
score``.

Probing the CRF from a real generated clip, not hardcoding it
-------------------------------------------------------------------
Verbatim idea from the source's ``probe_crf_context()``
(15_prepare_real_anchor.py:48-61): rather than a hand-picked CRF constant
(which silently drifts from whatever a generator's own encoder actually
used), probe a real generated clip's codec/pixel-format/fps/resolution with
``ffprobe`` and match the anchor's encode to it. ``kinescore anchor build
--probe-clip`` is this module's equivalent of that call; the alternative
(``--fps``/``--width``/``--height``/``--crf`` given directly) exists for a
caller who already knows the target and has no generated clip on hand to
probe, but is not the default path.
"""
from __future__ import annotations

import os
import subprocess

__all__ = ["probe_crf_context", "reencode_anchor_clip", "build_anchor"]


def probe_crf_context(clip_path: str) -> dict:
    """Probe one clip's codec/pix_fmt/fps/resolution via ``ffprobe``.

    Verbatim idea-port of the source's ``probe_crf_context()``
    (15_prepare_real_anchor.py:48-61), generalised from "the first
    ``full_gt.mp4`` found under a dreamdojo root" to "one caller-given clip
    path" -- the source's own glob-then-take-first is manifest-discovery
    logic that belongs to whichever caller has a manifest, not to the probe
    itself.

    Raises
    ------
    subprocess.CalledProcessError
        If ``ffprobe`` fails (missing binary, unreadable/corrupt file).
    ValueError
        If the file has no usable video stream (width/height/fps all zero).
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,pix_fmt,width,height,r_frame_rate",
         "-of", "default=nw=1", clip_path],
        capture_output=True, text=True, check=True,
    ).stdout
    d = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    rfr = d.get("r_frame_rate", "0/1").split("/")
    fps = float(rfr[0]) / float(rfr[1]) if len(rfr) == 2 and float(rfr[1]) else 0.0
    w, h = int(d.get("width", 0) or 0), int(d.get("height", 0) or 0)
    if fps <= 0 or w <= 0 or h <= 0:
        raise ValueError(
            f"{clip_path}: ffprobe returned no usable video stream "
            f"(fps={fps}, {w}x{h}) -- cannot probe a CRF context from it")
    return {"codec": d.get("codec_name", "h264"), "pix_fmt": d.get("pix_fmt", "yuv420p"),
            "fps": fps, "width": w, "height": h}


def reencode_anchor_clip(src: str, dst: str, *, fps: float, width: int,
                         height: int, pix_fmt: str, crf: int) -> None:
    """``ffmpeg`` re-encode ``src`` -> ``dst`` at a fixed fps/resolution/CRF.

    Verbatim port of the source's ``encode_real_dm()``
    (15_prepare_real_anchor.py:64-69, identical to
    34_prepare_real_dm16.py:36-40's ``encode()``): scale-and-drop-frames in
    one ``-vf`` pass, h264 at the given CRF, audio stripped (``-an``, since
    no metric in this package ever reads audio).

    Raises
    ------
    subprocess.CalledProcessError
        If ``ffmpeg`` fails.
    """
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-vf", f"scale={width}:{height},fps={fps}", "-c:v", "libx264",
         "-crf", str(crf), "-pix_fmt", pix_fmt, "-an", dst],
        check=True,
    )


def build_anchor(sources: list[str], out_dir: str, *, fps: float, width: int,
                 height: int, crf: int, pix_fmt: str = "yuv420p") -> dict:
    """Re-encode every clip in ``sources`` into ``out_dir`` at one fixed context.

    Resume-safe (mirrors the source's ``if not os.path.exists(dst):`` skip,
    15_prepare_real_anchor.py:162 / 34_prepare_real_dm16.py:82): a clip
    already re-encoded at ``out_dir`` is left alone rather than redone.

    Returns
    -------
    dict
        ``{"fps", "width", "height", "crf", "pix_fmt", "n_built", "n_skipped",
        "n_failed", "out_dir", "clips": [{"episode", "src", "dst"}, ...],
        "failures": [{"src", "error"}, ...]}``.
    """
    os.makedirs(out_dir, exist_ok=True)
    clips: list[dict] = []
    failures: list[dict] = []
    n_skipped = 0
    for src in sources:
        episode = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(out_dir, f"{episode}.mp4")
        if os.path.exists(dst):
            n_skipped += 1
            clips.append({"episode": episode, "src": src, "dst": dst})
            continue
        try:
            reencode_anchor_clip(src, dst, fps=fps, width=width, height=height,
                                 pix_fmt=pix_fmt, crf=crf)
        except Exception as exc:  # noqa: BLE001 -- one bad clip must not abort the batch
            failures.append({"src": src, "error": f"{type(exc).__name__}: {exc}"})
            continue
        clips.append({"episode": episode, "src": src, "dst": dst})
    return {
        "fps": fps, "width": width, "height": height, "crf": crf,
        "pix_fmt": pix_fmt, "out_dir": out_dir,
        "n_built": len(clips) - n_skipped, "n_skipped": n_skipped,
        "n_failed": len(failures), "clips": clips, "failures": failures,
    }
