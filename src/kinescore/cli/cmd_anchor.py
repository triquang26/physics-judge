"""``kinescore anchor build``: re-encode REAL footage to a generator's fps/resolution.

The actual ffmpeg/ffprobe work -- :func:`~kinescore.video.anchor.probe_crf_context`,
:func:`~kinescore.video.anchor.reencode_anchor_clip`,
:func:`~kinescore.video.anchor.build_anchor` -- lives in
:mod:`kinescore.video.anchor` (it is video I/O, not CLI logic, and is
reachable from Python without argparse; see that module's docstring for the
full "frame-rate-matched footage" rationale and the ``real_dm``/``real_dm16``
provenance). This module is the argparse shell: declare ``anchor build``'s
flags, resolve ``--probe-clip`` vs. ``--fps``/``--width``/``--height``,
glob ``--real-glob``, call :func:`~kinescore.video.anchor.build_anchor`, and
write the provenance sidecar.
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys

NAME = "anchor"
HELP = "re-encode real footage to a generator's fps/resolution for a fair comparison"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="anchor_action", metavar="action")

    build_p = actions.add_parser(
        "build", help="ffmpeg re-encode real clips to a target fps/resolution/CRF")
    build_p.add_argument("--real-glob", required=True,
                         help="glob of real source video files to re-encode "
                              "(e.g. 'data/real_teleop/*.mp4')")
    build_p.add_argument("--out-dir", required=True,
                         help="directory the re-encoded anchor clips + "
                              "provenance JSON are written to")
    build_p.add_argument("--probe-clip", default=None,
                         help="probe fps/width/height/pix_fmt/codec from this "
                              "REAL GENERATED clip instead of the --fps/"
                              "--width/--height flags below")
    build_p.add_argument("--fps", type=float, default=None)
    build_p.add_argument("--width", type=int, default=None)
    build_p.add_argument("--height", type=int, default=None)
    build_p.add_argument("--crf", type=int, default=23,
                         help="h264 CRF for the anchor encode (default: 23, "
                              "the source's default for both real_dm and "
                              "real_dm16)")
    build_p.add_argument("--pix-fmt", default="yuv420p")
    build_p.add_argument("--limit", type=int, default=0,
                         help="cap the number of clips re-encoded (0 = all "
                              "matched by --real-glob)")


def run(args: argparse.Namespace) -> int:
    action = getattr(args, "anchor_action", None)
    if action == "build":
        return _run_build(args)
    print("usage: kinescore anchor build --real-glob ... --out-dir ... "
          "(--probe-clip CLIP | --fps F --width W --height H)", file=sys.stderr)
    return 2


def _run_build(args: argparse.Namespace) -> int:
    import os

    from kinescore.cli._provenance import provenance_block, write_json
    from kinescore.video.anchor import build_anchor, probe_crf_context

    if args.probe_clip:
        try:
            ctx = probe_crf_context(args.probe_clip)
        except (subprocess.CalledProcessError, ValueError) as exc:
            print(f"[anchor] failed to probe --probe-clip {args.probe_clip!r}: "
                 f"{exc}", file=sys.stderr)
            return 1
        fps, width, height = ctx["fps"], ctx["width"], ctx["height"]
        pix_fmt = ctx["pix_fmt"]
        probe_source = args.probe_clip
    else:
        if args.fps is None or args.width is None or args.height is None:
            print("[anchor] need --probe-clip, OR all of --fps/--width/--height",
                 file=sys.stderr)
            return 2
        fps, width, height = args.fps, args.width, args.height
        pix_fmt = args.pix_fmt
        ctx = {"codec": None, "pix_fmt": pix_fmt, "fps": fps,
              "width": width, "height": height}
        probe_source = None

    sources = sorted(glob.glob(args.real_glob))
    if args.limit:
        sources = sources[: args.limit]
    if not sources:
        print(f"[anchor] no files matched --real-glob {args.real_glob!r}",
             file=sys.stderr)
        return 1

    print(f"[anchor] context: fps={fps} {width}x{height} pix_fmt={pix_fmt} "
         f"crf={args.crf}"
         + (f" (probed from {probe_source})" if probe_source else " (from CLI args)"))

    result = build_anchor(sources, args.out_dir, fps=fps, width=width,
                          height=height, crf=args.crf, pix_fmt=pix_fmt)

    prov = provenance_block(
        probe_source=probe_source, probe_context=ctx, real_glob=args.real_glob,
        n_sources=len(sources), **result)
    prov_path = os.path.join(args.out_dir, "anchor_provenance.json")
    write_json(prov_path, prov)

    print(f"[anchor] built {result['n_built']} (skipped {result['n_skipped']}, "
         f"failed {result['n_failed']}) / {len(sources)} clip(s) -> "
         f"{args.out_dir}")
    print(f"[anchor] provenance -> {prov_path}")
    if result["n_failed"]:
        for f in result["failures"][:10]:
            print(f"[anchor]   FAILED {f['src']}: {f['error']}", file=sys.stderr)
    return 0 if result["n_failed"] == 0 else 1
