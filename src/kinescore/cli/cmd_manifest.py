"""``kinescore manifest``: discover clips on disk into a probed manifest.

``kinescore.bench.manifest.build_manifest`` takes a list of *plugins* (plain
zero-argument callables yielding :class:`~kinescore.bench.manifest.DiscoveredClip`)
so that adding a new data-source layout never means editing shared code (see
that module's docstring for the hardcoded-cascade defect this replaced). This
command is the CLI's own plugin: a recursive glob under ``--root``, one
:class:`DiscoveredClip` per matched file. It is deliberately the simplest
possible plugin, not the only one this package could ever need -- a caller
with a more structured layout (paired gt/pred trees, several families in one
run) is expected to call :func:`kinescore.bench.manifest.build_manifest`
directly with its own plugins; this command exists for the common case of
"score every clip under one directory."
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

HELP = "discover clips under a directory into a probed manifest"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", required=True,
                        help="directory to search for clips")
    parser.add_argument("--pattern", default="*.mp4",
                        help="glob pattern relative to --root (default: "
                             "*.mp4; use e.g. '**/*.mp4' for a recursive search)")
    parser.add_argument("--method", default="method",
                        help="benchmark method name stamped on every "
                             "discovered clip (default: 'method')")
    parser.add_argument("--family", default="family",
                        help="dataset family name stamped on every "
                             "discovered clip (default: 'family')")
    parser.add_argument("--role", default="pred", choices=("gt", "pred", "real"),
                        help="role stamped on every discovered clip (default: pred)")
    parser.add_argument("--out", required=True,
                        help="output directory; the manifest is written as "
                             "<out>/bench_manifest.parquet (pandas/pyarrow "
                             "installed) or <out>/bench_manifest.json")
    parser.add_argument("--fps", type=float, default=None,
                        help="config-table fps hint for every clip, cross-"
                             "checked against ffprobe (kinescore.video.probe."
                             "resolve_timebase); omit to trust the probe alone")
    parser.add_argument("--n-views", type=int, default=1)
    parser.add_argument("--view-order", default=None,
                        help="comma-separated view names, length --n-views")
    parser.add_argument("--episode-cap", type=int, default=None,
                        help="stop after this many accepted clips from --family")
    parser.add_argument("--compute-sha1", action="store_true",
                        help="hash every clip's bytes (slow; off by default)")
    parser.add_argument("--on-error", choices=("skip", "raise"), default="skip",
                        help="what to do when a clip fails to probe (default: skip)")
    parser.add_argument("--no-pairing-report", action="store_true",
                        help="skip the gt/pred pairing sanity check")


def run(args: argparse.Namespace) -> int:
    from kinescore.bench.manifest import (
        DiscoveredClip,
        build_manifest,
        save_manifest,
        verify_manifest,
    )
    from kinescore.cli._provenance import provenance_block, write_json
    from kinescore.cli._scoring import view_layout_from_args

    view_layout = view_layout_from_args(args)

    search = os.path.join(args.root, args.pattern)
    recursive = "**" in args.pattern
    paths = sorted(glob.glob(search, recursive=recursive))
    if not paths:
        print(f"[manifest] no files matched {search!r}", file=sys.stderr)
        return 1

    def _plugin():
        for path in paths:
            episode = os.path.splitext(os.path.basename(path))[0]
            yield DiscoveredClip(
                method=args.method, family=args.family, episode=episode,
                role=args.role, path=path, pair_key=f"{args.family}/{episode}",
                fps_hint=args.fps, view_layout=view_layout)

    rows = build_manifest([_plugin], episode_cap=args.episode_cap,
                          compute_sha1=args.compute_sha1, on_error=args.on_error)
    if not rows:
        print("[manifest] every discovered clip failed to probe; nothing written",
              file=sys.stderr)
        return 1

    report = None if args.no_pairing_report else verify_manifest(rows)
    written = save_manifest(rows, args.out, pairing_report=report)
    print(f"[manifest] wrote {len(rows)} row(s) -> {written['manifest']}")
    if report is not None:
        print(f"[manifest] pairing: {report['n_pairs']} pair(s), "
              f"{report['n_mismatches']} mismatch(es) -> "
              f"{written.get('pairing_report', '(not written)')}")

    prov = provenance_block(
        n_rows=len(rows), root=args.root, pattern=args.pattern,
        view_layout=view_layout.key,
        dt_sources=sorted({r.get("dt_source") for r in rows}))
    write_json(os.path.join(args.out, "provenance.json"), prov)

    return 0 if (report is None or report["ok"]) else 1
