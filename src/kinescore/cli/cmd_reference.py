"""``kinescore reference``: build a :class:`~kinescore.reference.RealMotionReference`.

Only one action exists today (``build``), but the command is structured as
``reference <action>`` (a nested subparser, mirroring the ``reference build``
spelling everywhere else in this package's docs) rather than a flat ``kinescore
reference-build`` so a future ``reference inspect``/``reference diff`` has
somewhere to live without a new top-level command.
"""
from __future__ import annotations

import argparse
import sys

NAME = "reference"
HELP = "build a real-motion reference fingerprint from real clips"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from kinescore.cli._scoring import add_common_arguments

    actions = parser.add_subparsers(dest="reference_action", metavar="action")

    build_p = actions.add_parser(
        "build", help="score real clips and fit a RealMotionReference")
    build_p.add_argument("--manifest", required=True,
                         help="manifest file (.parquet or .json)")
    build_p.add_argument("--role", action="append", default=None,
                         help="manifest role(s) counted as real motion "
                              "(repeatable; default: gt and real)")
    add_common_arguments(build_p, require_reader=True)
    build_p.add_argument("--out", required=True,
                         help="output .pt path for the reference")
    build_p.add_argument("--n-q", type=int, default=100,
                         help="quantile points per motion quantity (default: 100)")
    build_p.add_argument("--max-frames", type=int, default=0)


def run(args: argparse.Namespace) -> int:
    action = getattr(args, "reference_action", None)
    if action == "build":
        return _run_build(args)
    print("usage: kinescore reference build --manifest ... --robot ... "
         "--reader ... --out ...", file=sys.stderr)
    return 2


def _run_build(args: argparse.Namespace) -> int:
    from kinescore.bench.manifest import load_manifest
    from kinescore.cli._provenance import provenance_block, write_json
    from kinescore.cli._scoring import (
        apply_resolved_timebase,
        build_scorer,
        view_layout_from_args,
    )
    from kinescore.core.clip import ClipSpec, TimebaseError
    from kinescore.reference import RealMotionReference
    from kinescore.video.reader import load_rgb

    roles = set(args.role) if args.role else {"gt", "real"}
    view_layout = view_layout_from_args(args)

    rows = load_manifest(args.manifest)
    real_rows = [r for r in rows if r.get("role") in roles]
    if not real_rows:
        print(f"[reference] no rows with role in {sorted(roles)} in "
             f"{args.manifest!r}", file=sys.stderr)
        return 1

    print(f"[reference] cross-checking timebase for {len(real_rows)} real "
         f"clip(s)...")
    try:
        real_rows = apply_resolved_timebase(real_rows, fps=args.fps,
                                            dt=args.dt, view_layout=view_layout)
    except (ValueError, TimebaseError) as exc:
        print(f"[reference] timebase error: {exc}", file=sys.stderr)
        raise

    dts = sorted({row["dt"] for row in real_rows})
    if len(dts) > 1:
        print(f"[reference] real clips do not share one dt: {dts}. A "
             f"reference is pinned to a single rate (see "
             f"RealMotionReference's D2 fix) -- pass --fps/--dt to force "
             f"every clip to the same rate, or split this manifest by rate "
             f"and build one reference per rate.", file=sys.stderr)
        return 1
    dt = dts[0]

    scorer = build_scorer(args, view_layout)

    residual_list: list[dict] = []
    samples_list: list[dict] = []
    n_failed = 0
    for row in real_rows:
        try:
            clip = ClipSpec.from_fps(
                path=row["path"], fps=float(row["fps"]),
                n_frames=int(row["n_frames"]), width=int(row["w"]),
                height=int(row["h"]), dt_source=row.get("dt_source", "table"),
                view_layout=view_layout, codec=row.get("codec"))
            frames = load_rgb(clip, max_frames=args.max_frames)
            scored = scorer.score(frames, clip)
        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            print(f"[reference] skip {row['path']}: {type(exc).__name__}: {exc}",
                 file=sys.stderr)
            continue
        residual_list.append(scored.result.scalars())
        samples_list.append(scored.result.perframe())

    if not residual_list:
        print("[reference] every real clip failed to score; no reference "
             "written", file=sys.stderr)
        return 1

    reference = RealMotionReference.build(
        scorer.suite, residual_list, samples_list, dt=dt, n_q=args.n_q)
    reference.save(args.out)
    print(f"[reference] built from {len(residual_list)}/{len(real_rows)} "
         f"real clip(s) ({n_failed} failed) -> {args.out}")
    print(f"[reference] {reference!r}")

    prov = provenance_block(
        suite_id=scorer.suite.suite_id, suite_name=scorer.suite.name,
        robot=args.robot, reader_id=scorer.reader.reader_id, dt=dt,
        n_real_clips=len(residual_list), n_failed=n_failed,
        n_terms=len(reference.term_keys), n_quantities=len(reference.quantity_keys))
    write_json(args.out + ".provenance.json", prov)
    return 0
