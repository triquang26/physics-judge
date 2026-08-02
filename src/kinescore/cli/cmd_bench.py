"""``kinescore bench``: expand the benchmark matrix and build a manifest per cell.

``bench run --config configs/benchmark.yaml`` is the "download data, drop it
in the folder, run one command" entry point the plan describes: it loads a
validated :class:`~kinescore.bench.config.BenchConfig`
(:mod:`kinescore.bench.config`) and a :class:`~kinescore.bench.robot_map.RobotMap`
(``--robot-map``, default ``configs/robot_map.yaml`` next to ``--config``),
expands them into cells (:mod:`kinescore.bench.matrix`), and for each non-N/A
cell dispatches to the matching :class:`~kinescore.bench.sources.base.ClipSource`
(looked up in :data:`~kinescore.bench.sources.DEFAULT_SOURCES` by generator
name) to discover clips into one combined manifest under ``<out>/<run_id>/``
via the existing :func:`kinescore.bench.manifest.build_manifest`.

Scoring and aggregation are other commands' jobs
(``kinescore score`` / ``kinescore aggregate`` already exist and operate on
any manifest, including this one); this command stops after the manifest
stage on purpose, but is structured -- one resolved cell list, one combined
manifest, one provenance block -- so a later ``bench run`` revision can chain
straight into scoring without restructuring what is built here.

N/A cells (declared in ``na_cells``, or a (robot, generator) pair
``robot_map.yaml`` does not claim) are reported as ``N/A`` explicitly, both
in the printed table and in ``--cells-out``'s JSON -- never as "0 rows
found", which would be indistinguishable from a real discovery failure.

``bench noise-floor`` builds the paired re-encode noise floor
(:mod:`kinescore.bench.noise_floor`) for one manifest's ground-truth clips --
see that module's docstring for why a PAIRED claim needs this ruler instead
of an absolute per-clip error bound. Wired here (not a standalone top-level
command) because it is a diagnostic ON a manifest ``bench run`` already
built, sharing that command's ``--config``/robot/reader plumbing via
``kinescore.cli._scoring``.
"""
from __future__ import annotations

import argparse
import json
import sys

NAME = "bench"
HELP = "expand the benchmark matrix and build per-cell manifests; also: noise-floor"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="bench_action", metavar="action")

    run_p = actions.add_parser(
        "run", help="expand the matrix and build a manifest per cell")
    run_p.add_argument("--config", required=True,
                       help="path to a benchmark.yaml (see configs/benchmark.yaml)")
    run_p.add_argument("--robot-map", default=None,
                       help="path to robot_map.yaml (default: robot_map.yaml "
                            "next to --config)")
    run_p.add_argument("--out", default=None,
                       help="output directory (default: "
                            "$KINESCORE_OUTPUT_DIR/<run_id>, or ./out/<run_id>)")
    run_p.add_argument("--only", action="append", default=None, metavar="AXIS=VALUE",
                       help="restrict to cells matching AXIS=VALUE (repeatable; "
                            "cells must match every --only given)")
    run_p.add_argument("--dry-run", action="store_true",
                       help="expand the matrix and print the cell table; "
                            "touch no video, build no manifest")
    run_p.add_argument("--cells-out", default=None,
                       help="write the resolved cell table (including N/A "
                            "cells) as JSON to this path")
    run_p.add_argument("--episode-cap", type=int, default=None,
                       help="override caps.episodes_per_cell from the config "
                            "for this invocation")
    run_p.add_argument("--compute-sha1", action="store_true",
                       help="hash every clip's bytes (slow; off by default)")
    run_p.add_argument("--on-error", choices=("skip", "raise"), default="skip",
                       help="what to do when a clip fails to probe (default: skip)")

    nf_p = actions.add_parser(
        "noise-floor", help="build the paired re-encode noise floor for a manifest's gt clips")
    nf_p.add_argument("--manifest", required=True,
                      help="manifest (.parquet/.json) from `kinescore bench run`/`manifest`")
    from kinescore.cli._scoring import add_common_arguments
    add_common_arguments(nf_p, require_reader=True)
    nf_p.add_argument("--scratch-dir", required=True,
                      help="scratch directory for temporary re-encoded clips")
    nf_p.add_argument("--out", required=True, help="noise_floor.json is written here")
    nf_p.add_argument("--crf-base", type=int, default=23)
    nf_p.add_argument("--crf-mod", type=int, default=12)


def run(args: argparse.Namespace) -> int:
    action = getattr(args, "bench_action", None)
    if action == "run":
        return _run_run(args)
    if action == "noise-floor":
        return _run_noise_floor(args)
    print("usage: kinescore bench run --config configs/benchmark.yaml [--dry-run]",
         file=sys.stderr)
    return 2


def _run_run(args: argparse.Namespace) -> int:  # noqa: C901
    import os

    from kinescore import paths
    from kinescore.bench.config import AXIS_VALUES, ConfigError, load_config
    from kinescore.bench.manifest import build_manifest, save_manifest, verify_manifest
    from kinescore.bench.matrix import (
        allow_patterns,
        cell_row,
        expand,
        matches_only,
        na_cells,
        parse_only_filters,
    )
    from kinescore.bench.robot_map import RobotMapError, load_robot_map
    from kinescore.bench.sources import DEFAULT_SOURCES
    from kinescore.cli._common import resolve_robot_map_path
    from kinescore.cli._provenance import provenance_block, write_json

    try:
        config = load_config(args.config)
    except (ConfigError, paths.MissingPathError) as exc:
        print(f"[bench] invalid config {args.config!r}: {exc}", file=sys.stderr)
        return 1

    robot_map_path = resolve_robot_map_path(args)
    try:
        robot_map = load_robot_map(robot_map_path)
    except (RobotMapError, OSError) as exc:
        print(f"[bench] invalid robot map {robot_map_path!r}: {exc}", file=sys.stderr)
        return 1

    try:
        filters = parse_only_filters(args.only, AXIS_VALUES)
    except ValueError as exc:
        print(f"[bench] {exc}", file=sys.stderr)
        return 2

    cells = [c for c in expand(config, robot_map) if matches_only(c, filters)]
    nas = [c for c in na_cells(config, robot_map) if matches_only(c, filters)]

    print(f"[bench] run_id={config.run_id!r} suite={config.suite!r} "
         f"seed={config.seed}")
    print(f"[bench] {len(cells)} cell(s) to build, {len(nas)} declared N/A")
    for cell in cells:
        print(f"[bench]   {cell.cell_id}  robot={cell.robot}  "
             f"view_layout={cell.view_layout.key}")
    for cell in nas:
        print(f"[bench]   {cell.cell_id}  N/A")

    if args.cells_out:
        rows = [cell_row(c, config, status="pending") for c in cells]
        rows += [cell_row(c, config, status="na") for c in nas]
        os.makedirs(os.path.dirname(args.cells_out) or ".", exist_ok=True)
        with open(args.cells_out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[bench] wrote cell table -> {args.cells_out}")

    if args.dry_run:
        print(f"[bench] --dry-run: {len(allow_patterns(config, robot_map))} "
             f"allow_patterns would be requested; no manifest built")
        return 0

    if not cells:
        print("[bench] no cells to build (matrix + --only filters produced an "
             "empty set); nothing to do", file=sys.stderr)
        return 1

    data_root = paths.env_path("KINESCORE_DATA_ROOT")
    out_dir = args.out or str(paths.output_dir() / config.run_id)
    episode_cap = (args.episode_cap if args.episode_cap is not None
                  else config.caps.get("episodes_per_cell"))

    all_rows: list[dict] = []
    n_rows_by_cell: dict[str, int] = {}
    for cell in cells:
        try:
            source = DEFAULT_SOURCES.get(cell.generator)
        except ValueError as exc:
            print(f"[bench] {exc} ({cell.cell_id}); skipping", file=sys.stderr)
            continue
        plugin = source.make_plugin(cell, str(data_root), config)
        rows = build_manifest([plugin], episode_cap=episode_cap,
                              compute_sha1=args.compute_sha1, on_error=args.on_error)
        n_rows_by_cell[cell.cell_id] = len(rows)
        print(f"[bench]   {cell.cell_id}: {len(rows)} row(s)")
        all_rows.extend(rows)

    if not all_rows:
        print("[bench] every cell discovered zero clips; nothing written",
             file=sys.stderr)
        return 1

    report = verify_manifest(all_rows)
    written = save_manifest(all_rows, out_dir, pairing_report=report)
    print(f"[bench] wrote {len(all_rows)} row(s) across {len(cells)} cell(s) "
         f"-> {written['manifest']}")
    print(f"[bench] pairing: {report['n_pairs']} pair(s), "
         f"{report['n_mismatches']} mismatch(es)")

    prov = provenance_block(
        run_id=config.run_id, seed=config.seed, config_path=args.config,
        robot_map_path=robot_map_path, suite=config.suite, rate_policy=config.rate_policy,
        n_cells=len(cells), n_na_cells=len(nas), n_rows=len(all_rows),
        n_rows_by_cell=n_rows_by_cell,
        na_cells=[c.cell_id for c in nas],
        allow_patterns=allow_patterns(config, robot_map))
    write_json(os.path.join(out_dir, "provenance.json"), prov)

    return 0 if report["ok"] else 1


def _run_noise_floor(args: argparse.Namespace) -> int:
    import os

    from kinescore.bench.manifest import load_manifest
    from kinescore.bench.noise_floor import build_noise_floor
    from kinescore.cli._provenance import provenance_block, write_json
    from kinescore.cli._scoring import build_scorer, view_layout_from_args
    from kinescore.core.clip import ClipSpec
    from kinescore.video.reader import load_rgb

    view_layout = view_layout_from_args(args)
    rows = load_manifest(args.manifest)
    gt_rows = [r for r in rows if r.get("role") == "gt"]
    if not gt_rows:
        print(f"[bench] manifest {args.manifest!r} has no role=gt rows", file=sys.stderr)
        return 1

    scorer = build_scorer(args, view_layout)

    def _score_fn(path: str, dt: float) -> dict:
        clip = ClipSpec(path=path, fps=1.0 / dt, n_frames=0, width=0, height=0,
                        dt=dt, view_layout=view_layout)
        frames = load_rgb(clip)
        return scorer.score(frames, clip).result.scalars()

    result = build_noise_floor(gt_rows, _score_fn, crf_base=args.crf_base,
                               crf_mod=args.crf_mod, scratch_dir=args.scratch_dir)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_json(args.out, result)
    print(f"[bench] noise floor over {result['n_clips']} clip(s) -> {args.out}")
    for metric, summary in result["summary"].items():
        print(f"[bench]   {metric}: null_p95={summary['null_p95']:.4g} "
             f"(n={summary['n']})")

    prov = provenance_block(manifest=args.manifest, robot=args.robot, reader=args.reader,
                            crf_base=args.crf_base, crf_mod=args.crf_mod,
                            n_clips=result["n_clips"])
    write_json(os.path.join(os.path.dirname(args.out) or ".",
                            "noise_floor_provenance.json"), prov)
    return 0
