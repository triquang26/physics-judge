"""``kinescore data``: pull the benchmark's video datasets from Hugging Face.

``kinescore data pull``
------------------------
The allow-pattern resolution, the HF repo table, and the actual
``snapshot_download`` calls live in :mod:`kinescore.bench.pull` -- see that
module's docstring for the full three-repo rationale, the
``humanoid``/``single_arm`` dataset-identity findings that drove the
coordinator directive to source those axes from local
``$KINESCORE_DROID_STD_DIR``/``$KINESCORE_TELEOP_GR1_DIR`` trees instead, and
exactly what :func:`~kinescore.bench.pull.resolve_allow_patterns` does and
does not know. This module is the argparse shell around it: parse
``--config``/``--repo``/``--dry-run``/etc., resolve the robot map and
allow-patterns, and either print the dry-run plan or call
:func:`~kinescore.bench.pull.pull_one` per dataset -- and, since
``kinescore.bench`` must not import ``kinescore.cli``
(``tests/test_import_layering.py``), build and write each dataset's
``provenance.json`` sidecar here from :func:`~kinescore.bench.pull.pull_one`'s
return value rather than inside that function.

``kinescore data ingest``/``kinescore data verify``
-------------------------------------------------------
Two more actions on this same ``data`` subcommand, downstream of ``pull``:
``ingest`` runs :class:`kinescore.bench.ingest.Ingestor` (walks the raw HF
mirror -- :class:`~kinescore.bench.layout.RawHFLayout` -- and symlinks it
into the canonical ``bench/<cache>/<robot>/<view>/<generator>/<horizon>/
episode_XXXX/{pred.mp4,gt.mp4}`` shape, see that module's docstring for the
``cell_card.json`` schema and the symlink-by-default/``--copy`` fallback);
``verify`` ffprobes every canonical clip against ``configs/data_spec.yaml``
(:func:`kinescore.bench.verify.verify_layout`) and exits non-zero, naming the
SPECIFIC clip path, on the first hard mismatch (width/height always; fps
unless the generator is ``fps_tolerant``; a broken symlink). Both actions
already delegate essentially everything to :mod:`kinescore.bench.ingest`/
:mod:`kinescore.bench.verify`; ``_run_ingest``/``_run_verify`` below are only
argument translation + printing, so they stay here rather than moving further.
"""
from __future__ import annotations

import argparse
import os
import sys

from kinescore.bench.pull import HF_REPOS

NAME = "data"
HELP = "pull benchmark video datasets from Hugging Face"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="data_action", metavar="action")

    pull = actions.add_parser(
        "pull", help="download the configured HF dataset repos")
    pull.add_argument("--config", required=True,
                      help="benchmark.yaml whose axes/sources determine "
                           "which files to pull")
    pull.add_argument("--dry-run", action="store_true",
                      help="print the resolved repos and allow_patterns; "
                           "download nothing")
    pull.add_argument("--repo", choices=sorted(HF_REPOS), default=None,
                      help="restrict to one dataset (default: all three)")
    pull.add_argument("--data-root", default=None,
                      help="override $KINESCORE_DATA_ROOT for this pull")
    pull.add_argument("--max-workers", type=int, default=8,
                      help="snapshot_download parallel downloads (default: 8)")
    pull.add_argument("--robot-map", default=None,
                      help="path to robot_map.yaml (default: robot_map.yaml "
                           "next to --config)")

    ingest = actions.add_parser(
        "ingest", help="materialise the raw HF mirror into the canonical layout")
    ingest.add_argument("--robot-map", required=True, help="path to robot_map.yaml")
    ingest.add_argument("--data-spec", required=True, help="path to data_spec.yaml")
    ingest.add_argument("--data-root", default=None,
                        help="raw layout root (default: $KINESCORE_DATA_ROOT)")
    ingest.add_argument("--out", required=True,
                        help="canonical layout root to materialise into")
    ingest.add_argument("--copy", action="store_true",
                        help="copy files instead of symlinking (fallback for "
                             "filesystems without symlink support; default: "
                             "symlink)")

    verify = actions.add_parser(
        "verify", help="ffprobe every canonical clip against data_spec.yaml")
    verify.add_argument("--data-spec", required=True, help="path to data_spec.yaml")
    verify.add_argument("--canonical-root", required=True,
                        help="canonical layout root written by `data ingest`")


def run(args: argparse.Namespace) -> int:
    action = getattr(args, "data_action", None)
    if action == "pull":
        return _run_pull(args)
    if action == "ingest":
        return _run_ingest(args)
    if action == "verify":
        return _run_verify(args)
    print("usage: kinescore data {pull,ingest,verify} ...", file=sys.stderr)
    return 2


def _run_ingest(args: argparse.Namespace) -> int:
    from kinescore import paths
    from kinescore.bench.data_spec import DataSpecError, load_data_spec
    from kinescore.bench.ingest import Ingestor
    from kinescore.bench.layout import CanonicalLayout, RawHFLayout
    from kinescore.bench.robot_map import RobotMapError, load_robot_map
    from kinescore.cli._provenance import provenance_block, write_json

    try:
        robot_map = load_robot_map(args.robot_map)
    except (RobotMapError, OSError) as exc:
        print(f"[data] invalid robot map {args.robot_map!r}: {exc}", file=sys.stderr)
        return 1
    try:
        data_spec = load_data_spec(args.data_spec)
    except (DataSpecError, OSError) as exc:
        print(f"[data] invalid data spec {args.data_spec!r}: {exc}", file=sys.stderr)
        return 1

    data_root = args.data_root or str(paths.env_path("KINESCORE_DATA_ROOT"))
    raw = RawHFLayout(data_root, robot_map, data_spec)
    canonical = CanonicalLayout(args.out)

    report = Ingestor(raw, canonical).run(copy=args.copy)
    for c in report.cells:
        note = ""
        if c.n_episodes_declared is not None and c.n_episodes_declared != c.n_episodes_actual:
            note = (f"  (DISAGREES with declared n_episodes_declared="
                    f"{c.n_episodes_declared})")
        print(f"[data]   {c.cell_id}: {c.n_episodes_actual} episode(s)"
             f"{f', {c.n_skipped_missing_gt} skipped for missing gt' if c.n_skipped_missing_gt else ''}"
             f"{note}")
    if report.unresolved:
        print(f"[data] {len(report.unresolved)} unresolved raw dir(s) (no robot "
             f"claims them):", file=sys.stderr)
        for u in report.unresolved:
            print(f"[data]   {u}", file=sys.stderr)

    print(f"[data] ingested {report.n_cells} cell(s), {report.n_episodes} episode(s) "
         f"-> {args.out}")

    prov = provenance_block(
        robot_map=args.robot_map, data_spec=args.data_spec, data_root=data_root,
        out=args.out, copy=args.copy, n_cells=report.n_cells,
        n_episodes=report.n_episodes,
        cells=[{"cell_id": c.cell_id, "n_episodes_actual": c.n_episodes_actual,
               "n_episodes_declared": c.n_episodes_declared,
               "n_skipped_missing_gt": c.n_skipped_missing_gt} for c in report.cells])
    write_json(os.path.join(args.out, "ingest_provenance.json"), prov)
    return 0


def _run_verify(args: argparse.Namespace) -> int:
    from kinescore.bench.data_spec import DataSpecError, load_data_spec
    from kinescore.bench.layout import CanonicalLayout
    from kinescore.bench.verify import verify_layout

    try:
        data_spec = load_data_spec(args.data_spec)
    except (DataSpecError, OSError) as exc:
        print(f"[data] invalid data spec {args.data_spec!r}: {exc}", file=sys.stderr)
        return 1

    canonical = CanonicalLayout(args.canonical_root)
    report = verify_layout(canonical, data_spec)

    print(f"[data] verified {report.n_cells} cell(s), {report.n_episodes} "
         f"episode(s), {report.n_clips_checked} clip(s)")
    for p in report.problems:
        print(f"[data]   FAIL {p.path}: {p.reason}", file=sys.stderr)

    if report.ok:
        print("[data] verify: OK")
        return 0
    print(f"[data] verify: {len(report.problems)} problem(s)", file=sys.stderr)
    return 1


def _run_pull(args: argparse.Namespace) -> int:
    from kinescore.bench.config import load_config
    from kinescore.bench.pull import pull_one, resolve_allow_patterns, resolve_data_root
    from kinescore.bench.robot_map import load_robot_map
    from kinescore.cli._common import resolve_robot_map_path
    from kinescore.cli._provenance import provenance_block, write_json

    config = load_config(args.config)
    robot_map = load_robot_map(resolve_robot_map_path(args))
    all_patterns = resolve_allow_patterns(config, robot_map)

    dataset_keys = [args.repo] if args.repo else sorted(HF_REPOS)

    if args.dry_run:
        print(f"[data] config: {args.config} (run_id={config.run_id})")
        data_root = resolve_data_root(args.data_root, dry_run=True)
        root_note = data_root or (
            "(unresolved -- set $KINESCORE_DATA_ROOT or pass --data-root)")
        print(f"[data] data_root: {root_note}")
        for key in dataset_keys:
            print(f"[data] {key}: repo_id={HF_REPOS[key]!r} "
                 f"allow_patterns={all_patterns[key]!r}")
        print("[data] --dry-run: downloaded nothing")
        return 0

    data_root = resolve_data_root(args.data_root, dry_run=False)
    assert data_root is not None  # resolve_data_root only returns None in dry-run

    results = []
    for key in dataset_keys:
        result = pull_one(key, patterns=all_patterns[key], data_root=data_root,
                          max_workers=args.max_workers)
        prov = provenance_block(
            run_id=config.run_id, config=args.config, dataset_key=key,
            repo_id=result["repo_id"], repo_type="dataset",
            revision=result["revision"], allow_patterns=all_patterns[key],
            n_files=result["n_files"], total_bytes=result["total_bytes"])
        write_json(os.path.join(result["local_dir"], "provenance.json"), prov)
        results.append(prov)

    n_files_total = sum(r["n_files"] for r in results)
    gb_total = sum(r["total_bytes"] for r in results) / 1e9
    print(f"[data] done: {len(results)} dataset(s), {n_files_total} file(s), "
         f"{gb_total:.2f} GB total")
    return 0
