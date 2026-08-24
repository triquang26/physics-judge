"""``kinescore pull``: fetch a declared corpus and pin the revision it came from."""
from __future__ import annotations

import argparse

NAME = "pull"
HELP = "download a declared corpus into $KINESCORE_DATA_ROOT"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--what", default="all",
                        help="source name from sources.yaml, or 'all'")
    parser.add_argument("--revision", default=None,
                        help="commit sha or branch (default: the revision "
                             "already recorded for this source, else main)")
    parser.add_argument("--sources", default=None, help="path to sources.yaml")
    parser.add_argument("--workers", type=int, default=8,
                        help="concurrent file downloads")
    parser.add_argument("--list", action="store_true",
                        help="print declared sources and what is on disk")


def _select(sources: dict, what: str) -> list:
    if what == "all":
        return list(sources.values())
    if what not in sources:
        raise SystemExit(
            f"unknown source {what!r}; declared: {sorted(sources)} (or 'all')")
    return [sources[what]]


def run(args: argparse.Namespace) -> int:
    import shutil

    from huggingface_hub import HfApi, snapshot_download

    from kinescore.cli._shared import now
    from kinescore.registry.sources import (
        load_sources,
        read_revisions,
        record_revision,
    )

    sources = load_sources(args.sources) if args.sources else load_sources()
    on_disk = read_revisions()

    if args.list:
        for name, spec in sorted(sources.items()):
            got = on_disk.get(name)
            state = (f"{got['revision'][:12]}  {got['n_files']} files  "
                     f"{got['pulled_at']}" if got else "not pulled")
            print(f"{name:8s} {spec.repo:38s} -> {spec.dest:8s} {state}")
        return 0

    for spec in _select(sources, args.what):
        dest = spec.local_dir
        # A source already on disk re-pulls at the revision it was pulled at,
        # so `pull` twice is the same data twice unless a revision is named.
        revision = args.revision or (on_disk.get(spec.name) or {}).get(
            "revision") or "main"
        staging = dest.parent / f".{spec.name}.staging"
        print(f"[pull] {spec.name}: {spec.repo}@{revision} -> {dest}")
        snapshot_download(
            repo_id=spec.repo, repo_type=spec.repo_type, revision=revision,
            allow_patterns=list(spec.include), local_dir=str(staging),
            max_workers=args.workers)
        resolved = HfApi().repo_info(
            spec.repo, repo_type=spec.repo_type, revision=revision).sha

        landed = staging / spec.strip_prefix if spec.strip_prefix else staging
        if not landed.is_dir():
            raise SystemExit(
                f"{spec.name}: expected {landed} after download but it is not "
                f"a directory; check `strip_prefix` in sources.yaml")
        dest.mkdir(parents=True, exist_ok=True)
        n_files = 0
        for src in sorted(landed.rglob("*")):
            if not src.is_file() or ".cache" in src.parts:
                continue
            out = dest / src.relative_to(landed)
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                out.unlink()
            shutil.move(str(src), str(out))
            n_files += 1
        shutil.rmtree(staging, ignore_errors=True)

        record_revision(spec, resolved, pulled_at=now(), n_files=n_files)
        print(f"[pull] {spec.name}: {n_files} files at {resolved[:12]} -> {dest}")
    return 0
