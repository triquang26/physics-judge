"""``kinescore push``: sync local artifacts to the bucket via ``hf sync``."""
from __future__ import annotations

import argparse

NAME = "push"
HELP = "sync trained readers, scored cells and web bundles to the bucket"

BUCKET = "hf://buckets/twanghcmut/hallucinate-bench"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from kinescore.cli._shared import add_config_arguments

    parser.add_argument("--reader", action="append", default=[],
                        help="reader id whose checkpoint goes to train/")
    parser.add_argument("--scores", action="append", default=[],
                        help="scored output dir; its basename is the cell id "
                             "under scores/")
    parser.add_argument("--web", action="append", default=[],
                        help="web bundle dir; its basename goes under web/")
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the syncs without running them")
    add_config_arguments(parser)


def jobs(args: argparse.Namespace, ckpt_dir) -> list[tuple[str, str]]:
    """``(local, bucket)`` pairs for every requested artifact."""
    from pathlib import Path

    out = []
    for reader_id in args.reader:
        out.append((str(Path(ckpt_dir) / f"{reader_id}.diff"),
                    f"{args.bucket}/train/{reader_id}/diffusion"))
    for scores in args.scores:
        cell_id = Path(scores).name
        out.append((scores, f"{args.bucket}/scores/{cell_id}/diffusion"))
    for web in args.web:
        out.append((web, f"{args.bucket}/web/{Path(web).name}"))
    return out


def _stage_reader(prefix: str, stage) -> str:
    """checkpoint.pt + meta.json for one reader, staged from ``<prefix>.pt``."""
    import json
    import shutil

    import torch

    ck_path = prefix + ".pt"
    ck = torch.load(ck_path, map_location="cpu")
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ck_path, stage / "checkpoint.pt")
    meta = {**(ck.get("cfg") or {}), **(ck.get("meta") or {})}
    meta.pop("head", None)
    (stage / "meta.json").write_text(json.dumps(meta, indent=1))
    log = prefix + ".train_log.jsonl"
    import os
    if os.path.isfile(log):
        shutil.copyfile(log, stage / "train_log.jsonl")
    return str(stage)


def run(args: argparse.Namespace) -> int:
    import os
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    from kinescore.paths import env_path

    if not (args.reader or args.scores or args.web):
        raise SystemExit("pass --reader, --scores and/or --web")
    hf = shutil.which("hf")
    if hf is None:
        raise SystemExit("no `hf` CLI on PATH -- install it first")
    if not args.dry_run and not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is not set -- export a write token, "
                         "never write it to a file")

    ckpt_dir = env_path("KINESCORE_CKPT_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        for i, (src, dest) in enumerate(jobs(args, ckpt_dir)):
            if src.startswith(str(ckpt_dir)) and src.endswith(".diff"):
                src = _stage_reader(src, Path(tmp) / str(i))
            if not Path(src).is_dir():
                raise SystemExit(f"nothing to push at {src}")
            print(f"[push] {src} -> {dest}")
            if not args.dry_run:
                subprocess.run([hf, "sync", src, dest], check=True)
    return 0
