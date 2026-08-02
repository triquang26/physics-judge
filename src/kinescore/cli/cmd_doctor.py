"""``kinescore doctor``: environment sanity check.

The checks themselves (torch/CUDA, ffprobe, asset hashes, env vars) and the
human/markdown renderers live in :mod:`kinescore.bench.doctor` -- see that
module's docstring for why it sits in ``bench/`` and for the hard
requirement this command is pinned against (``tests/test_cli_smoke.py``):
it must run to completion on an interpreter with none of
torch/pytorch_kinematics/transformers installed, and it must never touch the
network. This module is the argparse shell: pick ``--json``/``--markdown``/
human output and print it.
"""
from __future__ import annotations

import argparse
import json

NAME = "doctor"
HELP = "check the environment kinescore needs (torch, CUDA, ffprobe, assets)"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    fmt.add_argument("--markdown", action="store_true",
                     help="emit the README 'Tested environment' table")


def run(args: argparse.Namespace) -> int:
    from kinescore.bench.doctor import build_report, render_human, render_markdown

    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    elif args.markdown:
        print(render_markdown(report))
    else:
        print(render_human(report))
    return 0
