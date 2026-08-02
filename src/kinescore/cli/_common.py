"""Tiny argparse-``Namespace``-facing helpers shared by more than one ``cmd_*.py``.

Deliberately separate from :mod:`kinescore.cli._provenance` (output stamping)
and :mod:`kinescore.cli._scoring` (robot/reader/suite composition) -- this
module is for the CLI-layer glue that is genuinely about *argparse.Namespace*
shapes (``args.config``/``args.robot_map``) rather than benchmark logic, so it
has no business living in :mod:`kinescore.bench`. Not a subcommand itself (no
leading ``cmd_``), so ``kinescore.cli.main`` never registers it as one.
"""
from __future__ import annotations

import argparse
import os

__all__ = ["resolve_robot_map_path"]


def resolve_robot_map_path(args: argparse.Namespace) -> str:
    """``--robot-map`` if given, else ``robot_map.yaml`` next to ``--config``.

    Shared by ``kinescore bench run`` and ``kinescore data pull`` -- both
    take a ``benchmark.yaml`` (``--config``) and default to a sibling
    ``robot_map.yaml`` rather than requiring ``--robot-map`` to be spelled
    out every time.
    """
    if args.robot_map:
        return args.robot_map
    return os.path.join(os.path.dirname(os.path.abspath(args.config)), "robot_map.yaml")
