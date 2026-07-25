"""``kinescore``: argparse entry point (``pyproject.toml``'s ``[project.scripts]``).

Every subcommand lives in its own ``cmd_*.py`` module exposing two names:

* ``add_arguments(parser)`` -- declares that subcommand's flags. Runs at
  parser-construction time, i.e. on every invocation including
  ``kinescore --help``, so it must stay import-light (see ``kinescore.cli``'s
  docstring).
* ``run(args) -> int`` -- does the work, including every heavy import. Only
  called after argparse has successfully parsed the command line, so a typo'd
  flag fails fast with a normal argparse usage error instead of importing
  torch first and failing later.

``main.py`` never branches on *which* subcommand was chosen -- it wires
``sub.set_defaults(_run=module.run)`` uniformly and lets ``args._run(args)``
dispatch. A subcommand with its own nested actions (``reference build``) does
that branching itself, inside its own ``run(args)`` -- see
``kinescore.cli.cmd_reference`` -- rather than main.py special-casing it.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

from kinescore import __version__

__all__ = ["build_parser", "main"]

#: (subcommand name, module path). Order here is display order in --help.
_SUBCOMMANDS: tuple[tuple[str, str], ...] = (
    ("manifest", "kinescore.cli.cmd_manifest"),
    ("score", "kinescore.cli.cmd_score"),
    ("reference", "kinescore.cli.cmd_reference"),
    ("aggregate", "kinescore.cli.cmd_aggregate"),
    ("report", "kinescore.cli.cmd_report"),
    ("describe", "kinescore.cli.cmd_describe"),
    ("doctor", "kinescore.cli.cmd_doctor"),
    ("cache", "kinescore.cli.cmd_cache"),
    ("train", "kinescore.cli.cmd_train"),
    ("calibrate", "kinescore.cli.cmd_calibrate"),
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the full ``kinescore`` argparse tree.

    Imports every ``cmd_*`` module (cheap -- see module docstring) to collect
    its ``add_arguments``/``run``; none of those imports pull in torch or any
    other heavy/optional dependency, which is what keeps this function (and
    therefore ``kinescore --help``) instant.
    """
    parser = argparse.ArgumentParser(
        prog="kinescore",
        description="Physics-plausibility benchmark for AI-generated robot video.")
    parser.add_argument("--version", action="version",
                        version=f"kinescore {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    for name, module_path in _SUBCOMMANDS:
        module = importlib.import_module(module_path)
        sub = subparsers.add_parser(
            name, help=getattr(module, "HELP", ""),
            description=module.__doc__)
        module.add_arguments(sub)
        sub.set_defaults(_run=module.run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run = getattr(args, "_run", None)
    if run is None:
        parser.print_help()
        return 1
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
