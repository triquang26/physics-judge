"""``kinescore``: argparse entry point (``pyproject.toml``'s ``[project.scripts]``).

Every subcommand lives in its own ``cmd_*.py`` module exposing four names:

* ``NAME`` -- the subcommand string (``"score"``, ``"train-rawrad"``, ...).
* ``HELP`` -- one-line summary shown in ``kinescore --help``.
* ``add_arguments(parser)`` -- declares that subcommand's flags. Runs at
  parser-construction time, i.e. on every invocation including
  ``kinescore --help``, so it must stay import-light (see ``kinescore.cli``'s
  docstring).
* ``run(args) -> int`` -- does the work, including every heavy import. Only
  called after argparse has successfully parsed the command line, so a typo'd
  flag fails fast with a normal argparse usage error instead of importing
  torch first and failing later.

No class, no instance: every one of these subcommands is stateless (parse
flags, do work, return a code), so there is nothing a ``self`` would ever
carry between ``add_arguments`` and ``run`` -- a base class here would be
ceremony, not structure. ``main.py`` never branches on *which* subcommand was
chosen -- it wires ``sub.set_defaults(_run=module.run)`` uniformly and lets
``args._run(args)`` dispatch. A subcommand with its own nested actions
(``reference build``) does that branching itself, inside its own ``run(args)``
-- see ``kinescore.cli.cmd_reference`` -- rather than main.py special-casing it.

Registration is by discovery, not by a hand-maintained list
(:func:`_discover_commands`): every ``cmd_*.py`` file on disk is imported and
checked for all four names -- a module missing one raises immediately, naming
the file. There is no second place to remember to update, which is what makes
"a command exists on disk but ``kinescore <name>`` doesn't work" (the
``kinescore export`` defect this replaced -- see git history of
``tests/test_cli_registration.py``, since deleted) structurally impossible
rather than merely tested-for.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

from kinescore import __version__

__all__ = ["build_parser", "main"]

_CLI_DIR = Path(__file__).resolve().parent

#: Every attribute a ``cmd_*.py`` module must define to be a valid subcommand.
_REQUIRED_ATTRS: tuple[str, ...] = ("NAME", "HELP", "add_arguments", "run")

#: Preferred ``--help`` display order: the order a benchmark run actually
#: goes through (acquire the data, prepare a rate-matched anchor, discover
#: clips, score them, aggregate, then read the result), followed by the
#: reader-training side and the two inspection utilities. Purely cosmetic --
#: a ``cmd_*.py`` module not listed here still registers automatically (see
#: :func:`_discover_commands`) and is appended, alphabetically, after these;
#: this tuple affects display order only, never whether a command exists.
_DISPLAY_ORDER: tuple[str, ...] = (
    "data", "anchor", "manifest", "bench", "score", "reference", "aggregate",
    "export", "report", "rank", "describe", "doctor", "cache", "train-rawrad",
    "calibrate",
)


def _discover_commands() -> tuple[ModuleType, ...]:
    """Import every ``cli/cmd_*.py`` module and validate its subcommand shape.

    Cheap -- see this package's module docstring for why importing every
    ``cmd_*`` module never pulls in torch or any other heavy/optional
    dependency, which is what keeps this (and therefore ``kinescore --help``)
    instant.
    """
    found: dict[str, ModuleType] = {}
    for path in sorted(_CLI_DIR.glob("cmd_*.py")):
        module = importlib.import_module(f"kinescore.cli.{path.stem}")
        missing = [a for a in _REQUIRED_ATTRS if not hasattr(module, a)]
        if missing:
            raise RuntimeError(
                f"kinescore.cli.{path.stem} is missing {missing} -- every "
                f"cli/cmd_*.py module must define NAME/HELP/add_arguments/run")
        name = module.NAME
        if name in found:
            raise RuntimeError(
                f"duplicate command name {name!r}: kinescore.cli.{path.stem} "
                f"and {found[name].__name__}")
        found[name] = module

    ordered_names = [n for n in _DISPLAY_ORDER if n in found]
    ordered_names += sorted(set(found) - set(_DISPLAY_ORDER))
    return tuple(found[n] for n in ordered_names)


def build_parser() -> argparse.ArgumentParser:
    """Construct the full ``kinescore`` argparse tree."""
    parser = argparse.ArgumentParser(
        prog="kinescore",
        description="Physics-plausibility benchmark for AI-generated robot video.")
    parser.add_argument("--version", action="version",
                        version=f"kinescore {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    for module in _discover_commands():
        sub = subparsers.add_parser(
            module.NAME, help=module.HELP, description=module.__doc__)
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
