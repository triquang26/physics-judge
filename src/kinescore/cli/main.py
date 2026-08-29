"""``kinescore``: the command line entry point."""
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

#: What a ``cmd_*.py`` module must define.
_REQUIRED_ATTRS: tuple[str, ...] = ("NAME", "HELP", "add_arguments", "run")

#: ``--help`` order: the order a benchmark run goes through. A command not
#: listed still registers, and is appended alphabetically.
_DISPLAY_ORDER: tuple[str, ...] = (
    "pull", "data", "cache", "train", "score", "report", "render", "export", "push",
    "readers", "models", "ledger")


def _discover_commands() -> tuple[ModuleType, ...]:
    """Import every ``cli/cmd_*.py`` and validate its shape."""
    found: dict[str, ModuleType] = {}
    for path in sorted(_CLI_DIR.glob("cmd_*.py")):
        module = importlib.import_module(f"kinescore.cli.{path.stem}")
        missing = [a for a in _REQUIRED_ATTRS if not hasattr(module, a)]
        if missing:
            raise RuntimeError(
                f"kinescore.cli.{path.stem} is missing {missing}; every "
                f"cli/cmd_*.py must define NAME/HELP/add_arguments/run")
        if module.NAME in found:
            raise RuntimeError(
                f"duplicate command {module.NAME!r}: kinescore.cli.{path.stem} "
                f"and {found[module.NAME].__name__}")
        found[module.NAME] = module

    names = [n for n in _DISPLAY_ORDER if n in found]
    names += sorted(set(found) - set(_DISPLAY_ORDER))
    return tuple(found[n] for n in names)


def build_parser() -> argparse.ArgumentParser:
    """Construct the full ``kinescore`` argparse tree."""
    parser = argparse.ArgumentParser(
        prog="kinescore",
        description="Physics-plausibility benchmark for generated robot video.")
    parser.add_argument("--version", action="version",
                        version=f"kinescore {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    for module in _discover_commands():
        sub = subparsers.add_parser(module.NAME, help=module.HELP,
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
