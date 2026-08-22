"""Flags and lookups more than one subcommand needs."""
from __future__ import annotations

import argparse
import datetime as _dt

from kinescore.registry.cells import CellSpec, ReaderSpec, Registry, load_registry

__all__ = ["add_config_arguments", "load", "resolve_reader", "resolve_cell",
           "now"]


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """``--views``/``--robots``/``--cells``: point at other definition files."""
    parser.add_argument("--views", default=None, help="path to views.yaml")
    parser.add_argument("--robots", default=None, help="path to robots.yaml")
    parser.add_argument("--cells", default=None, help="path to cells.yaml")


def load(args: argparse.Namespace) -> Registry:
    """Read the definition files this invocation points at."""
    from kinescore.registry.cells import (
        DEFAULT_CELLS_PATH,
        DEFAULT_ROBOTS_PATH,
    )
    from kinescore.registry.views import DEFAULT_VIEWS_PATH

    return load_registry(
        cells_path=args.cells or DEFAULT_CELLS_PATH,
        robots_path=args.robots or DEFAULT_ROBOTS_PATH,
        views_path=args.views or DEFAULT_VIEWS_PATH,
    )


def resolve_reader(registry: Registry, reader_id: str) -> ReaderSpec:
    """Look up a reader, or exit with the list of declared ids."""
    try:
        return registry.reader(reader_id)
    except KeyError as exc:
        raise SystemExit(str(exc)) from None


def resolve_cell(registry: Registry, cell_id: str) -> CellSpec:
    """Look up a cell, or exit with the list of declared ids."""
    try:
        return registry.cell(cell_id)
    except KeyError as exc:
        raise SystemExit(str(exc)) from None


def now() -> str:
    """UTC timestamp for a run manifest."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
