"""``kinescore`` command-line interface: argparse subparsers, one per file.

Design rule this whole package follows (see ``kinescore.cli.main``): every
``cmd_*.py`` module's top level imports only the standard library plus other
``kinescore.cli`` helpers -- never ``torch``, ``pandas``, ``transformers`` or
anything in :mod:`kinescore.core`/``.robots``/``.readers``. Those go inside
each subcommand's ``run(args)`` function. That is what makes ``kinescore
--help`` and ``kinescore doctor`` instant and importable on an interpreter
that has none of the heavy optional dependencies installed -- the CLI package
itself never pays for a dependency a given invocation doesn't use.
"""
from __future__ import annotations
