"""The command line: one module per subcommand, discovered by filename.

A ``cmd_*.py`` module declares ``NAME``, ``HELP``, ``add_arguments(parser)``
and ``run(args) -> int``. ``add_arguments`` runs on every invocation including
``--help``, so it stays import-light; every heavy import happens inside
``run``, after argparse has accepted the command line.
"""
