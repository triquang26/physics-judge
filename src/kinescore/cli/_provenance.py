"""Shared provenance stamp for every CLI command that writes output.

Every command in this package that produces a file (``manifest``, ``score``,
``reference build``, ``aggregate``) writes a small ``provenance.json`` next to
it -- kinescore version, git sha, and whatever run-specific identity applies
(``suite_id``, ``robot``, ``reader_id``, resolved ``dt``/``dt_source``). This
is deliberately *not* folded into the primary output file: the primary output
(``results.jsonl``, a manifest, ``reference.pt``) has its own schema owned by
``kinescore.bench``/``kinescore.reference`` and this package must not invent a
second shape for it (see ``kinescore.bench.store``'s docstring for why "two
incompatible shapes for the same thing" is a defect class of its own). A
sidecar file is the boundary: the CLI's bookkeeping stays the CLI's.

Not a subcommand itself (no leading ``cmd_``), so ``kinescore.cli.main`` never
registers it as one.
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from kinescore import __version__

__all__ = ["git_sha", "provenance_block", "print_provenance", "write_json"]


def git_sha(cwd: str | None = None) -> str | None:
    """Short git commit hash of the checkout ``cwd`` (default: cwd) sits in.

    Best-effort: returns ``None`` (never raises) when git is not installed,
    the directory is not a repo, or the call times out -- a missing git sha
    should degrade a provenance record, not crash the command that requested
    it.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
            capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    sha = out.stdout.strip()
    return sha or None


def provenance_block(**extra: Any) -> dict[str, Any]:
    """``{"kinescore_version", "git_sha", **extra}`` -- the common envelope.

    ``extra`` is whatever the calling command additionally knows makes a
    result reproducible-or-diagnosable: ``suite_id``, ``robot``,
    ``reader_id``, the resolved ``dt``/``dt_source`` for the clips it just
    processed, and so on. Nothing here validates ``extra``'s keys -- every
    command's provenance is shaped differently, and forcing one schema on all
    of them would be the exact "one shape must fit every writer" mistake
    ``kinescore.bench.store`` documents fixing on the scoring side.
    """
    return {"kinescore_version": __version__, "git_sha": git_sha(), **extra}


def print_provenance(block: dict[str, Any]) -> None:
    """Print a provenance block as one JSON line, prefixed for grep-ability."""
    print("[provenance] " + json.dumps(block, sort_keys=True, default=str))


def write_json(path: str, obj: Any) -> None:
    """Write ``obj`` as pretty JSON to ``path``, creating parent directories.

    ``default=str`` so an accidental ``Path``/``numpy`` scalar that slipped
    into a provenance dict is coerced to a string instead of raising deep
    inside ``json.dump`` -- provenance is diagnostic output, and a crash
    while *writing the diagnostic* is worse than a slightly-ugly string.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
