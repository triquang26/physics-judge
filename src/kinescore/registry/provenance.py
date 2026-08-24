"""``run_manifest.json``: what produced an artifact.

Written by every command, next to whatever it wrote: the config files the run
read, the checkpoint it read or wrote, and the revision it ran at.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from kinescore import __version__

__all__ = ["sha256_file", "git_state", "run_manifest", "write_run_manifest"]

#: Bytes hashed per read.
_CHUNK = 1 << 20


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hex digest of a file, or ``""`` if it does not exist."""
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[3]), *args],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_state() -> dict[str, Any]:
    """The working tree's revision, read now."""
    return {"git_sha": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain"))}


def run_manifest(command: str, *, started_at: str, sources: tuple[str, ...] = (),
                 git: dict[str, Any] | None = None,
                 extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the record for one command invocation.

    ``git`` is the revision captured when the run started. Read at write time
    instead, a run that outlives a commit records the commit it did not run.
    """
    return {
        "command": command,
        "argv": list(sys.argv),
        "started_at": started_at,
        "kinescore_version": __version__,
        **(git if git is not None else git_state()),
        "host": platform.node(),
        "config_sha256": {os.path.basename(s): sha256_file(s) for s in sources},
        **(extra or {}),
    }


def write_run_manifest(directory: str | os.PathLike[str], manifest: dict[str, Any]
                       ) -> Path:
    """Write ``run_manifest.json`` into ``directory``, creating it if needed."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path
