"""Filesystem locations, resolved from environment variables.

Deliberately unlike the repo this pattern is borrowed from: **there are no
hardcoded fallback paths**. A prior extraction baked another user's home
directory into its defaults, so a fresh checkout resolved to directories that
did not exist and failed with confusing errors far from the cause.

Here an unset variable raises immediately and names the variable to set. The
only default is the current working directory, and only for outputs.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

__all__ = ["ENV_VARS", "env_path", "optional_env_path", "output_dir",
           "MissingPathError"]

#: Environment variable -> what it points at.
ENV_VARS = {
    "KINESCORE_ASSETS": "URDF and mesh trees (never vendored; GR-1 alone is ~285 MB)",
    "KINESCORE_CKPT_DIR": "trained pose-reader checkpoints",
    "KINESCORE_CACHE_DIR": "precomputed backbone feature caches",
    "KINESCORE_DATA_ROOT": "video clips to score",
    "KINESCORE_OUTPUT_DIR": "scoring outputs (defaults to ./out)",
}


class MissingPathError(RuntimeError):
    """Raised when a required location has not been configured."""


def env_path(key: str) -> Path:
    """Resolve a required path from the environment.

    Raises
    ------
    MissingPathError
        If unset, with the variable name and its purpose -- so the fix is
        obvious from the message alone.
    """
    if key not in ENV_VARS:
        raise KeyError(f"unknown path key {key!r}; known: {sorted(ENV_VARS)}")
    raw = os.environ.get(key)
    if not raw:
        raise MissingPathError(
            f"{key} is not set. It should point at: {ENV_VARS[key]}.\n"
            f"Set it in your shell or in .env (see .env.example).")
    return Path(raw).expanduser().resolve()


def optional_env_path(key: str) -> Optional[Path]:
    """Like :func:`env_path` but returns ``None`` instead of raising."""
    try:
        return env_path(key)
    except MissingPathError:
        return None


def output_dir() -> Path:
    """Where results are written. The one location with a sane default."""
    raw = os.environ.get("KINESCORE_OUTPUT_DIR", "out")
    p = Path(raw).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p
