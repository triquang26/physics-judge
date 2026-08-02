"""Expand ``${VAR_NAME}`` inside a parsed config's string values, fail-loud.

Used by :mod:`kinescore.bench.config` to resolve every ``${...}`` in a
freshly-parsed ``benchmark.yaml`` mapping before validation runs (see that
module's docstring's "Environment variables" section for the contract this
implements). Split into its own module because it is a self-contained
concern -- a generic dict/list/str walker with no knowledge of
``BenchConfig``'s five dataclasses -- not because it is reused anywhere else
yet.

``KINESCORE_*`` names (:data:`kinescore.paths.ENV_VARS`) are expanded via
:func:`kinescore.paths.env_path`, so an unset one raises the existing
:class:`~kinescore.paths.MissingPathError` -- naming the variable and what it
should point at -- rather than the config silently keeping the literal
``"${KINESCORE_DATA_ROOT}"`` string or resolving to an empty path. Any other
``${VAR}`` is resolved from ``os.environ`` with the same fail-loud behaviour.
"""
from __future__ import annotations

import os
import re
from typing import Any

from kinescore import paths

__all__ = ["expand_env"]

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_str(s: str) -> str:
    def _sub(m: re.Match) -> str:
        name = m.group(1)
        if name in paths.ENV_VARS:
            # Raises paths.MissingPathError, naming the variable, if unset --
            # never silently resolves to "" or the literal "${...}" text.
            return str(paths.env_path(name))
        val = os.environ.get(name)
        if val is None:
            raise paths.MissingPathError(
                f"{name} is not set (referenced as ${{{name}}} in the "
                f"benchmark config). Set it in your shell or in .env.")
        return val
    return _ENV_VAR_RE.sub(_sub, s)


def expand_env(value: Any) -> Any:
    """Recursively expand every ``${VAR}`` in ``value``'s string leaves.

    Walks straight through ``dict``/``list`` nesting; anything else (an
    ``int``, ``bool``, ``None``, ...) is returned unchanged. Always returns a
    new structure rather than mutating ``value`` in place, so a caller
    reusing the same raw dict after calling this is unaffected.
    """
    if isinstance(value, str):
        return _expand_str(value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value
