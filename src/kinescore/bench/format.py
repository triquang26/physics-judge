"""One shared scalar-cell formatter for the plain-text/markdown/HTML renderers.

:mod:`kinescore.bench.report` and :mod:`kinescore.bench.rank` both render
tables of ``None``/``float``/other-scalar cells and used to each carry their
own near-identical ``_fmt`` (same ``None``->``"-"``, same NaN handling,
different default ``digits``). One copy here, ``digits`` still a parameter so
each caller keeps its own default precision.

Deliberately not the same contract as :func:`kinescore.bench.csv_export._fmt`
-- that one renders a missing value as ``""`` (a CSV cell), not ``"-"`` (a
human-readable table cell), because a spreadsheet consumer needs a genuinely
empty cell, not a string that means "missing" only by convention. Two
different callers wanting two different "missing" spellings is a real
divergence, not accidental drift, so it is kept as its own function rather
than folded in here.
"""
from __future__ import annotations

import math

__all__ = ["fmt_value"]


def fmt_value(x, digits: int = 6) -> str:
    """Render one cell: ``None`` -> ``"-"``, NaN -> ``"NaN"``, else a compact repr."""
    if x is None:
        return "-"
    if isinstance(x, float):
        if math.isnan(x):
            return "NaN"
        return f"{x:.{digits}g}"
    return str(x)
