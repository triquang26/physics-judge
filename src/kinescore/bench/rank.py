"""``kinescore rank``'s engine: sort clips by ONE physical-unit ruler -- never a score.

Used by ``kinescore.cli.cmd_rank`` (the argparse shell); every function here
takes a joined :class:`pandas.DataFrame` (from
:func:`kinescore.bench.stats.load_scores`) or a plain list of rows, so it is
reachable and unit-testable from Python without argparse.

There is no composite score in this benchmark. The project page is explicit
about why: magnitude (physical units) and separation (AUROC) answer
different questions and are deliberately kept apart rather than squashed
into one 0-1 number (see ``bench/separation.py``'s module docstring for the
quote). "Rank videos by physics" therefore can only ever mean "sort by one
named ruler" -- ``kinescore rank`` makes ``--metric`` required and refuses to
invent a default, so a caller cannot accidentally sort by something that
does not exist (the PIS aggregate is legacy, frozen in
``kinescore.reference``, and is never reachable from here).

Two sort modes
---------------
* Default -- every individual scored clip (any role), sorted by its raw
  physical-unit value.
* ``--paired`` -- episodes, sorted by their own paired delta = phi(pred) -
  phi(gt), the same per-episode quantity
  ``kinescore.bench.stats.paired_deltas`` computes for the aggregate report.
  Answers "which generated clips took the biggest physics tax against their
  own ground truth", independent of how demanding the underlying episode is.

Both modes put the most physically-implausible clip/episode **first**,
oriented by the metric's declared ``direction`` (``docs/METRICS.md``) so
"worst" means the same thing whether the ruler is lower-better or
higher-better -- reading that off the registry, never a hardcoded metric-name
check, is what keeps this correct for every metric without special-casing.
"""
from __future__ import annotations

import csv
import io
import json

from kinescore.bench.format import fmt_value

__all__ = ["badness", "unpaired_rows", "paired_rows", "render_rows"]


def badness(value: float, direction: str) -> float:
    """Orient a raw value/delta so *higher = worse*, for a single sort key."""
    return value if direction == "lower_better" else -value


def unpaired_rows(df, col: str, methods) -> list[dict]:
    sub = df if methods is None else df[df["method"].astype(str).isin(methods)]
    sub = sub[sub[col].notna()]
    rows = []
    for _, r in sub.iterrows():
        rows.append({"method": str(r["method"]), "episode": str(r["episode"]),
                    "role": str(r["role"]), "value": float(r[col])})
    return rows


def paired_rows(df, col: str, methods) -> list[dict]:
    from kinescore.bench.stats import paired_deltas

    all_methods = methods or sorted(df["method"].dropna().astype(str).unique().tolist())
    rows = []
    for method in all_methods:
        ep, delta = paired_deltas(df, method, col)
        for e, d in zip(ep, delta, strict=True):
            rows.append({"method": method, "episode": str(e), "delta": float(d)})
    return rows


def render_rows(rows: list[dict], fmt: str, columns: list[str]) -> str:
    if fmt == "json":
        return json.dumps(rows, indent=2)
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in columns})
        return buf.getvalue()
    # table
    widths = {c: max(len(c), *(len(fmt_value(r.get(c))) for r in rows)) if rows else len(c)
             for c in columns}
    lines = ["  ".join(c.ljust(widths[c]) for c in columns),
             "  ".join("-" * widths[c] for c in columns)]
    for row in rows:
        lines.append("  ".join(fmt_value(row.get(c)).ljust(widths[c]) for c in columns))
    return "\n".join(lines) + "\n"
