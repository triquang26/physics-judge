"""``kinescore describe``: print the active metric suite.

``--json`` output is what generates ``docs/METRICS.md`` from -- its
``"metrics"`` list is exactly
:meth:`kinescore.core.suite.MetricSuite.describe`'s return value, unmodified,
so a doc generator (or a test) can rely on the two never drifting apart.
"""
from __future__ import annotations

import argparse
import json

HELP = "print every metric in a suite: key, units, dt_exponent, direction, PIS membership"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from kinescore.cli._suites import available_suites

    parser.add_argument("--suite", default="invariant_v1",
                        help=f"suite name (available: {', '.join(available_suites())})")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON instead of a table")


def run(args: argparse.Namespace) -> int:
    from kinescore.cli._suites import get_suite

    suite = get_suite(args.suite)
    metrics = suite.describe()

    if args.json:
        payload = {
            "suite_id": suite.suite_id, "suite_name": suite.name,
            "metrics": metrics,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"suite: {suite.name}  ({suite.suite_id})")
    print(f"{len(metrics)} metric(s), {len(suite.invariant_keys)} in PIS\n")
    header = f"{'key':<28} {'units':<10} {'dt^n':>5} {'direction':<14} {'pis':<4} requires"
    print(header)
    print("-" * len(header))
    for m in metrics:
        dt_exp = "-" if m["dt_exponent"] is None else str(m["dt_exponent"])
        pis = "yes" if m["in_pis"] else ""
        requires = ",".join(m["requires"]) or "-"
        print(f"{m['key']:<28} {m['units']:<10} {dt_exp:>5} "
             f"{m['direction']:<14} {pis:<4} {requires}")
    return 0
