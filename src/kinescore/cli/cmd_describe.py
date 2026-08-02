"""``kinescore describe``: print the active metric suite.

``--json`` output is what generates ``docs/METRICS.md`` from -- its
``"metrics"`` list is exactly
:meth:`kinescore.core.suite.MetricSuite.describe`'s return value, unmodified,
so a doc generator (or a test) can rely on the two never drifting apart.

``--reader`` (optional) surfaces what a *specific* checkpoint changes about
that suite's observability -- purely from the checkpoint's ``cfg`` dict, no
model construction, no backbone, so this stays cheap and network-free. The
one fact worth surfacing here (see ``legacy_docs/PROVENANCE.md`` D7): under a
``"raw_rad"`` reader (:class:`~kinescore.heads.heteroscedastic.ReadoutV2Head`,
the GR-1 production family) ``limit_violation_frac``/``limit_excess_rad``
become *observable* instead of the structural ``null`` a ``"squashed"``
reader always reports for them.
"""
from __future__ import annotations

import argparse
import json

NAME = "describe"
HELP = "print every metric in a suite: key, units, dt_exponent, direction, PIS membership"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from kinescore.bench.suites import available_suites

    parser.add_argument("--suite", default="invariant_v1",
                        help=f"suite name (available: {', '.join(available_suites())})")
    parser.add_argument("--reader", default=None,
                        help="optional checkpoint path; if given, also report "
                             "which reader family it routes to (see "
                             "kinescore.readers.checkpoint.load_reader) and "
                             "what that implies for limit_violation observability")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON instead of a table")


def _reader_info(path: str) -> dict:
    """Cheap cfg-only inspection of a checkpoint -- ``torch.load`` plus a dict
    lookup, no head/backbone construction. Mirrors the same auto-routing
    check ``kinescore.readers.checkpoint.load_reader`` uses
    (:func:`kinescore.readers.checkpoint_v2.is_readout_v2_cfg`), so the
    family this reports is exactly the family ``kinescore score --reader
    <path>`` would actually build.
    """
    import torch

    from kinescore.readers import checkpoint_v2

    ck = torch.load(path, map_location="cpu")
    cfg = dict(ck.get("cfg", {}))
    if checkpoint_v2.is_readout_v2_cfg(cfg):
        meta = dict(ck.get("meta", {}))
        return {
            "path": path,
            "head_family": "readout_v2 (heteroscedastic)",
            "limit_semantics": "raw_rad",
            "sigma_scale": meta.get("sigma_scale", 1.0),
            "note": ("limit_violation_frac/limit_excess_rad are OBSERVABLE "
                     "under this reader (q_raw is unsquashed, clamp overshoot "
                     "is the signal) -- see legacy_docs/PROVENANCE.md D7. A "
                     "sigma-gate (drop frames whose predicted aleatoric sigma "
                     "exceeds a threshold) is a caller-supplied "
                     "kinescore.core.scorer.Scorer(gate=...); `kinescore "
                     "score` does not enable one by default, so gate_coverage "
                     "is 1.0 for every clip unless you wire one up yourself."),
        }
    return {
        "path": path,
        "head_family": "attentive (legacy AttentivePoseHead cfg)",
        "limit_semantics": cfg.get("limit_semantics"),
        "note": ("this checkpoint's cfg is NOT a ReadoutV2Head cfg -- it is "
                 "the legacy AttentivePoseHead/PixelPhysicsJudge format "
                 "(judge_v3l/judge_v3l_mv/judge_reward, or anything written "
                 "by kinescore.readers.checkpoint.save). That format's only "
                 "reader (SquashedPoseReader) is no longer part of this "
                 "package -- see legacy_docs/PROVENANCE.md's D7 addendum -- so "
                 "`kinescore score --reader <this file>` now raises "
                 "NotImplementedError rather than building a reader whose "
                 "limit_violation_frac/limit_excess_rad were structurally "
                 "UNOBSERVABLE by construction anyway. Retrain via "
                 "`kinescore train-rawrad`."),
    }


def run(args: argparse.Namespace) -> int:
    from kinescore.bench.suites import get_suite

    suite = get_suite(args.suite)
    metrics = suite.describe()
    reader_info = _reader_info(args.reader) if getattr(args, "reader", None) else None

    if args.json:
        payload = {
            "suite_id": suite.suite_id, "suite_name": suite.name,
            "metrics": metrics,
        }
        if reader_info is not None:
            payload["reader"] = reader_info
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
    if reader_info is not None:
        print(f"\nreader: {reader_info['path']}")
        print(f"  head_family:     {reader_info['head_family']}")
        print(f"  limit_semantics: {reader_info['limit_semantics']}")
        print(f"  {reader_info['note']}")
    return 0
