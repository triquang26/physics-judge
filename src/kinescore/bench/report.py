"""``stats.json`` -> a self-contained HTML or Markdown summary.

Used by ``kinescore report`` (:mod:`kinescore.cli.cmd_report`, a thin
argparse shell around :func:`render_markdown`/:func:`render_html` below) and
directly importable/unit-testable without argparse.

Deliberately template-free (an f-string builds the HTML): pulling in Jinja2
for one table would be a new dependency for a document that is a handful of
rows, and "self-contained" is a hard requirement here -- the HTML output must
open correctly as a single file with no external CSS/JS, which an f-string
trivially satisfies and a templating engine adds no value for.

Two more tables, matching the project page and not a composite score
----------------------------------------------------------------------
Beyond the existing paired-tax table, this renders the two sections
``kinescore aggregate`` (``cli/cmd_aggregate.py``) now also writes into
``stats.json``, both produced by :mod:`kinescore.bench.separation`:

* **Separation** -- one row per (method, metric): magnitude (real/generated
  median, in physical units) *and* separation (AUROC, oriented so 1.0 always
  means "generated is worse") side by side, exactly the page's framing that
  these answer different questions and neither replaces the other. Rows
  scoring ~0.50 are **not** filtered out -- the page's own point is that a
  ruler correctly reporting "can't tell them apart" on an axis it has no
  business detecting is itself evidence the ruler is a specific diagnostic,
  not a blanket alarm. A metric whose ``dt_exponent`` is ``None`` (read from
  the metric registry via the row's own ``rate_comparable`` field -- never a
  hardcoded metric-name list, so a metric added or reclassified elsewhere is
  picked up automatically) is marked not comparable across frame rates.
* **Cache ranking** -- mean rank across physical axes vs. a baseline (Exp8),
  from :func:`kinescore.bench.separation.rank_caches`.
"""
from __future__ import annotations

import html
import json
import os

from kinescore.bench.format import fmt_value

__all__ = [
    "load_stats", "rows", "separation_rows", "cache_rank_rows", "rate_flag",
    "render_markdown", "render_html",
]


def load_stats(path: str) -> dict:
    if os.path.isdir(path):
        path = os.path.join(path, "stats.json")
    with open(path) as f:
        return json.load(f)


def rows(stats: dict) -> list[dict]:
    return stats.get("results", [])


def separation_rows(stats: dict) -> list[dict]:
    return stats.get("separation", [])


def cache_rank_rows(stats: dict) -> list[dict]:
    return stats.get("cache_ranking", [])


def rate_flag(row: dict) -> str:
    """Human-readable cross-frame-rate comparability marker for one row.

    Reads ``row["rate_comparable"]`` (``dt_exponent is not None``, stamped by
    ``bench/separation.py`` from the metric registry) rather than testing
    ``row["metric"]`` against a hardcoded list -- a metric later reclassified
    (or a new one added) as ``dt_exponent=None`` shows up here automatically.
    """
    if row.get("rate_comparable"):
        return ""
    return "not comparable across frame rates"


def render_markdown(stats: dict) -> str:
    prov = stats.get("provenance", {})
    lines = [
        "# kinescore report", "",
        f"kinescore {prov.get('kinescore_version', '?')} "
        f"(git {prov.get('git_sha') or 'unknown'})",
        "",
        "| method | metric | n | median delta | CI low | CI high | p | warning |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows(stats):
        ci = r.get("ci", {})
        w = r.get("wilcoxon", {})
        lines.append(
            f"| {r.get('method')} | {r.get('metric')} | {r.get('n')} | "
            f"{fmt_value(r.get('median'), 4)} | {fmt_value(ci.get('lo'), 4)} | "
            f"{fmt_value(ci.get('hi'), 4)} | {fmt_value(w.get('p'), 4)} | "
            f"{r.get('warning') or ''} |")

    sep_rows = separation_rows(stats)
    lines += [
        "", "## Separation (magnitude + AUROC per method/ruler)", "",
        "No composite score exists -- see docs/METRICS.md. Rows scoring "
        "~0.50 separation are kept deliberately: a ruler that honestly "
        "can't tell an axis apart is a diagnostic finding, not noise to "
        "filter.", "",
        "| method | metric | units | n | real median | gen median | "
        "delta median | CI low | CI high | p | p (Holm) | cliff's delta | "
        "frac worse | separation | verdict | noise floor | above noise | "
        "cross-rate | reason |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sep_rows:
        lines.append(
            f"| {r.get('method')} | {r.get('metric')} | {r.get('units')} | "
            f"{r.get('n')} | {fmt_value(r.get('real_median'), 4)} | "
            f"{fmt_value(r.get('gen_median'), 4)} | {fmt_value(r.get('delta_median'), 4)} | "
            f"{fmt_value(r.get('ci_lo'), 4)} | {fmt_value(r.get('ci_hi'), 4)} | "
            f"{fmt_value(r.get('p'), 4)} | {fmt_value(r.get('p_holm'), 4)} | "
            f"{fmt_value(r.get('cliffs_delta'), 4)} | {fmt_value(r.get('frac_worse'), 4)} | "
            f"{fmt_value(r.get('separation'), 4)} | {r.get('verdict') or '-'} | "
            f"{fmt_value(r.get('noise_floor'), 4)} | {r.get('above_noise')} | "
            f"{rate_flag(r)} | {r.get('reason') or ''} |")

    rank_rows = cache_rank_rows(stats)
    lines += [
        "", "## Cache ranking (mean rank across physical axes)", "",
        "| method | mean rank | n axes | per-axis rank | per-axis extra "
        "cost vs baseline |",
        "|---|---|---|---|---|",
    ]
    for r in rank_rows:
        axis_ranks = ", ".join(f"{k}={v:g}" for k, v in
                               (r.get("axis_ranks") or {}).items())
        extra_cost = ", ".join(f"{k}={v:+.4g}" for k, v in
                               (r.get("axis_extra_cost") or {}).items())
        lines.append(
            f"| {r.get('method')} | {fmt_value(r.get('mean_rank'), 4)} | "
            f"{r.get('n_axes')} | {axis_ranks or '-'} | {extra_cost or '-'} |")

    return "\n".join(lines) + "\n"


def render_html(stats: dict) -> str:
    prov = stats.get("provenance", {})
    body_rows = []
    for r in rows(stats):
        ci = r.get("ci", {})
        w = r.get("wilcoxon", {})
        warn = html.escape(r.get("warning") or "")
        warn_cell = f'<td class="warn">{warn}</td>' if warn else "<td></td>"
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(str(r.get('method')))}</td>"
            f"<td>{html.escape(str(r.get('metric')))}</td>"
            f"<td>{r.get('n')}</td>"
            f"<td>{fmt_value(r.get('median'), 4)}</td>"
            f"<td>{fmt_value(ci.get('lo'), 4)}</td>"
            f"<td>{fmt_value(ci.get('hi'), 4)}</td>"
            f"<td>{fmt_value(w.get('p'), 4)}</td>"
            f"{warn_cell}"
            "</tr>")
    rows_html = "\n".join(body_rows) if body_rows else (
        '<tr><td colspan="8">no summaries</td></tr>')

    sep_body = []
    for r in separation_rows(stats):
        rate_flag_ = html.escape(rate_flag(r))
        reason = html.escape(r.get("reason") or "")
        reason_cell = f'<td class="warn">{reason}</td>' if reason else "<td></td>"
        sep_body.append(
            "<tr>"
            f"<td>{html.escape(str(r.get('method')))}</td>"
            f"<td>{html.escape(str(r.get('metric')))}</td>"
            f"<td>{html.escape(str(r.get('units')))}</td>"
            f"<td>{r.get('n')}</td>"
            f"<td>{fmt_value(r.get('real_median'), 4)}</td>"
            f"<td>{fmt_value(r.get('gen_median'), 4)}</td>"
            f"<td>{fmt_value(r.get('delta_median'), 4)}</td>"
            f"<td>{fmt_value(r.get('ci_lo'), 4)}</td>"
            f"<td>{fmt_value(r.get('ci_hi'), 4)}</td>"
            f"<td>{fmt_value(r.get('p'), 4)}</td>"
            f"<td>{fmt_value(r.get('p_holm'), 4)}</td>"
            f"<td>{fmt_value(r.get('cliffs_delta'), 4)}</td>"
            f"<td>{fmt_value(r.get('frac_worse'), 4)}</td>"
            f"<td>{fmt_value(r.get('separation'), 4)}</td>"
            f"<td>{html.escape(str(r.get('verdict') or '-'))}</td>"
            f"<td>{fmt_value(r.get('noise_floor'), 4)}</td>"
            f"<td>{r.get('above_noise')}</td>"
            f"<td class=\"warn\">{rate_flag_}</td>"
            f"{reason_cell}"
            "</tr>")
    sep_rows_html = "\n".join(sep_body) if sep_body else (
        '<tr><td colspan="18">no separation rows</td></tr>')

    rank_body = []
    for r in cache_rank_rows(stats):
        axis_ranks = ", ".join(f"{k}={v:g}" for k, v in
                               (r.get("axis_ranks") or {}).items())
        extra_cost = ", ".join(f"{k}={v:+.4g}" for k, v in
                               (r.get("axis_extra_cost") or {}).items())
        rank_body.append(
            "<tr>"
            f"<td>{html.escape(str(r.get('method')))}</td>"
            f"<td>{fmt_value(r.get('mean_rank'), 4)}</td>"
            f"<td>{r.get('n_axes')}</td>"
            f"<td>{html.escape(axis_ranks or '-')}</td>"
            f"<td>{html.escape(extra_cost or '-')}</td>"
            "</tr>")
    rank_rows_html = "\n".join(rank_body) if rank_body else (
        '<tr><td colspan="5">no cache ranking rows</td></tr>')

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>kinescore report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: right; }}
th {{ background: #f0f0f0; text-align: center; }}
td:first-child, td:nth-child(2) {{ text-align: left; }}
td.warn {{ text-align: left; color: #a33; font-size: 0.85em; }}
caption {{ text-align: left; margin-bottom: 0.5rem; color: #555; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>kinescore report</h1>
<p>kinescore {html.escape(str(prov.get('kinescore_version', '?')))}
 (git {html.escape(str(prov.get('git_sha') or 'unknown'))})</p>
<table>
<caption>Paired physics-tax summary per method/metric (median pred&#8722;gt delta,
bootstrap CI, Wilcoxon p). See legacy_docs/SCHEMA.md for column provenance.</caption>
<thead><tr>
<th>method</th><th>metric</th><th>n</th><th>median delta</th>
<th>CI low</th><th>CI high</th><th>p</th><th>warning</th>
</tr></thead>
<tbody>
{rows_html}
</tbody>
</table>
<h2>Separation</h2>
<table>
<caption>Magnitude (physical units) and separation (AUROC, oriented so 1.0 =
generated is worse) side by side -- there is no composite score. Rows scoring
~0.50 are kept: a ruler honestly reporting "can't tell them apart" on an axis
it has no business detecting is a finding, not noise. A row marked
"not comparable across frame rates" has <code>dt_exponent=None</code> in the
metric registry (docs/METRICS.md) -- do not compare it across clips scored at
different fps.</caption>
<thead><tr>
<th>method</th><th>metric</th><th>units</th><th>n</th>
<th>real median</th><th>gen median</th><th>delta median</th>
<th>CI low</th><th>CI high</th><th>p</th><th>p (Holm)</th>
<th>cliff's delta</th><th>frac worse</th><th>separation</th>
<th>verdict</th><th>noise floor</th><th>above noise</th>
<th>cross-rate</th><th>reason</th>
</tr></thead>
<tbody>
{sep_rows_html}
</tbody>
</table>
<h2>Cache ranking</h2>
<table>
<caption>Mean rank across physical axes (jerk / balance margin / joint
limits, or whichever metrics were aggregated), rank 1 = best. Extra cost is
a point estimate of oriented tax relative to the baseline method -- see
kinescore.bench.separation.extra_cost_vs_baseline for the CI'd version of
any one axis.</caption>
<thead><tr>
<th>method</th><th>mean rank</th><th>n axes</th><th>per-axis rank</th>
<th>per-axis extra cost vs baseline</th>
</tr></thead>
<tbody>
{rank_rows_html}
</tbody>
</table>
</body>
</html>
"""
