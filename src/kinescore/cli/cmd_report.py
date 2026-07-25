"""``kinescore report``: ``stats.json`` -> a self-contained HTML or Markdown summary.

Deliberately template-free (an f-string builds the HTML): pulling in Jinja2
for one table would be a new dependency for a document that is a handful of
rows, and "self-contained" is a hard requirement here -- the HTML output must
open correctly as a single file with no external CSS/JS, which an f-string
trivially satisfies and a templating engine adds no value for.
"""
from __future__ import annotations

import argparse
import html
import json
import os

HELP = "render stats.json as a self-contained HTML or Markdown summary"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("stats", help="stats.json from `kinescore aggregate`, "
                                      "or the directory containing one")
    parser.add_argument("--format", choices=("html", "markdown"), default="html")
    parser.add_argument("--out", default=None,
                        help="output path (default: alongside stats.json as "
                             "report.html / report.md)")


def _load_stats(path: str) -> dict:
    if os.path.isdir(path):
        path = os.path.join(path, "stats.json")
    with open(path) as f:
        return json.load(f)


def _fmt(x, digits: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        if x != x:  # NaN
            return "NaN"
        return f"{x:.{digits}g}"
    return str(x)


def _rows(stats: dict) -> list[dict]:
    return stats.get("results", [])


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
    for r in _rows(stats):
        ci = r.get("ci", {})
        w = r.get("wilcoxon", {})
        lines.append(
            f"| {r.get('method')} | {r.get('metric')} | {r.get('n')} | "
            f"{_fmt(r.get('median'))} | {_fmt(ci.get('lo'))} | "
            f"{_fmt(ci.get('hi'))} | {_fmt(w.get('p'))} | "
            f"{r.get('warning') or ''} |")
    return "\n".join(lines) + "\n"


def render_html(stats: dict) -> str:
    prov = stats.get("provenance", {})
    body_rows = []
    for r in _rows(stats):
        ci = r.get("ci", {})
        w = r.get("wilcoxon", {})
        warn = html.escape(r.get("warning") or "")
        warn_cell = f'<td class="warn">{warn}</td>' if warn else "<td></td>"
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(str(r.get('method')))}</td>"
            f"<td>{html.escape(str(r.get('metric')))}</td>"
            f"<td>{r.get('n')}</td>"
            f"<td>{_fmt(r.get('median'))}</td>"
            f"<td>{_fmt(ci.get('lo'))}</td>"
            f"<td>{_fmt(ci.get('hi'))}</td>"
            f"<td>{_fmt(w.get('p'))}</td>"
            f"{warn_cell}"
            "</tr>")
    rows_html = "\n".join(body_rows) if body_rows else (
        '<tr><td colspan="8">no summaries</td></tr>')
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>kinescore report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }}
table {{ border-collapse: collapse; width: 100%; }}
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
bootstrap CI, Wilcoxon p). See docs/SCHEMA.md for column provenance.</caption>
<thead><tr>
<th>method</th><th>metric</th><th>n</th><th>median delta</th>
<th>CI low</th><th>CI high</th><th>p</th><th>warning</th>
</tr></thead>
<tbody>
{rows_html}
</tbody>
</table>
</body>
</html>
"""


def run(args: argparse.Namespace) -> int:
    stats = _load_stats(args.stats)
    stats_dir = args.stats if os.path.isdir(args.stats) else os.path.dirname(args.stats) or "."

    if args.format == "markdown":
        text = render_markdown(stats)
        default_name = "report.md"
    else:
        text = render_html(stats)
        default_name = "report.html"

    out_path = args.out or os.path.join(stats_dir, default_name)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
    print(f"[report] wrote {out_path}")
    return 0
