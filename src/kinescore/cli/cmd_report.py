"""``kinescore report``: ``stats.json`` -> a self-contained HTML or Markdown summary.

The rendering logic (:func:`~kinescore.bench.report.render_markdown`,
:func:`~kinescore.bench.report.render_html`, and the ``stats.json`` section
accessors they use) lives in :mod:`kinescore.bench.report` -- see that
module's docstring for the two extra sections (separation, cache ranking)
and why rows scoring ~0.50 separation are kept rather than filtered. This
module is the argparse shell: parse ``--format``/``--out``, load
``stats.json``, call the right renderer, write the result.
"""
from __future__ import annotations

import argparse
import os

NAME = "report"
HELP = "render stats.json as a self-contained HTML or Markdown summary"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("stats", help="stats.json from `kinescore aggregate`, "
                                      "or the directory containing one")
    parser.add_argument("--format", choices=("html", "markdown"), default="html")
    parser.add_argument("--out", default=None,
                        help="output path (default: alongside stats.json as "
                             "report.html / report.md)")


def run(args: argparse.Namespace) -> int:
    from kinescore.bench.report import load_stats, render_html, render_markdown

    stats = load_stats(args.stats)
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
