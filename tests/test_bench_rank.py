"""Direct, argparse-free tests for :mod:`kinescore.bench.rank`.

``kinescore.cli.cmd_rank`` is a thin shell around ``badness``/
``unpaired_rows``/``paired_rows``/``render_rows`` here; the CLI-level
behaviour (argparse wiring, error messages, worst-first sorting end to end)
is already exercised through ``args._run`` in ``tests/test_rank.py``. This
file is the library-level coverage the move from ``cli/cmd_rank.py`` is
supposed to buy: every function reachable and testable on a plain
``pandas.DataFrame``/list of dicts, with no ``argparse.Namespace`` involved.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pandas")

import pandas as pd

from kinescore.bench.rank import badness, paired_rows, render_rows, unpaired_rows


class TestBadness:
    def test_lower_better_is_identity(self):
        assert badness(5.0, "lower_better") == 5.0
        assert badness(-2.0, "lower_better") == -2.0

    def test_higher_better_is_negated(self):
        assert badness(5.0, "higher_better") == -5.0


class TestUnpairedRows:
    def _df(self):
        return pd.DataFrame([
            {"method": "dense", "episode": "ep0", "role": "pred", "metrics.j": 1.0},
            {"method": "dense", "episode": "ep1", "role": "pred", "metrics.j": None},
            {"method": "dicache", "episode": "ep0", "role": "pred", "metrics.j": 9.0},
        ])

    def test_drops_missing_values(self):
        rows = unpaired_rows(self._df(), "metrics.j", None)
        assert len(rows) == 2
        assert all(r["value"] is not None for r in rows)

    def test_method_filter(self):
        rows = unpaired_rows(self._df(), "metrics.j", ["dicache"])
        assert len(rows) == 1
        assert rows[0]["method"] == "dicache"


class TestPairedRows:
    def test_computes_per_episode_delta(self):
        df = pd.DataFrame([
            {"method": "dense", "episode": "ep0", "role": "gt", "metrics.j": 10.0},
            {"method": "dense", "episode": "ep0", "role": "pred", "metrics.j": 15.0},
        ])
        rows = paired_rows(df, "metrics.j", None)
        assert len(rows) == 1
        assert rows[0]["delta"] == pytest.approx(5.0)


class TestRenderRows:
    ROWS = [{"rank": 1, "method": "dense", "value": 9.0},
            {"rank": 2, "method": "dicache", "value": 3.0}]
    COLUMNS = ["rank", "method", "value"]

    def test_json_round_trips(self):
        import json

        text = render_rows(self.ROWS, "json", self.COLUMNS)
        assert json.loads(text) == self.ROWS

    def test_csv_has_a_header_and_every_row(self):
        import csv
        import io

        text = render_rows(self.ROWS, "csv", self.COLUMNS)
        rows = list(csv.DictReader(io.StringIO(text)))
        assert len(rows) == 2
        assert rows[0]["method"] == "dense"

    def test_table_columns_are_aligned_and_include_header(self):
        text = render_rows(self.ROWS, "table", self.COLUMNS)
        lines = text.splitlines()
        assert lines[0].split()[:3] == ["rank", "method", "value"]
        assert len(lines) == 4  # header + separator + 2 rows

    def test_empty_rows_still_render_a_header(self):
        text = render_rows([], "table", self.COLUMNS)
        assert "rank" in text.splitlines()[0]
