"""``kinescore ledger`` reports the artifacts that exist, and nothing else."""
from __future__ import annotations

import json

import pytest

from kinescore.cli import cmd_ledger

pytestmark = pytest.mark.unit


class _Reader:
    def __init__(self, tmp_path, reader_id="robot.corpus.view", status=""):
        self.reader_id = reader_id
        self.status = status
        self.train_tree = tmp_path / "tree"
        self.cache_dir = tmp_path / "cache"
        self.checkpoint_path = tmp_path / f"{reader_id}.pt"


class _Cell:
    def __init__(self, tmp_path, reader, cell_id="emb.view.model"):
        self.cell_id = cell_id
        self.reader = reader
        self.output_dir = tmp_path / "out" / cell_id


class TestTicks:
    def test_missing_artifacts_read_as_absent(self, tmp_path):
        reader = _Reader(tmp_path)
        rows = cmd_ledger._reader_rows(
            type("R", (), {"readers": {reader.reader_id: reader}})())
        assert rows[0]["tree"] is False
        assert rows[0]["cached"] == 0
        assert "val_mm" not in rows[0]

    def test_cached_episodes_are_counted_across_splits(self, tmp_path):
        reader = _Reader(tmp_path)
        for split, n in (("train", 3), ("val", 2)):
            d = reader.cache_dir / split
            d.mkdir(parents=True)
            for i in range(n):
                (d / f"{i}.pt").touch()
        rows = cmd_ledger._reader_rows(
            type("R", (), {"readers": {reader.reader_id: reader}})())
        assert rows[0]["cached"] == 5


class TestStaleness:
    def _scored(self, tmp_path, sha):
        reader = _Reader(tmp_path)
        cell = _Cell(tmp_path, reader)
        cell.output_dir.mkdir(parents=True)
        (cell.output_dir / "summary.json").write_text(json.dumps(
            {"n_clips": 34, "n_failed": 0, "n_scored_by_role": {"dense": 34},
             "checkpoint_sha256": sha}))
        return reader, cell

    def _rows(self, reader, cell, trained_sha):
        registry = type("R", (), {"cells": {cell.cell_id: cell}})()
        return cmd_ledger._cell_rows(
            registry, {reader.reader_id: {"sha256": trained_sha}})

    def test_score_matching_the_checkpoint_is_not_stale(self, tmp_path):
        reader, cell = self._scored(tmp_path, "abc")
        assert self._rows(reader, cell, "abc")[0]["stale"] is False

    def test_score_from_a_replaced_checkpoint_is_stale(self, tmp_path):
        reader, cell = self._scored(tmp_path, "abc")
        assert self._rows(reader, cell, "def")[0]["stale"] is True

    def test_an_unscored_cell_is_not_reported_stale(self, tmp_path):
        reader = _Reader(tmp_path)
        cell = _Cell(tmp_path, reader)
        assert self._rows(reader, cell, "abc")[0]["stale"] is False


class TestTable:
    def test_columns_a_row_lacks_render_as_missing(self):
        out = cmd_ledger._table([{"a": 1}], ("a", "b"))
        assert out.splitlines()[-1].split() == ["1", cmd_ledger._MISSING]

    def test_booleans_render_as_ticks(self):
        assert cmd_ledger._fmt(True) == "yes"
        assert cmd_ledger._fmt(False) == cmd_ledger._MISSING
