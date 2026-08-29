"""``kinescore ledger`` reports the artifacts that exist, and nothing else."""
from __future__ import annotations

import json

import pytest

from kinescore.cli import cmd_ledger

pytestmark = pytest.mark.unit


class _Reader:
    def __init__(self, tmp_path, reader_id="robot.corpus.view"):
        self.reader_id = reader_id
        self.robot = "robot"
        self.corpus = "corpus"
        self.status = ""
        self.train_tree = tmp_path / "tree"
        self.cache_dir = tmp_path / "cache"
        self.checkpoint_path = tmp_path / f"{reader_id}.pt"


class _Cell:
    def __init__(self, tmp_path, cell_id="emb.view.model"):
        self.cell_id = cell_id
        self.output_dir = tmp_path / "out" / cell_id


class _Registry:
    def __init__(self, reader, cells=()):
        self.readers = {reader.reader_id: reader}
        self._cells = tuple(cells)

    def cells_for_reader(self, reader_id):
        return self._cells


def _summary(cell, sha, n_clips=34):
    cell.output_dir.mkdir(parents=True)
    (cell.output_dir / "summary.json").write_text(json.dumps(
        {"n_clips": n_clips, "n_failed": 0, "n_scored_by_role": {"dense": n_clips},
         "checkpoint": "/ckpt/robot.diff.pt", "checkpoint_sha256": sha}))


class TestReaderRow:
    def test_a_reader_with_no_artifacts_reports_none(self, tmp_path):
        row = cmd_ledger._rows(_Registry(_Reader(tmp_path)))[0]
        assert row["tree"] is False
        assert row["cached"] == 0
        assert row["scores"] is None
        assert "val_mm" not in row

    def test_cached_episodes_are_counted_per_split(self, tmp_path):
        reader = _Reader(tmp_path)
        for split, n in (("train", 3), ("val", 2)):
            (reader.cache_dir / split).mkdir(parents=True)
            for i in range(n):
                (reader.cache_dir / split / f"{i}.pt").touch()
        row = cmd_ledger._rows(_Registry(reader))[0]
        assert row["cached"] == 5
        assert row["cached_by_split"] == {"train": 3, "val": 2}

    def test_the_robot_and_corpus_come_from_the_reader(self, tmp_path):
        row = cmd_ledger._rows(_Registry(_Reader(tmp_path)))[0]
        assert (row["robot"], row["corpus"]) == ("robot", "corpus")


class TestScoresColumn:
    def test_every_scored_cell_of_a_reader_is_listed(self, tmp_path):
        reader = _Reader(tmp_path)
        cells = [_Cell(tmp_path, "a.b.dreamgen"), _Cell(tmp_path, "a.b.dreamdojo")]
        for c in cells:
            _summary(c, "abc", n_clips=16)
        row = cmd_ledger._rows(_Registry(reader, cells))[0]
        assert row["scores"] == "a.b.dreamgen, a.b.dreamdojo"
        assert row["clips"] == 32

    def test_an_unscored_cell_is_left_out_of_the_column(self, tmp_path):
        reader = _Reader(tmp_path)
        scored, bare = _Cell(tmp_path, "a.b.one"), _Cell(tmp_path, "a.b.two")
        _summary(scored, "abc")
        row = cmd_ledger._rows(_Registry(reader, [scored, bare]))[0]
        assert row["scores"] == "a.b.one"
        assert [c["scored"] for c in row["cells"]] == [True, False]


class TestScoreProvenance:
    def _row(self, tmp_path, scored_sha, trained_sha):
        import torch

        reader = _Reader(tmp_path)
        torch.save({"cfg": {"head": "diffusion"},
                    "meta": {"val_mm": 78.6, "n_train_episodes": 227}},
                   reader.checkpoint_path)
        cell = _Cell(tmp_path)
        _summary(cell, scored_sha)
        rows = cmd_ledger._rows(_Registry(reader, [cell]))
        return rows[0], trained_sha

    def test_the_head_and_scores_are_read_off_the_checkpoint(self, tmp_path):
        row, _ = self._row(tmp_path, "abc", "abc")
        assert row["head"] == "diffusion"
        assert row["val_mm"] == 78.6
        assert row["train_ep"] == 227

    def test_a_score_from_another_checkpoint_is_flagged(self, tmp_path):
        row, _ = self._row(tmp_path, "not-the-readers-sha", "")
        assert row["cells"][0]["off_reader"] is True


class TestTable:
    def test_columns_a_row_lacks_render_as_missing(self):
        out = cmd_ledger._table([{"a": 1}], ("a", "b"))
        assert out.splitlines()[-1].split() == ["1", cmd_ledger._MISSING]

    def test_booleans_render_as_ticks(self):
        assert cmd_ledger._fmt(True) == "yes"
        assert cmd_ledger._fmt(False) == cmd_ledger._MISSING
