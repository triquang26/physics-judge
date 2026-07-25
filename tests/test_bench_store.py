"""bench.runner + bench.store: resume skips scored rows, failures are recorded.

Exercises the two properties the runner exists to guarantee:

* ``--resume`` skips clips whose ``(path, suite_id, reader_id)`` already
  appears in ``results.jsonl``, so a killed run can pick back up without
  rescoring everything.
* a clip that raises during scoring still produces a row -- with
  ``status="failed"`` and the suite's full metric key set filled with
  ``None`` -- instead of silently shrinking the results file (a
  silently-shorter results file is how a benchmark lies).

Uses a fake Scorer (duck-typed to what ``kinescore.bench.store.failed_record``
and ``kinescore.bench.runner.run`` actually touch) and monkeypatches
``kinescore.video.reader.load_rgb`` so this stays in the CPU tier -- no real
video files or ffmpeg needed.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from kinescore.bench.runner import run
from kinescore.bench.store import ResultsStore, assert_uniform_schema
from kinescore.core.metric import BaseMetric, MetricContext, MetricSpec
from kinescore.core.scorer import ScoredClip
from kinescore.core.suite import MetricSuite


class _SpeedMetric(BaseMetric):
    spec = MetricSpec(key="dummy_speed", units="m/s", dt_exponent=1,
                      direction="lower_better", requires=frozenset())

    def _compute(self, ctx: MetricContext):
        return self._ok(9.9)


def _suite() -> MetricSuite:
    return MetricSuite("test_suite_v1", [_SpeedMetric()])


class _FakeScorer:
    """Duck-typed Scorer: real .suite, fake .reader/.robot, scripted .score()."""

    def __init__(self, fail_paths=frozenset()):
        self.suite = _suite()
        self.reader = SimpleNamespace(reader_id="fake_reader",
                                      limit_semantics="raw_rad")
        self.robot = SimpleNamespace(name="fake_robot")
        self.fail_paths = fail_paths

    def score(self, frames, clip):
        if clip.path in self.fail_paths:
            raise RuntimeError(f"scripted failure for {clip.path}")
        ctx = MetricContext(dt=clip.dt, P=torch.zeros(1, frames.shape[0], 3, 3))
        result = self.suite.evaluate(ctx)
        return ScoredClip(clip=clip, result=result, robot=self.robot.name,
                          reader_id=self.reader.reader_id,
                          limit_semantics=self.reader.limit_semantics,
                          n_frames_scored=frames.shape[0])


def _rows(paths):
    return [{"method": "m", "family": "f", "episode": f"ep{i}", "role": "pred",
            "path": p, "n_frames": 4, "fps": 10.0, "w": 8, "h": 8, "dt": 0.1,
            "pair_key": f"m/ep{i}", "dt_source": "ffprobe"}
           for i, p in enumerate(paths)]


@pytest.fixture(autouse=True)
def _fake_decode(monkeypatch):
    def fake_load_rgb(clip, max_frames=0):
        return torch.rand(clip.n_frames, 3, clip.height, clip.width)
    monkeypatch.setattr("kinescore.video.reader.load_rgb", fake_load_rgb)


class TestFailureRecording:
    def test_failed_clip_still_produces_a_row(self, tmp_path):
        rows = _rows(["/tmp/a.mp4", "/tmp/b.mp4"])
        scorer = _FakeScorer(fail_paths={"/tmp/b.mp4"})
        results_path = str(tmp_path / "results.jsonl")

        summary = run(rows, scorer, results_path, force=True)

        assert summary == {"n_total": 2, "n_scored": 1, "n_skipped": 0, "n_failed": 1}
        store = ResultsStore(results_path)
        records = list(store.iter_records())
        assert len(records) == 2  # NOT silently shorter than n_total

        by_path = {r["clip"]["path"]: r for r in records}
        assert by_path["/tmp/a.mp4"]["status"] == "ok"
        assert by_path["/tmp/b.mp4"]["status"] == "failed"

    def test_failed_row_has_full_null_metrics_matching_ok_row_keys(self, tmp_path):
        rows = _rows(["/tmp/a.mp4", "/tmp/b.mp4"])
        scorer = _FakeScorer(fail_paths={"/tmp/b.mp4"})
        results_path = str(tmp_path / "results.jsonl")
        run(rows, scorer, results_path, force=True)

        records = list(ResultsStore(results_path).iter_records())
        assert_uniform_schema(records)  # would raise on a key-set mismatch

        failed = next(r for r in records if r["status"] == "failed")
        assert failed["metrics"] == {"dummy_speed": None}
        assert "dummy_speed" in failed["metrics_unavailable"]

    def test_ok_row_has_a_real_value(self, tmp_path):
        rows = _rows(["/tmp/a.mp4"])
        scorer = _FakeScorer()
        results_path = str(tmp_path / "results.jsonl")
        run(rows, scorer, results_path, force=True)
        [rec] = list(ResultsStore(results_path).iter_records())
        assert rec["metrics"]["dummy_speed"] == pytest.approx(9.9)


class TestResume:
    def test_resume_skips_already_scored_rows(self, tmp_path):
        rows = _rows(["/tmp/a.mp4", "/tmp/b.mp4"])
        scorer = _FakeScorer()
        results_path = str(tmp_path / "results.jsonl")

        first = run(rows, scorer, results_path, force=True)
        assert first["n_scored"] == 2

        second = run(rows, scorer, results_path, resume=True)
        assert second["n_scored"] == 0
        assert second["n_skipped"] == 2

        records = list(ResultsStore(results_path).iter_records())
        assert len(records) == 2  # not duplicated

    def test_resume_only_scores_new_rows(self, tmp_path):
        results_path = str(tmp_path / "results.jsonl")
        scorer = _FakeScorer()
        run(_rows(["/tmp/a.mp4"]), scorer, results_path, force=True)

        second = run(_rows(["/tmp/a.mp4", "/tmp/c.mp4"]), scorer, results_path,
                    resume=True)
        assert second["n_skipped"] == 1
        assert second["n_scored"] == 1

        records = list(ResultsStore(results_path).iter_records())
        assert {r["clip"]["path"] for r in records} == {"/tmp/a.mp4", "/tmp/c.mp4"}

    def test_resume_and_force_are_mutually_exclusive(self, tmp_path):
        with pytest.raises(ValueError):
            run([], _FakeScorer(), str(tmp_path / "results.jsonl"),
               resume=True, force=True)

    def test_force_truncates_prior_results(self, tmp_path):
        results_path = str(tmp_path / "results.jsonl")
        scorer = _FakeScorer()
        run(_rows(["/tmp/a.mp4"]), scorer, results_path, force=True)
        run(_rows(["/tmp/b.mp4"]), scorer, results_path, force=True)
        records = list(ResultsStore(results_path).iter_records())
        assert {r["clip"]["path"] for r in records} == {"/tmp/b.mp4"}
