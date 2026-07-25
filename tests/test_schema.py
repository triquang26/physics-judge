"""The one canonical result schema round-trips, and the key set is exact.

Builds records the way ``kinescore.bench.runner`` does -- a real
``ScoredClip.to_record()`` for a successful clip, and
``kinescore.bench.store.failed_record`` for a clip that failed before a
``SuiteResult`` existed -- and checks the property the whole ``bench.store``
module exists to guarantee: both have *exactly* the same metrics key set for
the same ``suite_id``, unavailable metrics serialise as ``null`` (Python
``None``) never ``0``, and a record round-trips through ``results.jsonl``
unchanged.

Uses small local dummy metrics rather than anything from
``kinescore.metrics`` (a separate, not-owned subpackage) so this test only
depends on the frozen ``kinescore.core`` contracts plus ``kinescore.bench``.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from kinescore.bench.store import (
    SCHEMA_VERSION,
    STATUSES,
    ResultsStore,
    assert_uniform_schema,
    failed_record,
    flatten,
    record_key,
)
from kinescore.core.clip import ClipSpec
from kinescore.core.metric import BaseMetric, MetricContext, MetricSpec
from kinescore.core.scorer import ScoredClip
from kinescore.core.suite import MetricSuite


class _AvailableMetric(BaseMetric):
    spec = MetricSpec(key="dummy_speed", units="m/s", dt_exponent=1,
                      direction="lower_better", requires=frozenset(),
                      description="test-only metric, always computable")

    def _compute(self, ctx: MetricContext):
        return self._ok(1.23)


class _UnavailableMetric(BaseMetric):
    # requires "q_raw", which the test context never provides -> always
    # unavailable, exercising the null-not-omitted path.
    spec = MetricSpec(key="dummy_limit_violation", units="fraction",
                      dt_exponent=0, direction="lower_better",
                      requires=frozenset({"q_raw"}),
                      description="test-only metric, never computable here")

    def _compute(self, ctx: MetricContext):  # pragma: no cover - never reached
        return self._ok(0.0)


def _suite() -> MetricSuite:
    return MetricSuite("test_suite_v1", [_AvailableMetric(), _UnavailableMetric()])


def _ok_record() -> dict:
    suite = _suite()
    ctx = MetricContext(dt=0.1, P=torch.zeros(1, 5, 3, 3))
    result = suite.evaluate(ctx)
    clip = ClipSpec.from_fps(path="/tmp/ok.mp4", fps=10.0, n_frames=5,
                             width=64, height=64)
    scored = ScoredClip(clip=clip, result=result, robot="test_robot",
                        reader_id="test_reader", limit_semantics="raw_rad",
                        n_frames_scored=5)
    record = dict(scored.to_record())
    record["status"] = "ok"
    return record


def _failed_record() -> dict:
    suite = _suite()
    fake_scorer = SimpleNamespace(
        suite=suite,
        robot=SimpleNamespace(name="test_robot"),
        reader=SimpleNamespace(reader_id="test_reader", limit_semantics="raw_rad"),
    )
    clip_row = ClipSpec.from_fps(path="/tmp/bad.mp4", fps=10.0, n_frames=5,
                                 width=64, height=64).as_row()
    return failed_record(clip_row, fake_scorer, "error:RuntimeError:decode failed")


class TestRecordShape:
    def test_ok_record_has_expected_top_level_keys(self):
        rec = _ok_record()
        assert set(rec) >= {"clip", "run", "coverage", "metrics",
                            "metrics_unavailable", "status"}

    def test_available_metric_has_a_value(self):
        rec = _ok_record()
        assert rec["metrics"]["dummy_speed"] == pytest.approx(1.23)

    def test_unavailable_metric_is_null_not_zero(self):
        rec = _ok_record()
        assert rec["metrics"]["dummy_limit_violation"] is None
        assert rec["metrics"]["dummy_limit_violation"] != 0

    def test_unavailable_metric_has_a_reason(self):
        rec = _ok_record()
        assert "dummy_limit_violation" in rec["metrics_unavailable"]
        assert rec["metrics_unavailable"]["dummy_limit_violation"]
        # the available metric must NOT appear in metrics_unavailable at all
        assert "dummy_speed" not in rec["metrics_unavailable"]

    def test_failed_record_has_same_metric_key_set_as_ok_record(self):
        ok = _ok_record()
        failed = _failed_record()
        assert set(ok["metrics"]) == set(failed["metrics"])

    def test_failed_record_metrics_are_all_null(self):
        failed = _failed_record()
        assert all(v is None for v in failed["metrics"].values())

    def test_failed_record_status_is_failed(self):
        assert _failed_record()["status"] == "failed"

    def test_status_is_one_of_the_declared_values(self):
        assert _ok_record()["status"] in STATUSES
        assert _failed_record()["status"] in STATUSES


class TestAssertUniformSchema:
    def test_accepts_matching_key_sets(self):
        assert_uniform_schema([_ok_record(), _failed_record()])

    def test_rejects_mismatched_key_sets(self):
        ok = _ok_record()
        broken = dict(ok)
        broken["metrics"] = dict(ok["metrics"])
        broken["metrics"].pop("dummy_speed")
        with pytest.raises(ValueError, match="inconsistent"):
            assert_uniform_schema([ok, broken])


class TestFlatten:
    def test_produces_dotted_keys(self):
        flat = flatten(_ok_record())
        assert flat["clip.dt"] == pytest.approx(0.1)
        assert flat["run.suite_id"].startswith("sha256:")
        assert flat["metrics.dummy_speed"] == pytest.approx(1.23)
        assert flat["metrics.dummy_limit_violation"] is None

    def test_is_the_only_flattener_used_for_both_ok_and_failed(self):
        # Same function, same key shape, regardless of status.
        flat_ok = flatten(_ok_record())
        flat_failed = flatten(_failed_record())
        metric_keys_ok = {k for k in flat_ok if k.startswith("metrics.")}
        metric_keys_failed = {k for k in flat_failed if k.startswith("metrics.")}
        assert metric_keys_ok == metric_keys_failed


class TestResultsStoreRoundTrip:
    def test_append_and_read_back_preserves_record(self, tmp_path):
        store = ResultsStore(str(tmp_path / "results.jsonl"))
        rec = _ok_record()
        store.append(rec)
        [back] = list(store.iter_records())
        assert back["schema_version"] == SCHEMA_VERSION
        assert back["metrics"] == rec["metrics"]
        assert back["metrics"]["dummy_limit_violation"] is None
        assert back["status"] == "ok"

    def test_append_rejects_missing_status(self, tmp_path):
        store = ResultsStore(str(tmp_path / "results.jsonl"))
        rec = _ok_record()
        del rec["status"]
        with pytest.raises(ValueError):
            store.append(rec)

    def test_append_is_valid_json_lines(self, tmp_path):
        path = tmp_path / "results.jsonl"
        store = ResultsStore(str(path))
        store.append(_ok_record())
        store.append(_failed_record())
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            json.loads(line)  # must not raise

    def test_record_key_identifies_clip_suite_and_reader(self):
        rec = _ok_record()
        key = record_key(rec)
        assert key == (rec["clip"]["path"], rec["run"]["suite_id"],
                       rec["run"]["reader_id"])
