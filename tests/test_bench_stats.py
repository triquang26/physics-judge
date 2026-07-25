"""bench.stats: ported paired-statistics spine + the new suite-mixing guard.

The numeric functions (``wilcoxon_signed``, ``bootstrap_ci``, ``holm``,
``auroc``, ``cliffs_delta``) are a verbatim port and are covered lightly here
(the source ships its own ``_selftest`` with planted-effect gates; this file
does not re-derive that, it just pins the public behaviour). The two things
that actually changed get the real coverage: :func:`load_scores` joining a
results store against a manifest, and :func:`aggregate`'s refusal to pool
mixed ``suite_id`` rows.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pandas")
pytest.importorskip("scipy")

import numpy as np
import torch

from kinescore.bench import stats
from kinescore.bench.store import ResultsStore
from kinescore.core.clip import ClipSpec
from kinescore.core.metric import BaseMetric, MetricContext, MetricSpec
from kinescore.core.scorer import ScoredClip
from kinescore.core.suite import MetricSuite


class _JerkMetric(BaseMetric):
    spec = MetricSpec(key="jerk", units="m/s^3", dt_exponent=3,
                      direction="lower_better", requires=frozenset())

    def __init__(self, value):
        self._value = value

    def _compute(self, ctx):
        return self._ok(self._value)


def _suite(value, name="suite_a") -> MetricSuite:
    return MetricSuite(name, [_JerkMetric(value)])


def _scored_record(path, value, suite_name="suite_a"):
    # A fresh suite per call so each record's metric carries its own `value`
    # -- suite_id only hashes the suite's *name and declared key set*, not
    # any metric's computed value, so this still shares one suite_id across
    # all records built with the same suite_name.
    suite = _suite(value, name=suite_name)
    ctx = MetricContext(dt=0.1, P=torch.zeros(1, 4, 3, 3))
    result = suite.evaluate(ctx)
    clip = ClipSpec.from_fps(path=path, fps=10.0, n_frames=4, width=8, height=8)
    scored = ScoredClip(clip=clip, result=result, robot="r", reader_id="reader_a",
                        limit_semantics="raw_rad", n_frames_scored=4)
    rec = dict(scored.to_record())
    rec["status"] = "ok"
    return rec


def _manifest_row(path, method, episode, role, pair_key):
    return {"method": method, "family": "f", "episode": episode, "role": role,
           "path": path, "pair_key": pair_key}


@pytest.fixture()
def toy_store_and_manifest(tmp_path):
    """3 episodes of method 'demo': pred jerk = gt jerk + 40 (planted effect)."""
    results_path = str(tmp_path / "results.jsonl")
    store = ResultsStore(results_path)
    manifest = []
    for i in range(6):
        ep = f"ep{i:02d}"
        gt_path, pred_path = f"/tmp/{ep}_gt.mp4", f"/tmp/{ep}_pred.mp4"
        store.append(_scored_record(gt_path, 100.0 + i))
        store.append(_scored_record(pred_path, 140.0 + i))
        manifest.append(_manifest_row(gt_path, "demo", ep, "gt", f"demo/{ep}"))
        manifest.append(_manifest_row(pred_path, "demo", ep, "pred", f"demo/{ep}"))
    return results_path, manifest


class TestLoadScores:
    def test_joins_manifest_identity_onto_flattened_scores(self, toy_store_and_manifest):
        results_path, manifest = toy_store_and_manifest
        df = stats.load_scores(results_path, manifest)
        assert set(df["role"].unique()) == {"gt", "pred"}
        assert "metrics.jerk" in df.columns
        assert len(df) == 12

    def test_accepts_a_manifest_list_directly_not_only_a_path(self, toy_store_and_manifest):
        results_path, manifest = toy_store_and_manifest
        df = stats.load_scores(results_path, list(manifest))
        assert not df.empty


class TestPairedDeltasAndWilcoxon:
    def test_planted_effect_is_detected(self, toy_store_and_manifest):
        results_path, manifest = toy_store_and_manifest
        df = stats.load_scores(results_path, manifest)
        ep, delta = stats.paired_deltas(df, "demo", "metrics.jerk")
        assert delta.size == 6
        assert np.allclose(delta, 40.0)
        w = stats.wilcoxon_signed(delta)
        assert w["median"] == pytest.approx(40.0)


class TestAggregateSuiteGuard:
    def test_refuses_to_pool_mixed_suite_ids_by_default(self, tmp_path):
        results_path = str(tmp_path / "results.jsonl")
        store = ResultsStore(results_path)
        manifest = []
        for i, suite_name in enumerate(["suite_a", "suite_b"]):  # different suite_id
            ep = f"ep{i}"
            gt, pred = f"/tmp/{ep}_gt.mp4", f"/tmp/{ep}_pred.mp4"
            store.append(_scored_record(gt, 100.0, suite_name))
            store.append(_scored_record(pred, 140.0, suite_name))
            manifest.append(_manifest_row(gt, "demo", ep, "gt", f"demo/{ep}"))
            manifest.append(_manifest_row(pred, "demo", ep, "pred", f"demo/{ep}"))

        df = stats.load_scores(results_path, manifest)
        with pytest.raises(ValueError, match="suite_id"):
            stats.aggregate(df, "demo", "metrics.jerk")

    def test_allow_mixed_suites_proceeds_with_a_warning(self, tmp_path):
        results_path = str(tmp_path / "results.jsonl")
        store = ResultsStore(results_path)
        manifest = []
        for i, suite_name in enumerate(["suite_a", "suite_b"]):
            ep = f"ep{i}"
            gt, pred = f"/tmp/{ep}_gt.mp4", f"/tmp/{ep}_pred.mp4"
            store.append(_scored_record(gt, 100.0, suite_name))
            store.append(_scored_record(pred, 140.0, suite_name))
            manifest.append(_manifest_row(gt, "demo", ep, "gt", f"demo/{ep}"))
            manifest.append(_manifest_row(pred, "demo", ep, "pred", f"demo/{ep}"))

        df = stats.load_scores(results_path, manifest)
        out = stats.aggregate(df, "demo", "metrics.jerk", allow_mixed_suites=True)
        assert out["warning"] is not None
        assert "MIXED SUITES" in out["warning"]

    def test_single_suite_has_no_warning(self, toy_store_and_manifest):
        results_path, manifest = toy_store_and_manifest
        df = stats.load_scores(results_path, manifest)
        out = stats.aggregate(df, "demo", "metrics.jerk")
        assert out["warning"] is None
        assert out["median"] == pytest.approx(40.0)


class TestHolm:
    def test_monotone_and_preserves_order(self):
        adj = stats.holm([0.01, 0.04, 0.5])
        assert adj[0] <= adj[1] <= adj[2]
        assert np.all(adj >= np.array([0.01, 0.04, 0.5]) - 1e-9)


class TestAuroc:
    def test_separated_distributions_score_high(self):
        rng = np.random.default_rng(0)
        a = stats.auroc(rng.normal(0, 1, 50), rng.normal(3, 1, 50))
        assert a > 0.9

    def test_all_ties_is_an_honest_half(self):
        a = stats.auroc(np.zeros(20), np.zeros(20))
        assert a == pytest.approx(0.5, abs=1e-6)
