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

import json

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


class TestReportCrossRateFlagReadFromRegistryNotHardcoded:
    """Definition-of-done item: a `dt_exponent is None` metric must be marked
    not-cross-rate-comparable in `kinescore report`'s output, and that must
    come from the separation row's own `rate_comparable` field (in turn read
    off the metric registry by `bench/separation.py`) -- not a hardcoded
    metric-name list in `bench/report.py`. Proven here with made-up metric
    names no report code could plausibly have hardcoded.
    """

    @staticmethod
    def _stats_blob():
        return {
            "provenance": {"kinescore_version": "0.0.0", "git_sha": "abc123"},
            "results": [],
            "separation": [
                {"method": "dense", "metric": "totally_made_up_ratefree_metric",
                 "units": "u", "n": 10, "real_median": 1.0, "gen_median": 2.0,
                 "delta_median": 1.0, "ci_lo": 0.5, "ci_hi": 1.5, "p": 0.01,
                 "p_holm": 0.02, "cliffs_delta": 0.6, "frac_worse": 0.9,
                 "separation": 0.75, "verdict": "a clear signal",
                 "noise_floor": None, "above_noise": None,
                 "rate_comparable": False, "reason": None},
                {"method": "dense", "metric": "totally_made_up_powerlaw_metric",
                 "units": "u", "n": 10, "real_median": 1.0, "gen_median": 2.0,
                 "delta_median": 1.0, "ci_lo": 0.5, "ci_hi": 1.5, "p": 0.01,
                 "p_holm": 0.02, "cliffs_delta": 0.6, "frac_worse": 0.9,
                 "separation": 0.75, "verdict": "a clear signal",
                 "noise_floor": None, "above_noise": None,
                 "rate_comparable": True, "reason": None},
            ],
            "cache_ranking": [],
        }

    def test_markdown_flags_only_the_non_comparable_row(self):
        from kinescore.bench import report as bench_report

        text = bench_report.render_markdown(self._stats_blob())
        assert "totally_made_up_ratefree_metric" in text
        assert "not comparable across frame rates" in text
        lines = text.splitlines()
        rate_free_line = next(
            ln for ln in lines if "totally_made_up_ratefree_metric" in ln)
        powerlaw_line = next(
            ln for ln in lines if "totally_made_up_powerlaw_metric" in ln)
        assert "not comparable across frame rates" in rate_free_line
        assert "not comparable across frame rates" not in powerlaw_line

    def test_html_flags_only_the_non_comparable_row(self):
        from kinescore.bench import report as bench_report

        html_text = bench_report.render_html(self._stats_blob())
        # The caption also explains the marker in prose (one occurrence);
        # count only the rendered table CELLS carrying the marker.
        assert html_text.count(
            '<td class="warn">not comparable across frame rates</td>') == 1


def _raw_record(path, metrics):
    """A results.jsonl row with hand-picked metric values, bypassing
    MetricSuite.evaluate entirely -- unlike `_scored_record` above (whose
    ``"jerk"`` key is a test-local stub never added to
    ``kinescore.core.metric.REGISTRY``), this uses real, globally registered
    metric keys so `bench/separation.py`'s `get_metric(...)` lookups
    (units/direction/dt_exponent) resolve to the actual shipped declarations.
    """
    return {
        "clip": {"path": path, "fps": 10.0, "dt": 0.1, "n_frames": 4,
                "width": 8, "height": 8, "dt_source": "synthetic",
                "view_layout": "1x1", "stride": 1, "codec": "h264", "sha1": None},
        "run": {"robot": "r", "reader_id": "test", "limit_semantics": "raw_rad",
                "suite_id": "sha256:realmetrics", "suite_name": "test_suite"},
        "coverage": {"n_frames_scored": 4, "gate_coverage": 1.0},
        "metrics": dict(metrics),
        "metrics_unavailable": {},
        "status": "ok",
    }


class TestAggregateAndReportEndToEnd:
    """Full pipeline on a synthetic run directory: `kinescore aggregate`
    writes `separation`/`cache_ranking` into stats.json (via
    `bench/separation.py`), and `kinescore report` renders both. Uses real
    registered metric keys -- including `sparc` (dt_exponent=None) -- so the
    report's cross-rate flag is exercised end to end via the real registry,
    not just via a hand-built stats.json.
    """

    def test_full_pipeline(self, tmp_path):
        import argparse

        import kinescore.metrics  # noqa: F401
        from kinescore.cli import cmd_aggregate, cmd_report

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        results_path = str(run_dir / "results.jsonl")
        store = ResultsStore(results_path)
        manifest = []
        for i in range(8):
            ep = f"ep{i:03d}"
            gt_path, pred_path = f"/tmp/{ep}_gt.mp4", f"/tmp/{ep}_pred.mp4"
            store.append(_raw_record(gt_path, {"mean_jerk_mps3": 100.0 + i,
                                               "sparc": -1.0 - 0.01 * i}))
            store.append(_raw_record(pred_path, {"mean_jerk_mps3": 140.0 + i,
                                                 "sparc": -1.5 - 0.01 * i}))
            manifest.append(_manifest_row(gt_path, "demo", ep, "gt", f"demo/{ep}"))
            manifest.append(_manifest_row(pred_path, "demo", ep, "pred", f"demo/{ep}"))
        manifest_path = str(run_dir / "bench_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        parser = argparse.ArgumentParser()
        cmd_aggregate.add_arguments(parser)
        args = parser.parse_args([str(run_dir), "--baseline", "demo",
                                  "--min-episodes", "2"])
        rc = cmd_aggregate.run(args)
        assert rc == 0

        with open(run_dir / "stats.json") as f:
            blob = json.load(f)
        assert "separation" in blob
        assert "cache_ranking" in blob
        sep_rows = {r["metric"]: r for r in blob["separation"]}
        assert set(sep_rows) == {"mean_jerk_mps3", "sparc"}

        jerk_row = sep_rows["mean_jerk_mps3"]
        assert jerk_row["reason"] is None
        assert jerk_row["delta_median"] == pytest.approx(40.0)
        assert jerk_row["rate_comparable"] is True

        sparc_row = sep_rows["sparc"]
        assert sparc_row["reason"] is None
        assert sparc_row["rate_comparable"] is False

        parser2 = argparse.ArgumentParser()
        cmd_report.add_arguments(parser2)
        args2 = parser2.parse_args([str(run_dir / "stats.json"), "--format", "markdown"])
        rc2 = cmd_report.run(args2)
        assert rc2 == 0
        with open(run_dir / "report.md") as f:
            report_text = f.read()
        assert "Separation" in report_text
        assert "Cache ranking" in report_text
        assert "not comparable across frame rates" in report_text
        lines = report_text.splitlines()
        sparc_line = next(ln for ln in lines if "| sparc |" in ln)
        jerk_line = next(ln for ln in lines if "| mean_jerk_mps3 |" in ln)
        assert "not comparable across frame rates" in sparc_line
        assert "not comparable across frame rates" not in jerk_line
