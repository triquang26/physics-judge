"""Tests for ``bench/separation.py``: paired tax + oriented AUROC separation.

Builds ``results.jsonl`` rows directly (bypassing ``MetricSuite.evaluate``,
unlike ``test_bench_stats.py``'s stub-metric fixture) so each test can plant
an exact ``metrics.<key>`` value per clip -- everything ``separation.py``
reads comes off the joined DataFrame, not off how a value was computed, so
this is a faithful and much cheaper way to control the inputs. Real registry
metric keys are used throughout (``mean_jerk_mps3``, ``limit_headroom_rad``,
``sparc``) specifically so ``compute_separation``'s ``get_metric(...).spec``
lookups exercise the real, shipped ``units``/``direction``/``dt_exponent``
declarations rather than a stand-in.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pandas")
pytest.importorskip("scipy")

import numpy as np

import kinescore.metrics  # noqa: F401  (side effect: populates the metric registry)
from kinescore.bench import separation, stats
from kinescore.bench.store import ResultsStore

MIN_FRAMES = 4  # >= mean_jerk_mps3's min_frames


def _record(path: str, metrics: dict) -> dict:
    return {
        "clip": {"path": path, "fps": 10.0, "dt": 0.1, "n_frames": MIN_FRAMES,
                 "width": 8, "height": 8, "dt_source": "synthetic",
                 "view_layout": "1x1", "stride": 1, "codec": "h264", "sha1": None},
        "run": {"robot": "r", "reader_id": "test", "limit_semantics": "raw_rad",
                "suite_id": "sha256:test", "suite_name": "test_suite"},
        "coverage": {"n_frames_scored": MIN_FRAMES, "gate_coverage": 1.0},
        "metrics": dict(metrics),
        "metrics_unavailable": {},
        "status": "ok",
    }


def _manifest_row(path: str, method: str, episode: str, role: str) -> dict:
    return {"method": method, "family": "f", "episode": episode, "role": role,
           "path": path, "pair_key": f"{method}/{episode}"}


def _build_df(tmp_path, method: str, metric_key: str, gt_values, pred_values):
    """One method, len(gt_values) episodes, paired gt/pred on ``metric_key``."""
    results_path = str(tmp_path / "results.jsonl")
    store = ResultsStore(results_path)
    manifest = []
    for i, (gv, pv) in enumerate(zip(gt_values, pred_values, strict=True)):
        ep = f"ep{i:03d}"
        gt_path, pred_path = f"/tmp/{method}_{ep}_gt.mp4", f"/tmp/{method}_{ep}_pred.mp4"
        store.append(_record(gt_path, {metric_key: float(gv)}))
        store.append(_record(pred_path, {metric_key: float(pv)}))
        manifest.append(_manifest_row(gt_path, method, ep, "gt"))
        manifest.append(_manifest_row(pred_path, method, ep, "pred"))
    return stats.load_scores(results_path, manifest)


class TestComputeSeparationHappyPath:
    def test_planted_jerk_tax(self, tmp_path):
        rng = np.random.default_rng(0)
        gt = 100.0 + rng.normal(0, 1, 20)
        pred = gt + 5.0  # constant, planted tax
        df = _build_df(tmp_path, "demo", "mean_jerk_mps3", gt, pred)

        row = separation.compute_separation(df, "demo", "mean_jerk_mps3")

        assert row.reason is None
        assert row.n == 20
        assert row.delta_median == pytest.approx(5.0, abs=1e-6)
        assert row.ci_lo <= 5.0 <= row.ci_hi
        assert row.units == "m/s^3"
        assert row.direction == "lower_better"
        assert row.dt_exponent == 3
        assert row.rate_comparable is True
        assert row.frac_worse == pytest.approx(1.0)  # every episode's pred is worse
        assert row.separation is not None and row.separation > 0.9
        assert row.verdict == "a strong signal"
        assert row.real_median is not None and row.gen_median is not None
        assert row.gen_median > row.real_median

    def test_noise_floor_above_and_below(self, tmp_path):
        gt = np.full(10, 100.0)
        pred = gt + 5.0
        df = _build_df(tmp_path, "demo", "mean_jerk_mps3", gt, pred)

        above = separation.compute_separation(df, "demo", "mean_jerk_mps3",
                                               noise_floor=1.0)
        assert above.above_noise is True

        below = separation.compute_separation(df, "demo", "mean_jerk_mps3",
                                               noise_floor=100.0)
        assert below.above_noise is False

        unknown = separation.compute_separation(df, "demo", "mean_jerk_mps3")
        assert unknown.above_noise is None  # no noise_floor given -> unknown, not False


class TestUnavailableRowsCarryReasonNeverZero:
    def test_below_min_episodes(self, tmp_path):
        gt = np.array([100.0, 101.0, 102.0])
        pred = gt + 5.0
        df = _build_df(tmp_path, "demo", "mean_jerk_mps3", gt, pred)

        row = separation.compute_separation(df, "demo", "mean_jerk_mps3",
                                             min_episodes=5)

        assert row.n == 3
        assert row.reason is not None
        assert row.reason.startswith("too_few_episodes:3<5")
        # Every numeric field is None -- never a fabricated 0.0.
        for field in ("delta_median", "ci_lo", "ci_hi", "p", "cliffs_delta",
                      "frac_worse", "separation", "real_median", "gen_median",
                      "above_noise", "verdict"):
            assert getattr(row, field) is None, field

    def test_metric_column_missing_from_results(self, tmp_path):
        gt = np.full(10, 1.0)
        pred = gt + 1.0
        df = _build_df(tmp_path, "demo", "mean_jerk_mps3", gt, pred)

        row = separation.compute_separation(df, "demo", "rigidity_residual_mm")

        assert row.reason is not None
        assert row.reason.startswith("metric_unavailable:")
        assert row.delta_median is None
        # units/direction/dt_exponent are still filled from the registry --
        # a row that cannot be computed still describes what was requested.
        assert row.units == "mm"
        assert row.dt_exponent == 0

    def test_unregistered_metric_key(self, tmp_path):
        gt = np.full(10, 1.0)
        pred = gt + 1.0
        df = _build_df(tmp_path, "demo", "mean_jerk_mps3", gt, pred)

        row = separation.compute_separation(df, "demo", "not_a_real_metric")

        assert row.reason is not None
        assert "not_a_real_metric" in row.reason
        assert row.delta_median is None
        assert row.n == 0


class TestAurocOrientation:
    """Definition-of-done #2: orientation must not silently invert for a
    ``higher_better`` metric."""

    def test_higher_better_metric_real_better_yields_above_half(self, tmp_path):
        rng = np.random.default_rng(1)
        # limit_headroom_rad is higher_better: real (gt) genuinely has more
        # headroom (bigger margin) than generated -- the ruler should say
        # "generated is worse" i.e. separation > 0.5.
        gt = 0.5 + rng.normal(0, 0.02, 40)
        pred = 0.2 + rng.normal(0, 0.02, 40)
        df = _build_df(tmp_path, "demo", "limit_headroom_rad", gt, pred)

        row = separation.compute_separation(df, "demo", "limit_headroom_rad")

        assert row.direction == "higher_better"
        assert row.reason is None
        assert row.separation is not None
        assert row.separation > 0.5

    def test_flipping_the_data_flips_separation_below_half(self, tmp_path):
        rng = np.random.default_rng(1)
        # Same metric, same magnitude of separation, but now the GENERATED
        # clip has the larger (better) margin -- the ruler is now "wrong",
        # and must report separation < 0.5, not silently stay > 0.5.
        gt = 0.2 + rng.normal(0, 0.02, 40)
        pred = 0.5 + rng.normal(0, 0.02, 40)
        df = _build_df(tmp_path, "demo", "limit_headroom_rad", gt, pred)

        row = separation.compute_separation(df, "demo", "limit_headroom_rad")

        assert row.separation is not None
        assert row.separation < 0.5

    def test_lower_better_metric_orientation_for_symmetry(self, tmp_path):
        rng = np.random.default_rng(2)
        gt = rng.normal(0, 1, 40)
        pred = gt + 10.0  # generated is much worse (higher jerk)
        df = _build_df(tmp_path, "demo", "mean_jerk_mps3", gt, pred)
        row = separation.compute_separation(df, "demo", "mean_jerk_mps3")
        assert row.separation > 0.5

    def test_all_ties_is_an_honest_half_not_a_crash(self, tmp_path):
        gt = np.full(10, 3.0)
        pred = np.full(10, 3.0)
        df = _build_df(tmp_path, "demo", "mean_jerk_mps3", gt, pred)
        row = separation.compute_separation(df, "demo", "mean_jerk_mps3")
        assert row.separation == pytest.approx(0.5, abs=1e-6)
        assert row.verdict == "can't tell them apart"


class TestVerdictBands:
    def test_bands(self):
        assert separation._verdict(0.9) == "a strong signal"
        assert separation._verdict(0.80) == "a strong signal"
        assert separation._verdict(0.7) == "a clear signal"
        assert separation._verdict(0.60) == "a clear signal"
        assert separation._verdict(0.5) == "can't tell them apart"
        assert separation._verdict(0.2) == "can't tell them apart"  # inverted, still an honest label
        assert separation._verdict(None) is None
        assert separation._verdict(float("nan")) is None


class TestRateComparableReadFromRegistry:
    def test_dt_exponent_none_metric_is_flagged_not_rate_comparable(self, tmp_path):
        gt = np.full(10, 0.5)
        pred = np.full(10, 0.6)
        df = _build_df(tmp_path, "demo", "sparc", gt, pred)
        row = separation.compute_separation(df, "demo", "sparc")
        assert row.dt_exponent is None
        assert row.rate_comparable is False

    def test_dt_exponent_int_metric_is_rate_comparable(self, tmp_path):
        gt = np.full(10, 0.5)
        pred = np.full(10, 0.6)
        df = _build_df(tmp_path, "demo", "mean_jerk_mps3", gt, pred)
        row = separation.compute_separation(df, "demo", "mean_jerk_mps3")
        assert row.dt_exponent == 3
        assert row.rate_comparable is True


class TestSeparationTableHolmAdjustment:
    def test_holm_adjusted_p_is_never_less_than_raw_p(self, tmp_path):
        results_path = str(tmp_path / "results.jsonl")
        store = ResultsStore(results_path)
        manifest = []
        metrics = ["mean_jerk_mps3", "rigidity_residual_mm"]
        for i in range(10):
            ep = f"ep{i:03d}"
            gt_path = f"/tmp/{ep}_gt.mp4"
            pred_path = f"/tmp/{ep}_pred.mp4"
            store.append(_record(gt_path, {"mean_jerk_mps3": 10.0 + i,
                                           "rigidity_residual_mm": 1.0 + 0.1 * i}))
            store.append(_record(pred_path, {"mean_jerk_mps3": 20.0 + i,
                                             "rigidity_residual_mm": 1.2 + 0.1 * i}))
            manifest.append(_manifest_row(gt_path, "demo", ep, "gt"))
            manifest.append(_manifest_row(pred_path, "demo", ep, "pred"))
        df = stats.load_scores(results_path, manifest)

        rows = separation.separation_table(df, ["demo"], metrics)

        assert len(rows) == 2
        for row in rows:
            assert row.reason is None
            assert row.p_holm is not None
            assert row.p_holm >= row.p - 1e-12

    def test_unavailable_row_has_no_p_holm(self, tmp_path):
        gt = np.array([1.0, 2.0])  # below default min_episodes
        pred = gt + 1.0
        df = _build_df(tmp_path, "demo", "mean_jerk_mps3", gt, pred)
        rows = separation.separation_table(df, ["demo"], ["mean_jerk_mps3"])
        assert rows[0].reason is not None
        assert rows[0].p_holm is None


class TestRankCaches:
    """rank_caches operates purely on SeparationRow point estimates, so rows
    are built directly for full control over the ranking scenario."""

    @staticmethod
    def _row(method, metric, direction, delta_median):
        return separation.SeparationRow(
            method=method, metric=metric, units="u", dt_exponent=0,
            direction=direction, rate_comparable=True, n=50,
            real_median=0.0, gen_median=delta_median, delta_median=delta_median,
            ci_lo=delta_median - 1, ci_hi=delta_median + 1, p=0.01, p_holm=0.01,
            cliffs_delta=0.5, frac_worse=0.9, separation=0.7, noise_floor=None,
            above_noise=None, verdict="a clear signal", reason=None)

    def test_lower_is_better_ranking(self):
        rows = [
            self._row("dense", "mean_jerk_mps3", "lower_better", 5.0),
            self._row("dicache", "mean_jerk_mps3", "lower_better", 10.4),
            self._row("fastercache", "mean_jerk_mps3", "lower_better", 10.5),
        ]
        ranked = separation.rank_caches(rows, baseline="dense")
        by_method = {r.method: r for r in ranked}

        assert by_method["dense"].axis_ranks["mean_jerk_mps3"] == 1.0
        assert by_method["dicache"].axis_ranks["mean_jerk_mps3"] == 2.0
        assert by_method["fastercache"].axis_ranks["mean_jerk_mps3"] == 3.0
        assert ranked[0].method == "dense"  # best mean rank first
        # extra cost vs baseline: dense excluded, others positive (worse)
        assert "mean_jerk_mps3" not in by_method["dense"].axis_extra_cost
        assert by_method["dicache"].axis_extra_cost["mean_jerk_mps3"] == pytest.approx(5.4)
        assert by_method["fastercache"].axis_extra_cost["mean_jerk_mps3"] == pytest.approx(5.5)

    def test_higher_is_better_axis_flips_ranking_order(self):
        # com_margin_m: a *more negative* delta is worse -> should rank worst.
        rows = [
            self._row("dense", "com_margin_m", "higher_better", -0.01),
            self._row("dicache", "com_margin_m", "higher_better", -0.05),
        ]
        ranked = separation.rank_caches(rows, baseline="dense")
        by_method = {r.method: r for r in ranked}
        assert by_method["dense"].axis_ranks["com_margin_m"] == 1.0
        assert by_method["dicache"].axis_ranks["com_margin_m"] == 2.0

    def test_mean_rank_across_multiple_axes(self):
        rows = [
            self._row("dense", "mean_jerk_mps3", "lower_better", 5.0),
            self._row("dense", "com_margin_m", "higher_better", -0.01),
            self._row("dicache", "mean_jerk_mps3", "lower_better", 10.0),
            self._row("dicache", "com_margin_m", "higher_better", -0.05),
        ]
        ranked = separation.rank_caches(rows, baseline="dense")
        by_method = {r.method: r for r in ranked}
        assert by_method["dense"].mean_rank == pytest.approx(1.0)
        assert by_method["dicache"].mean_rank == pytest.approx(2.0)
        assert by_method["dense"].n_axes == 2
        assert ranked[0].method == "dense"

    def test_single_method_axis_is_not_rankable(self):
        rows = [self._row("dense", "mean_jerk_mps3", "lower_better", 5.0)]
        ranked = separation.rank_caches(rows, baseline="dense")
        assert ranked[0].axis_ranks == {}
        assert ranked[0].mean_rank is None
        assert ranked[0].n_axes == 0


class TestExtraCostVsBaseline:
    def test_matches_stats_second_difference(self, tmp_path):
        results_path = str(tmp_path / "results.jsonl")
        store = ResultsStore(results_path)
        manifest = []
        for i in range(10):
            ep = f"ep{i:03d}"
            for method, tax in (("dense", 5.0), ("dicache", 10.5)):
                gt_path = f"/tmp/{method}_{ep}_gt.mp4"
                pred_path = f"/tmp/{method}_{ep}_pred.mp4"
                store.append(_record(gt_path, {"mean_jerk_mps3": 100.0 + i}))
                store.append(_record(pred_path, {"mean_jerk_mps3": 100.0 + i + tax}))
                manifest.append(_manifest_row(gt_path, method, ep, "gt"))
                manifest.append(_manifest_row(pred_path, method, ep, "pred"))
        df = stats.load_scores(results_path, manifest)

        row = separation.extra_cost_vs_baseline(df, "dicache", "dense", "mean_jerk_mps3")
        expected = stats.second_difference(df, "dicache", "dense", "metrics.mean_jerk_mps3")

        assert row.reason is None
        assert row.median == pytest.approx(expected["median"])
        assert row.ci_lo == pytest.approx(expected["ci"]["lo"])
        assert row.paired is True
        assert row.units == "m/s^3"
        assert row.median == pytest.approx(5.5, abs=1e-6)

    def test_unavailable_metric_carries_reason(self, tmp_path):
        gt = np.full(6, 1.0)
        pred = gt + 1.0
        df = _build_df(tmp_path, "dicache", "mean_jerk_mps3", gt, pred)
        row = separation.extra_cost_vs_baseline(df, "dicache", "dense", "rigidity_residual_mm")
        assert row.reason is not None
        assert row.median is None
