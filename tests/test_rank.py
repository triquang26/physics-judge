"""Tests for ``kinescore rank`` (``cli/cmd_rank.py``).

Exercises the ``HELP``/``add_arguments``/``run`` trio the same way every
other ``cli/cmd_*.py`` module is used by ``cli/main.py`` -- build a real
``argparse.ArgumentParser``, parse a real argv list, call ``run(args)`` --
rather than hand-building a ``Namespace``, so a flag added to
``add_arguments`` but never wired into ``run`` (or vice versa) would show up
here the same way it would on the command line.
"""
from __future__ import annotations

import argparse
import json

import pytest

pytest.importorskip("pandas")
pytest.importorskip("scipy")

import kinescore.metrics  # noqa: F401  (side effect: populates the metric registry)
from kinescore.bench.store import ResultsStore
from kinescore.cli import cmd_rank

MIN_FRAMES = 4


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


def _make_run_dir(tmp_path, entries):
    """``entries``: iterable of (method, episode, role, metrics_dict)."""
    d = tmp_path / "run"
    d.mkdir()
    store = ResultsStore(str(d / "results.jsonl"))
    manifest = []
    for method, episode, role, metrics in entries:
        path = f"/tmp/{method}_{episode}_{role}.mp4"
        store.append(_record(path, metrics))
        manifest.append(_manifest_row(path, method, episode, role))
    with open(d / "bench_manifest.json", "w") as f:
        json.dump(manifest, f)
    return str(d)


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    cmd_rank.add_arguments(parser)
    return parser.parse_args(argv)


class TestHelpAndArgparseWiring:
    def test_help_is_a_string(self):
        assert isinstance(cmd_rank.HELP, str) and cmd_rank.HELP

    def test_metric_is_required(self):
        parser = argparse.ArgumentParser()
        cmd_rank.add_arguments(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["some_dir"])  # no --metric


class TestUnknownMetricFailsWithValidKeys:
    def test_unregistered_metric_lists_registered_keys(self, tmp_path, capsys):
        run_dir = _make_run_dir(tmp_path, [
            ("dense", "ep0", "gt", {"mean_jerk_mps3": 1.0}),
            ("dense", "ep0", "pred", {"mean_jerk_mps3": 2.0}),
        ])
        args = _parse_args([run_dir, "--metric", "not_a_real_metric"])
        rc = cmd_rank.run(args)
        assert rc == 2
        err = capsys.readouterr().err
        assert "not_a_real_metric" in err
        assert "registered:" in err
        assert "mean_jerk_mps3" in err


class TestUnpairedSort:
    def test_worst_first_for_lower_better_metric(self, tmp_path, capsys):
        run_dir = _make_run_dir(tmp_path, [
            ("dense", "ep0", "gt", {"mean_jerk_mps3": 1.0}),
            ("dense", "ep0", "pred", {"mean_jerk_mps3": 9.0}),
            ("dense", "ep1", "gt", {"mean_jerk_mps3": 3.0}),
            ("dense", "ep1", "pred", {"mean_jerk_mps3": 5.0}),
        ])
        args = _parse_args([run_dir, "--metric", "mean_jerk_mps3", "--format", "json"])
        rc = cmd_rank.run(args)
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        values = [r["value"] for r in out]
        assert values == sorted(values, reverse=True)  # worst (highest) first
        assert out[0]["value"] == pytest.approx(9.0)
        assert out[0]["rank"] == 1

    def test_worst_first_for_higher_better_metric_is_smallest_value(self, tmp_path, capsys):
        run_dir = _make_run_dir(tmp_path, [
            ("dense", "ep0", "gt", {"limit_headroom_rad": 0.5}),
            ("dense", "ep0", "pred", {"limit_headroom_rad": 0.05}),
            ("dense", "ep1", "gt", {"limit_headroom_rad": 0.4}),
        ])
        args = _parse_args([run_dir, "--metric", "limit_headroom_rad", "--format", "json"])
        cmd_rank.run(args)
        out = json.loads(capsys.readouterr().out)
        # smallest headroom (0.05) is worst -> first
        assert out[0]["value"] == pytest.approx(0.05)
        assert out[-1]["value"] == pytest.approx(0.5)

    def test_top_limits_row_count(self, tmp_path, capsys):
        entries = []
        for i in range(5):
            entries.append(("dense", f"ep{i}", "pred", {"mean_jerk_mps3": float(i)}))
        run_dir = _make_run_dir(tmp_path, entries)
        args = _parse_args([run_dir, "--metric", "mean_jerk_mps3", "--top", "2",
                            "--format", "json"])
        cmd_rank.run(args)
        out = json.loads(capsys.readouterr().out)
        assert len(out) == 2
        assert out[0]["value"] == pytest.approx(4.0)
        assert out[1]["value"] == pytest.approx(3.0)

    def test_method_filter(self, tmp_path, capsys):
        run_dir = _make_run_dir(tmp_path, [
            ("dense", "ep0", "pred", {"mean_jerk_mps3": 1.0}),
            ("dicache", "ep0", "pred", {"mean_jerk_mps3": 99.0}),
        ])
        args = _parse_args([run_dir, "--metric", "mean_jerk_mps3",
                            "--method", "dense", "--format", "json"])
        cmd_rank.run(args)
        out = json.loads(capsys.readouterr().out)
        assert len(out) == 1
        assert out[0]["method"] == "dense"

    def test_metric_absent_from_results_is_an_error(self, tmp_path, capsys):
        run_dir = _make_run_dir(tmp_path, [
            ("dense", "ep0", "gt", {"mean_jerk_mps3": 1.0}),
            ("dense", "ep0", "pred", {"mean_jerk_mps3": 2.0}),
        ])
        args = _parse_args([run_dir, "--metric", "rigidity_residual_mm",
                            "--format", "json"])
        rc = cmd_rank.run(args)
        assert rc == 1
        assert "rigidity_residual_mm" in capsys.readouterr().err


class TestPairedSort:
    def test_paired_delta_worst_first(self, tmp_path, capsys):
        run_dir = _make_run_dir(tmp_path, [
            ("dense", "ep0", "gt", {"mean_jerk_mps3": 10.0}),
            ("dense", "ep0", "pred", {"mean_jerk_mps3": 12.0}),  # delta +2
            ("dense", "ep1", "gt", {"mean_jerk_mps3": 10.0}),
            ("dense", "ep1", "pred", {"mean_jerk_mps3": 20.0}),  # delta +10
        ])
        args = _parse_args([run_dir, "--metric", "mean_jerk_mps3", "--paired",
                            "--format", "json"])
        rc = cmd_rank.run(args)
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out[0]["episode"] == "ep1"
        assert out[0]["delta"] == pytest.approx(10.0)
        assert out[1]["delta"] == pytest.approx(2.0)


class TestOutputFormats:
    def test_table_format_has_a_header_and_a_row(self, tmp_path, capsys):
        run_dir = _make_run_dir(tmp_path, [
            ("dense", "ep0", "pred", {"mean_jerk_mps3": 1.0}),
        ])
        args = _parse_args([run_dir, "--metric", "mean_jerk_mps3", "--format", "table"])
        cmd_rank.run(args)
        out = capsys.readouterr().out
        assert "rank" in out
        assert "value" in out

    def test_csv_format_is_parseable(self, tmp_path, capsys):
        import csv
        import io

        run_dir = _make_run_dir(tmp_path, [
            ("dense", "ep0", "pred", {"mean_jerk_mps3": 1.0}),
            ("dense", "ep1", "pred", {"mean_jerk_mps3": 2.0}),
        ])
        args = _parse_args([run_dir, "--metric", "mean_jerk_mps3", "--format", "csv"])
        cmd_rank.run(args)
        out = capsys.readouterr().out
        reader = csv.DictReader(io.StringIO(out))
        rows = list(reader)
        assert len(rows) == 2
        assert set(rows[0]) >= {"rank", "method", "episode", "role", "value"}


class TestEmptyResults:
    def test_missing_results_file_is_an_error(self, tmp_path, capsys):
        d = tmp_path / "empty"
        d.mkdir()
        args = _parse_args([str(d), "--metric", "mean_jerk_mps3"])
        rc = cmd_rank.run(args)
        assert rc == 1
        assert "results.jsonl" in capsys.readouterr().err
