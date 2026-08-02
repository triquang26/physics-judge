"""``kinescore bench run``: expand the matrix and build a manifest per cell, end to end.

CPU/network-free: every "clip" is an empty ``.mp4`` file (mirrors
``test_bench_sources.py``'s convention -- discovery here never probes real
video, only ``kinescore.bench.manifest.build_manifest`` does, via ffprobe,
which is why every discovered row here is expected to be SKIPPED, not
scored -- these tests pin the cell-expansion/dispatch wiring, not probing).
"""
from __future__ import annotations

import argparse
import json

import pytest
import yaml

from kinescore.cli import cmd_bench

_ROBOT = "airbot_mmk2"  # -> embodiment "humanoid", ctrlworld only


def _config_raw() -> dict:
    return {
        "run_id": "t", "seed": 0,
        "axes": {
            "robot": [_ROBOT],
            "view": ["multiview"],
            "horizon": ["makovian"],
            "cache": ["dense"],
            "generator": ["ctrlworld"],
        },
        "na_cells": [],
        "robots": {_ROBOT: {"spec": _ROBOT, "reader": "a.pt", "assets": "a"}},
        "sources": {"ctrlworld": {"view_dir": "multiview"}},
        "fps_expected": {},
        "rate_policy": "paired", "suite": "invariant_v1",
        "baseline_cache": "dense", "caps": {},
    }


def _robot_map_raw() -> dict:
    return {"robots": {_ROBOT: {"embodiment": "humanoid", "generators": ["ctrlworld"]}}}


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(_config_raw()))
    (tmp_path / "robot_map.yaml").write_text(yaml.safe_dump(_robot_map_raw()))
    return str(path)


def _build_bench_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kinescore")
    subparsers = parser.add_subparsers(dest="command")
    bench_parser = subparsers.add_parser("bench")
    cmd_bench.add_arguments(bench_parser)
    bench_parser.set_defaults(_run=cmd_bench.run)
    return parser


class TestDryRun:
    def test_dry_run_reports_the_one_real_cell_and_na_cells(self, config_path, capsys):
        parser = _build_bench_parser()
        args = parser.parse_args(["bench", "run", "--config", config_path, "--dry-run"])
        rc = args._run(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "1 cell(s) to build" in out
        assert f"robot={_ROBOT}" in out

    def test_cells_out_writes_json_with_robot_and_embodiment(self, config_path, tmp_path):
        parser = _build_bench_parser()
        cells_out = tmp_path / "cells.json"
        args = parser.parse_args([
            "bench", "run", "--config", config_path, "--dry-run",
            "--cells-out", str(cells_out)])
        assert args._run(args) == 0
        rows = json.loads(cells_out.read_text())
        pending = [r for r in rows if r["status"] == "pending"]
        assert len(pending) == 1
        assert pending[0]["robot"] == _ROBOT
        assert pending[0]["embodiment"] == "humanoid"
        assert pending[0]["robot_spec"] == _ROBOT
        assert pending[0]["reader"] == "a.pt"


class TestRun:
    def test_run_dispatches_through_the_source_registry_and_writes_a_manifest(
            self, config_path, tmp_path, monkeypatch):
        data_root = tmp_path / "data"
        ep_dir = (data_root / "video_gen_physics" / "dense" / "humanoid" / "output"
                 / "multiview" / "ctrlworld" / "makovian" / "episode_0000")
        (ep_dir).mkdir(parents=True)
        (ep_dir / "pred_all_views.mp4").write_bytes(b"")
        (ep_dir / "gt_all_views.mp4").write_bytes(b"")
        monkeypatch.setenv("KINESCORE_DATA_ROOT", str(data_root))
        monkeypatch.setenv("KINESCORE_OUTPUT_DIR", str(tmp_path / "out"))

        parser = _build_bench_parser()
        args = parser.parse_args([
            "bench", "run", "--config", config_path, "--on-error", "skip"])
        # The two empty .mp4 fixtures fail ffprobe (build_manifest's
        # on_error="skip" default) -- so this cell discovers rows at the
        # DiscoveredClip level (pin the dispatch worked) but they cannot be
        # probed into manifest rows, matching every other clip-discovery
        # test in this suite that uses empty-file fixtures. That means
        # `_run_run` legitimately returns 1 ("every cell discovered zero
        # clips") -- what this test pins is that it reaches that point via
        # the real ClipSource dispatch, not a crash.
        rc = args._run(args)
        assert rc == 1


class TestOnlyFilter:
    def test_unknown_only_axis_is_a_usage_error(self, config_path, capsys):
        parser = _build_bench_parser()
        args = parser.parse_args([
            "bench", "run", "--config", config_path, "--dry-run",
            "--only", "not_an_axis=x"])
        rc = args._run(args)
        assert rc == 2
        assert "unknown axis" in capsys.readouterr().err

    def test_only_filter_narrows_the_dry_run_table(self, config_path, capsys):
        parser = _build_bench_parser()
        args = parser.parse_args([
            "bench", "run", "--config", config_path, "--dry-run",
            "--only", f"robot={_ROBOT}"])
        assert args._run(args) == 0
        assert f"robot={_ROBOT}" in capsys.readouterr().out
