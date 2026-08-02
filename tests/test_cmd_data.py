"""``kinescore data pull``/``ingest``/``verify``, through the CLI.

The allow-pattern generation / ``_count_local`` library logic (formerly
defined directly on ``cli.cmd_data``) now lives in :mod:`kinescore.bench.pull`
and is tested directly, without argparse, in ``tests/test_bench_pull.py``.
This file is the CLI-layer coverage: argument parsing/dispatch, ``--dry-run``,
provenance-JSON wiring, and (for ``ingest``/``verify``, which were already
thin wrappers over :mod:`kinescore.bench.ingest`/:mod:`kinescore.bench.verify`)
an end-to-end round trip through ``args._run``. Anything that actually talks
to Hugging Face is marked ``@pytest.mark.net`` and skipped by default (see
``pyproject.toml``'s ``addopts``); the ingest/verify round trip needs a real
``ffmpeg``/``ffprobe`` binary and is marked ``@pytest.mark.ffmpeg``
(auto-skipped only when that binary is absent -- see ``tests/conftest.py``).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pytest
import yaml

from kinescore.cli import cmd_data
from kinescore.cli.cmd_data import HF_REPOS

_ROBOT_A = "franka_panda"   # -> embodiment "single_arm"
_ROBOT_B = "airbot_mmk2"    # -> embodiment "humanoid"

_MINIMAL_CONFIG: dict = {
    "run_id": "test_run",
    "seed": 0,
    "axes": {
        "robot": [_ROBOT_A, _ROBOT_B],
        "view": ["multiview"],
        "horizon": ["makovian"],
        "cache": ["dense"],
        "generator": ["ctrlworld"],
    },
    "na_cells": [],
    "robots": {
        _ROBOT_A: {"spec": _ROBOT_A, "reader": "single_arm.pt", "assets": "franka"},
        _ROBOT_B: {"spec": _ROBOT_B, "reader": "humanoid.pt", "assets": "grx"},
    },
    "sources": {
        "ctrlworld": {"view_dir": "multiview"},
    },
    "fps_expected": {},
    "rate_policy": "paired",
    "suite": "invariant_v1",
    "baseline_cache": "dense",
    "caps": {},
}

_MINIMAL_ROBOT_MAP: dict = {
    "robots": {
        _ROBOT_A: {"embodiment": "single_arm", "generators": ["ctrlworld"]},
        _ROBOT_B: {"embodiment": "humanoid", "generators": ["ctrlworld"]},
    },
}


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(_MINIMAL_CONFIG))
    # robot_map.yaml next to the config -- _resolve_robot_map_path's default
    # (no --robot-map given) -- mirrors real usage (configs/robot_map.yaml
    # sits next to configs/benchmark.yaml).
    (tmp_path / "robot_map.yaml").write_text(yaml.safe_dump(_MINIMAL_ROBOT_MAP))
    return str(path)


@pytest.fixture
def loaded_config(config_path):
    from kinescore.bench.config import load_config

    return load_config(config_path)


@pytest.fixture
def robot_map():
    from kinescore.bench.robot_map import parse_robot_map

    return parse_robot_map(_MINIMAL_ROBOT_MAP)


def _build_data_parser() -> argparse.ArgumentParser:
    """Mirror how ``cli.main`` would wire this subcommand (see its
    docstring: ``sub.set_defaults(_run=module.run)`` uniformly), without
    depending on whether ``main.py`` has actually registered ``data`` yet --
    that registration belongs to a different file in this change.
    """
    parser = argparse.ArgumentParser(prog="kinescore")
    subparsers = parser.add_subparsers(dest="command")
    data_parser = subparsers.add_parser("data")
    cmd_data.add_arguments(data_parser)
    data_parser.set_defaults(_run=cmd_data.run)
    return parser


class TestRealPullPathStubbed:
    """Exercises ``pull_one``/the non-dry-run branch of ``_run_pull``
    against a fully stubbed ``huggingface_hub`` -- no network, but real
    coverage of the code path ``@pytest.mark.net`` alone would leave
    untested by default (see ``pyproject.toml``'s network-free default
    tier).
    """

    def test_pull_writes_files_and_provenance_json(
            self, config_path, tmp_path, monkeypatch):
        import json
        import types

        written: list[str] = []

        class _FakeInfo:
            sha = "deadbeef"

        class _FakeHfApi:
            def dataset_info(self, repo_id):
                written.append(repo_id)
                return _FakeInfo()

        def _fake_snapshot_download(*, repo_id, repo_type, revision, local_dir,
                                    allow_patterns, max_workers):
            assert repo_type == "dataset"
            assert revision == "deadbeef"
            os.makedirs(local_dir, exist_ok=True)
            (Path(local_dir) / "clip.mp4").write_bytes(b"z" * 42)
            return local_dir

        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.HfApi = _FakeHfApi
        fake_hf.snapshot_download = _fake_snapshot_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        parser = _build_data_parser()
        args = parser.parse_args([
            "data", "pull", "--config", config_path,
            "--repo", "cosmos_synthetic_data", "--data-root", str(tmp_path)])
        rc = args._run(args)
        assert rc == 0
        assert written == [HF_REPOS["cosmos_synthetic_data"]]

        local_dir = tmp_path / "cosmos_synthetic_data"
        assert (local_dir / "clip.mp4").is_file()
        prov = json.loads((local_dir / "provenance.json").read_text())
        assert prov["repo_id"] == HF_REPOS["cosmos_synthetic_data"]
        assert prov["revision"] == "deadbeef"
        assert prov["n_files"] == 1
        assert prov["total_bytes"] == 42
        assert prov["allow_patterns"] == ["**"]

    def test_missing_data_root_without_dry_run_raises(
            self, config_path, monkeypatch):
        monkeypatch.delenv("KINESCORE_DATA_ROOT", raising=False)
        parser = _build_data_parser()
        args = parser.parse_args([
            "data", "pull", "--config", config_path,
            "--repo", "cosmos_synthetic_data"])
        from kinescore.paths import MissingPathError

        with pytest.raises(MissingPathError):
            args._run(args)


class TestDryRun:
    def test_dry_run_prints_all_three_repos_and_downloads_nothing(
            self, config_path, tmp_path, monkeypatch, capsys):
        # A stub that errors if the dry-run path ever reaches for the
        # network client -- the guard this test exists to pin.
        import types

        def _boom(*a, **k):
            raise AssertionError("dry-run must not touch huggingface_hub")

        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.snapshot_download = _boom
        fake_hf.HfApi = _boom
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        parser = _build_data_parser()
        args = parser.parse_args([
            "data", "pull", "--config", config_path, "--dry-run",
            "--data-root", str(tmp_path)])
        rc = args._run(args)
        assert rc == 0

        out = capsys.readouterr().out
        for repo_id in HF_REPOS.values():
            assert repo_id in out
        assert "--dry-run" in out or "dry-run" in out
        # tmp_path holds config_path's benchmark.yaml; no dataset
        # subdirectory (what a real pull would create) may appear.
        for key in HF_REPOS:
            assert not (tmp_path / key).exists()

    def test_dry_run_restricts_to_one_repo_with_repo_flag(
            self, config_path, tmp_path, capsys):
        parser = _build_data_parser()
        args = parser.parse_args([
            "data", "pull", "--config", config_path, "--dry-run",
            "--repo", "cosmos_synthetic_data", "--data-root", str(tmp_path)])
        rc = args._run(args)
        assert rc == 0

        out = capsys.readouterr().out
        assert HF_REPOS["cosmos_synthetic_data"] in out
        assert HF_REPOS["video_gen_physics"] not in out
        assert HF_REPOS["video_gen_physics_real_video"] not in out

    def test_dry_run_without_data_root_or_env_var_does_not_crash(
            self, config_path, monkeypatch, capsys):
        monkeypatch.delenv("KINESCORE_DATA_ROOT", raising=False)
        parser = _build_data_parser()
        args = parser.parse_args(
            ["data", "pull", "--config", config_path, "--dry-run"])
        rc = args._run(args)
        assert rc == 0
        assert "KINESCORE_DATA_ROOT" in capsys.readouterr().out

    def test_no_action_prints_usage_and_returns_nonzero(self, capsys):
        parser = _build_data_parser()
        args = parser.parse_args(["data"])
        rc = args._run(args)
        assert rc != 0
        assert "usage" in capsys.readouterr().err.lower()


class TestHelp:
    def test_pull_help_exits_zero(self, capsys):
        parser = _build_data_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["data", "pull", "--help"])
        assert exc.value.code == 0
        assert "usage:" in capsys.readouterr().out

    def test_ingest_help_exits_zero(self, capsys):
        parser = _build_data_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["data", "ingest", "--help"])
        assert exc.value.code == 0
        assert "usage:" in capsys.readouterr().out

    def test_verify_help_exits_zero(self, capsys):
        parser = _build_data_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["data", "verify", "--help"])
        assert exc.value.code == 0
        assert "usage:" in capsys.readouterr().out


class TestIngestAndVerifyCli:
    """End-to-end through the CLI (not the library) against a tiny synthetic
    raw tree -- proves ``data ingest`` and ``data verify`` are wired
    together and reachable via ``args._run``, not just importable.
    """

    def _write_robot_map(self, tmp_path):
        path = tmp_path / "robot_map.yaml"
        path.write_text(yaml.safe_dump(_MINIMAL_ROBOT_MAP))
        return str(path)

    def _write_data_spec(self, tmp_path):
        path = tmp_path / "data_spec.yaml"
        path.write_text(yaml.safe_dump({
            "generators": {
                "ctrlworld": {
                    "shape": "episode_dir", "pred_filename": "pred_all_views.mp4",
                    "gt_filename": "gt_all_views.mp4", "has_iter_level": False,
                    "width": 4, "height": 4, "n_views": 1, "has_ground_truth": True,
                    "fps": 5.0, "fps_tolerant": True,
                },
            },
            "exclude_globs": [],
            "robots": {},
        }))
        return str(path)

    def _write_raw_episode(self, data_root: Path):
        ep = (data_root / "video_gen_physics" / "dense" / "humanoid" / "output"
             / "multiview" / "ctrlworld" / "makovian" / "episode_0000")
        ep.mkdir(parents=True)
        self._write_tiny_mp4(ep / "pred_all_views.mp4")
        self._write_tiny_mp4(ep / "gt_all_views.mp4")

    @staticmethod
    def _write_tiny_mp4(path: Path) -> None:
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", "color=c=black:s=4x4:d=0.2:r=5", "-pix_fmt", "yuv420p", str(path)],
            check=True)

    @pytest.mark.ffmpeg
    def test_ingest_then_verify_round_trip(self, tmp_path, monkeypatch):
        data_root = tmp_path / "raw"
        self._write_raw_episode(data_root)
        robot_map_path = self._write_robot_map(tmp_path)
        data_spec_path = self._write_data_spec(tmp_path)
        canon_root = tmp_path / "canon"

        monkeypatch.setenv("KINESCORE_DATA_ROOT", str(data_root))
        parser = _build_data_parser()

        ingest_args = parser.parse_args([
            "data", "ingest", "--robot-map", robot_map_path,
            "--data-spec", data_spec_path, "--out", str(canon_root)])
        assert ingest_args._run(ingest_args) == 0
        assert (canon_root / "dense" / "airbot_mmk2" / "multiview" / "ctrlworld"
               / "makovian" / "cell_card.json").is_file()

        verify_args = parser.parse_args([
            "data", "verify", "--data-spec", data_spec_path,
            "--canonical-root", str(canon_root)])
        assert verify_args._run(verify_args) == 0


@pytest.mark.net
class TestRealPull:
    """Hits Hugging Face for real. Run explicitly with ``-m net``."""

    def test_pull_cosmos_synthetic_data_end_to_end(self, config_path, tmp_path):
        parser = _build_data_parser()
        args = parser.parse_args([
            "data", "pull", "--config", config_path,
            "--repo", "cosmos_synthetic_data", "--data-root", str(tmp_path)])
        rc = args._run(args)
        assert rc == 0

        local_dir = tmp_path / "cosmos_synthetic_data"
        assert local_dir.is_dir()
        prov_path = local_dir / "provenance.json"
        assert prov_path.is_file()

        import json

        prov = json.loads(prov_path.read_text())
        assert prov["repo_id"] == HF_REPOS["cosmos_synthetic_data"]
        assert prov["n_files"] > 0
        assert prov["total_bytes"] > 0
        assert prov["allow_patterns"] == ["**"]

    def test_pull_is_idempotent_on_a_warm_cache(self, config_path, tmp_path):
        parser = _build_data_parser()
        args = parser.parse_args([
            "data", "pull", "--config", config_path,
            "--repo", "cosmos_synthetic_data", "--data-root", str(tmp_path)])
        assert args._run(args) == 0
        first_n_files = len(list((tmp_path / "cosmos_synthetic_data").rglob("*")))
        assert args._run(args) == 0  # second pull must not fail or duplicate
        second_n_files = len(list((tmp_path / "cosmos_synthetic_data").rglob("*")))
        assert first_n_files == second_n_files
