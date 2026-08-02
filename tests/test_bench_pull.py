"""Direct, argparse-free tests for :mod:`kinescore.bench.pull`.

``kinescore.cli.cmd_data``'s ``pull`` action is a thin shell around
``resolve_allow_patterns``/``resolve_data_root``/``_pull_one`` here (``data
ingest``/``data verify`` were already thin over
``kinescore.bench.ingest``/``kinescore.bench.verify`` and are tested through
the CLI in ``tests/test_cmd_data.py``). This file is the library-level
coverage for the allow-pattern generation and local file counting -- the two
things that must be right *before* any bytes move, since a wrong
``allow_patterns`` is exactly how a run accidentally pulls hundreds of GB of
``dicache``/``fastercache``/etc.
"""
from __future__ import annotations

import sys

import pytest
import yaml

from kinescore.bench import pull

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


class TestFallbackAllowPatterns:
    def test_covers_exactly_the_three_known_repos(self, loaded_config, robot_map):
        patterns = pull._fallback_allow_patterns(loaded_config, robot_map)
        assert set(patterns) == set(pull.HF_REPOS)

    def test_video_gen_physics_is_cache_cross_embodiment(self, loaded_config, robot_map):
        patterns = pull._fallback_allow_patterns(loaded_config, robot_map)
        assert patterns["video_gen_physics"] == [
            "dense/humanoid/**", "dense/single_arm/**"]

    def test_real_video_is_per_embodiment_only_no_droid(self, loaded_config, robot_map):
        # single_arm has no top-level dir of its own in
        # video_gen_physics_real_video (verified live) -- and the fallback
        # does NOT add droid_1.0.1_20chunks/** for it: coordinator directive
        # is to use $KINESCORE_DROID_STD_DIR (a local, already-present,
        # same-fps DROID Franka tree) instead of pulling that 44 GB from HF.
        # See kinescore.bench.pull's module docstring and .env.
        patterns = pull._fallback_allow_patterns(loaded_config, robot_map)
        assert patterns["video_gen_physics_real_video"] == [
            "humanoid/**", "single_arm/**"]
        assert "droid_1.0.1_20chunks/**" not in patterns["video_gen_physics_real_video"]

    def test_real_video_is_per_embodiment_with_a_single_robot(self, config_path, tmp_path):
        from kinescore.bench.config import load_config
        from kinescore.bench.robot_map import parse_robot_map

        raw = dict(_MINIMAL_CONFIG)
        raw["axes"] = dict(raw["axes"], robot=[_ROBOT_B])
        raw["robots"] = {_ROBOT_B: _MINIMAL_CONFIG["robots"][_ROBOT_B]}
        raw_path = config_path.replace("benchmark.yaml", "single_robot.yaml")
        with open(raw_path, "w") as f:
            yaml.safe_dump(raw, f)
        config = load_config(raw_path)
        rm = parse_robot_map(_MINIMAL_ROBOT_MAP)
        patterns = pull._fallback_allow_patterns(config, rm)
        assert patterns["video_gen_physics_real_video"] == ["humanoid/**"]

    def test_cosmos_is_pulled_whole(self, loaded_config, robot_map):
        patterns = pull._fallback_allow_patterns(loaded_config, robot_map)
        assert patterns["cosmos_synthetic_data"] == ["**"]

    def test_scales_with_multiple_cache_values(self, config_path, robot_map):
        from kinescore.bench.config import load_config

        raw = dict(_MINIMAL_CONFIG)
        raw["axes"] = dict(raw["axes"], cache=["dense", "dicache"])
        raw_path = config_path.replace("benchmark.yaml", "multi_cache.yaml")
        with open(raw_path, "w") as f:
            yaml.safe_dump(raw, f)
        config = load_config(raw_path)
        patterns = pull._fallback_allow_patterns(config, robot_map)
        assert patterns["video_gen_physics"] == [
            "dense/humanoid/**", "dense/single_arm/**",
            "dicache/humanoid/**", "dicache/single_arm/**"]


class TestResolveAllowPatterns:
    def test_falls_back_when_bench_matrix_has_no_allow_patterns(
            self, loaded_config, robot_map, monkeypatch):
        # Simulate "kinescore.bench.matrix does not exist yet" regardless of
        # whether it has actually landed in this checkout by the time the
        # test runs elsewhere in the session.
        monkeypatch.setitem(sys.modules, "kinescore.bench.matrix", None)
        patterns = pull.resolve_allow_patterns(loaded_config, robot_map)
        assert patterns == pull._fallback_allow_patterns(loaded_config, robot_map)

    def test_real_bench_matrix_overrides_video_gen_physics_entry_only(
            self, loaded_config, robot_map):
        # No monkeypatching: exercises whatever kinescore.bench.matrix
        # actually is in this checkout. Its allow_patterns is source-aware
        # (na_cells, per-generator view_dir/gt_from) so it need not equal
        # the coarse local fallback -- but it must still be a list of
        # HF-repo-relative globs (no "video_gen_physics/" prefix), and the
        # other two repos must be untouched (bench.matrix does not index
        # them).
        fallback = pull._fallback_allow_patterns(loaded_config, robot_map)
        patterns = pull.resolve_allow_patterns(loaded_config, robot_map)

        assert set(patterns) == set(pull.HF_REPOS)
        assert patterns["video_gen_physics_real_video"] == \
            fallback["video_gen_physics_real_video"]
        assert patterns["cosmos_synthetic_data"] == fallback["cosmos_synthetic_data"]
        for pattern in patterns["video_gen_physics"]:
            assert not pattern.startswith("video_gen_physics/")

    def test_prefers_bench_matrix_when_available(self, loaded_config, robot_map, monkeypatch):
        import types

        fake_matrix = types.ModuleType("kinescore.bench.matrix")
        sentinel_raw = ["video_gen_physics/sentinel/humanoid/**"]
        fake_matrix.allow_patterns = lambda config, rm: sentinel_raw
        monkeypatch.setitem(sys.modules, "kinescore.bench.matrix", fake_matrix)

        patterns = pull.resolve_allow_patterns(loaded_config, robot_map)
        assert patterns["video_gen_physics"] == ["sentinel/humanoid/**"]
        # The other two repos are unaffected by bench.matrix -- still the
        # local fallback's answer.
        fallback = pull._fallback_allow_patterns(loaded_config, robot_map)
        assert patterns["video_gen_physics_real_video"] == \
            fallback["video_gen_physics_real_video"]
        assert patterns["cosmos_synthetic_data"] == fallback["cosmos_synthetic_data"]

    def test_raises_if_bench_matrix_returns_pattern_outside_its_repo_prefix(
            self, loaded_config, robot_map, monkeypatch):
        import types

        fake_matrix = types.ModuleType("kinescore.bench.matrix")
        fake_matrix.allow_patterns = lambda config, rm: ["not_video_gen_physics/x/**"]
        monkeypatch.setitem(sys.modules, "kinescore.bench.matrix", fake_matrix)

        with pytest.raises(ValueError, match="video_gen_physics/"):
            pull.resolve_allow_patterns(loaded_config, robot_map)


class TestCountLocal:
    def test_counts_files_and_bytes_excluding_hf_cache_dir(self, tmp_path):
        (tmp_path / "high").mkdir()
        (tmp_path / "high" / "a.mp4").write_bytes(b"x" * 10)
        (tmp_path / "high" / "a.txt").write_bytes(b"y" * 5)
        cache_dir = tmp_path / ".cache" / "huggingface"
        cache_dir.mkdir(parents=True)
        (cache_dir / "metadata.json").write_bytes(b"z" * 999)

        n_files, total_bytes = pull._count_local(str(tmp_path))
        assert n_files == 2
        assert total_bytes == 15

    def test_empty_dir_is_zero_zero(self, tmp_path):
        assert pull._count_local(str(tmp_path)) == (0, 0)

    def test_skips_a_file_that_vanishes_mid_walk(self, tmp_path, monkeypatch):
        import os

        (tmp_path / "a.mp4").write_bytes(b"x" * 10)
        (tmp_path / "b.mp4").write_bytes(b"y" * 20)

        real_getsize = os.path.getsize

        def _flaky_getsize(path):
            if os.path.basename(path) == "a.mp4":
                raise OSError("vanished between listdir and stat")
            return real_getsize(path)

        monkeypatch.setattr(os.path, "getsize", _flaky_getsize)
        n_files, total_bytes = pull._count_local(str(tmp_path))
        assert n_files == 1
        assert total_bytes == 20


class TestResolveDataRoot:
    def test_explicit_data_root_wins(self):
        assert pull.resolve_data_root("/explicit", dry_run=False) == "/explicit"

    def test_falls_back_to_env_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KINESCORE_DATA_ROOT", str(tmp_path))
        assert pull.resolve_data_root(None, dry_run=False) == str(tmp_path)

    def test_missing_env_var_without_dry_run_raises(self, monkeypatch):
        monkeypatch.delenv("KINESCORE_DATA_ROOT", raising=False)
        from kinescore.paths import MissingPathError

        with pytest.raises(MissingPathError):
            pull.resolve_data_root(None, dry_run=False)

    def test_missing_env_var_during_dry_run_returns_none(self, monkeypatch, capsys):
        monkeypatch.delenv("KINESCORE_DATA_ROOT", raising=False)
        assert pull.resolve_data_root(None, dry_run=True) is None
        assert "KINESCORE_DATA_ROOT" in capsys.readouterr().err
