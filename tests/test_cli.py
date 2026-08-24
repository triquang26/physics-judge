"""``kinescore.cli.main``: every command on disk is reachable, and reachable once.

The parser is built by importing ``cli/cmd_*.py`` rather than from a list, so
these pin the contract that discovery relies on: the stages a benchmark run
goes through are present, each exposes the attributes discovery requires, and
the tree builds without a backbone, a GPU or a corpus.
"""
from __future__ import annotations

import pytest

from kinescore.cli.main import _discover_commands, build_parser, main

STAGES = ("pull", "data", "cache", "train", "score", "report")
QUERIES = ("readers", "models")


def _commands():
    return {m.NAME: m for m in _discover_commands()}


class TestDiscovery:
    def test_every_stage_is_registered(self):
        assert set(_commands()) == set(STAGES + QUERIES)

    def test_help_order_follows_the_pipeline(self):
        # pull -> ... -> report is the order a run goes through, and the order
        # --help must list, so the listing reads as instructions.
        assert [m.NAME for m in _discover_commands()] == list(STAGES + QUERIES)

    @pytest.mark.parametrize("name", STAGES + QUERIES)
    def test_command_exposes_what_discovery_requires(self, name):
        module = _commands()[name]
        for attr in ("NAME", "HELP", "add_arguments", "run"):
            assert hasattr(module, attr), f"{name} is missing {attr}"

    @pytest.mark.parametrize("name", STAGES + QUERIES)
    def test_command_help_is_a_sentence_not_a_placeholder(self, name):
        help_text = _commands()[name].HELP
        assert help_text and help_text[0].islower() and len(help_text) > 10

    def test_names_are_unique(self):
        names = [m.NAME for m in _discover_commands()]
        assert len(names) == len(set(names))


class TestParser:
    def test_tree_builds_without_a_corpus(self):
        assert build_parser().prog == "kinescore"

    @pytest.mark.parametrize("name", STAGES + QUERIES)
    def test_every_stage_parses_and_binds_its_runner(self, name):
        args = build_parser().parse_args([name] + _required_args(name))
        assert args._run is _commands()[name].run

    def test_version_exits_cleanly(self):
        with pytest.raises(SystemExit) as exc:
            build_parser().parse_args(["--version"])
        assert exc.value.code == 0

    def test_unknown_command_is_rejected(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["polish"])

    def test_no_command_prints_help_and_fails(self, capsys):
        assert main([]) == 1
        assert "kinescore" in capsys.readouterr().out


def _required_args(name: str) -> list[str]:
    """The minimum arguments a stage needs to parse."""
    return {"cache": ["--reader", "r"], "train": ["--reader", "r"]}.get(name, [])


class TestScorePreconditions:
    """``score`` names the missing thing and the command that makes it.

    Both checks run before the backbone, the robot or a single clip is
    touched, so a run that cannot succeed says why in a line rather than
    surfacing a loader traceback.
    """

    @pytest.fixture(autouse=True)
    def _paths(self, tmp_path, monkeypatch):
        for key in ("KINESCORE_DATA_ROOT", "KINESCORE_CACHE_DIR",
                    "KINESCORE_CKPT_DIR", "KINESCORE_ASSETS"):
            monkeypatch.setenv(key, str(tmp_path / key.lower()))

    def _args(self, **over):
        import argparse

        from kinescore.registry.cells import (
            DEFAULT_CELLS_PATH,
            DEFAULT_ROBOTS_PATH,
        )
        from kinescore.registry.views import DEFAULT_VIEWS_PATH

        base = {
            "cell": "single_arm.mv3_row.ctrlworld", "list": False,
            "videos": None, "checkpoint": None, "out": None,
            "calibration_clips": 24, "percentile": 99.0, "max_frames": 0,
            "limit": 0, "device": "cpu", "views": str(DEFAULT_VIEWS_PATH),
            "robots": str(DEFAULT_ROBOTS_PATH),
            "cells": str(DEFAULT_CELLS_PATH),
        }
        return argparse.Namespace(**{**base, **over})

    def test_an_untrained_reader_names_the_train_command(self):
        from kinescore.cli import cmd_score

        with pytest.raises(SystemExit) as e:
            cmd_score.run(self._args())

        assert ("kinescore train --reader "
                "franka_panda.single_arm_mv.mv3_row") in str(e.value)

    def test_a_missing_checkpoint_is_reported_before_anything_is_built(
            self, tmp_path):
        from kinescore.cli import cmd_score

        with pytest.raises(SystemExit) as e:
            cmd_score.run(self._args(checkpoint=str(tmp_path / "nope.pt")))

        assert "no reader checkpoint at" in str(e.value)
        assert str(tmp_path / "nope.pt") in str(e.value)


class TestPartialCacheGate:
    """Training reads the cache, so a cache short of the tree is refused.

    An interrupted cache stage leaves a directory that loads and trains
    perfectly well on whatever episodes reached it, reporting a validation
    number that looks ordinary. The counts are compared instead.
    """

    def _tree(self, root, n_train, n_val):
        for split, n in (("train", n_train), ("val", n_val)):
            (root / "videos" / split).mkdir(parents=True)
            (root / "annotation" / split).mkdir(parents=True)
            for i in range(n):
                (root / "videos" / split / f"ep{i}.mp4").write_bytes(b"")
                (root / "annotation" / split / f"ep{i}.json").write_text("{}")

    def _cache(self, root, n_train, n_val):
        for split, n in (("train", n_train), ("val", n_val)):
            (root / split).mkdir(parents=True)
            for i in range(n):
                (root / split / f"ep{i}.pt").write_bytes(b"")

    def test_a_complete_cache_reports_nothing(self, tmp_path):
        from kinescore.cli.cmd_train import _uncached

        self._tree(tmp_path / "tree", 8, 2)
        self._cache(tmp_path / "cache", 8, 2)

        assert _uncached(tmp_path / "tree", tmp_path / "cache") == []

    def test_a_short_split_is_named_with_both_counts(self, tmp_path):
        from kinescore.cli.cmd_train import _uncached

        self._tree(tmp_path / "tree", 8, 2)
        self._cache(tmp_path / "cache", 5, 2)

        assert _uncached(tmp_path / "tree", tmp_path / "cache") == [
            ("train", 5, 8)]

    def test_every_short_split_is_reported(self, tmp_path):
        from kinescore.cli.cmd_train import _uncached

        self._tree(tmp_path / "tree", 8, 4)
        self._cache(tmp_path / "cache", 5, 1)

        assert _uncached(tmp_path / "tree", tmp_path / "cache") == [
            ("train", 5, 8), ("val", 1, 4)]

    def test_a_clip_without_an_annotation_is_not_expected_in_the_cache(
            self, tmp_path):
        from kinescore.cli.cmd_train import _uncached

        self._tree(tmp_path / "tree", 8, 2)
        (tmp_path / "tree" / "annotation" / "train" / "ep7.json").unlink()
        self._cache(tmp_path / "cache", 7, 2)

        assert _uncached(tmp_path / "tree", tmp_path / "cache") == []
