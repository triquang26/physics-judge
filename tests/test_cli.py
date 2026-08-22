"""``kinescore.cli.main``: every command on disk is reachable, and reachable once.

The parser is built by importing ``cli/cmd_*.py`` rather than from a list, so
these pin the contract that discovery relies on: the four stages a benchmark
run goes through are present, each exposes the attributes discovery requires,
and the tree builds without a backbone, a GPU or a corpus.
"""
from __future__ import annotations

import pytest

from kinescore.cli.main import _discover_commands, build_parser, main

STAGES = ("data", "cache", "train", "score")


def _commands():
    return {m.NAME: m for m in _discover_commands()}


class TestDiscovery:
    def test_the_four_stages_are_registered(self):
        assert set(_commands()) == set(STAGES)

    def test_help_order_follows_the_pipeline(self):
        # data -> cache -> train -> score is the order a run goes through, and
        # the order --help must list, so the listing reads as instructions.
        assert [m.NAME for m in _discover_commands()] == list(STAGES)

    @pytest.mark.parametrize("name", STAGES)
    def test_command_exposes_what_discovery_requires(self, name):
        module = _commands()[name]
        for attr in ("NAME", "HELP", "add_arguments", "run"):
            assert hasattr(module, attr), f"{name} is missing {attr}"

    @pytest.mark.parametrize("name", STAGES)
    def test_command_help_is_a_sentence_not_a_placeholder(self, name):
        help_text = _commands()[name].HELP
        assert help_text and help_text[0].islower() and len(help_text) > 10

    def test_names_are_unique(self):
        names = [m.NAME for m in _discover_commands()]
        assert len(names) == len(set(names))


class TestParser:
    def test_tree_builds_without_a_corpus(self):
        assert build_parser().prog == "kinescore"

    @pytest.mark.parametrize("name", STAGES)
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
