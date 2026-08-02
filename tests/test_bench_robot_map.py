"""kinescore.bench.robot_map: the (embodiment, generator) -> robot table."""
from __future__ import annotations

from pathlib import Path

import pytest

from kinescore.bench.robot_map import RobotMapError, load_robot_map, parse_robot_map

REPO_ROOT = Path(__file__).resolve().parents[1]


def _raw(**overrides) -> dict:
    raw = {
        "robots": {
            "fourier_gr1": {"embodiment": "humanoid", "generators": ["dreamgen", "dreamdojo"]},
            "airbot_mmk2": {"embodiment": "humanoid", "generators": ["ctrlworld"]},
        },
    }
    raw.update(overrides)
    return raw


class TestParsing:
    def test_valid_map_parses(self):
        rm = parse_robot_map(_raw())
        assert rm.embodiment_of("fourier_gr1") == "humanoid"
        assert rm.generators_of("airbot_mmk2") == ("ctrlworld",)

    def test_missing_robots_key_is_rejected(self):
        with pytest.raises(RobotMapError, match="robots"):
            parse_robot_map({})

    def test_entry_missing_generators_is_rejected(self):
        raw = _raw()
        del raw["robots"]["airbot_mmk2"]["generators"]
        with pytest.raises(RobotMapError, match="generators"):
            parse_robot_map(raw)

    def test_empty_generators_list_is_rejected(self):
        raw = _raw()
        raw["robots"]["airbot_mmk2"]["generators"] = []
        with pytest.raises(RobotMapError):
            parse_robot_map(raw)


class TestResolve:
    def test_resolves_the_one_robot_that_claims_the_pair(self):
        rm = parse_robot_map(_raw())
        assert rm.resolve(embodiment="humanoid", generator="ctrlworld") == "airbot_mmk2"
        assert rm.resolve(embodiment="humanoid", generator="dreamgen") == "fourier_gr1"

    def test_the_exact_bug_this_table_exists_to_fix(self):
        # `embodiment=humanoid` alone is ambiguous between two real robots;
        # `resolve` must disambiguate via `generator`, never return the
        # wrong one or silently mix clips from both.
        rm = parse_robot_map(_raw())
        assert rm.resolve(embodiment="humanoid", generator="ctrlworld") != \
            rm.resolve(embodiment="humanoid", generator="dreamdojo")

    def test_unclaimed_pair_resolves_to_none_not_an_error(self):
        rm = parse_robot_map(_raw())
        assert rm.resolve(embodiment="bimanual", generator="ctrlworld") is None
        assert rm.resolve(embodiment="humanoid", generator="cosmos") is None

    def test_embodiment_of_and_generators_of_raise_keyerror_for_unknown_robot(self):
        rm = parse_robot_map(_raw())
        with pytest.raises(KeyError):
            rm.embodiment_of("not_a_robot")
        with pytest.raises(KeyError):
            rm.generators_of("not_a_robot")


class TestLoadFromYaml:
    def test_round_trip_through_a_written_yaml_file(self, tmp_path):
        import yaml

        path = tmp_path / "robot_map.yaml"
        path.write_text(yaml.safe_dump(_raw()))
        rm = load_robot_map(path)
        assert rm.embodiment_of("fourier_gr1") == "humanoid"

    def test_the_committed_configs_robot_map_yaml_loads_cleanly(self):
        rm = load_robot_map(REPO_ROOT / "configs" / "robot_map.yaml")
        # The exact four-robot table this task's restructuring is about --
        # see kinescore.bench.cell's module docstring.
        assert set(rm.robots) == {
            "fourier_gr1", "airbot_mmk2", "franka_panda", "aloha_bimanual"}
        assert rm.resolve(embodiment="humanoid", generator="ctrlworld") == "airbot_mmk2"
        assert rm.resolve(embodiment="humanoid", generator="dreamgen") == "fourier_gr1"
        assert rm.resolve(embodiment="humanoid", generator="dreamdojo") == "fourier_gr1"
        assert rm.resolve(embodiment="single_arm", generator="ctrlworld") == "franka_panda"
        assert rm.resolve(embodiment="bimanual", generator="dreamgen") == "aloha_bimanual"
