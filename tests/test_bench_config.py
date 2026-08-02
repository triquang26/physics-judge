"""kinescore.bench.config: validate benchmark.yaml hard and early.

Every test here works on a plain dict via ``from_dict`` -- no file I/O needed
to exercise validation -- except the round-trip tests at the bottom, which
prove :func:`load_config` reads real YAML (including the repo's own
``configs/benchmark.yaml``) into the same shape.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from kinescore.bench.config import (
    AXIS_VALUES,
    VIEW_DIR_VALUES,
    BenchConfig,
    ConfigError,
    RobotConfig,
    from_dict,
    load_config,
)
from kinescore.paths import MissingPathError, optional_env_path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Two registered robots (see kinescore.robots.available_robots()) used
#: throughout instead of the old embodiment-keyed fixture -- picked because
#: neither needs pytorch_kinematics to construct (synthetic_2r is the only
#: other zero-dependency one, and franka_panda's own fixture already exists
#: in kinescore.robots without any extra setup).
_ROBOT_A = "franka_panda"
_ROBOT_B = "airbot_mmk2"


def _valid_raw() -> dict:
    return {
        "run_id": "dense_two_robots",
        "seed": 0,
        "axes": {
            "robot": [_ROBOT_A, _ROBOT_B],
            "view": ["multiview", "singleview"],
            "horizon": ["makovian", "non_makovian"],
            "cache": ["dense"],
            "generator": ["ctrlworld", "dreamgen", "dreamdojo"],
        },
        "na_cells": [
            {"generator": "dreamdojo", "view": "multiview"},
            {"generator": "ctrlworld", "view": "singleview"},
        ],
        "robots": {
            _ROBOT_A: {"spec": _ROBOT_A, "reader": "a.pt", "assets": "a"},
            _ROBOT_B: {"spec": _ROBOT_B, "reader": "b.pt", "assets": "b"},
        },
        "sources": {
            "dreamgen": {"iter": "iter_000113000", "view_dir": "singleview",
                        "gt_from": "input"},
            "dreamdojo": {"iter": "iter_000050000", "view_dir": "singleview"},
            "ctrlworld": {"view_dir": "multiview"},
        },
        "fps_expected": {"dreamgen": 16.0, "dreamdojo": 10.0},
        "rate_policy": "paired",
        "suite": "invariant_v1",
        "baseline_cache": "dense",
        "caps": {"episodes_per_cell": 150},
    }


class TestValidConfig:
    def test_parses_into_frozen_dataclass(self):
        config = from_dict(_valid_raw())
        assert isinstance(config, BenchConfig)
        assert config.run_id == "dense_two_robots"
        assert config.axes.robot == (_ROBOT_A, _ROBOT_B)
        assert config.axes.generator == ("ctrlworld", "dreamgen", "dreamdojo")
        assert len(config.na_cells) == 2
        assert config.robots[_ROBOT_A].spec == _ROBOT_A
        assert config.sources["dreamgen"].gt_from == "input"
        assert config.sources["ctrlworld"].iter is None
        assert config.fps_expected == {"dreamgen": 16.0, "dreamdojo": 10.0}
        assert config.baseline_cache == "dense"
        assert config.caps == {"episodes_per_cell": 150}

    def test_is_frozen(self):
        config = from_dict(_valid_raw())
        with pytest.raises(Exception):  # noqa: B017 -- dataclasses.FrozenInstanceError
            config.run_id = "other"  # type: ignore[misc]

    def test_rate_policy_resample_form_is_accepted(self):
        raw = _valid_raw()
        raw["rate_policy"] = "resample:10"
        config = from_dict(raw)
        assert config.rate_policy == "resample:10"

    def test_view_axis_never_includes_the_disk_only_single_view_spelling(self):
        # "single_view" is a real ON-DISK directory name (see
        # VIEW_DIR_VALUES) but must never be a valid `view` AXIS value --
        # the axis has exactly two canonical members.
        assert AXIS_VALUES["view"] == frozenset({"multiview", "singleview"})
        assert "single_view" in VIEW_DIR_VALUES
        assert "single_view" not in AXIS_VALUES["view"]

    def test_robot_axis_values_come_from_the_live_robot_registry(self):
        from kinescore.robots import available_robots
        assert AXIS_VALUES["robot"] == frozenset(available_robots())


class TestUnknownTopLevelKey:
    """``from_dict`` must reject a config root key it does not read.

    Regression test for the bug this module's docstring already promised to
    prevent but did not: every field was read via ``raw.get(...)``, so a
    typo like ``rate_polciy`` silently fell back to the ``rate_policy``
    default with no error.
    """

    def test_typo_d_key_is_rejected(self):
        raw = _valid_raw()
        raw["rate_polciy"] = "paired"
        with pytest.raises(ConfigError, match="unknown top-level key"):
            from_dict(raw)

    def test_error_names_the_offending_key(self):
        raw = _valid_raw()
        raw["totally_bogus_key"] = 1
        with pytest.raises(ConfigError, match="totally_bogus_key"):
            from_dict(raw)

    def test_every_documented_key_is_accepted(self):
        # The converse: every key _valid_raw() sets is a real, accepted key
        # -- this would fail loudly if a future field rename forgot to
        # update _TOP_LEVEL_KEYS.
        from_dict(_valid_raw())  # must not raise


class TestAxesValidation:
    def test_unknown_axis_value_names_the_valid_set(self):
        raw = _valid_raw()
        raw["axes"]["robot"] = [_ROBOT_A, "quadruped_bot"]
        with pytest.raises(ConfigError, match=r"quadruped_bot.*valid values are"):
            from_dict(raw)

    def test_missing_axis_key_is_an_error(self):
        raw = _valid_raw()
        del raw["axes"]["cache"]
        with pytest.raises(ConfigError, match="axes:"):
            from_dict(raw)

    def test_empty_axis_list_is_an_error(self):
        raw = _valid_raw()
        raw["axes"]["cache"] = []
        with pytest.raises(ConfigError, match="axes.cache"):
            from_dict(raw)


class TestNaCellsValidation:
    def test_unknown_na_cell_axis_key_is_rejected(self):
        raw = _valid_raw()
        raw["na_cells"].append({"not_an_axis": "x"})
        with pytest.raises(ConfigError, match="unknown axis key"):
            from_dict(raw)

    def test_unknown_na_cell_axis_value_is_rejected(self):
        raw = _valid_raw()
        raw["na_cells"].append({"generator": "not_a_generator"})
        with pytest.raises(ConfigError, match="not a valid generator|not_a_generator"):
            from_dict(raw)

    def test_na_cells_may_be_omitted(self):
        raw = _valid_raw()
        del raw["na_cells"]
        config = from_dict(raw)
        assert config.na_cells == ()

    def test_na_cell_rule_matches_on_robot_key(self):
        from kinescore.bench.config import NaCellRule
        rule = NaCellRule(axes={"generator": "dreamdojo"})
        assert rule.matches(robot=_ROBOT_A, view="multiview", horizon="makovian",
                            cache="dense", generator="dreamdojo")
        assert not rule.matches(robot=_ROBOT_A, view="multiview", horizon="makovian",
                                cache="dense", generator="ctrlworld")


class TestSourcesValidation:
    def test_missing_source_for_axes_generator_is_an_error(self):
        raw = _valid_raw()
        del raw["sources"]["dreamdojo"]
        with pytest.raises(ConfigError, match="dreamdojo"):
            from_dict(raw)

    def test_iter_as_a_list_is_rejected(self):
        # The exact trap: two iter_ values silently mixing checkpoints into
        # one number must be a hard config error, not accepted as "a list of
        # iters to glob".
        raw = _valid_raw()
        raw["sources"]["dreamdojo"]["iter"] = ["iter_000050000", "iter_000090000"]
        with pytest.raises(ConfigError, match="single string"):
            from_dict(raw)

    def test_iter_as_a_list_nested_inside_the_per_cell_mapping_is_also_rejected(self):
        # The same trap, one level deeper: a list under robot->horizon must
        # be rejected exactly like a top-level list is.
        raw = _valid_raw()
        raw["sources"]["dreamdojo"]["iter"] = {
            _ROBOT_A: {"makovian": ["iter_000050000", "iter_000090000"]},
        }
        with pytest.raises(ConfigError, match="single string"):
            from_dict(raw)

    def test_iter_as_a_number_is_rejected(self):
        raw = _valid_raw()
        raw["sources"]["dreamdojo"]["iter"] = 50000
        with pytest.raises(ConfigError, match="single string"):
            from_dict(raw)

    def test_iter_as_a_nested_robot_horizon_mapping_is_accepted(self):
        raw = _valid_raw()
        raw["sources"]["dreamgen"]["iter"] = {
            _ROBOT_A: {"makovian": "iter_000090000", "non_makovian": "iter_000090000"},
            _ROBOT_B: {"makovian": "iter_000090000_static16fps"},
        }
        config = from_dict(raw)
        source = config.sources["dreamgen"]
        assert source.resolve_iter(robot=_ROBOT_A, horizon="makovian") == "iter_000090000"
        assert source.resolve_iter(robot=_ROBOT_A, horizon="non_makovian") == "iter_000090000"
        assert (source.resolve_iter(robot=_ROBOT_B, horizon="makovian")
               == "iter_000090000_static16fps")

    def test_nested_iter_mapping_missing_a_cell_resolves_to_none_not_an_error(self):
        # resolve_iter is a pure, always-succeeding lookup -- turning a
        # missing entry into a hard error is bench.matrix.expand's job (it
        # alone knows whether the missing cell is a declared N/A one).
        raw = _valid_raw()
        raw["sources"]["dreamgen"]["iter"] = {_ROBOT_A: {"makovian": "iter_000090000"}}
        config = from_dict(raw)
        source = config.sources["dreamgen"]
        assert source.resolve_iter(robot=_ROBOT_A, horizon="non_makovian") is None
        assert source.resolve_iter(robot=_ROBOT_B, horizon="makovian") is None

    def test_nested_iter_mapping_unknown_robot_key_is_rejected(self):
        raw = _valid_raw()
        raw["sources"]["dreamgen"]["iter"] = {"quadruped_bot": {"makovian": "iter_000090000"}}
        with pytest.raises(ConfigError, match="quadruped_bot"):
            from_dict(raw)

    def test_nested_iter_mapping_unknown_horizon_key_is_rejected(self):
        raw = _valid_raw()
        raw["sources"]["dreamgen"]["iter"] = {_ROBOT_A: {"quarterly": "iter_000090000"}}
        with pytest.raises(ConfigError, match="quarterly"):
            from_dict(raw)

    def test_flat_string_iter_still_works_via_resolve_iter(self):
        # Backward compatibility: a single string still applies to every
        # (robot, horizon) cell, unchanged from before this schema extension.
        raw = _valid_raw()
        config = from_dict(raw)
        source = config.sources["dreamgen"]
        assert source.resolve_iter(robot=_ROBOT_A, horizon="makovian") == source.iter
        assert source.resolve_iter(robot=_ROBOT_B, horizon="non_makovian") == source.iter

    def test_no_iter_at_all_resolves_to_none_for_every_cell(self):
        raw = _valid_raw()
        config = from_dict(raw)
        source = config.sources["ctrlworld"]
        assert source.iter is None
        assert source.resolve_iter(robot=_ROBOT_A, horizon="makovian") is None

    def test_bad_view_dir_is_rejected(self):
        raw = _valid_raw()
        raw["sources"]["ctrlworld"]["view_dir"] = "both_views"
        with pytest.raises(ConfigError, match="view_dir"):
            from_dict(raw)

    def test_gt_from_other_than_input_is_rejected(self):
        raw = _valid_raw()
        raw["sources"]["dreamgen"]["gt_from"] = "output"
        with pytest.raises(ConfigError, match="gt_from"):
            from_dict(raw)


class TestRobotsValidation:
    def test_missing_robot_for_axes_robot_is_an_error(self):
        raw = _valid_raw()
        del raw["robots"][_ROBOT_B]
        with pytest.raises(ConfigError, match=_ROBOT_B):
            from_dict(raw)

    def test_incomplete_robot_entry_is_an_error(self):
        raw = _valid_raw()
        del raw["robots"][_ROBOT_A]["assets"]
        with pytest.raises(ConfigError, match="spec, reader, assets"):
            from_dict(raw)

    def test_unregistered_robot_key_is_rejected(self):
        raw = _valid_raw()
        raw["axes"]["robot"] = [_ROBOT_A]
        raw["robots"] = {"not_a_real_robot": {"spec": _ROBOT_A, "reader": "a.pt", "assets": "a"}}
        with pytest.raises(ConfigError, match="not a registered robot"):
            from_dict(raw)

    def test_unregistered_spec_is_rejected(self):
        # The exact bug this validation exists to close: `spec` used to
        # accept any free string (e.g. "aloha", never a registered robot
        # package) and silently score nothing.
        raw = _valid_raw()
        raw["robots"][_ROBOT_A]["spec"] = "aloha"
        with pytest.raises(ConfigError, match="not a registered robot"):
            from_dict(raw)

    def test_spec_need_not_equal_the_mapping_key_as_long_as_both_are_registered(self):
        raw = _valid_raw()
        raw["robots"][_ROBOT_A]["spec"] = _ROBOT_B  # a different, but real, registry key
        config = from_dict(raw)
        assert config.robots[_ROBOT_A].spec == _ROBOT_B


class TestReaderStatus:
    """``robots.<robot>.reader_status`` -- the schema-level marking that
    replaced naming a checkpoint filename that may or may not exist on disk.

    Added after ``configs/benchmark.yaml`` shipped ``franka_panda: {reader:
    single_arm.pt}`` and ``aloha_bimanual: {reader: bimanual.pt}`` -- neither
    file ever existed, and ``kinescore.bench.config._parse_robots`` had no
    way to say so; it only validated ``reader`` was *a string*, never that it
    named something real or was honestly marked otherwise. See
    ``kinescore.bench.config.RobotConfig.reader_status``.
    """

    def test_default_status_is_accepted(self):
        config = from_dict(_valid_raw())
        assert config.robots[_ROBOT_A].reader_status == "accepted"
        assert config.robots[_ROBOT_A].reader_note is None

    def test_unknown_reader_status_is_rejected(self):
        raw = _valid_raw()
        raw["robots"][_ROBOT_A]["reader_status"] = "probably_fine"
        with pytest.raises(ConfigError, match="reader_status"):
            from_dict(raw)

    def test_untrained_status_requires_reader_to_be_null(self):
        # The exact bug this field exists to close: naming a filename for a
        # checkpoint that was never trained is a phantom path.
        raw = _valid_raw()
        raw["robots"][_ROBOT_A]["reader_status"] = "untrained"
        # reader is still "a.pt" from _valid_raw() -- not null.
        with pytest.raises(ConfigError, match="untrained"):
            from_dict(raw)

    def test_untrained_status_with_null_reader_is_accepted(self):
        raw = _valid_raw()
        raw["robots"][_ROBOT_A]["reader_status"] = "untrained"
        raw["robots"][_ROBOT_A]["reader"] = None
        config = from_dict(raw)
        assert config.robots[_ROBOT_A].reader is None
        assert config.robots[_ROBOT_A].reader_status == "untrained"

    def test_failing_gate_status_keeps_a_real_reader_filename(self):
        raw = _valid_raw()
        raw["robots"][_ROBOT_A]["reader_status"] = "failing_gate"
        raw["robots"][_ROBOT_A]["reader_note"] = "val 162.10mm vs ~19-20mm band"
        config = from_dict(raw)
        assert config.robots[_ROBOT_A].reader == "a.pt"
        assert config.robots[_ROBOT_A].reader_status == "failing_gate"
        assert "162.10mm" in config.robots[_ROBOT_A].reader_note

    def test_accepted_or_failing_gate_status_rejects_a_null_reader(self):
        # The converse of test_untrained_status_requires_reader_to_be_null:
        # a null reader with a status other than "untrained" is just as
        # dishonest the other way (claims a working/failing checkpoint that
        # does not exist).
        raw = _valid_raw()
        raw["robots"][_ROBOT_A]["reader"] = None
        with pytest.raises(ConfigError, match="non-empty checkpoint"):
            from_dict(raw)

    def test_reader_note_must_be_a_string_if_given(self):
        raw = _valid_raw()
        raw["robots"][_ROBOT_A]["reader_note"] = 42
        with pytest.raises(ConfigError, match="reader_note"):
            from_dict(raw)

    def test_unknown_key_in_a_robot_entry_is_rejected(self):
        raw = _valid_raw()
        raw["robots"][_ROBOT_A]["reader_stauts"] = "accepted"  # typo
        with pytest.raises(ConfigError, match="unknown key"):
            from_dict(raw)


class TestFpsExpectedValidation:
    @pytest.mark.parametrize("bad", [0.0, -1.0, "16.0", None, True])
    def test_non_positive_or_non_numeric_fps_is_rejected(self, bad):
        raw = _valid_raw()
        raw["fps_expected"]["dreamgen"] = bad
        with pytest.raises(ConfigError, match="positive float"):
            from_dict(raw)

    def test_fps_expected_may_be_omitted(self):
        raw = _valid_raw()
        del raw["fps_expected"]
        config = from_dict(raw)
        assert config.fps_expected == {}

    def test_nested_per_robot_fps_resolves_by_robot(self):
        raw = _valid_raw()
        raw["fps_expected"]["dreamdojo"] = {_ROBOT_A: 15.0, _ROBOT_B: 10.0}
        config = from_dict(raw)
        assert config.resolve_fps(generator="dreamdojo", robot=_ROBOT_A) == 15.0
        assert config.resolve_fps(generator="dreamdojo", robot=_ROBOT_B) == 10.0

    def test_nested_fps_unknown_robot_key_is_rejected(self):
        raw = _valid_raw()
        raw["fps_expected"]["dreamdojo"] = {"quadruped_bot": 10.0}
        with pytest.raises(ConfigError, match="quadruped_bot"):
            from_dict(raw)


class TestRatePolicyAndBaselineCache:
    def test_bad_rate_policy_is_rejected(self):
        raw = _valid_raw()
        raw["rate_policy"] = "sometimes"
        with pytest.raises(ConfigError, match="rate_policy"):
            from_dict(raw)

    def test_baseline_cache_must_be_in_axes_cache(self):
        raw = _valid_raw()
        raw["baseline_cache"] = "dicache"
        with pytest.raises(ConfigError, match="baseline_cache"):
            from_dict(raw)


class TestCapsValidation:
    def test_non_positive_cap_is_rejected(self):
        raw = _valid_raw()
        raw["caps"]["episodes_per_cell"] = 0
        with pytest.raises(ConfigError, match="positive integer"):
            from_dict(raw)

    def test_caps_may_be_omitted(self):
        raw = _valid_raw()
        del raw["caps"]
        config = from_dict(raw)
        assert config.caps == {}


class TestEnvVarExpansion:
    def test_known_kinescore_var_expands_via_paths(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KINESCORE_ASSETS", str(tmp_path))
        raw = _valid_raw()
        raw["robots"][_ROBOT_A]["assets"] = "${KINESCORE_ASSETS}/grx"
        config = from_dict(raw)
        assert config.robots[_ROBOT_A].assets == str(tmp_path.resolve()) + "/grx"

    def test_unset_known_var_raises_missing_path_error(self, monkeypatch):
        monkeypatch.delenv("KINESCORE_ASSETS", raising=False)
        raw = _valid_raw()
        raw["robots"][_ROBOT_A]["assets"] = "${KINESCORE_ASSETS}/grx"
        with pytest.raises(MissingPathError, match="KINESCORE_ASSETS"):
            from_dict(raw)

    def test_unset_arbitrary_var_also_raises(self, monkeypatch):
        monkeypatch.delenv("SOME_UNSET_VAR", raising=False)
        raw = _valid_raw()
        raw["run_id"] = "${SOME_UNSET_VAR}"
        with pytest.raises(MissingPathError, match="SOME_UNSET_VAR"):
            from_dict(raw)

    def test_set_arbitrary_var_expands_from_os_environ(self, monkeypatch):
        monkeypatch.setenv("SOME_SET_VAR", "custom_run")
        raw = _valid_raw()
        raw["run_id"] = "${SOME_SET_VAR}"
        config = from_dict(raw)
        assert config.run_id == "custom_run"

    def test_deep_copy_is_untouched_by_expansion(self):
        # _expand_env must not mutate the caller's dict in place in a way
        # that surprises a caller reusing the same raw dict.
        raw = _valid_raw()
        before = copy.deepcopy(raw)
        from_dict(raw)
        assert raw == before


class TestLoadConfigFromYaml:
    def test_round_trip_through_a_written_yaml_file(self, tmp_path):
        import yaml

        raw = _valid_raw()
        path = tmp_path / "benchmark.yaml"
        path.write_text(yaml.safe_dump(raw))
        config = load_config(path)
        assert config.run_id == raw["run_id"]
        assert config.path == str(path)

    def test_the_committed_configs_benchmark_yaml_loads_cleanly(self):
        # The concrete first-run config this task ships must itself validate
        # -- no env vars referenced, so no environment setup needed.
        config = load_config(REPO_ROOT / "configs" / "benchmark.yaml")
        assert config.run_id
        assert set(config.axes.generator) <= AXIS_VALUES["generator"]
        assert config.na_cells
        # All four registry robots are covered now that aloha_bimanual has a
        # real spec/reader entry (kinescore.robots.available_robots()
        # includes it) -- see the reconciliation task that folded it in from
        # configs/benchmark_bimanual.yaml.
        assert set(config.axes.robot) == {
            "fourier_gr1", "airbot_mmk2", "franka_panda", "aloha_bimanual"}

    def test_aloha_bimanual_cells_are_real_not_na_and_use_the_pinned_iters(self):
        # Fix-4 reconciliation pin: aloha_bimanual's three real cells
        # (ctrlworld x multiview, dreamgen/dreamdojo x singleview) must
        # expand as genuine cells -- not silently swallowed into na_cells --
        # with the measured HF-tree-crawl iters attached; its dreamgen x
        # multiview and ctrlworld x singleview slices must stay N/A, exactly
        # like the other three robots.
        from kinescore.bench.matrix import expand, na_cells
        from kinescore.bench.robot_map import load_robot_map

        config = load_config(REPO_ROOT / "configs" / "benchmark.yaml")
        robot_map = load_robot_map(REPO_ROOT / "configs" / "robot_map.yaml")
        cells = {c.cell_id: c for c in expand(config, robot_map)
                 if c.robot == "aloha_bimanual"}
        assert set(cells) == {
            "dense/aloha_bimanual/multiview/ctrlworld/makovian",
            "dense/aloha_bimanual/multiview/ctrlworld/non_makovian",
            "dense/aloha_bimanual/singleview/dreamgen/makovian",
            "dense/aloha_bimanual/singleview/dreamgen/non_makovian",
            "dense/aloha_bimanual/singleview/dreamdojo/makovian",
            "dense/aloha_bimanual/singleview/dreamdojo/non_makovian",
        }
        assert cells["dense/aloha_bimanual/singleview/dreamgen/makovian"].iter == \
            "iter_000110000_bimanual16fps"
        assert cells["dense/aloha_bimanual/singleview/dreamdojo/makovian"].iter == \
            "iter_000100000"

        na_ids = {c.cell_id for c in na_cells(config, robot_map)
                  if c.robot == "aloha_bimanual"}
        assert "dense/aloha_bimanual/multiview/dreamgen/makovian" in na_ids
        assert "dense/aloha_bimanual/singleview/ctrlworld/makovian" in na_ids

    def test_aloha_bimanual_allow_patterns_are_derived_for_data_pull(self):
        # The whole point of wiring aloha_bimanual into axes.robot: `kinescore
        # data pull` must now derive real bimanual/** patterns for it (it
        # derived none before this robot was in the axis) -- this is what
        # lets a future `data pull` actually fetch dense/bimanual/output/**,
        # which is not downloaded yet.
        from kinescore.bench.matrix import allow_patterns
        from kinescore.bench.robot_map import load_robot_map

        config = load_config(REPO_ROOT / "configs" / "benchmark.yaml")
        robot_map = load_robot_map(REPO_ROOT / "configs" / "robot_map.yaml")
        patterns = allow_patterns(config, robot_map)
        assert ("video_gen_physics/dense/bimanual/output/multiview/ctrlworld/"
                "makovian/**") in patterns
        assert ("video_gen_physics/dense/bimanual/output/singleview/dreamgen/"
                "makovian/**") in patterns
        # dreamdojo resolves through the "single_view" (underscored) literal
        # for aloha_bimanual specifically -- the per-robot view_dir override
        # (see kinescore.bench.config.SourceConfig.resolve_view_dir) that
        # picks the ONE dreamdojo subtree with real full_gt.mp4 pairs; the
        # "singleview" spelling is pred-only for this robot and would derive
        # a pattern for data that can never be scored (see
        # configs/benchmark.yaml's header comment, point 2).
        assert ("video_gen_physics/dense/bimanual/output/single_view/dreamdojo/"
                "makovian/**") in patterns
        assert ("video_gen_physics/dense/bimanual/output/singleview/dreamdojo/"
                "makovian/**") not in patterns

    def test_the_bimanual_template_config_also_loads_cleanly_if_robot_is_registered(self):
        # Not run by default; validates only once kinescore.robots registers
        # "aloha_bimanual" (it does not yet -- see that config's own header
        # comment). Skipped, not xfail'd, so this file does not need
        # updating the day that robot lands; it will just start running.
        from kinescore.robots import available_robots
        if "aloha_bimanual" not in available_robots():
            pytest.skip("aloha_bimanual not yet registered in kinescore.robots")
        config = load_config(REPO_ROOT / "configs" / "benchmark_bimanual.yaml")
        assert set(config.axes.robot) == {"aloha_bimanual"}
        assert config.robots["aloha_bimanual"].spec == "aloha_bimanual"

    def test_the_bimanual_template_config_yaml_is_at_least_well_formed(self):
        # Even without the robot registered, the YAML itself must parse and
        # every axis value the file uses that ISN'T robot-registry-gated
        # must be valid -- catches a typo anywhere else in the file.
        import yaml
        with open(REPO_ROOT / "configs" / "benchmark_bimanual.yaml") as f:
            raw = yaml.safe_load(f)
        assert raw["axes"]["robot"] == ["aloha_bimanual"]
        assert set(raw["axes"]["generator"]) <= AXIS_VALUES["generator"]
        assert set(raw["axes"]["view"]) <= AXIS_VALUES["view"]
        assert set(raw["axes"]["horizon"]) <= AXIS_VALUES["horizon"]


class TestShippedConfigsReadersAreResolvableOrHonestlyMarked:
    """Every ``robots.*.reader`` in every shipped bench config must either
    resolve to a checkpoint file :func:`kinescore.readers.load_reader`
    accepts, or be explicitly marked ``reader_status: failing_gate`` /
    ``untrained`` -- see ``kinescore.bench.config.RobotConfig.reader_status``.

    This is the regression pin for the exact incident that motivated the
    field: ``configs/benchmark.yaml`` shipped ``franka_panda: {reader:
    single_arm.pt}`` (deleted -- the squashed AttentivePoseHead reader) and
    ``aloha_bimanual: {reader: bimanual.pt}`` (never trained), neither
    resolvable, both silent about it -- a config a reader trusted at face
    value. The set of configs to check is discovered from the filesystem
    (every ``configs/benchmark*.yaml`` that parses as a ``BenchConfig``),
    not hand-typed, so a future config added to this family is covered
    without editing this file.
    """

    @staticmethod
    def _shipped_bench_configs() -> list[Path]:
        candidates = sorted((REPO_ROOT / "configs").glob("benchmark*.yaml"))
        out = []
        for path in candidates:
            try:
                load_config(path)
            except Exception:
                continue  # not a BenchConfig-shaped file (or needs an unset env var)
            out.append(path)
        return out

    @staticmethod
    def _check_readers_resolvable(robots: dict, ckpt_dir: Path,
                                  *, source_name: str = "<config>") -> list[str]:
        """Pure checker: one ``{robot: RobotConfig}`` mapping against one
        checkpoint directory -> a list of problem strings (empty = clean).

        Kept separate from any single config/filesystem so
        :func:`test_a_phantom_reader_is_caught` can feed it a synthetic,
        deliberately-broken mapping and assert on exactly what comes back,
        the same call this class makes for real against every shipped
        config's actual ``robots``.
        """
        import torch

        from kinescore.readers import checkpoint_v2

        problems = []
        for robot, robot_cfg in robots.items():
            if robot_cfg.reader_status == "untrained":
                if robot_cfg.reader is not None:
                    problems.append(
                        f"{source_name}: robots.{robot} is reader_status="
                        f"'untrained' but names reader={robot_cfg.reader!r}")
                continue
            if not robot_cfg.reader:
                problems.append(
                    f"{source_name}: robots.{robot} is reader_status="
                    f"{robot_cfg.reader_status!r} but has no reader filename")
                continue
            ckpt_path = ckpt_dir / robot_cfg.reader
            if not ckpt_path.is_file():
                problems.append(
                    f"{source_name}: robots.{robot}.reader="
                    f"{robot_cfg.reader!r} does not exist at {ckpt_path}")
                continue
            ck = torch.load(str(ckpt_path), map_location="cpu")
            cfg = dict(ck.get("cfg", {}))
            if not checkpoint_v2.is_readout_v2_cfg(cfg):
                problems.append(
                    f"{source_name}: robots.{robot}.reader="
                    f"{robot_cfg.reader!r} exists but is not a ReadoutV2Head "
                    f"checkpoint -- load_reader would raise "
                    f"NotImplementedError on it")
        return problems

    def test_discovery_finds_at_least_the_two_known_configs(self):
        # Guards the glob itself: if this ever finds zero files, every check
        # below passes vacuously and hides a real regression.
        names = {p.name for p in self._shipped_bench_configs()}
        assert {"benchmark.yaml", "benchmark_bimanual.yaml"} <= names

    def test_every_named_reader_resolves_and_loads(self):
        # Filesystem-facing: every reader named as "accepted" or
        # "failing_gate" (i.e. every reader that is NOT "untrained") must
        # actually exist under KINESCORE_CKPT_DIR and be a checkpoint format
        # kinescore.readers.load_reader can route (not the removed
        # AttentivePoseHead/SquashedPoseReader format -- see
        # legacy_docs/PROVENANCE.md's D7 addendum). Skips (does not silently
        # pass) if KINESCORE_CKPT_DIR is unset -- there is nothing to check
        # against without it.
        ckpt_dir = optional_env_path("KINESCORE_CKPT_DIR")
        if ckpt_dir is None:
            pytest.skip("KINESCORE_CKPT_DIR not set -- cannot verify "
                       "checkpoint files on disk")

        problems = []
        for path in self._shipped_bench_configs():
            config = load_config(path)
            problems += self._check_readers_resolvable(
                config.robots, ckpt_dir, source_name=path.name)
        assert not problems, "\n".join(problems)

    def test_a_phantom_reader_is_caught(self, tmp_path):
        # The checker must actually fail a missing/untrained-but-named
        # checkpoint, not just pass real ones -- this is what would have
        # caught single_arm.pt / bimanual.pt before this pass fixed them.
        robots = {
            "franka_panda": RobotConfig(spec="franka_panda", reader="ghost.pt",
                                        assets="franka"),
            "aloha_bimanual": RobotConfig(spec="aloha_bimanual", reader="ghost2.pt",
                                          assets="aloha", reader_status="untrained"),
        }
        problems = self._check_readers_resolvable(robots, tmp_path,
                                                   source_name="synthetic.yaml")
        assert len(problems) == 2
        assert any("ghost.pt" in p and "does not exist" in p for p in problems)
        assert any("ghost2.pt" in p and "untrained" in p for p in problems)

    def test_a_real_checkpoint_passes(self, tmp_path):
        # Counterpart to the phantom-path test: the checker must not flag a
        # real, load_reader-routable checkpoint. Builds one via
        # kinescore.readers.checkpoint_v2.save (the same helper
        # tests/test_checkpoint_v2.py uses) rather than depending on any
        # checkpoint file actually existing on this machine, so it runs even
        # when KINESCORE_CKPT_DIR is unset.
        pytest.importorskip("torch")
        from kinescore.core.clip import ViewLayout
        from kinescore.heads.heteroscedastic import ReadoutV2Head
        from kinescore.readers.checkpoint_v2 import save

        head = ReadoutV2Head(in_dim=12, d_model=16, n_heads=2, temporal_nhead=2,
                             ff=16, n_temporal_layers=1, t_max=8, n_out=8)
        ckpt_path = tmp_path / "real.pt"
        save(str(ckpt_path), head, view_layout=ViewLayout(n_views=1),
            robot_name="franka_panda")
        robots = {"franka_panda": RobotConfig(spec="franka_panda", reader="real.pt",
                                              assets="franka")}
        assert self._check_readers_resolvable(robots, tmp_path) == []
