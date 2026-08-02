"""kinescore.bench.matrix: expand a BenchConfig into cells (and only this module knows the grid exists)."""
from __future__ import annotations

import pytest

from kinescore.bench.cell import Cell
from kinescore.bench.config import AXIS_VALUES, ConfigError, from_dict
from kinescore.bench.matrix import (
    allow_patterns,
    cell_row,
    expand,
    matches_only,
    na_cells,
    parse_only_filters,
)
from kinescore.bench.robot_map import RobotEntry, RobotMap
from kinescore.core.clip import ViewLayout

_ROBOT_A = "franka_panda"  # -> embodiment "single_arm" in the fixture robot_map
_ROBOT_B = "airbot_mmk2"   # -> embodiment "humanoid" in the fixture robot_map


def _touch_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def _robot_map(**overrides: RobotEntry) -> RobotMap:
    # Both fixture robots claim all three generators by default, matching
    # the old embodiment-keyed fixture's arithmetic (no extra N/A cells
    # introduced by the robot table itself) -- individual tests override
    # a robot's `generators` to exercise the auto-N/A behaviour.
    robots = {
        _ROBOT_A: RobotEntry(embodiment="single_arm",
                             generators=("ctrlworld", "dreamgen", "dreamdojo")),
        _ROBOT_B: RobotEntry(embodiment="humanoid",
                             generators=("ctrlworld", "dreamgen", "dreamdojo")),
    }
    robots.update(overrides)
    return RobotMap(robots=robots)


def _raw(**overrides) -> dict:
    raw = {
        "run_id": "t",
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
    raw.update(overrides)
    return raw


class TestExpand:
    def test_cell_count_is_the_cartesian_product_minus_na_cells(self):
        config = from_dict(_raw())
        rm = _robot_map()
        # 2 robot x 2 view x 2 horizon x 1 cache x 3 generator = 24 raw
        # combinations; na_cells removes generator=dreamdojo&view=multiview
        # (2*1*2*1*1=4) and generator=ctrlworld&view=singleview (2*1*2*1*1=4).
        cells = expand(config, rm)
        assert len(cells) == 24 - 4 - 4

    def test_no_cell_matches_a_declared_na_rule(self):
        config = from_dict(_raw())
        rm = _robot_map()
        for cell in expand(config, rm):
            assert not (cell.generator == "dreamdojo" and cell.view == "multiview")
            assert not (cell.generator == "ctrlworld" and cell.view == "singleview")

    def test_every_cell_is_a_unique_five_axis_combination(self):
        config = from_dict(_raw())
        rm = _robot_map()
        cells = expand(config, rm)
        keys = {(c.robot, c.view, c.horizon, c.cache, c.generator) for c in cells}
        assert len(keys) == len(cells)

    def test_cell_carries_resolved_embodiment_from_the_robot_map(self):
        config = from_dict(_raw())
        rm = _robot_map()
        cells = expand(config, rm)
        robot_a_cell = next(c for c in cells if c.robot == _ROBOT_A)
        assert robot_a_cell.embodiment == "single_arm"
        robot_b_cell = next(c for c in cells if c.robot == _ROBOT_B)
        assert robot_b_cell.embodiment == "humanoid"

    def test_a_robot_generator_pair_the_robot_map_does_not_claim_is_auto_na(self):
        # airbot_mmk2 restricted to ctrlworld only (the real dataset shape --
        # see kinescore.bench.cell's module docstring) must silently drop
        # every dreamgen/dreamdojo cell for it, with NO na_cells rule needed.
        config = from_dict(_raw())
        rm = _robot_map(**{_ROBOT_B: RobotEntry(embodiment="humanoid",
                                                generators=("ctrlworld",))})
        cells = expand(config, rm)
        assert not any(c.robot == _ROBOT_B and c.generator != "ctrlworld" for c in cells)
        assert any(c.robot == _ROBOT_B and c.generator == "ctrlworld" for c in cells)

    def test_view_layout_matches_the_shared_dataset_wide_convention(self):
        config = from_dict(_raw())
        rm = _robot_map()
        cells = expand(config, rm)
        single = next(c for c in cells if c.view == "singleview")
        multi = next(c for c in cells if c.view == "multiview")
        assert single.view_layout == ViewLayout(n_views=1)
        assert multi.view_layout.n_views == 3
        assert multi.view_layout.key == "3x?:exterior_1+exterior_2+wrist"

    def test_cell_id_is_a_slash_joined_short_identity_in_path_axis_order(self):
        config = from_dict(_raw())
        rm = _robot_map()
        cell = expand(config, rm)[0]
        assert cell.cell_id == (f"{cell.cache}/{cell.robot}/{cell.view}/"
                                f"{cell.generator}/{cell.horizon}")

    def test_family_is_stable_and_parseable(self):
        config = from_dict(_raw())
        rm = _robot_map()
        cell = expand(config, rm)[0]
        parsed = dict(kv.split("=") for kv in cell.family.split("|"))
        assert parsed == {
            "cache": cell.cache, "robot": cell.robot, "view": cell.view,
            "generator": cell.generator, "horizon": cell.horizon,
            "embodiment": cell.embodiment,
        }

    def test_restricting_axes_shrinks_the_grid(self):
        raw = _raw()
        raw["axes"]["generator"] = ["ctrlworld"]
        raw["axes"]["view"] = ["multiview"]
        config = from_dict(raw)
        cells = expand(config, _robot_map())
        assert cells and all(c.generator == "ctrlworld" for c in cells)
        assert all(c.view == "multiview" for c in cells)


class TestNaCells:
    def test_na_cells_returns_exactly_the_excluded_cells(self):
        config = from_dict(_raw())
        rm = _robot_map()
        excluded = na_cells(config, rm)
        included = expand(config, rm)
        assert len(excluded) == 8  # see TestExpand's arithmetic
        assert {c.generator for c in excluded} <= {"dreamdojo", "ctrlworld"}
        included_keys = {(c.robot, c.view, c.horizon, c.cache, c.generator)
                         for c in included}
        excluded_keys = {(c.robot, c.view, c.horizon, c.cache, c.generator)
                         for c in excluded}
        assert included_keys.isdisjoint(excluded_keys)

    def test_na_cell_still_carries_a_resolved_embodiment_and_view_layout(self):
        # N/A cells are reported, not dropped from view -- they need the
        # same shape as a real cell so a report can print an N/A row.
        config = from_dict(_raw())
        rm = _robot_map()
        na = na_cells(config, rm)[0]
        assert isinstance(na, Cell)
        assert na.embodiment
        assert na.view_layout.n_views >= 1

    def test_no_na_cells_declared_means_only_robot_map_exclusions_remain(self):
        raw = _raw()
        del raw["na_cells"]
        config = from_dict(raw)
        rm = _robot_map()
        assert na_cells(config, rm) == []
        assert len(expand(config, rm)) == 24


class TestPartialNaMatch:
    def test_na_rule_matches_regardless_of_the_other_axes(self):
        raw = _raw()
        raw["na_cells"] = [{"cache": "dense"}]  # matches every cell
        config = from_dict(raw)
        rm = _robot_map()
        assert expand(config, rm) == []
        assert len(na_cells(config, rm)) == 24


class TestCellResolvesIter:
    def test_flat_string_iter_resolves_the_same_for_every_cell(self):
        config = from_dict(_raw())  # sources.dreamdojo.iter == "iter_000050000"
        rm = _robot_map()
        cells = expand(config, rm)
        dreamdojo_cells = [c for c in cells if c.generator == "dreamdojo"]
        assert dreamdojo_cells  # sanity: the fixture's na_cells doesn't drop all of them
        assert all(c.iter == "iter_000050000" for c in dreamdojo_cells)

    def test_nested_iter_resolves_per_robot_and_horizon(self):
        raw = _raw()
        raw["sources"]["dreamdojo"]["iter"] = {
            _ROBOT_B: {"makovian": "iter_000050000", "non_makovian": "iter_000060000"},
            _ROBOT_A: {"makovian": "iter_000030000", "non_makovian": "iter_000030000"},
        }
        config = from_dict(raw)
        rm = _robot_map()
        by_cell = {(c.robot, c.horizon): c.iter
                  for c in expand(config, rm) if c.generator == "dreamdojo"}
        assert by_cell[(_ROBOT_B, "makovian")] == "iter_000050000"
        assert by_cell[(_ROBOT_B, "non_makovian")] == "iter_000060000"
        assert by_cell[(_ROBOT_A, "makovian")] == "iter_000030000"

    def test_ctrlworld_cell_iter_is_always_none(self):
        config = from_dict(_raw())
        rm = _robot_map()
        ctrlworld_cells = [c for c in expand(config, rm) if c.generator == "ctrlworld"]
        assert ctrlworld_cells
        assert all(c.iter is None for c in ctrlworld_cells)

    def test_cell_constructed_by_hand_defaults_iter_to_none(self):
        # Existing call sites (bench.sources tests) construct Cell directly
        # without an iter kwarg -- must keep working unchanged.
        cell = Cell(cache="dense", robot=_ROBOT_B, view="multiview",
                   generator="ctrlworld", horizon="makovian", embodiment="humanoid",
                   view_layout=ViewLayout(n_views=3))
        assert cell.iter is None


class TestExpandHardValidation:
    """``expand(config, robot_map, data_root=...)`` hard-fails a cell whose
    resolved iter directory does not exist on disk -- the fix for the exact
    bug the plan describes: pinning ``iter_000113000`` (which existed for
    neither robot) produced zero rows silently instead of failing loudly.
    """

    def test_missing_data_root_arg_performs_no_disk_access(self, tmp_path):
        # Default behaviour (no data_root) must stay a pure function of
        # config -- required for callers that run before any data exists
        # (cli.cmd_data's allow_patterns path) and for every other test in
        # this file, none of which touch a filesystem.
        raw = _raw()
        raw["sources"]["dreamdojo"]["iter"] = "iter_999999999"  # never created anywhere
        config = from_dict(raw)
        cells = expand(config, _robot_map())  # no data_root -- must not raise
        assert any(c.generator == "dreamdojo" for c in cells)

    def test_existing_iter_dir_passes_validation(self, tmp_path):
        raw = _raw()
        raw["axes"]["generator"] = ["dreamdojo"]
        raw["axes"]["view"] = ["singleview"]
        config = from_dict(raw)
        rm = _robot_map()
        _touch_dir(tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
                  / "singleview" / "dreamdojo" / "makovian" / "iter_000050000")
        _touch_dir(tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
                  / "singleview" / "dreamdojo" / "non_makovian" / "iter_000050000")
        _touch_dir(tmp_path / "video_gen_physics" / "dense" / "single_arm" / "output"
                  / "singleview" / "dreamdojo" / "makovian" / "iter_000050000")
        _touch_dir(tmp_path / "video_gen_physics" / "dense" / "single_arm" / "output"
                  / "singleview" / "dreamdojo" / "non_makovian" / "iter_000050000")
        cells = expand(config, rm, data_root=tmp_path)
        assert any(c.generator == "dreamdojo" for c in cells)

    def test_missing_iter_dir_raises_naming_the_iters_that_do_exist(self, tmp_path):
        raw = _raw()
        raw["axes"]["robot"] = [_ROBOT_B]  # -> embodiment "humanoid", isolates the horizon_dir below
        raw["axes"]["generator"] = ["dreamdojo"]
        raw["axes"]["view"] = ["singleview"]
        raw["sources"]["dreamdojo"]["iter"] = "iter_000113000"  # does not exist
        config = from_dict(raw)
        rm = _robot_map()
        horizon_dir = (tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
                      / "singleview" / "dreamdojo" / "makovian")
        _touch_dir(horizon_dir / "iter_000050000")
        _touch_dir(horizon_dir / "iter_000090000")

        with pytest.raises(ConfigError) as exc_info:
            expand(config, rm, data_root=tmp_path)
        message = str(exc_info.value)
        assert "iter_000113000" in message
        assert "iter_000050000" in message
        assert "iter_000090000" in message

    def test_nested_iter_mapping_missing_a_cell_raises_at_expand_time(self, tmp_path):
        # A non-N/A cell whose nested iter mapping has no entry must fail
        # here -- this is the difference between "config bug" (this test)
        # and "declared N/A" (never reaches _validate_iter_on_disk at all).
        raw = _raw()
        raw["axes"]["robot"] = [_ROBOT_B]  # -> embodiment "humanoid", isolates the horizon_dir below
        raw["axes"]["generator"] = ["dreamdojo"]
        raw["axes"]["view"] = ["singleview"]
        raw["sources"]["dreamdojo"]["iter"] = {_ROBOT_B: {"makovian": "iter_000050000"}}
        config = from_dict(raw)
        rm = _robot_map()
        _touch_dir(tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
                  / "singleview" / "dreamdojo" / "makovian" / "iter_000050000")

        with pytest.raises(ConfigError, match="non_makovian"):
            expand(config, rm, data_root=tmp_path)

    def test_generator_with_no_iter_concept_is_never_checked(self, tmp_path):
        # ctrlworld's SourceConfig.iter is None -- expand(data_root=...) must
        # not require any ctrlworld directory to exist on disk to pass.
        raw = _raw()
        raw["axes"]["generator"] = ["ctrlworld"]
        raw["axes"]["view"] = ["multiview"]
        config = from_dict(raw)
        cells = expand(config, _robot_map(), data_root=tmp_path)  # tmp_path is empty
        assert cells and all(c.generator == "ctrlworld" for c in cells)


class TestThreeAxisNaRule:
    """``na_cells`` matches on however many axis keys are present -- proven
    here with all THREE of robot/horizon/generator at once, for the real gap
    the live inventory found: dreamgen has no non_makovian directory for
    franka_panda at all.
    """

    def _raw_with_extra_na(self):
        raw = _raw()
        raw["na_cells"].append(
            {"generator": "dreamgen", "horizon": "non_makovian", "robot": _ROBOT_A})
        return raw

    def test_three_axis_rule_excludes_only_the_exact_combination(self):
        raw = self._raw_with_extra_na()
        config = from_dict(raw)
        rm = _robot_map()
        cells = expand(config, rm)
        assert not any(c.generator == "dreamgen" and c.horizon == "non_makovian"
                      and c.robot == _ROBOT_A for c in cells)
        # The other three (robot, horizon) combinations for dreamgen must be
        # unaffected by the three-axis rule.
        assert any(c.generator == "dreamgen" and c.horizon == "non_makovian"
                  and c.robot == _ROBOT_B for c in cells)
        assert any(c.generator == "dreamgen" and c.horizon == "makovian"
                  and c.robot == _ROBOT_A for c in cells)

    def test_three_axis_rule_appears_in_na_cells(self):
        raw = self._raw_with_extra_na()
        config = from_dict(raw)
        rm = _robot_map()
        na = na_cells(config, rm)
        assert any(c.generator == "dreamgen" and c.horizon == "non_makovian"
                  and c.robot == _ROBOT_A for c in na)

    def test_three_axis_rule_does_not_affect_ctrlworld_or_dreamdojo(self):
        # A 3-key rule naming (dreamgen, non_makovian, robot_a) must not
        # accidentally match cells that share only 1 or 2 of those values.
        raw = self._raw_with_extra_na()
        config = from_dict(raw)
        rm = _robot_map()
        cells = expand(config, rm)
        assert any(c.generator == "ctrlworld" and c.horizon == "non_makovian"
                  and c.robot == _ROBOT_A for c in cells)


class TestAllowPatterns:
    def test_dense_only_config_never_mentions_another_cache_name(self):
        config = from_dict(_raw())  # axes.cache == ["dense"]
        patterns = allow_patterns(config, _robot_map())
        other_caches = AXIS_VALUES["cache"] - {"dense"}
        for pattern in patterns:
            for other in other_caches:
                assert f"/{other}/" not in pattern, (other, pattern)

    def test_patterns_are_scoped_to_the_pinned_view_dir_not_a_glob(self):
        # The only "*" allowed is the trailing "/**" that scopes everything
        # below one resolved cell's directory -- view_dir/generator/horizon
        # segments themselves must be literal, never a wildcard.
        config = from_dict(_raw())
        patterns = allow_patterns(config, _robot_map())
        assert any("/singleview/" in p for p in patterns)
        assert any("/multiview/" in p for p in patterns)
        for pattern in patterns:
            assert pattern.endswith("/**")
            assert "*" not in pattern[: -len("/**")]

    def test_dreamgen_gt_from_input_adds_a_baseline_cache_input_pattern(self):
        config = from_dict(_raw())
        patterns = allow_patterns(config, _robot_map())
        assert any("/dense/" in p and "/input/" in p and "dreamgen" in p
                  for p in patterns)

    def test_multiple_requested_cache_values_all_appear_and_nothing_else_does(self):
        raw = _raw()
        raw["axes"]["cache"] = ["dense", "worldcache"]
        raw["baseline_cache"] = "dense"
        config = from_dict(raw)
        patterns = allow_patterns(config, _robot_map())
        mentioned = {p.split("/")[1] for p in patterns}
        assert mentioned == {"dense", "worldcache"}


class TestCellRow:
    """Direct, argparse-free coverage for the row-building/filtering helpers
    ``kinescore bench run --cells-out``/``--only`` uses -- formerly private
    to ``cli/cmd_bench.py``, moved here since neither takes an
    ``argparse.Namespace`` and both are genuinely cell-table logic.
    """

    def test_row_carries_the_cell_and_robot_config_fields(self):
        config = from_dict(_raw())
        rm = _robot_map()
        cell = expand(config, rm)[0]
        row = cell_row(cell, config, status="pending")
        assert row["robot"] == cell.robot
        assert row["cell_id"] == cell.cell_id
        assert row["status"] == "pending"
        assert "n_rows" not in row

    def test_n_rows_included_only_when_given(self):
        config = from_dict(_raw())
        rm = _robot_map()
        cell = expand(config, rm)[0]
        row = cell_row(cell, config, status="scored", n_rows=7)
        assert row["n_rows"] == 7


class TestParseOnlyFilters:
    def test_empty_only_is_no_filters(self):
        assert parse_only_filters(None, AXIS_VALUES) == []
        assert parse_only_filters([], AXIS_VALUES) == []

    def test_parses_axis_value_pairs(self):
        filters = parse_only_filters(["cache=dense"], AXIS_VALUES)
        assert filters == [("cache", "dense")]

    def test_missing_equals_sign_raises(self):
        with pytest.raises(ValueError, match="AXIS=VALUE"):
            parse_only_filters(["cache_dense"], AXIS_VALUES)

    def test_unknown_axis_raises(self):
        with pytest.raises(ValueError, match="unknown axis"):
            parse_only_filters(["not_an_axis=x"], AXIS_VALUES)

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError, match="unknown cache value"):
            parse_only_filters(["cache=not_a_cache_value"], AXIS_VALUES)


class TestMatchesOnly:
    def test_no_filters_matches_everything(self):
        config = from_dict(_raw())
        rm = _robot_map()
        cell = expand(config, rm)[0]
        assert matches_only(cell, []) is True

    def test_matching_filter(self):
        config = from_dict(_raw())
        rm = _robot_map()
        cell = expand(config, rm)[0]
        assert matches_only(cell, [("cache", cell.cache)]) is True

    def test_non_matching_filter(self):
        config = from_dict(_raw())
        rm = _robot_map()
        cell = expand(config, rm)[0]
        assert matches_only(cell, [("robot", "definitely_not_this_robot")]) is False
