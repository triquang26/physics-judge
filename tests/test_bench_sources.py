"""kinescore.bench.sources: pure discovery over synthetic directory trees.

Every source here only globs paths and yields
:class:`~kinescore.bench.manifest.DiscoveredClip` -- it never probes a real
video file (probing is centralised in
:func:`kinescore.bench.manifest.build_manifest`, per that module's
docstring). So these tests touch no ffmpeg/ffprobe and stay in the default
CPU-only, network-free tier: every "clip" is an empty file with a ``.mp4``
name, since only its *path* is ever inspected here.

``cosmos``/``lerobot`` (no ``Cell``, not part of the generator matrix) were
deleted as orphaned discovery code during the class-based port -- see
``kinescore.bench.sources``'s module docstring -- so their test classes are
gone too, not adapted.
"""
from __future__ import annotations

import pytest

from kinescore.bench.cell import Cell
from kinescore.bench.config import from_dict
from kinescore.bench.sources import (
    CtrlWorldSource,
    DreamDojoSource,
    DreamGenSource,
    dreamgen as dreamgen_mod,
)
from kinescore.bench.sources.ctrlworld import _VIEW_LAYOUT as _CTRLWORLD_VIEW_LAYOUT
from kinescore.core.clip import ViewLayout

_MULTIVIEW = ViewLayout(n_views=3, order=("exterior_1", "exterior_2", "wrist"))
_SINGLEVIEW = ViewLayout(n_views=1)

#: Two registered robots (kinescore.robots.available_robots()), standing in
#: for the old "humanoid"/"bimanual" fixture keys. _ROBOT_HUMANOID's
#: embodiment is literally "humanoid" so path fixtures below read naturally.
_ROBOT_HUMANOID = "airbot_mmk2"
_ROBOT_SINGLE_ARM = "franka_panda"

ctrlworld = CtrlWorldSource()
dreamdojo = DreamDojoSource()
dreamgen = DreamGenSource()


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _raw(**overrides) -> dict:
    raw = {
        "run_id": "t", "seed": 0,
        "axes": {
            "robot": [_ROBOT_HUMANOID, _ROBOT_SINGLE_ARM],
            "view": ["multiview", "singleview"],
            "horizon": ["makovian", "non_makovian"],
            "cache": ["dense"],
            "generator": ["ctrlworld", "dreamgen", "dreamdojo"],
        },
        "na_cells": [{"generator": "dreamdojo", "view": "multiview"},
                    {"generator": "ctrlworld", "view": "singleview"}],
        "robots": {
            _ROBOT_HUMANOID: {"spec": _ROBOT_HUMANOID, "reader": "humanoid.pt", "assets": "grx"},
            _ROBOT_SINGLE_ARM: {"spec": _ROBOT_SINGLE_ARM, "reader": "single_arm.pt",
                               "assets": "franka"},
        },
        "sources": {
            "dreamgen": {"iter": "iter_000113000", "view_dir": "singleview",
                        "gt_from": "input"},
            "dreamdojo": {"iter": "iter_000050000", "view_dir": "singleview"},
            "ctrlworld": {"view_dir": "multiview"},
        },
        "fps_expected": {"dreamgen": 16.0, "dreamdojo": 10.0},
        "rate_policy": "paired", "suite": "invariant_v1",
        "baseline_cache": "dense", "caps": {"episodes_per_cell": 150},
    }
    raw.update(overrides)
    return raw


def _cell(*, generator, view, horizon="makovian", embodiment="humanoid",
         robot=_ROBOT_HUMANOID, cache="dense", view_layout=None) -> Cell:
    return Cell(cache=cache, robot=robot, view=view, generator=generator, horizon=horizon,
               embodiment=embodiment,
               view_layout=view_layout or (_MULTIVIEW if view == "multiview" else _SINGLEVIEW))


class TestCtrlworld:
    def test_discovers_the_multiview_pair_not_full_pred(self, tmp_path):
        config = from_dict(_raw())
        cell = _cell(generator="ctrlworld", view="multiview")
        ep_dir = (tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
                 / "multiview" / "ctrlworld" / "makovian" / "episode_taskA__001")
        _touch(ep_dir / "pred_all_views.mp4")
        _touch(ep_dir / "gt_all_views.mp4")
        _touch(ep_dir / "full_pred.mp4")

        rows = list(ctrlworld.make_plugin(cell, str(tmp_path), config)())
        assert len(rows) == 2
        roles = {r.role: r for r in rows}
        assert set(roles) == {"pred", "gt"}
        assert roles["pred"].path.endswith("pred_all_views.mp4")
        assert roles["gt"].path.endswith("gt_all_views.mp4")
        assert roles["pred"].pair_key == roles["gt"].pair_key
        assert roles["pred"].family == cell.family
        # NOT cell.view_layout: ctrlworld overrides the dataset-wide
        # height-stack convention with its own measured packing (width
        # stack, wrist panel dropped) -- see
        # kinescore.bench.sources.ctrlworld's module docstring and
        # legacy_docs/DECISIONS.md D-G.
        assert roles["pred"].view_layout == _CTRLWORLD_VIEW_LAYOUT
        assert roles["pred"].view_layout != cell.view_layout
        assert roles["pred"].view_layout.packing == "width"
        assert roles["pred"].view_layout.n_views == 2
        assert roles["pred"].view_layout.panel_indices == (0, 1)
        assert roles["pred"].fps_hint is None  # ctrlworld has no fps_expected entry
        assert all("full_pred" not in r.path for r in rows)

    def test_episode_missing_gt_is_skipped(self, tmp_path):
        config = from_dict(_raw())
        cell = _cell(generator="ctrlworld", view="multiview")
        ep_dir = (tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
                 / "multiview" / "ctrlworld" / "makovian" / "episode_taskB__002")
        _touch(ep_dir / "pred_all_views.mp4")  # no gt_all_views.mp4

        rows = list(ctrlworld.make_plugin(cell, str(tmp_path), config)())
        assert rows == []

    def test_wrong_generator_cell_raises(self, tmp_path):
        config = from_dict(_raw())
        cell = _cell(generator="dreamgen", view="singleview")
        with pytest.raises(ValueError, match="ctrlworld"):
            ctrlworld.make_plugin(cell, str(tmp_path), config)

    def test_a_stray_file_matching_the_episode_glob_is_not_a_directory_and_is_skipped(
            self, tmp_path):
        # "episode_*" can match a non-directory entry (e.g. a leftover log
        # file); it must be skipped, not treated as an episode.
        config = from_dict(_raw())
        cell = _cell(generator="ctrlworld", view="multiview")
        root = (tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
               / "multiview" / "ctrlworld" / "makovian")
        root.mkdir(parents=True)
        (root / "episode_not_a_dir").write_bytes(b"")
        ep_dir = root / "episode_taskA__001"
        _touch(ep_dir / "pred_all_views.mp4")
        _touch(ep_dir / "gt_all_views.mp4")

        rows = list(ctrlworld.make_plugin(cell, str(tmp_path), config)())
        assert len(rows) == 2


class TestDreamdojo:
    def test_pins_the_configured_iter_and_ignores_others(self, tmp_path):
        config = from_dict(_raw())
        cell = _cell(generator="dreamdojo", view="singleview")
        base = (tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
               / "singleview" / "dreamdojo" / "makovian")
        pinned = base / "iter_000050000"
        _touch(pinned / "0000_pred.mp4")
        _touch(pinned / "0000_gt.mp4")
        _touch(pinned / "0001_pred.mp4")
        _touch(pinned / "0001_gt.mp4")
        other = base / "iter_000090000"
        _touch(other / "0000_pred.mp4")
        _touch(other / "0000_gt.mp4")
        _touch(other / "0002_pred.mp4")
        _touch(other / "0002_gt.mp4")

        rows = list(dreamdojo.make_plugin(cell, str(tmp_path), config)())
        assert len(rows) == 4  # 2 episodes x {pred, gt}, from the pinned iter only
        assert all("iter_000050000" in r.path for r in rows)
        assert not any("iter_000090000" in r.path for r in rows)
        episodes = {r.episode for r in rows}
        # "flat:" prefix -- see TestDreamdojoShapes for why: a "dir" shape
        # episode can share the same bare numeric index.
        assert episodes == {"flat:0000", "flat:0001"}
        for r in rows:
            assert r.fps_hint == pytest.approx(10.0)  # from fps_expected.dreamdojo

    def test_episode_missing_gt_is_skipped(self, tmp_path):
        config = from_dict(_raw())
        cell = _cell(generator="dreamdojo", view="singleview")
        pinned = (tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
                 / "singleview" / "dreamdojo" / "makovian" / "iter_000050000")
        _touch(pinned / "0000_pred.mp4")  # no 0000_gt.mp4

        rows = list(dreamdojo.make_plugin(cell, str(tmp_path), config)())
        assert rows == []

    def test_missing_iter_pin_raises(self, tmp_path):
        raw = _raw()
        raw["sources"]["dreamdojo"]["iter"] = None
        config = from_dict(raw)
        cell = _cell(generator="dreamdojo", view="singleview")
        with pytest.raises(ValueError, match="iter"):
            dreamdojo.make_plugin(cell, str(tmp_path), config)

    def test_wrong_generator_cell_raises(self, tmp_path):
        config = from_dict(_raw())
        cell = _cell(generator="ctrlworld", view="multiview")
        with pytest.raises(ValueError, match="dreamdojo"):
            dreamdojo.make_plugin(cell, str(tmp_path), config)

    def test_a_pred_file_not_matching_the_numbered_pattern_is_ignored(self, tmp_path):
        config = from_dict(_raw())
        cell = _cell(generator="dreamdojo", view="singleview")
        pinned = (tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
                 / "singleview" / "dreamdojo" / "makovian" / "iter_000050000")
        _touch(pinned / "extra_pred.mp4")  # matches *_pred.mp4 but not \d+_pred.mp4
        _touch(pinned / "0000_pred.mp4")
        _touch(pinned / "0000_gt.mp4")

        rows = list(dreamdojo.make_plugin(cell, str(tmp_path), config)())
        assert {r.episode for r in rows} == {"flat:0000"}

    def test_iter_may_be_a_nested_robot_horizon_mapping(self, tmp_path):
        # Per the live inventory, the right dreamdojo iter genuinely differs
        # per (robot, horizon) -- not just per generator.
        raw = _raw()
        raw["sources"]["dreamdojo"]["iter"] = {
            _ROBOT_HUMANOID: {"makovian": "iter_000050000", "non_makovian": "iter_000060000"},
        }
        config = from_dict(raw)
        makovian_cell = _cell(generator="dreamdojo", view="singleview", horizon="makovian")
        non_mak_cell = _cell(generator="dreamdojo", view="singleview", horizon="non_makovian")
        base = (tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
               / "singleview" / "dreamdojo")
        _touch(base / "makovian" / "iter_000050000" / "0000_pred.mp4")
        _touch(base / "makovian" / "iter_000050000" / "0000_gt.mp4")
        _touch(base / "non_makovian" / "iter_000060000" / "0000_pred.mp4")
        _touch(base / "non_makovian" / "iter_000060000" / "0000_gt.mp4")

        mak_rows = list(dreamdojo.make_plugin(makovian_cell, str(tmp_path), config)())
        non_mak_rows = list(dreamdojo.make_plugin(non_mak_cell, str(tmp_path), config)())
        assert len(mak_rows) == 2 and all("iter_000050000" in r.path for r in mak_rows)
        assert len(non_mak_rows) == 2 and all("iter_000060000" in r.path for r in non_mak_rows)

    def test_nested_iter_with_no_entry_for_this_cell_raises(self, tmp_path):
        raw = _raw()
        raw["sources"]["dreamdojo"]["iter"] = {_ROBOT_HUMANOID: {"makovian": "iter_000050000"}}
        config = from_dict(raw)
        # _ROBOT_SINGLE_ARM has no entry at all in the nested map above.
        cell = _cell(generator="dreamdojo", view="singleview", embodiment="single_arm",
                    robot=_ROBOT_SINGLE_ARM)
        with pytest.raises(ValueError, match=_ROBOT_SINGLE_ARM):
            dreamdojo.make_plugin(cell, str(tmp_path), config)


class TestDreamdojoShapes:
    """The dreamdojo "flat" vs "dir" export shapes coexist in one iter dir.

    See ``bench.sources.dreamdojo``'s module docstring for the full
    investigation: comparing LFS sha256 hashes for every overlapping index
    established the two shapes are NOT duplicates (only 1/118 and 4/118
    accidental matches, not systematic), and that a THIRD sub-shape --
    ``episode_<id>/full_pred.mp4`` with no ``full_gt.mp4`` sibling -- has no
    ground truth anywhere and must be dropped, not counted.
    """

    def _pinned(self, tmp_path):
        return (tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
               / "singleview" / "dreamdojo" / "makovian" / "iter_000050000")

    def test_flat_and_dir_shapes_with_the_same_bare_index_both_discovered(self, tmp_path):
        pinned = self._pinned(tmp_path)
        _touch(pinned / "0000_pred.mp4")
        _touch(pinned / "0000_gt.mp4")
        _touch(pinned / "episode_0000" / "full_pred.mp4")
        _touch(pinned / "episode_0000" / "full_gt.mp4")
        config = from_dict(_raw())
        cell = _cell(generator="dreamdojo", view="singleview")

        rows = list(dreamdojo.make_plugin(cell, str(tmp_path), config)())
        episodes = {r.episode for r in rows}
        assert episodes == {"flat:0000", "dir:0000"}

    def test_same_bare_index_in_two_shapes_never_collides_into_one_pair_key(self, tmp_path):
        # The non-collision property the coordinator asked to pin explicitly:
        # two DIFFERENT clips (different shape, same nominal index) must
        # never share a pair_key, or bench.stats.paired_deltas would inner-
        # join a prediction from one export against a ground truth from the
        # unrelated other one.
        pinned = self._pinned(tmp_path)
        _touch(pinned / "0000_pred.mp4")
        _touch(pinned / "0000_gt.mp4")
        _touch(pinned / "episode_0000" / "full_pred.mp4")
        _touch(pinned / "episode_0000" / "full_gt.mp4")
        config = from_dict(_raw())
        cell = _cell(generator="dreamdojo", view="singleview")

        rows = list(dreamdojo.make_plugin(cell, str(tmp_path), config)())
        pair_keys_by_episode = {r.episode: r.pair_key for r in rows}
        assert len(set(pair_keys_by_episode.values())) == 2
        assert pair_keys_by_episode["flat:0000"] != pair_keys_by_episode["dir:0000"]
        # Every gt/pred pair within one shape still pairs correctly.
        by_key = {}
        for r in rows:
            by_key.setdefault(r.pair_key, set()).add(r.role)
        assert all(roles == {"pred", "gt"} for roles in by_key.values())

    def test_dir_episode_missing_full_gt_is_dropped_and_logged(self, tmp_path, capsys):
        pinned = self._pinned(tmp_path)
        _touch(pinned / "episode_000200" / "full_pred.mp4")  # no full_gt.mp4 sibling
        config = from_dict(_raw())
        cell = _cell(generator="dreamdojo", view="singleview")

        rows = list(dreamdojo.make_plugin(cell, str(tmp_path), config)())
        assert rows == []  # unscoreable -- no ground truth exists for it anywhere
        err = capsys.readouterr().err
        assert "1 dir episode(s) dropped" in err

    def test_dir_episode_with_both_files_is_discovered_with_dir_prefix(self, tmp_path):
        pinned = self._pinned(tmp_path)
        _touch(pinned / "episode_000200" / "full_pred.mp4")
        _touch(pinned / "episode_000200" / "full_gt.mp4")
        config = from_dict(_raw())
        cell = _cell(generator="dreamdojo", view="singleview")

        rows = list(dreamdojo.make_plugin(cell, str(tmp_path), config)())
        assert len(rows) == 2
        assert {r.episode for r in rows} == {"dir:000200"}
        roles = {r.role: r for r in rows}
        assert roles["pred"].path.endswith("episode_000200/full_pred.mp4")
        assert roles["gt"].path.endswith("episode_000200/full_gt.mp4")

    def test_six_digit_and_four_digit_dir_ids_never_collide(self, tmp_path):
        # Confirmed on real data: a 4-digit "dir" export index (e.g. 0200)
        # and a 6-digit one (000200) can both exist; int(...)-normalising
        # either would collide them. The raw captured digit string must be
        # kept verbatim.
        pinned = self._pinned(tmp_path)
        _touch(pinned / "episode_0200" / "full_pred.mp4")
        _touch(pinned / "episode_0200" / "full_gt.mp4")
        _touch(pinned / "episode_000200" / "full_pred.mp4")
        _touch(pinned / "episode_000200" / "full_gt.mp4")
        config = from_dict(_raw())
        cell = _cell(generator="dreamdojo", view="singleview")

        rows = list(dreamdojo.make_plugin(cell, str(tmp_path), config)())
        episodes = {r.episode for r in rows}
        assert episodes == {"dir:0200", "dir:000200"}
        pair_keys = {r.pair_key for r in rows}
        assert len(pair_keys) == 2


class TestDreamgen:
    def _layout(self, tmp_path, *, embodiment="humanoid", extra_task_dirs=(),
               extra_input_siblings=()):
        output_iter = (tmp_path / "video_gen_physics" / "dense" / embodiment / "output"
                      / "singleview" / "dreamgen" / "makovian" / "iter_000113000")
        _touch(output_iter / "taskA" / "episode_000000.mp4")
        _touch(output_iter / "taskA" / "episode_000000.txt")
        for name in extra_task_dirs:
            _touch(output_iter / name / "episode_999999.mp4")

        input_root = (tmp_path / "video_gen_physics" / "dense" / embodiment / "input"
                     / "singleview" / "dreamgen")
        _touch(input_root / "makovian" / "taskA" / "episode_000000.mp4")
        for name in extra_input_siblings:
            _touch(input_root / name / "irrelevant.mp4")

    def test_discovers_the_pred_gt_pair_from_the_sibling_input_tree(self, tmp_path):
        self._layout(tmp_path)
        config = from_dict(_raw())
        cell = _cell(generator="dreamgen", view="singleview")

        rows = list(dreamgen.make_plugin(cell, str(tmp_path), config)())
        assert len(rows) == 2
        roles = {r.role: r for r in rows}
        assert roles["pred"].path.endswith(
            "output/singleview/dreamgen/makovian/iter_000113000/taskA/episode_000000.mp4")
        assert roles["gt"].path.endswith(
            "input/singleview/dreamgen/makovian/taskA/episode_000000.mp4")
        assert roles["pred"].pair_key == roles["gt"].pair_key
        for r in rows:
            assert r.fps_hint == pytest.approx(16.0)

    def test_singleview_cell_uses_the_plain_singleview_layout(self, tmp_path):
        self._layout(tmp_path)
        config = from_dict(_raw())
        cell = _cell(generator="dreamgen", view="singleview")

        rows = list(dreamgen.make_plugin(cell, str(tmp_path), config)())
        assert all(r.view_layout == ViewLayout(n_views=1) for r in rows)

    def test_multiview_cell_uses_the_real_grid2x2_layout_not_height_stack(
            self, tmp_path):
        # dreamgen's real multiview packing is a 2x2 grid (measured at
        # 768x432), not the dataset-wide height-stack `cell.view_layout`
        # convention -- see the module docstring's `_MULTIVIEW_LAYOUT` and
        # legacy_docs/DECISIONS.md D-G. The on-disk layout here still lives under
        # the config's pinned `view_dir` (independent of the `view` axis --
        # see the module docstring), only `cell.view` changes.
        self._layout(tmp_path)
        config = from_dict(_raw())
        cell = _cell(generator="dreamgen", view="multiview")

        rows = list(dreamgen.make_plugin(cell, str(tmp_path), config)())
        assert len(rows) == 2
        assert cell.view_layout != dreamgen_mod._MULTIVIEW_LAYOUT  # sanity: genuinely different
        for r in rows:
            assert r.view_layout == dreamgen_mod._MULTIVIEW_LAYOUT
            assert r.view_layout.packing == "grid2x2"
            assert r.view_layout.n_views == 4

    def test_underscore_prefixed_directory_is_skipped_and_logged(self, tmp_path, capsys):
        self._layout(tmp_path, extra_task_dirs=["_fps_compare"])
        config = from_dict(_raw())
        cell = _cell(generator="dreamgen", view="singleview")

        rows = list(dreamgen.make_plugin(cell, str(tmp_path), config)())
        assert all("_fps_compare" not in r.path for r in rows)
        assert all("999999" not in r.episode for r in rows)
        err = capsys.readouterr().err
        assert "_fps_compare" not in err or "skipped" in err  # a summary, not necessarily the name
        assert "skipped 1" in err

    def test_view_named_siblings_under_input_are_never_touched(self, tmp_path):
        # single_arm's real input/ tree has exterior_1_left / exterior_2_left
        # / global siblings of the horizon directories. Even though this
        # fixture uses "humanoid", the plugin must never list the parent
        # dreamgen/ directory -- it goes straight to .../makovian/ -- so
        # these siblings existing must have zero effect on the result.
        self._layout(tmp_path, extra_input_siblings=["exterior_1_left", "global"])
        config = from_dict(_raw())
        cell = _cell(generator="dreamgen", view="singleview")

        rows = list(dreamgen.make_plugin(cell, str(tmp_path), config)())
        assert len(rows) == 2
        assert not any("exterior_1_left" in r.path or "global" in r.path for r in rows)

    def test_unrecognised_horizon_segment_raises_instead_of_mislabelling(self, tmp_path):
        config = from_dict(_raw())
        cell = _cell(generator="dreamgen", view="singleview", horizon="exterior_1_left")
        with pytest.raises(ValueError, match="not one of"):
            dreamgen.make_plugin(cell, str(tmp_path), config)

    def test_missing_iter_pin_raises(self, tmp_path):
        raw = _raw()
        raw["sources"]["dreamgen"]["iter"] = None
        config = from_dict(raw)
        cell = _cell(generator="dreamgen", view="singleview")
        with pytest.raises(ValueError, match="iter"):
            dreamgen.make_plugin(cell, str(tmp_path), config)

    def test_wrong_generator_cell_raises(self, tmp_path):
        config = from_dict(_raw())
        cell = _cell(generator="dreamdojo", view="singleview")
        with pytest.raises(ValueError, match="dreamgen"):
            dreamgen.make_plugin(cell, str(tmp_path), config)

    def test_a_stray_file_under_the_iter_dir_is_not_a_task_and_is_skipped(self, tmp_path):
        self._layout(tmp_path)
        output_iter = (tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
                      / "singleview" / "dreamgen" / "makovian" / "iter_000113000")
        (output_iter / "stray_file.txt").write_bytes(b"")
        config = from_dict(_raw())
        cell = _cell(generator="dreamgen", view="singleview")

        rows = list(dreamgen.make_plugin(cell, str(tmp_path), config)())
        assert len(rows) == 2  # unaffected by the stray non-directory entry

    def test_episode_with_no_matching_gt_in_input_tree_is_skipped(self, tmp_path):
        output_iter = (tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
                      / "singleview" / "dreamgen" / "makovian" / "iter_000113000")
        _touch(output_iter / "taskA" / "episode_000001.mp4")  # no matching gt
        config = from_dict(_raw())
        cell = _cell(generator="dreamgen", view="singleview")

        rows = list(dreamgen.make_plugin(cell, str(tmp_path), config)())
        assert rows == []

    def test_iter_may_be_a_nested_robot_horizon_mapping(self, tmp_path):
        # Per the live inventory: fourier_gr1-equivalent humanoid dreamgen
        # uses iter_000090000 at both horizons; the single_arm robot's
        # makovian cell uses a differently-named _static16fps iteration --
        # one flat string cannot express both.
        raw = _raw()
        raw["sources"]["dreamgen"]["iter"] = {
            _ROBOT_HUMANOID: {"makovian": "iter_000090000", "non_makovian": "iter_000090000"},
            _ROBOT_SINGLE_ARM: {"makovian": "iter_000090000_static16fps"},
        }
        config = from_dict(raw)

        humanoid_cell = _cell(generator="dreamgen", view="singleview", embodiment="humanoid",
                              robot=_ROBOT_HUMANOID)
        single_arm_cell = _cell(generator="dreamgen", view="singleview",
                                embodiment="single_arm", robot=_ROBOT_SINGLE_ARM)
        for embodiment, iterd in [("humanoid", "iter_000090000"),
                                  ("single_arm", "iter_000090000_static16fps")]:
            output_iter = (tmp_path / "video_gen_physics" / "dense" / embodiment / "output"
                          / "singleview" / "dreamgen" / "makovian" / iterd)
            _touch(output_iter / "taskA" / "episode_000000.mp4")
            input_root = (tmp_path / "video_gen_physics" / "dense" / embodiment / "input"
                         / "singleview" / "dreamgen" / "makovian" / "taskA")
            _touch(input_root / "episode_000000.mp4")

        humanoid_rows = list(dreamgen.make_plugin(humanoid_cell, str(tmp_path), config)())
        single_arm_rows = list(dreamgen.make_plugin(single_arm_cell, str(tmp_path), config)())
        assert len(humanoid_rows) == 2
        assert all("iter_000090000/" in r.path for r in humanoid_rows
                  if r.role == "pred")
        assert len(single_arm_rows) == 2
        assert all("iter_000090000_static16fps" in r.path for r in single_arm_rows
                  if r.role == "pred")

    def test_nested_iter_with_no_entry_for_this_cell_raises(self, tmp_path):
        raw = _raw()
        raw["sources"]["dreamgen"]["iter"] = {
            _ROBOT_HUMANOID: {"makovian": "iter_000090000"},
        }  # no non_makovian entry
        config = from_dict(raw)
        cell = _cell(generator="dreamgen", view="singleview", horizon="non_makovian")
        with pytest.raises(ValueError, match="non_makovian"):
            dreamgen.make_plugin(cell, str(tmp_path), config)
