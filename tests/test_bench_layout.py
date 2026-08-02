"""kinescore.bench.layout: DataLayout, the raw HF mirror and canonical shapes.

CPU/network-free: every "clip" is an empty ``.mp4`` file, since discovery
here only ever inspects paths (mirrors ``test_bench_sources.py``'s own
convention). Content-probing (ffprobe) tests live in
``test_bench_ingest.py``/``test_bench_verify.py``, marked ``@pytest.mark.ffmpeg``.
"""
from __future__ import annotations

import json
from pathlib import Path

from kinescore.bench.cell import Cell
from kinescore.bench.data_spec import load_data_spec, parse_data_spec
from kinescore.bench.layout import (
    CELL_CARD_NAME,
    CanonicalLayout,
    RawHFLayout,
    _glob_excluded,
    sanitize_episode,
)
from kinescore.bench.robot_map import parse_robot_map
from kinescore.core.clip import ViewLayout

REPO_ROOT = Path(__file__).resolve().parents[1]


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _robot_map():
    return parse_robot_map({
        "robots": {
            "airbot_mmk2": {"embodiment": "humanoid", "generators": ["ctrlworld"]},
            "fourier_gr1": {"embodiment": "humanoid", "generators": ["dreamgen", "dreamdojo"]},
            "aloha_bimanual": {"embodiment": "bimanual",
                              "generators": ["ctrlworld", "dreamgen", "dreamdojo"]},
        },
    })


def _data_spec(**overrides):
    raw = {
        "generators": {
            "ctrlworld": {
                "shape": "episode_dir", "pred_filename": "pred_all_views.mp4",
                "gt_filename": "gt_all_views.mp4", "has_iter_level": False,
                "width": 960, "height": 192, "n_views": 3, "has_ground_truth": True,
                "fps": 5.0, "fps_tolerant": True,
            },
            "dreamgen": {
                "shape": "task_episode", "pred_glob": "episode_*.mp4", "gt_filename": None,
                "has_iter_level": True, "width": 768, "height": 432, "n_views": 1,
                "has_ground_truth": False, "fps": 16.0, "fps_tolerant": False,
            },
            "dreamdojo": {
                "shape": "flat_or_dir", "flat_pred_glob": "*_pred.mp4",
                "flat_gt_glob": "*_gt.mp4", "dir_pred_filename": "full_pred.mp4",
                "dir_gt_filename": "full_gt.mp4", "has_iter_level": True,
                "width": 640, "height": 480, "n_views": 1, "has_ground_truth": True,
                "fps": 10.0, "fps_tolerant": False,
            },
        },
        "exclude_globs": ["**/tmp/**", "**/_fps_compare/**", "**/*_static16fps/**"],
        "robots": {},
    }
    raw.update(overrides)
    return parse_data_spec(raw)


class TestGlobExcluded:
    def test_double_star_matches_the_bare_directory_itself(self):
        # Must match BEFORE descending -- a caller checks the directory
        # itself, not just its future children.
        assert _glob_excluded("a/b/tmp", ("**/tmp/**",))

    def test_double_star_matches_nested_children(self):
        assert _glob_excluded("a/b/tmp/c/d.mp4", ("**/tmp/**",))

    def test_similar_but_different_name_does_not_match(self):
        assert not _glob_excluded("a/b/tmpx/c", ("**/tmp/**",))

    def test_literal_question_mark_in_a_character_class_is_not_a_wildcard(self):
        pat = "**/gr00t[?]rlkey=x"
        assert _glob_excluded("gr00t?rlkey=x", (pat,))
        assert not _glob_excluded("gr00tArlkey=x", (pat,))

    def test_no_patterns_never_excludes_anything(self):
        assert not _glob_excluded("anything/at/all", ())


class TestExcludeGlobsMatchRealDataSpec:
    """Pins the Fix-3 reconciliation outcome against the REAL, committed
    ``configs/data_spec.yaml`` -- not a local fixture -- so a future edit to
    that file cannot silently reintroduce the tension documented in
    ``kinescore.bench.layout``'s module docstring (an exclude glob eating a
    real, human-verified, pinned ``iter_*`` directory).

    ``**/*_static16fps/**``/``**/*bimanual16fps/**`` were removed entirely
    (see ``configs/data_spec.yaml``'s header comment for why): the source
    plugins that actually drive ``kinescore bench run`` resolve
    ``sources.<generator>.iter`` as an explicit whitelist and never consult
    ``exclude_globs``, so the only place either glob had teeth was
    ``RawHFLayout``'s ingest-time auto-pick, where it was pure harm. An
    unrelated, unambiguous ablation directory (``base_2b``, from
    ``**/base_2b/**``) must stay excluded -- this is not "delete every
    glob", only the two that collided with a real pin.
    """

    def _spec(self):
        return load_data_spec(REPO_ROOT / "configs" / "data_spec.yaml")

    def test_franka_panda_dreamgen_makovian_static16fps_pin_is_reachable(self):
        # dense/single_arm/output/singleview/dreamgen/makovian/
        # iter_000090000_static16fps -- the human-verified scoring pin
        # (configs/benchmark.yaml sources.dreamgen.iter.franka_panda.makovian),
        # 120 real episodes.
        spec = self._spec()
        path = ("video_gen_physics/dense/single_arm/output/singleview/"
               "dreamgen/makovian/iter_000090000_static16fps/global/"
               "episode_000000.mp4")
        assert not _glob_excluded(path, spec.exclude_globs)

    def test_aloha_bimanual_dreamgen_bimanual16fps_pin_is_reachable(self):
        # dense/bimanual/output/singleview/dreamgen/makovian/
        # iter_000110000_bimanual16fps -- the ONLY iter present for
        # bimanual/singleview/dreamgen (configs/benchmark_bimanual.yaml
        # sources.dreamgen.iter.aloha_bimanual.makovian), 300 episodes
        # across both horizons.
        spec = self._spec()
        path = ("video_gen_physics/dense/bimanual/output/singleview/"
               "dreamgen/makovian/iter_000110000_bimanual16fps/taskA/"
               "episode_000000.mp4")
        assert not _glob_excluded(path, spec.exclude_globs)

    def test_unpinned_ablation_base_2b_is_still_excluded(self):
        spec = self._spec()
        path = ("video_gen_physics/dense/humanoid/output/singleview/"
               "dreamgen/base_2b/episode_000000.mp4")
        assert _glob_excluded(path, spec.exclude_globs)


class TestSanitizeEpisode:
    def test_slashes_and_colons_become_underscores(self):
        assert sanitize_episode("taskA/episode_000012") == "episode_taskA__episode_000012"
        assert sanitize_episode("flat:0000") == "episode_flat_0000"

    def test_already_episode_prefixed_is_not_double_prefixed(self):
        assert sanitize_episode("episode_0000") == "episode_0000"


class TestRawHFLayoutCells:
    def test_discovers_ctrlworld_cell_with_no_iter_level(self, tmp_path):
        _touch(tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
              / "multiview" / "ctrlworld" / "makovian" / "episode_0000"
              / "pred_all_views.mp4")
        layout = RawHFLayout(str(tmp_path), _robot_map(), _data_spec())
        cells = list(layout.cells())
        assert len(cells) == 1
        cell = cells[0]
        assert cell.robot == "airbot_mmk2"  # humanoid x ctrlworld -> airbot_mmk2
        assert cell.embodiment == "humanoid"
        assert cell.view == "multiview"
        assert cell.iter is None

    def test_single_view_and_singleview_become_two_distinct_cells(self, tmp_path):
        # The exact trap: both spellings coexist as REAL, different
        # subtrees and must never be merged into one cell.
        for view_dir in ("singleview", "single_view"):
            _touch(tmp_path / "video_gen_physics" / "dense" / "bimanual" / "output"
                  / view_dir / "dreamdojo" / "makovian" / "iter_000100000"
                  / "episode_0000" / "full_pred.mp4")
        layout = RawHFLayout(str(tmp_path), _robot_map(), _data_spec())
        cells = [c for c in layout.cells() if c.generator == "dreamdojo"]
        views = {c.view for c in cells}
        assert views == {"singleview", "single_view"}
        assert len(cells) == 2

    def test_robot_map_auto_excludes_a_generator_this_robot_never_produces(self, tmp_path):
        # airbot_mmk2 only claims ctrlworld -- a dreamgen tree under the
        # SAME humanoid embodiment must not resolve to airbot_mmk2 or
        # fourier_gr1 by accident; it resolves correctly to fourier_gr1
        # because robot_map disambiguates on generator.
        _touch(tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
              / "singleview" / "dreamgen" / "makovian" / "iter_000090000"
              / "taskA" / "episode_000000.mp4")
        layout = RawHFLayout(str(tmp_path), _robot_map(), _data_spec())
        cells = list(layout.cells())
        assert len(cells) == 1
        assert cells[0].robot == "fourier_gr1"

    def test_unclaimed_embodiment_generator_pair_yields_no_cell(self, tmp_path):
        _touch(tmp_path / "video_gen_physics" / "dense" / "single_arm" / "output"
              / "multiview" / "ctrlworld" / "makovian" / "episode_0000"
              / "pred_all_views.mp4")  # single_arm not in this test's robot_map
        layout = RawHFLayout(str(tmp_path), _robot_map(), _data_spec())
        assert list(layout.cells()) == []

    def test_dreamgen_camera_named_siblings_are_never_treated_as_a_horizon(self, tmp_path):
        base = (tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
               / "singleview" / "dreamgen")
        _touch(base / "makovian" / "iter_000090000" / "taskA" / "episode_000000.mp4")
        _touch(base / "exterior_1_left" / "irrelevant.mp4")  # camera-named sibling, not a horizon
        layout = RawHFLayout(str(tmp_path), _robot_map(), _data_spec())
        cells = list(layout.cells())
        assert len(cells) == 1
        assert cells[0].horizon == "makovian"

    def test_excluded_fps_compare_directory_never_yields_a_cell(self, tmp_path):
        _touch(tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
              / "singleview" / "dreamgen" / "_fps_compare" / "iter_000090000"
              / "taskA" / "episode_0.mp4")
        layout = RawHFLayout(str(tmp_path), _robot_map(), _data_spec())
        assert list(layout.cells()) == []

    def test_iter_auto_pick_prefers_the_most_populated_candidate(self, tmp_path):
        base = (tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
               / "singleview" / "dreamgen" / "makovian")
        _touch(base / "iter_000070000" / "taskA" / "episode_0.mp4")
        for i in range(3):
            _touch(base / "iter_000090000" / "taskA" / f"episode_{i}.mp4")
        layout = RawHFLayout(str(tmp_path), _robot_map(), _data_spec())
        cells = list(layout.cells())
        assert len(cells) == 1
        assert cells[0].iter == "iter_000090000"

    def test_static16fps_iter_directory_is_excluded_per_data_spec(self, tmp_path):
        # The documented tension (see kinescore.bench.layout's module
        # docstring): applied literally, this exclusion also removes a
        # real, legitimately-used iter -- pinned here as the CURRENT
        # behaviour, not silently special-cased.
        base = (tmp_path / "video_gen_physics" / "dense" / "single_arm" / "output"
               / "singleview" / "dreamgen" / "makovian")
        _touch(base / "iter_000090000_static16fps" / "taskA" / "episode_0.mp4")
        layout = RawHFLayout(str(tmp_path), _robot_map(), _data_spec())
        assert list(layout.cells()) == []

    def test_validate_reports_an_unresolvable_embodiment_generator_pair(self, tmp_path):
        (tmp_path / "video_gen_physics" / "dense" / "single_arm" / "output"
        / "multiview" / "ctrlworld").mkdir(parents=True)
        layout = RawHFLayout(str(tmp_path), _robot_map(), _data_spec())
        problems = layout.validate()
        assert any("single_arm" in p and "ctrlworld" in p for p in problems)


class TestRawHFLayoutEpisodes:
    def test_episode_dir_shape_drops_a_pred_missing_gt_when_gt_is_expected(self, tmp_path):
        base = (tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
               / "multiview" / "ctrlworld" / "makovian")
        _touch(base / "episode_0000" / "pred_all_views.mp4")  # no gt sibling
        _touch(base / "episode_0001" / "pred_all_views.mp4")
        _touch(base / "episode_0001" / "gt_all_views.mp4")
        layout = RawHFLayout(str(tmp_path), _robot_map(), _data_spec())
        cell = next(c for c in layout.cells() if c.generator == "ctrlworld")
        episodes = list(layout.episodes(cell))
        assert len(episodes) == 1
        assert episodes[0].episode == "episode_0001"

    def test_task_episode_shape_never_drops_a_pred_for_missing_gt(self, tmp_path):
        # dreamgen has NO ground truth by design -- a prediction is always
        # kept, unlike ctrlworld/dreamdojo.
        base = (tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
               / "singleview" / "dreamgen" / "makovian" / "iter_000090000")
        _touch(base / "taskA" / "episode_000000.mp4")
        _touch(base / "episode_000001.mp4")  # taskless, humanoid-shape trap
        layout = RawHFLayout(str(tmp_path), _robot_map(), _data_spec())
        cell = next(c for c in layout.cells() if c.generator == "dreamgen")
        episodes = list(layout.episodes(cell))
        assert {e.episode for e in episodes} == {"taskA/episode_000000", "episode_000001"}
        assert all(e.gt_path is None for e in episodes)

    def test_flat_or_dir_shape_discovers_both_and_drops_pred_only_dir_orphan(self, tmp_path):
        base = (tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
               / "singleview" / "dreamdojo" / "makovian" / "iter_000050000")
        _touch(base / "0000_pred.mp4")
        _touch(base / "0000_gt.mp4")
        _touch(base / "episode_0001" / "full_pred.mp4")
        _touch(base / "episode_0001" / "full_gt.mp4")
        _touch(base / "episode_000200" / "full_pred.mp4")  # orphan -- no gt anywhere
        layout = RawHFLayout(str(tmp_path), _robot_map(), _data_spec())
        cell = next(c for c in layout.cells() if c.generator == "dreamdojo")
        episodes = {e.episode: e for e in layout.episodes(cell)}
        assert set(episodes) == {"flat:0000", "dir:0001"}


class TestCanonicalLayout:
    def _cell(self):
        return Cell(cache="dense", robot="airbot_mmk2", view="multiview",
                   generator="ctrlworld", horizon="makovian", embodiment="humanoid",
                   view_layout=ViewLayout(n_views=1))

    def test_cell_dir_follows_path_axis_order(self, tmp_path):
        canon = CanonicalLayout(str(tmp_path))
        cell = self._cell()
        assert canon.cell_dir(cell) == str(
            tmp_path / "dense" / "airbot_mmk2" / "multiview" / "ctrlworld" / "makovian")

    def test_episode_dir_sanitizes_the_episode_id(self, tmp_path):
        canon = CanonicalLayout(str(tmp_path))
        cell = self._cell()
        assert canon.episode_dir(cell, "flat:0000").endswith("episode_flat_0000")

    def test_cells_reads_back_from_cell_card_json(self, tmp_path):
        canon = CanonicalLayout(str(tmp_path))
        cell = self._cell()
        cell_dir = canon.cell_dir(cell)
        import os
        os.makedirs(cell_dir, exist_ok=True)
        card = {"cache": cell.cache, "robot": cell.robot, "view": cell.view,
               "generator": cell.generator, "horizon": cell.horizon,
               "embodiment": cell.embodiment, "iter": None}
        with open(os.path.join(cell_dir, CELL_CARD_NAME), "w") as f:
            json.dump(card, f)

        reread = list(canon.cells())
        assert len(reread) == 1
        assert reread[0].robot == "airbot_mmk2"
        assert reread[0].embodiment == "humanoid"

    def test_empty_root_yields_no_cells_and_no_problems(self, tmp_path):
        canon = CanonicalLayout(str(tmp_path / "does_not_exist_yet"))
        assert list(canon.cells()) == []
        assert canon.validate() == []

    def test_validate_detects_a_broken_symlink(self, tmp_path):
        canon = CanonicalLayout(str(tmp_path))
        cell = self._cell()
        ep_dir = canon.episode_dir(cell, "episode_0000")
        import os
        os.makedirs(ep_dir, exist_ok=True)
        os.symlink("/nonexistent/target.mp4", os.path.join(ep_dir, "pred.mp4"))
        problems = canon.validate()
        assert any("broken symlink" in p for p in problems)

    def test_validate_detects_a_malformed_cell_card(self, tmp_path):
        canon = CanonicalLayout(str(tmp_path))
        cell = self._cell()
        cell_dir = canon.cell_dir(cell)
        import os
        os.makedirs(cell_dir, exist_ok=True)
        with open(os.path.join(cell_dir, CELL_CARD_NAME), "w") as f:
            json.dump({"cache": "dense"}, f)  # missing most required keys
        problems = canon.validate()
        assert any("missing key" in p for p in problems)
