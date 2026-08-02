"""kinescore.bench.data_spec: the per-generator file/format contract."""
from __future__ import annotations

from pathlib import Path

import pytest

from kinescore.bench.data_spec import DataSpecError, load_data_spec, parse_data_spec

REPO_ROOT = Path(__file__).resolve().parents[1]


def _raw(**overrides) -> dict:
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
                "fps_by_robot": {"fourier_gr1": 10.0, "franka_panda": 15.0},
                "fps_tolerant": False,
            },
        },
        "exclude_globs": ["**/tmp/**"],
        "robots": {},
    }
    raw.update(overrides)
    return raw


class TestParsing:
    def test_valid_spec_parses(self):
        spec = parse_data_spec(_raw())
        assert spec.generators["ctrlworld"].width == 960
        assert spec.exclude_globs == ("**/tmp/**",)

    def test_missing_generators_is_rejected(self):
        with pytest.raises(DataSpecError, match="generators"):
            parse_data_spec({})

    def test_unknown_shape_is_rejected(self):
        raw = _raw()
        raw["generators"]["ctrlworld"]["shape"] = "something_else"
        with pytest.raises(DataSpecError, match="shape"):
            parse_data_spec(raw)

    def test_episode_dir_shape_requires_pred_filename(self):
        raw = _raw()
        del raw["generators"]["ctrlworld"]["pred_filename"]
        with pytest.raises(DataSpecError, match="pred_filename"):
            parse_data_spec(raw)

    def test_task_episode_shape_requires_pred_glob(self):
        raw = _raw()
        del raw["generators"]["dreamgen"]["pred_glob"]
        with pytest.raises(DataSpecError, match="pred_glob"):
            parse_data_spec(raw)

    def test_flat_or_dir_shape_requires_all_four_filename_fields(self):
        raw = _raw()
        del raw["generators"]["dreamdojo"]["dir_gt_filename"]
        with pytest.raises(DataSpecError, match="dir_gt_filename"):
            parse_data_spec(raw)

    def test_fps_and_fps_by_robot_are_mutually_exclusive(self):
        raw = _raw()
        raw["generators"]["ctrlworld"]["fps_by_robot"] = {"franka_panda": 5.0}
        with pytest.raises(DataSpecError, match="fps"):
            parse_data_spec(raw)

    def test_neither_fps_nor_fps_by_robot_is_rejected(self):
        raw = _raw()
        del raw["generators"]["ctrlworld"]["fps"]
        with pytest.raises(DataSpecError, match="fps"):
            parse_data_spec(raw)


class TestResolveFps:
    def test_flat_fps_ignores_robot(self):
        spec = parse_data_spec(_raw())
        gspec = spec.generators["ctrlworld"]
        assert gspec.resolve_fps(robot="franka_panda") == 5.0
        assert gspec.resolve_fps(robot=None) == 5.0

    def test_per_robot_fps_resolves_by_robot(self):
        spec = parse_data_spec(_raw())
        gspec = spec.generators["dreamdojo"]
        assert gspec.resolve_fps(robot="fourier_gr1") == 10.0
        assert gspec.resolve_fps(robot="franka_panda") == 15.0
        assert gspec.resolve_fps(robot="unknown_robot") is None


class TestLoadFromYaml:
    def test_round_trip_through_a_written_yaml_file(self, tmp_path):
        import yaml

        path = tmp_path / "data_spec.yaml"
        path.write_text(yaml.safe_dump(_raw()))
        spec = load_data_spec(path)
        assert "ctrlworld" in spec.generators

    def test_the_committed_configs_data_spec_yaml_loads_cleanly(self):
        spec = load_data_spec(REPO_ROOT / "configs" / "data_spec.yaml")
        assert set(spec.generators) == {"ctrlworld", "dreamgen", "dreamdojo"}
        assert spec.generators["ctrlworld"].width == 960
        assert spec.generators["ctrlworld"].height == 192
        assert spec.generators["dreamgen"].has_ground_truth is False
        assert spec.generators["dreamdojo"].fps_by_robot == {
            "fourier_gr1": 10.0, "franka_panda": 15.0}
        assert spec.exclude_globs  # non-empty -- the traps list
        assert any("wget-log" in g for g in spec.exclude_globs)
