"""Adapter: a LeRobot v2 tree in, ``RawEpisode``s out, nothing written."""
from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kinescore.adapters import available_adapters, get_adapter
from kinescore.adapters.base import RawEpisode, SkippedEpisode
from kinescore.registry.cells import TrainSource

CAMS = ("cam_high", "cam_low")


def _dataset(root, split, task="", *, cameras=CAMS, state=None, extra=None,
             fps=10.0, videos=None, n_episodes=1):
    """Write one LeRobot v2 dataset under ``root/split[/task]``."""
    d = root / split / task if task else root / split
    (d / "meta").mkdir(parents=True)
    (d / "meta" / "info.json").write_text(json.dumps({
        "fps": fps,
        "features": {f"observation.images.{c}": {"dtype": "video"}
                     for c in cameras},
    }))
    state = np.zeros((4, 8), dtype=np.float32) if state is None else np.asarray(state)
    for i in range(n_episodes):
        stem = f"episode_{i:06d}"
        columns = {"observation.state": [row.tolist() for row in state]}
        for name, value in (extra or {}).items():
            columns[name] = [np.atleast_1d(v).tolist() for v in np.asarray(value)]
        (d / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns),
                       d / "data" / "chunk-000" / f"{stem}.parquet")
        for cam in (cameras if videos is None else videos):
            v = d / "videos" / "chunk-000" / f"observation.images.{cam}"
            v.mkdir(parents=True, exist_ok=True)
            (v / f"{stem}.mp4").write_bytes(b"")
    return d


def _source(root, **kwargs):
    kwargs.setdefault("cameras", CAMS)
    kwargs.setdefault("corpus", "test_corpus")
    return TrainSource(adapter="lerobot", root=str(root), **kwargs)


def _read(root, **kwargs):
    return list(get_adapter("lerobot").episodes(_source(root, **kwargs)))


class TestLayouts:
    def test_a_flat_split_is_one_dataset(self, tmp_path):
        _dataset(tmp_path, "multiview", n_episodes=2)
        entries = _read(tmp_path)
        assert [e.episode_id for e in entries] == [
            "multiview__episode_000000", "multiview__episode_000001"]

    def test_a_task_grouped_split_is_one_dataset_per_task(self, tmp_path):
        _dataset(tmp_path, "multiview", "close_box")
        _dataset(tmp_path, "multiview", "open_drawer")
        assert [e.episode_id for e in _read(tmp_path)] == [
            "multiview__close_box__episode_000000",
            "multiview__open_drawer__episode_000000"]

    def test_scene_key_is_the_dataset_not_the_episode(self, tmp_path):
        _dataset(tmp_path, "multiview", "close_box", n_episodes=2)
        assert {e.scene_key for e in _read(tmp_path)} == {"multiview__close_box"}

    def test_fps_comes_from_info_json(self, tmp_path):
        _dataset(tmp_path, "multiview", fps=20.0)
        (episode,) = _read(tmp_path)
        assert episode.fps == 20.0


class TestColumns:
    def test_joint_columns_are_selected_in_order(self, tmp_path):
        _dataset(tmp_path, "mv", state=np.arange(4 * 8).reshape(4, 8))
        cols = (0, 1, 2, 5, 4)
        (episode,) = _read(tmp_path, joint_columns=cols)
        assert episode.joints.shape == (4, 5)
        assert episode.joints[0].tolist() == [float(c) for c in cols]

    def test_no_joint_columns_takes_the_whole_state(self, tmp_path):
        _dataset(tmp_path, "mv", state=np.zeros((4, 8)))
        (episode,) = _read(tmp_path)
        assert episode.joints.shape == (4, 8)

    def test_a_gripper_column_is_sliced_out_of_the_state(self, tmp_path):
        _dataset(tmp_path, "mv", state=np.arange(4 * 8).reshape(4, 8))
        (episode,) = _read(tmp_path, joint_columns=(0, 1), gripper_column=7)
        assert episode.gripper is not None
        assert episode.gripper.tolist() == [7.0, 15.0, 23.0, 31.0]

    def test_a_gripper_field_is_read_from_its_own_column(self, tmp_path):
        _dataset(tmp_path, "mv", extra={"gripper": np.arange(4.0)})
        (episode,) = _read(tmp_path, gripper_field="gripper")
        assert episode.gripper is not None
        assert episode.gripper.tolist() == [0.0, 1.0, 2.0, 3.0]

    def test_a_different_joint_field_is_read(self, tmp_path):
        _dataset(tmp_path, "mv", extra={"state.joints": np.zeros((4, 6))})
        (episode,) = _read(tmp_path, joint_field="state.joints")
        assert episode.joints.shape == (4, 6)

    def test_no_gripper_declared_is_no_gripper(self, tmp_path):
        _dataset(tmp_path, "mv")
        (episode,) = _read(tmp_path)
        assert episode.gripper is None


class TestViews:
    def test_views_follow_the_declared_camera_order(self, tmp_path):
        _dataset(tmp_path, "mv", cameras=("a", "b", "c"))
        (episode,) = _read(tmp_path, cameras=("c", "a", "b"))
        assert list(episode.views) == ["c", "a", "b"]

    def test_views_point_at_the_per_camera_files(self, tmp_path):
        _dataset(tmp_path, "mv")
        (episode,) = _read(tmp_path)
        assert episode.packed is None
        assert episode.views["cam_high"].endswith(
            "videos/chunk-000/observation.images.cam_high/episode_000000.mp4")


class TestCameraAlternates:
    """One panel, two names: corpora that merged two collection runs."""

    def test_the_first_name_present_is_taken(self, tmp_path):
        _dataset(tmp_path, "mv", cameras=("cam_high_rgb", "cam_wrist"))
        (episode,) = _read(tmp_path,
                           cameras=("cam_head_rgb|cam_high_rgb", "cam_wrist"))
        assert list(episode.views) == ["cam_high_rgb", "cam_wrist"]

    def test_either_naming_reaches_the_same_panel_order(self, tmp_path):
        _dataset(tmp_path / "a", "mv", cameras=("cam_head_rgb", "cam_wrist"))
        _dataset(tmp_path / "b", "mv", cameras=("cam_high_rgb", "cam_wrist"))
        cameras = ("cam_head_rgb|cam_high_rgb", "cam_wrist")
        (first,) = _read(tmp_path / "a", cameras=cameras)
        (second,) = _read(tmp_path / "b", cameras=cameras)
        assert list(first.views)[1] == list(second.views)[1] == "cam_wrist"

    def test_a_dataset_with_neither_name_is_skipped(self, tmp_path):
        _dataset(tmp_path, "mv", cameras=("cam_other", "cam_wrist"))
        (entry,) = _read(tmp_path,
                         cameras=("cam_head_rgb|cam_high_rgb", "cam_wrist"))
        assert isinstance(entry, SkippedEpisode)
        assert "cam_head_rgb|cam_high_rgb" in entry.reason


class TestSkips:
    def test_a_camera_the_dataset_lacks_skips_the_dataset(self, tmp_path):
        _dataset(tmp_path, "mv", cameras=("cam_high",))
        (entry,) = _read(tmp_path)
        assert isinstance(entry, SkippedEpisode)
        assert "cam_low" in entry.reason

    def test_a_missing_joint_field_is_reported(self, tmp_path):
        _dataset(tmp_path, "mv")
        (entry,) = _read(tmp_path, joint_field="observation.nope")
        assert isinstance(entry, SkippedEpisode)
        assert "no column" in entry.reason

    def test_joint_columns_past_the_state_are_reported(self, tmp_path):
        _dataset(tmp_path, "mv", state=np.zeros((4, 4)))
        (entry,) = _read(tmp_path, joint_columns=(0, 9))
        assert isinstance(entry, SkippedEpisode)
        assert "joint_columns reach index 9" in entry.reason

    def test_a_gripper_column_past_the_state_is_reported(self, tmp_path):
        _dataset(tmp_path, "mv", state=np.zeros((4, 4)))
        (entry,) = _read(tmp_path, gripper_column=9)
        assert isinstance(entry, SkippedEpisode)
        assert "gripper_column 9" in entry.reason

    def test_a_missing_video_is_reported(self, tmp_path):
        _dataset(tmp_path, "mv", videos=("cam_high",))
        (entry,) = _read(tmp_path)
        assert isinstance(entry, SkippedEpisode)
        assert "cam_low" in entry.reason

    def test_an_unreadable_parquet_is_reported(self, tmp_path):
        d = _dataset(tmp_path, "mv")
        (d / "data" / "chunk-000" / "episode_000000.parquet").write_bytes(b"nope")
        (entry,) = _read(tmp_path)
        assert isinstance(entry, SkippedEpisode)
        assert "parquet unreadable" in entry.reason

    def test_one_bad_dataset_does_not_stop_the_others(self, tmp_path):
        _dataset(tmp_path, "a", cameras=("cam_high",))
        _dataset(tmp_path, "b")
        kinds = [type(e) for e in _read(tmp_path)]
        assert kinds == [SkippedEpisode, RawEpisode]


class TestRefusals:
    def test_a_missing_root_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            _read(tmp_path / "absent")

    def test_no_cameras_declared_raises(self, tmp_path):
        _dataset(tmp_path, "mv")
        with pytest.raises(ValueError, match="needs `cameras:`"):
            _read(tmp_path, cameras=())

    def test_repeated_cameras_are_rejected_at_declaration(self, tmp_path):
        with pytest.raises(ValueError, match="cameras repeat"):
            _source(tmp_path, cameras=("cam_high", "cam_high"))


class TestRegistry:
    def test_lerobot_is_the_shipped_adapter(self):
        assert available_adapters() == ("lerobot",)

    def test_an_unknown_adapter_lists_what_exists(self):
        with pytest.raises(ValueError, match="available"):
            get_adapter("no_such_adapter")
