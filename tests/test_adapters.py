"""Adapters: one corpus shape in, ``RawEpisode``s out, nothing written.

Every fixture below is a directory tree built in ``tmp_path``, so these run
without the published corpus.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from kinescore.adapters import available_adapters, get_adapter
from kinescore.adapters.base import RawEpisode, SkippedEpisode
from kinescore.registry.cells import TrainSource


def _episode(root, subset, name, *, states=None, packed=True, views=0,
             field="states"):
    d = root / subset / f"episode_{name}"
    d.mkdir(parents=True)
    meta = {"fps": 10.0}
    if states is not None:
        meta[field] = np.asarray(states).tolist()
    (d / "metadata.json").write_text(json.dumps(meta))
    if packed:
        (d / "full_gt.mp4").write_bytes(b"")
    for i in range(views):
        (d / f"view_{i}.mp4").write_bytes(b"")
    return d


class TestCtrlWorldTeleop:
    def _source(self, root, **kwargs) -> TrainSource:
        return TrainSource(adapter="ctrlworld_teleop", root=str(root), **kwargs)

    def test_selects_the_declared_joint_columns(self, tmp_path):
        _episode(tmp_path, "task_a", "task_a__0",
                 states=np.arange(3 * 14).reshape(3, 14))
        adapter = get_adapter("ctrlworld_teleop")
        cols = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)

        (episode,) = list(adapter.episodes(
            self._source(tmp_path, joint_columns=cols)))

        assert isinstance(episode, RawEpisode)
        assert episode.joints.shape == (3, 12)
        assert episode.joints[0].tolist() == [float(c) for c in cols]

    def test_prefers_the_packed_frame(self, tmp_path):
        _episode(tmp_path, "task_a", "task_a__0",
                 states=np.zeros((3, 7)), packed=True, views=3)
        (episode,) = list(get_adapter("ctrlworld_teleop").episodes(
            self._source(tmp_path)))
        assert episode.packed and episode.packed.endswith("full_gt.mp4")
        assert not episode.views

    def test_falls_back_to_per_camera_files(self, tmp_path):
        _episode(tmp_path, "task_a", "task_a__0",
                 states=np.zeros((3, 7)), packed=False, views=3)
        (episode,) = list(get_adapter("ctrlworld_teleop").episodes(
            self._source(tmp_path)))
        assert episode.packed is None
        assert sorted(episode.views) == ["view_0", "view_1", "view_2"]

    def test_scene_key_drops_the_episode_index(self, tmp_path):
        _episode(tmp_path, "task_a", "close_box__7", states=np.zeros((3, 7)))
        (episode,) = list(get_adapter("ctrlworld_teleop").episodes(
            self._source(tmp_path)))
        assert episode.scene_key == "close_box"

    def test_a_different_joint_field_is_read(self, tmp_path):
        _episode(tmp_path, "task_a", "task_a__0",
                 states=np.zeros((3, 8)), field="joints")
        (episode,) = list(get_adapter("ctrlworld_teleop").episodes(
            self._source(tmp_path, joint_field="joints",
                         joint_columns=tuple(range(7)))))
        assert episode.joints.shape == (3, 7)

    def test_a_gripper_column_is_carried(self, tmp_path):
        _episode(tmp_path, "task_a", "task_a__0",
                 states=np.arange(3 * 8).reshape(3, 8), field="joints")
        (episode,) = list(get_adapter("ctrlworld_teleop").episodes(
            self._source(tmp_path, joint_field="joints",
                         joint_columns=tuple(range(7)), gripper_column=7)))
        assert episode.gripper is not None
        assert episode.gripper.shape == (3,)

    @pytest.mark.parametrize("make,reason", [
        (lambda root: _episode(root, "t", "t__0", states=None), "no 'states'"),
        (lambda root: _episode(root, "t", "t__0", states=np.zeros(3)),
         "expected [T, J]"),
        (lambda root: _episode(root, "t", "t__0", states=np.zeros((3, 2))),
         "joint_columns reach index"),
        (lambda root: _episode(root, "t", "t__0", states=np.zeros((3, 14)),
                               packed=False), "neither full_gt.mp4"),
    ])
    def test_unusable_episodes_are_reported_not_dropped(self, tmp_path, make,
                                                        reason):
        make(tmp_path)
        (entry,) = list(get_adapter("ctrlworld_teleop").episodes(
            self._source(tmp_path, joint_columns=(0, 1, 2, 3, 4, 5, 7))))
        assert isinstance(entry, SkippedEpisode)
        assert reason in entry.reason

    def test_unreadable_metadata_is_reported(self, tmp_path):
        d = tmp_path / "t" / "episode_t__0"
        d.mkdir(parents=True)
        (d / "metadata.json").write_text("{not json")
        (d / "full_gt.mp4").write_bytes(b"")
        (entry,) = list(get_adapter("ctrlworld_teleop").episodes(
            self._source(tmp_path)))
        assert isinstance(entry, SkippedEpisode)
        assert "unreadable" in entry.reason


class TestCanonicalTree:
    def _tree(self, tmp_path, joint_source="real"):
        for split in ("train", "val"):
            (tmp_path / "videos" / split).mkdir(parents=True)
            (tmp_path / "annotation" / split).mkdir(parents=True)
            (tmp_path / "videos" / split / "ep0.mp4").write_bytes(b"")
            (tmp_path / "annotation" / split / "ep0.json").write_text(json.dumps({
                "joint_source": joint_source,
                "observation.state.joint_position": np.zeros((4, 7)).tolist(),
                "fps": 10.0, "scene_key": "task",
            }))
        return TrainSource(adapter="canonical", root=str(tmp_path))

    def test_reads_a_materialised_tree(self, tmp_path):
        entries = list(get_adapter("canonical").episodes(self._tree(tmp_path)))
        assert len(entries) == 2
        assert all(isinstance(e, RawEpisode) for e in entries)
        assert all(e.packed for e in entries)

    def test_synthetic_joints_are_refused(self, tmp_path):
        entries = list(get_adapter("canonical").episodes(
            self._tree(tmp_path, joint_source="synthetic")))
        assert all(isinstance(e, SkippedEpisode) for e in entries)


class TestRegistry:
    def test_both_shipped_adapters_are_discoverable(self):
        assert {"ctrlworld_teleop", "canonical"} <= set(available_adapters())

    def test_an_unknown_adapter_lists_what_exists(self):
        with pytest.raises(ValueError, match="available"):
            get_adapter("no_such_adapter")
